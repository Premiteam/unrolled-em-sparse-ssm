from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import glob
import time

import numpy as np

from unrolled_em import anytime as at
from unrolled_em import core as ls
from unrolled_em.experiments import common as cm
from unrolled_em.experiments import exp_hand as EH

V2 = os.path.join(cm.HERE, "results", "v2")
V2C = os.path.join(cm.HERE, "results", "v2c")
G3 = os.path.join(V2, "g3")
P = "learned_tied_L4"
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
CFG = {"smoke": False}
STOP_EPS = (1e-3, 1e-4, 1e-5, 1e-6)
SR_ARMS = [P, "EM_ss_K50", "SQpkg_ss", "DAAREM_tun"]


def out_root():
    return os.path.join(cm.HERE, "results", "review_b_smoke" if CFG["smoke"] else "review_b")


def _f(part, name):
    return os.path.join(out_root(), part, name)


def choice():
    c = cm.load(os.path.join(V2C, "choice.json"))
    return int(c["r"]), float(c["amax0"])


def _ok(X):
    return bool(np.all(np.isfinite(X))) and float(np.max(np.abs(np.linalg.eigvals(X)))) < 1.0


def _layers(path):
    from unrolled_em.experiments import method as me
    return me.load_layers(path)[:cm.L_MAIN]


def _test_ii(name, j):
    old = cm.load(os.path.join(V2, "testII", f"{name}_{j:02d}.json"))
    z = np.load(os.path.join(V2, "refs", "testII", f"{name}_{j:02d}.npz"))
    sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
    lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
    refs = {"exact": (z["A_R"], old["F_R"]), "ss": (z["A_Rss"], old["F_Rss"])}
    return old, sys, y, yv, lam, A0, taus, refs


def _rich(y, lam, refs, own):
    return {"y": y, "lam": lam, "refs": refs, "own": own}


def _arm_out(rec, t, extra=None):
    o = {"hits": cm.hits_from_trace(rec.trace, rec.taus), "q": rec.q, "sec": time.time() - t,
         "trace": [(int(c), float(m)) for c, m in rec.trace]}
    if rec.rich is not None:
        o["extra"] = rec.extra
    if extra:
        o.update(extra)
    return o


def run_switch(layers, sys, y, lam, A0, r, amax0, max_E=cm.MAX_E, log=None):
    from unrolled_em.experiments import method as me
    log = [] if log is None else log
    eta = me.ETA
    orig = me.estep
    exact = lambda X, s, yy: ls.estep_exact(X, s.C, s.q, s.r, s.P1, yy)
    S = {"at": None}
    try:
        A = A0.copy()
        st = me.estep(A, sys, y)
        n = 1

        def maybe_switch(failed, A, st, n):
            if failed and S["at"] is None:
                S["at"] = n
                me.estep = exact
                log.append("switch")
                st = me.estep(A, sys, y)
                n += 1
            return st, n

        Rp = np.zeros_like(A)
        L = len(layers)
        k, consec = 0, 0
        handed = False
        while n + 1 <= max_E:
            if k >= L and consec >= r:
                handed = True
                break
            o = me.layer_step(layers[min(k, L - 1)], A, st, Rp, sys, y, lam)
            n += o["nE"]
            A, st, Rp = o["A"], o["st"], o["Rp"]
            log.append(o["reject"] or "ok")
            if k >= L:
                consec = consec + 1 if o["reject"] else 0
            k += 1
            failed = bool(o["reject"]) and not (o["F_new"] <= o["F_old"] - eta * o["delta"])
            st, n = maybe_switch(failed, A, st, n)
        if not handed:
            return S
        log.append("handover")
        amax = amax0
        while n < max_E:
            F_A = me.fobj(A, st, lam)
            B1 = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
            dA = max(me.surrogate(A, st["Phi"], st["Psi"], sys.q, lam) - me.surrogate(B1, st["Phi"], st["Psi"], sys.q, lam), 0.0)
            st1 = EH._estep(me, B1, sys, y); n += 1
            if st1 is None or n >= max_E:
                return S
            B2 = ls.mstep(B1, st1["Phi"], st1["Psi"], sys.q, lam, K=50)
            rr_, vv_ = B1 - A, B2 - 2 * B1 + A
            rr, vv = float(np.sum(rr_ * rr_)), float(np.sum(vv_ * vv_))
            a = float(np.sqrt(rr / vv)) if vv > 0 else 1.0
            a = max(1.0, min(amax, a))
            acc, why = False, "em"
            if abs(a - 1.0) > 0.01:
                X = A + 2 * a * rr_ + a * a * vv_
                why = "rho"
                if _ok(X):
                    sX = EH._estep(me, X, sys, y); n += 1
                    why = "obj"
                    if sX is not None and n < max_E:
                        Xt = ls.mstep(X, sX["Phi"], sX["Psi"], sys.q, lam, K=50)
                        if _ok(Xt):
                            sXt = EH._estep(me, Xt, sys, y); n += 1
                            if sXt is not None:
                                acc = me.fobj(Xt, sXt, lam) <= F_A - eta * dA
            if acc:
                A, st = Xt, sXt
                log.append("sq_ok")
            else:
                if n >= max_E:
                    return S
                s2 = EH._estep(me, B2, sys, y); n += 1
                if s2 is None:
                    return S
                F1, F2 = me.fobj(B1, st1, lam), me.fobj(B2, s2, lam)
                d1 = max(me.surrogate(B1, st1["Phi"], st1["Psi"], sys.q, lam) - me.surrogate(B2, st1["Phi"], st1["Psi"], sys.q, lam), 0.0)
                failed = not (F1 <= F_A - eta * dA) or not (F2 <= F1 - eta * d1)
                A, st = B2, s2
                log.append(f"sq_{why}")
                if why != "em":
                    if a == amax:
                        amax = max(amax0, amax / 4.0)
                    a = 1.0
                st, n = maybe_switch(failed, A, st, n)
            if a == amax:
                amax *= 4.0
            if rr == 0.0:
                return S
        return S
    finally:
        me.estep = orig


def job_switch(key):
    name, j = key
    fn = f"{name}_{j:02d}.json"
    if os.path.exists(_f("switch", fn)):
        return
    old, sys, y, yv, lam, A0, taus, refs = _test_ii(name, j)
    r, a = choice()
    layers = _layers(os.path.join(V2, "models", f"{name}_learned.pt"))
    out = {"cell": name, "j": j, "taus": taus, "arms": {}}
    rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus, rich=_rich(y, lam, refs, "exact"), stop_dist=0.01, stop_gap=1e-5)
    log, t = [], time.time()
    S = {}
    at.run(lambda: S.update(run_switch(layers, sys, y, lam, A0, r, a, log=log) or {}), sys, lam, 50, rec, A0)
    out["arms"]["switch"] = _arm_out(rec, t, {"switched_at": S.get("at"), "log": log})
    if j == 0:
        rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus)
        t = time.time()
        EH.VARIANTS["_cur"] = {"r": r, "amax0": a, "safe": True}
        at.run(lambda: EH.run_hand(layers, sys, y, lam, A0, "_cur"), sys, lam, 50, rec, A0)
        hits = cm.hits_from_trace(rec.trace, taus)
        paper = cm.load(os.path.join(V2C, "testII", fn))["arms"][P]["hits"]
        out["repro"] = {"hits": hits, "paper": paper, "same": hits == paper}
    cm.dump(out, _f("switch", fn))
    h = out["arms"]["switch"]["hits"]
    print(f"  switch {name} {j:02d}: α1.1/1.01 = {h[I11]}/{h[I101]}  切换于 {S.get('at')}"
          + (f"  复现 {out['repro']['same']}" if "repro" in out else ""), flush=True)


def qn_zal(sys, y, A0, lam, q=1, kind="ss", max_E=cm.MAX_E):
    from unrolled_em.experiments import exp_sqbt as SB
    E = SB._E(sys, y, lam, kind)
    x = A0.copy()
    fx = E.obj(x)
    U, V = [], []
    while E.n < max_E:
        n0 = E.n
        f1 = E.F(x)
        if f1 is None:
            break
        f2 = E.F(f1)
        if f2 is None:
            break
        u, v = (f1 - x).ravel(), (f2 - f1).ravel()
        U.append(u); V.append(v)
        U, V = U[-q:], V[-q:]
        Um, Vm = np.stack(U, 1), np.stack(V, 1)
        acc = False
        try:
            coef = np.linalg.solve(Um.T @ Um - Um.T @ Vm, Um.T @ u)
            xn = (f1.ravel() + Vm @ coef).reshape(x.shape)
            if _ok(xn):
                fn_ = E.obj(xn)
                acc = bool(np.isfinite(fn_) and fn_ <= fx)
        except np.linalg.LinAlgError:
            pass
        if acc:
            x, fx = xn, fn_
        else:
            x, fx = f2, E.obj(f2)
        if E.n == n0:
            break
    return x


def _new_arm(arm, sys, y, lam, A0, max_E=cm.MAX_E):
    from unrolled_em.experiments import exp_sqbt as SB
    if arm == "SQpkg16_ss":
        return lambda: SB.sq_pkg(sys, y, A0, lam, kind="ss", max_E=max_E, step_max0=16.0)
    if arm == "SQpkg_ss":
        return lambda: SB.sq_pkg(sys, y, A0, lam, kind="ss", max_E=max_E)
    if arm.startswith("QN"):
        q = int(arm[2:].split("_")[0])
        return lambda: qn_zal(sys, y, A0, lam, q=q, kind="ss", max_E=max_E)
    raise ValueError(arm)


def job_arms(key, part, arms):
    name, j = key
    fn = f"{name}_{j:02d}.json"
    if os.path.exists(_f(part, fn)):
        return
    old, sys, y, yv, lam, A0, taus, refs = _test_ii(name, j)
    out = {"cell": name, "j": j, "taus": taus, "arms": {}}
    for arm in arms:
        rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus, rich=_rich(y, lam, refs, "ss"), stop_dist=0.01, stop_gap=1e-5)
        t = time.time()
        at.run(_new_arm(arm, sys, y, lam, A0), sys, lam, 50, rec, A0)
        out["arms"][arm] = _arm_out(rec, t)
    cm.dump(out, _f(part, fn))
    print(f"  {part} {name} {j:02d}: " + "  ".join(f"{a} {o['hits'][I11]}/{o['hits'][I101]}" for a, o in out["arms"].items()), flush=True)


def job_stoprule(key):
    from unrolled_em.experiments import run as R
    name, j = key
    fn = f"{name}_{j:02d}.json"
    if os.path.exists(_f("stoprule", fn)):
        return
    old, sys, y, yv, lam, A0, taus, refs = _test_ii(name, j)
    r, a = choice()
    models = R.load_models(name)
    models["learned"] = models["learned"][:cm.L_MAIN]
    out = {"cell": name, "j": j, "taus": taus, "arms": {}}
    for arm in SR_ARMS:
        rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus, rich=_rich(y, lam, refs, "ss"), stop_dist=1e-3, stop_gap=1e-7)
        t = time.time()
        if arm == P:
            EH.VARIANTS["_cur"] = {"r": r, "amax0": a, "safe": True}
            fn_run = lambda: EH.run_hand(models["learned"], sys, y, lam, A0, "_cur")
        elif arm == "SQpkg_ss":
            fn_run = _new_arm(arm, sys, y, lam, A0)
        else:
            fn_run = R.algo_for(arm, sys, y, lam, A0, models, [])
        at.run(fn_run, sys, lam, 50, rec, A0)
        out["arms"][arm] = _arm_out(rec, t)
    cm.dump(out, _f("stoprule", fn))
    print(f"  stoprule {name} {j:02d}: " + "  ".join(f"{x} {len(o['trace'])}" for x, o in out["arms"].items()), flush=True)


def job_d512(j):
    cell = cm.big_cell(512)
    fn = f"{cell['name']}_{j:02d}.json"
    if os.path.exists(_f("d512", fn)):
        return
    old = cm.load(os.path.join(G3, "D", fn))
    z = np.load(os.path.join(G3, "refs", f"D_{fn[:-5]}.npz"))
    sys, y, yv, _ = cm.setup(cell, "test1g", j)
    lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
    rec = cm.Recorder(sys, yv, taus[-1], "ss", taus=taus)
    t = time.time()
    cap = 6 if CFG["smoke"] else cm.MAX_E
    at.run(_new_arm("SQpkg_ss", sys, y, lam, A0, max_E=cap), sys, lam, 50, rec, A0)
    out = {"cell": cell["name"], "j": j, "taus": taus, "arms": {"SQpkg_ss": _arm_out(rec, t)}}
    cm.dump(out, _f("d512", fn))
    h = out["arms"]["SQpkg_ss"]["hits"]
    print(f"  d512 {j:02d}: SQpkg_ss {h[I11]}/{h[I101]}", flush=True)


def part_t512():
    from unrolled_em.experiments import run as R
    r, a = choice()
    cell = cm.big_cell(512)
    orig = R.algo_for
    lay = _layers(os.path.join(G3, "models", "F256_500_learned.pt"))
    da_cfg = R.load_models("F128_500")["daarem"]

    def af(name, sys, y, lam, A0, models, log):
        if name == P:
            EH.VARIANTS["_cur"] = {"r": r, "amax0": a, "safe": True}
            return lambda: EH.run_hand(lay, sys, y, lam, A0, "_cur", log=log)
        if name == "SQpkg_ss":
            return _new_arm(name, sys, y, lam, A0)
        return orig(name, sys, y, lam, A0, models, log)
    R.algo_for = af
    try:
        js = [0] if CFG["smoke"] else list(range(cm.N_D))
        for j in js:
            fn = f"{cell['name']}_{j:02d}.json"
            if os.path.exists(_f("t512", fn)) or not os.path.exists(_f("d512", fn)):
                continue
            old = cm.load(os.path.join(G3, "D", fn))
            hits = {P: cm.load(os.path.join(V2C, "D", fn))["arms"]["from_F256"]["hits"],
                    "SQpkg_ss": cm.load(_f("d512", fn))["arms"]["SQpkg_ss"]["hits"],
                    "DAAREM_tun": old["arms"]["DAAREM_tun"]["hits"]}
            z = np.load(os.path.join(G3, "refs", f"D_{fn[:-5]}.npz"))
            sys, y, yv, _ = cm.setup(cell, "test1g", j)
            lam, A0 = float(z["lam"]), z["A0"]
            out = {"cell": cell["name"], "j": j, "arms": {}}
            for m in ((1.1,) if CFG["smoke"] else (1.1, 1.01)):
                i = cm.MULTS.index(m)
                for arm in hits:
                    k = hits[arm][i]
                    slot = out["arms"].setdefault(arm, {})
                    if k is None:
                        slot[str(m)] = {"h": None, "t": None}
                        continue
                    _, tt, cnt = R.replay(arm, sys, y, lam, A0, {"learned": lay, "daarem": da_cfg}, k)
                    slot[str(m)] = {"h": k, "t": tt, "cnt": cnt}
            cm.dump(out, _f("t512", fn))
            print(f"  t512 {j:02d}: " + "  ".join(f"{x} " + "/".join("-" if v['t'] is None else f"{v['t']:.1f}s" for v in o.values())
                                                for x, o in out["arms"].items()), flush=True)
    finally:
        R.algo_for = orig


def jobs_for(part):
    if part in ("switch", "sq16", "qn", "stoprule"):
        J = [(c, j) for c in cm.CELLS for j in range(cm.N_TEST_II) if os.path.exists(os.path.join(V2, "testII", f"{c}_{j:02d}.json"))]
        J.sort(key=lambda x: -cm.base_cell(x[0])["n"])
    elif part == "d512":
        J = [j for j in range(cm.N_D) if os.path.exists(os.path.join(G3, "D", f"F512_500_{j:02d}.json"))]
    else:
        J = []
    if CFG["smoke"]:
        J = [x for x in J if (x[0] == "F32_500" and x[1] == 0)] if part != "d512" else J[:1]
    return J


def _work(args):
    part, key, smoke = args
    CFG["smoke"] = smoke
    if part == "switch":
        job_switch(key)
    elif part == "sq16":
        job_arms(key, "sq16", ["SQpkg16_ss"])
    elif part == "qn":
        job_arms(key, "qn", ["QN1_ss", "QN3_ss"])
    elif part == "stoprule":
        job_stoprule(key)
    elif part == "d512":
        job_d512(key)
    return key


def part_run(part, workers):
    if part == "t512":
        part_t512()
        return
    jobs = [(part, k, CFG["smoke"]) for k in jobs_for(part)]
    print(f"[{time.strftime('%H:%M:%S')}] {part}：{len(jobs)} 个作业", flush=True)
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for _ in pool.imap_unordered(_work, jobs):
                pass
    else:
        for jb in jobs:
            _work(jb)


def _pool_arms(c, j):
    fn = f"{c}_{j:02d}.json"
    arms = dict(cm.load(os.path.join(V2, "testII", fn))["arms"])
    arms.update({a: cm.load(os.path.join(V2, "sqbt", "testII", fn))["arms"][a] for a in EH.NEW_SQ})
    arms[P] = cm.load(os.path.join(V2C, "testII", fn))["arms"][P]
    return arms


def _xh(arm, col, thr):
    return EH._xh(arm, col, thr)


def _gstats(H, me, pool):
    st = {a: cm.ratio_stats(H[a], H[me]) for a in pool}
    o = min(pool, key=lambda a: st[a]["G"])
    return {"G": st[o]["G"], "ci": st[o]["ci"], "vs": o, "median": cm.med(H[me]), "fail": sum(x is None for x in H[me])}


def summary():
    res = {}
    S = {}
    for c in cm.CELLS:
        fs = sorted(glob.glob(_f("switch", f"{c}_*.json")))
        if not fs:
            continue
        rows = [(cm.load(f), _pool_arms(c, cm.load(f)["j"])) for f in fs]
        e = {"n": len(rows), "switched": sum(d["arms"]["switch"]["switched_at"] is not None for d, _ in rows),
             "switch_at_median": (lambda v: cm.med(v) if v else None)([d["arms"]["switch"]["switched_at"] for d, _ in rows if d["arms"]["switch"]["switched_at"] is not None])}
        for met, i in (("a11", I11), ("a101", I101)):
            H = {a: [arms[a]["hits"][i] for _, arms in rows] for a in EH.POOL + [P]}
            H["switch"] = [d["arms"]["switch"]["hits"][i] for d, _ in rows]
            e[met] = {"switch": _gstats(H, "switch", EH.POOL), "paper": _gstats(H, P, EH.POOL),
                      "paper_over_switch": cm.ratio_stats(H[P], H["switch"])}
        for met, spec in (("dist1_exact", (1, 0.01)), ("gap5_exact", (3, 1e-5))):
            hs = [_xh(d["arms"]["switch"], *spec) for d, _ in rows]
            e[met] = {"median": cm.med(hs), "fail": sum(x is None for x in hs)}
        repro = [d["repro"]["same"] for d, _ in rows if "repro" in d]
        e["repro_all_same"] = all(repro) if repro else None
        S[c] = e
    res["switch"] = S
    for part, arms in (("sq16", ["SQpkg16_ss"]), ("qn", ["QN1_ss", "QN3_ss"])):
        T = {}
        for c in cm.CELLS:
            fs = sorted(glob.glob(_f(part, f"{c}_*.json")))
            if not fs:
                continue
            rows = []
            for f in fs:
                d = cm.load(f)
                base = _pool_arms(c, d["j"])
                base.update(d["arms"])
                rows.append(base)
            e = {"n": len(rows)}
            for met, spec in (("a11", I11), ("a101", I101), ("dist1", (1, 0.01)), ("gap5", (3, 1e-5))):
                if isinstance(spec, int):
                    H = {a: [r[a]["hits"][spec] for r in rows] for a in EH.POOL + arms + [P]}
                    pool_old = EH.POOL
                else:
                    H = {a: [_xh(r[a], *spec) for r in rows] for a in EH.POOL_SS + arms + [P]}
                    pool_old = EH.POOL_SS
                e[met] = {"paper_vs_old_pool": _gstats(H, P, pool_old), "paper_vs_pool_plus_new": _gstats(H, P, pool_old + arms),
                          "new_medians": {a: cm.med(H[a]) for a in arms}, "new_fail": {a: sum(x is None for x in H[a]) for a in arms}}
            T[c] = e
        res[part] = T
    SR = {}
    for c in cm.CELLS:
        fs = sorted(glob.glob(_f("stoprule", f"{c}_*.json")))
        if not fs:
            continue
        e = {}
        for arm in SR_ARMS:
            for eps in STOP_EPS:
                cs, als, ds, ok101, ok1 = [], [], [], 0, 0
                for f in fs:
                    d = cm.load(f)
                    taus = d["taus"]
                    ER = (taus[I11] - 1e-4) / 1.1
                    ex = d["arms"][arm].get("extra") or []
                    tr = dict(d["arms"][arm]["trace"])
                    stop = None
                    g = {e[0]: e for e in ex}
                    for c_ in sorted(g):
                        if c_ >= 4 and (c_ - 3) in g and abs(g[c_][3] - g[c_ - 3][3]) <= eps:
                            stop = g[c_]
                            break
                    if stop is None and ex:
                        stop = ex[-1]
                    if stop is None:
                        continue
                    c_, m_ = stop[0], tr.get(stop[0])
                    cs.append(c_)
                    if m_ is not None and ER > 0:
                        als.append(m_ / ER)
                        ok101 += m_ <= taus[I101]
                    ds.append(stop[1])
                    ok1 += stop[1] <= 0.01
                e[f"{arm}/{eps:g}"] = {"E_median": cm.med(cs), "alpha_median": float(np.median(als)) if als else None,
                                       "dist_median": float(np.median(ds)) if ds else None,
                                       "frac_alpha101": ok101 / max(len(fs), 1), "frac_dist1": ok1 / max(len(fs), 1)}
        SR[c] = e
    res["stoprule"] = SR
    D = {}
    fs = sorted(glob.glob(_f("d512", "*.json")))
    if fs:
        rows = []
        for f in fs:
            fn = os.path.basename(f)
            arms = dict(cm.load(os.path.join(G3, "D", fn))["arms"])
            arms["SQpkg_ss"] = cm.load(f)["arms"]["SQpkg_ss"]
            arms[P] = cm.load(os.path.join(V2C, "D", fn))["arms"]["from_F256"]
            rows.append(arms)
        pool = [a for a in rows[0] if a not in (P, "from_F256") and isinstance(rows[0][a], dict) and "hits" in rows[0][a]]
        for met, i in (("a11", I11), ("a101", I101)):
            H = {a: [r[a]["hits"][i] for r in rows] for a in pool + [P]}
            D[met] = {"paper": _gstats(H, P, pool), "medians": {a: cm.med(H[a]) for a in pool + [P]}}
    tf = sorted(glob.glob(_f("t512", "*.json")))
    if tf:
        TT = [cm.load(f) for f in tf]
        for m in ("1.1", "1.01"):
            for o in ("SQpkg_ss", "DAAREM_tun"):
                ok = [t for t in TT if t["arms"][P].get(m, {}).get("t") and t["arms"][o].get(m, {}).get("t")]
                if ok:
                    D[f"time_{m}_{o}"] = {"G_T": float(np.exp(np.mean([np.log(t["arms"][o][m]["t"] / t["arms"][P][m]["t"]) for t in ok]))),
                                          "median_t_prop": float(np.median([t["arms"][P][m]["t"] for t in ok])),
                                          "median_t_other": float(np.median([t["arms"][o][m]["t"] for t in ok])), "n": len(ok)}
    res["n512"] = D
    cm.dump(res, os.path.join(out_root(), "summary.json"))
    import json
    print(json.dumps(res, indent=1, default=str)[:20000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["switch", "sq16", "qn", "stoprule", "d512", "t512", "summary", "smoke"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.part == "smoke":
        CFG["smoke"] = True
        for p in ("switch", "sq16", "qn", "stoprule", "d512", "t512"):
            part_run(p, 1)
        summary()
        return
    if a.part == "summary":
        summary()
        return
    part_run(a.part, a.workers)


if __name__ == "__main__":
    main()
