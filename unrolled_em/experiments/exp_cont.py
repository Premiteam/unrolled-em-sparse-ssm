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
SUPP = os.path.join(cm.HERE, "results", "v2_supp")
G3 = os.path.join(V2, "g3")
CFG = {"smoke": False}
GRID_R, GRID_A = (2, 3, 5, 8), (1.0, 4.0, 16.0)
P = "learned_tied_L4"
SQ_ARMS = ["SQbt3_ss", "SQbt1_ss", "SQbt2_ss", "SQpkg_ss", "SQbt3_exact"]
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
PARTS = ["valsel", "testII", "supp", "testI", "B", "C", "heat", "ge", "smax", "D", "timing"]
EXTRA_PARTS = ["timing2"]


def out_root():
    return os.path.join(cm.HERE, "results", "v2c_smoke" if CFG["smoke"] else "v2c")


def choice():
    f = os.path.join(out_root(), "choice.json")
    if not os.path.exists(f):
        raise SystemExit("先跑 --part valsel")
    c = cm.load(f)
    return int(c["r"]), float(c["amax0"])


def _set_smax(v):
    from unrolled_em.experiments import smax_exp as SX
    SX.set_smax(v)


def run_new(layers, sys, y, yv, lam, A0, taus, r, amax0, kind="exact", rich=None):
    cfg = {"r": r, "amax0": amax0, "safe": True}
    rec = cm.Recorder(sys, yv, taus[-1], kind, taus=taus, rich=rich,
                      stop_dist=0.01 if rich else None, stop_gap=1e-5 if rich else None)
    log = []
    t = time.time()
    EH.VARIANTS["_cur"] = cfg
    at.run(lambda: EH.run_hand(layers, sys, y, lam, A0, "_cur", log=log), sys, lam, 50, rec, A0)
    tags = [x[0] for x in log]
    out = {"hits": cm.hits_from_trace(rec.trace, taus), "q": rec.q, "sec": time.time() - t,
           "handover_at": next((i for i, x in enumerate(tags) if x == "handover"), None),
           "n_layer_steps": sum(1 for x in tags if not x.startswith("sq_") and x != "handover"),
           "layer_reject": sum(1 for x in tags if x in ("rho", "obj", "eta")),
           "sq_ok": sum(1 for x in tags if x == "sq_ok"), "sq_fail": sum(1 for x in tags if x in ("sq_obj", "sq_rho"))}
    if rich:
        out["extra"] = rec.extra
    return out


def run_sq(arm, sys, y, yv, lam, A0, taus, kind="exact"):
    from unrolled_em.experiments import exp_sqbt as SB
    rec = cm.Recorder(sys, yv, taus[-1], kind, taus=taus)
    t = time.time()
    at.run(SB.algo(arm, sys, y, lam, A0, []), sys, lam, SB.K_M, rec, A0)
    return {"hits": cm.hits_from_trace(rec.trace, taus), "q": rec.q, "sec": time.time() - t}


def _learned(path, L=cm.L_MAIN):
    from unrolled_em.experiments import method as me
    return me.load_layers(path)[:L]


def _done(part, name):
    return os.path.exists(os.path.join(out_root(), part, name))


def _save(part, name, obj):
    cm.dump(obj, os.path.join(out_root(), part, name))


def _rich(y, lam, A_R, F_R, A_Rss, F_Rss):
    return {"y": y, "lam": lam, "refs": {"exact": (A_R, F_R), "ss": (A_Rss, F_Rss)}, "own": "ss"}


def _h(res):
    return f"{res['hits'][I11]}/{res['hits'][I101]}"


def valsel_jobs():
    jobs = []
    for c in cm.CELLS:
        for j in range(cm.N_TRVA[c][1]):
            if os.path.exists(os.path.join(V2, "data", f"{c}_val_{j:02d}.npz")):
                jobs.append(("pred", c, j))
    for c in ("F32_500", "F64_500"):
        for j in range(cm.N_TRVA[c][1]):
            if os.path.exists(os.path.join(SUPP, "data", f"{c}_val_{j:02d}.npz")):
                jobs.append(("supp", c, j))
    if CFG["smoke"]:
        jobs = [jb for jb in jobs if jb[1] in ("F32_500", "P32_500") and jb[2] == 0]
    return sorted(jobs, key=lambda x: -cm.base_cell(x[1])["n"])


def valsel_job(args):
    lt, name, j = args
    fn = f"{lt}_{name}_{j:02d}.json"
    if _done("valsel", fn):
        return
    _set_smax(10.0)
    t0 = time.time()
    root = V2 if lt == "pred" else SUPP
    z = np.load(os.path.join(root, "data", f"{name}_val_{j:02d}.npz"))
    sys, y, yv, _ = cm.setup(cm.base_cell(name), "val", j)
    lam, A_R, A0 = float(z["lam"]), z["A_R"], z["A0"]
    taus = [cm.tau_of(ls.excess_nll(A_R, sys, yv), m) for m in cm.MULTS]
    F_R = -ls.loglik(A_R, sys.C, sys.q, sys.r, sys.P1, y) + lam * float(np.sum(np.abs(A_R)))
    from unrolled_em.experiments import run as R
    A_Rss, _, _ = cm.reference(sys, y, A0, lam, kind="ss", max_E=R.REF_SS_MAX_E, rel_tol=R.REF_SS_TOL)
    F_Rss = -cm.loglik_ss(A_Rss, sys.C, sys.q, sys.r, y) + lam * float(np.sum(np.abs(A_Rss)))
    rich = _rich(y, lam, A_R, F_R, A_Rss, F_Rss)
    layers = _learned(os.path.join(root, "models", f"{name}_learned.pt"))
    runs = {}
    for r in GRID_R:
        for a in GRID_A:
            runs[f"r{r}_a{int(a)}"] = run_new(layers, sys, y, yv, lam, A0, taus, r, a, rich=rich)
    _save("valsel", fn, {"lt": lt, "cell": name, "j": j, "runs": runs, "sec": time.time() - t0})
    print(f"  valsel {lt} {name} {j:02d}: {time.time() - t0:.0f}s", flush=True)


def _stop_hit(res):
    h = next((e[0] for e in (res.get("extra") or []) if e[1] <= 0.01 and e[3] <= 1e-5), None)
    return h if (h is not None and h <= cm.MAX_E) else None


def _gm(hs):
    v = [1 if h == 0 else (cm.FAIL if h is None else h) for h in hs]
    return float(np.exp(np.mean(np.log(v))))


def valsel_choose():
    fs = sorted(glob.glob(os.path.join(out_root(), "valsel", "*.json")))
    L = [cm.load(f) for f in fs]
    keys = list(L[0]["runs"])
    gm_stop = {k: _gm([_stop_hit(d["runs"][k]) for d in L]) for k in keys}
    gm_101 = {k: _gm([d["runs"][k]["hits"][I101] for d in L]) for k in keys}
    best = min(gm_stop.values())
    near = [k for k in keys if gm_stop[k] <= best * 1.01]
    parse = lambda k: (int(k.split("_")[0][1:]), float(k.split("_")[1][1:]))
    pick = sorted(near, key=lambda k: (-parse(k)[0], parse(k)[1]))[0]
    r, a = parse(pick)
    res = {"r": r, "amax0": a, "key": pick, "n_val": len(L), "gm_stop": gm_stop, "gm_alpha101": gm_101,
           "criterion": "geometric mean E-steps to reach distance 1% and objective gap 1e-5 (fail = 101); ties within 1%: larger r, then smaller amax0"}
    cm.dump(res, os.path.join(out_root(), "choice.json"))
    print("验证集（到 1% 且差距 1e-5 的几何平均 E 步 / α=1.01）：")
    for k in keys:
        print(f"  {k:8s} {gm_stop[k]:7.2f}  {gm_101[k]:6.2f}" + ("   ← 选定" if k == pick else ""))
    print(f"选定 r = {r}，α_max0 = {a:g}（{len(L)} 个验证系统）", flush=True)


def job(args):
    part, key = args
    _set_smax(10.0)
    r, a = choice()
    t0 = time.time()
    if part in ("testII", "supp"):
        name, j = key
        fn = f"{name}_{j:02d}.json"
        if _done(part, fn):
            return
        root = V2 if part == "testII" else SUPP
        old = cm.load(os.path.join(root, "testII", fn))
        z = np.load(os.path.join(root, "refs", "testII", fn[:-5] + ".npz"))
        sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
        lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
        rich = _rich(y, lam, z["A_R"], old["F_R"], z["A_Rss"], old["F_Rss"])
        arms = {P: run_new(_learned(os.path.join(root, "models", f"{name}_learned.pt")), sys, y, yv, lam, A0, taus, r, a, rich=rich)}
        src = {"F64_500": ("F32_500", "tied_from32_L4"), "F128_500": ("F64_500", "tied_from64_L4")}.get(name)
        if part == "testII" and src:
            arms[src[1]] = run_new(_learned(os.path.join(V2, "models", f"{src[0]}_learned.pt")), sys, y, yv, lam, A0, taus, r, a, rich=rich)
        msg = f"{P} {_h(arms[P])}（原 {_h(old['arms'][P])}）"
    elif part == "testI":
        from unrolled_em.experiments import method as me
        from unrolled_em.experiments import run as R
        from unrolled_em.experiments import exp_class as EC
        name, j = key
        fn = f"{name}_{j:02d}.json"
        if _done(part, fn):
            return
        old = cm.load(os.path.join(V2, "testI", fn))
        z = np.load(os.path.join(V2, "refs", "testI", fn[:-5] + ".npz"))
        sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1c", j)
        lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
        a_alt = 16.0 if a != 16.0 else 1.0
        full = me.load_layers(os.path.join(V2, "models", f"{name}_learned.pt"))
        arms = {}
        for L in range(1, cm.L_MAX + 1):
            arms[f"learned_tied_L{L}"] = run_new(full[:L], sys, y, yv, lam, A0, taus, r, a)
            arms[f"learned_alt_L{L}"] = run_new(full[:L], sys, y, yv, lam, A0, taus, r, a_alt)
        mods = R.load_models(name)
        for arm in ("scalar", "ait_t", "mu_t"):
            arms[f"{arm}_tied"] = run_new(mods[arm][:cm.L_ABL], sys, y, yv, lam, A0, taus, r, a)
        for rule in ("sec", "ait", "mu"):
            arms[f"{rule}0_tied"] = run_new(me.zero_layers(rule, cm.L_ABL, mods.get("mu_variant")), sys, y, yv, lam, A0, taus, r, a)
        for mode in ("class", "linear"):
            arms[mode] = run_new(EC.load_alt(os.path.join(V2, "classabl", "models", f"{name}_{mode}.pt")), sys, y, yv, lam, A0, taus, r, a)
        for arm in SQ_ARMS:
            arms[arm] = run_sq(arm, sys, y, yv, lam, A0, taus)
        msg = f"L4 {_h(arms[P])}（原 {_h(old['arms'][P])}）  scalar {_h(arms['scalar_tied'])}  class {_h(arms['class'])}  SQpkg {_h(arms['SQpkg_ss'])}"
    elif part == "B":
        src, shift, j = key
        fn = f"{src}_{shift}_{j:02d}.json"
        if _done(part, fn):
            return
        old = cm.load(os.path.join(V2, "B", fn))
        z = np.load(os.path.join(V2, "refs", "B", fn[:-5] + ".npz"))
        sys, y, yv, _ = cm.setup(cm.shift_cell(src, shift), f"g_{shift}", j)
        arms = {P: run_new(_learned(os.path.join(V2, "models", f"{src}_learned.pt")), sys, y, yv, float(z["lam"]), z["A0"], old["taus"], r, a)}
        msg = f"{P} {_h(arms[P])}（原 {_h(old['arms'][P])}）"
    elif part == "C":
        j = key
        cell = cm.big_cell(256)
        fn = f"{cell['name']}_{j:02d}.json"
        if _done(part, fn):
            return
        old = cm.load(os.path.join(V2, "C", fn))
        z = np.load(os.path.join(V2, "refs", "C", fn[:-5] + ".npz"))
        sys, y, yv, _ = cm.setup(cell, "test1g", j)
        lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
        arms = {P: run_new(_learned(os.path.join(V2, "models", "F128_500_learned.pt")), sys, y, yv, lam, A0, taus, r, a, kind="ss"),
                "native_F256": run_new(_learned(os.path.join(G3, "models", "F256_500_learned.pt")), sys, y, yv, lam, A0, taus, r, a, kind="ss")}
        msg = f"F128 模型 {_h(arms[P])}（原 {_h(old['arms'][P])}）  原生 {_h(arms['native_F256'])}"
    elif part == "D":
        j = key
        cell = cm.big_cell(512)
        fn = f"{cell['name']}_{j:02d}.json"
        if _done(part, fn):
            return
        old = cm.load(os.path.join(G3, "D", fn))
        z = np.load(os.path.join(G3, "refs", f"D_{fn[:-5]}.npz"))
        sys, y, yv, _ = cm.setup(cell, "test1g", j)
        arms = {"from_F256": run_new(_learned(os.path.join(G3, "models", "F256_500_learned.pt")), sys, y, yv, float(z["lam"]), z["A0"], old["taus"], r, a, kind="ss")}
        msg = f"from_F256 {_h(arms['from_F256'])}（原 {_h(old['arms']['from_F256'])}）"
    elif part == "heat":
        from unrolled_em.experiments import group3 as GG
        from unrolled_em.experiments import run as R
        hname, j = key
        fn = f"{hname}_{j:02d}.json"
        if _done(part, fn):
            return
        old = cm.load(os.path.join(G3, "H_A7", fn))
        z = np.load(os.path.join(G3, "refs", f"H_{hname}_{j:02d}.npz"))
        cell = GG.heat_cell(hname)
        sys, y, yv, _ = GG.heat_setup(cell, "test", j)
        lam, A0 = float(z["lam"]), z["A0"]
        A_Rss, _, _ = cm.reference(sys, y, A0, lam, kind="ss", max_E=R.REF_SS_MAX_E, rel_tol=R.REF_SS_TOL)
        F_Rss = -cm.loglik_ss(A_Rss, sys.C, sys.q, sys.r, y) + lam * float(np.sum(np.abs(A_Rss)))
        rich = _rich(y, lam, z["A_R"], old["F_R"], A_Rss, F_Rss)
        kind = GG.kind_for(cell)
        src = GG.TRANSFER_H[hname]
        arms = {P: run_new(GG.load_learned(hname)[:cm.L_MAIN], sys, y, yv, lam, A0, old["taus"], r, a, kind=kind, rich=rich),
                f"from_{src}": run_new(GG.load_learned(src)[:cm.L_MAIN], sys, y, yv, lam, A0, old["taus"], r, a, kind=kind, rich=rich)}
        arms["F_Rss_check"] = abs(F_Rss - old["F_Rss"]) / abs(old["F_Rss"])
        msg = f"{P} {_h(arms[P])}"
    elif part == "ge":
        from unrolled_em.experiments import run as R
        j = key
        fn = f"test_{j:02d}.json"
        if _done(part, fn):
            return
        old = cm.load(os.path.join(V2, "ge", fn))
        z = np.load(os.path.join(V2, "refs", "ge", fn[:-5] + ".npz"))
        sys, y, yv, _ = R.ge_setup("test", j)
        mods = R.load_models(R.GE["name"])
        lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
        arms = {P: run_new(mods["learned"][:cm.L_MAIN], sys, y, yv, lam, A0, taus, r, a),
                "scalar_tied": run_new(mods["scalar"][:cm.L_ABL], sys, y, yv, lam, A0, taus, r, a)}
        msg = f"{P} {_h(arms[P])}（原 {_h(old['arms'][P])}）"
    elif part == "smax":
        name, j = key
        fn = f"{name}_{j:02d}.json"
        if _done(part, fn):
            return
        old = cm.load(os.path.join(V2, "testII", fn))
        z = np.load(os.path.join(V2, "refs", "testII", fn[:-5] + ".npz"))
        sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
        _set_smax(20.0)
        try:
            arms = {"smax20": run_new(_learned(os.path.join(V2, "smax20", "models", f"{name}_learned.pt")), sys, y, yv,
                                      float(z["lam"]), z["A0"], old["taus"], r, a)}
        finally:
            _set_smax(10.0)
        msg = f"smax20 {_h(arms['smax20'])}"
    else:
        raise ValueError(part)
    _save(part, fn, {"key": list(key) if isinstance(key, tuple) else key, "r": r, "amax0": a, "arms": arms, "sec": time.time() - t0})
    print(f"  {part} {fn[:-5]}: {msg} | {time.time() - t0:.0f}s", flush=True)


def jobs_for(part):
    sm = CFG["smoke"]
    J = []
    if part in ("testII", "supp"):
        root = V2 if part == "testII" else SUPP
        cells = cm.CELLS if part == "testII" else ["F32_500", "F64_500"]
        for c in cells:
            for j in range(cm.N_TEST_II):
                if os.path.exists(os.path.join(root, "testII", f"{c}_{j:02d}.json")):
                    J.append((c, j))
    elif part == "testI":
        J = [(c, j) for c in cm.CELLS for j in range(cm.N_TEST_I) if os.path.exists(os.path.join(V2, "testI", f"{c}_{j:02d}.json"))]
    elif part == "B":
        J = [(s, sh, j) for s in cm.SOURCES_B for sh in cm.SHIFTS for j in range(cm.N_B)
             if os.path.exists(os.path.join(V2, "B", f"{s}_{sh}_{j:02d}.json"))]
    elif part == "C":
        J = [j for j in range(cm.N_C) if os.path.exists(os.path.join(V2, "C", f"F256_500_{j:02d}.json"))]
    elif part == "D":
        J = [j for j in range(cm.N_D) if os.path.exists(os.path.join(G3, "D", f"F512_500_{j:02d}.json"))]
    elif part == "heat":
        J = [(h, j) for h in ("HF64_500", "HF128_500") for j in range(30) if os.path.exists(os.path.join(G3, "H_A7", f"{h}_{j:02d}.json"))]
    elif part == "ge":
        J = [j for j in range(50) if os.path.exists(os.path.join(V2, "ge", f"test_{j:02d}.json"))]
    elif part == "smax":
        J = [(c, j) for c in ("P32_500", "P64_500", "F64_500") for j in range(cm.N_TEST_II)
             if os.path.exists(os.path.join(V2, "smax20", "testII", f"{c}_{j:02d}.json"))]
    if sm:
        J = J[:1] if part not in ("testII",) else [x for x in J if x[1] == 0][:3]
    return J


def _cost(part, k):
    n = {"F32_500": 32, "F64_500": 64, "F128_500": 128, "P32_500": 32, "P64_500": 64, "HF64_500": 64, "HF128_500": 128}
    if part in ("C", "D"):
        return 10 ** 9
    if isinstance(k, tuple):
        return n.get(k[0], 64) ** 3
    return 1


def _init(cfg):
    CFG.update(cfg)


def pmap(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init, initargs=(dict(CFG),)) as pool:
            yield from pool.imap_unordered(fn, jobs, chunksize=1)
    else:
        for jb in jobs:
            yield fn(jb)


def part_run(part, workers):
    if part == "valsel":
        J = valsel_jobs()
        todo = [jb for jb in J if not _done("valsel", f"{jb[0]}_{jb[1]}_{jb[2]:02d}.json")]
        print(f"[{time.strftime('%H:%M:%S')}] valsel：{len(J)} 个验证系统，待跑 {len(todo)}", flush=True)
        for _ in pmap(valsel_job, todo, workers):
            pass
        valsel_choose()
        return
    if part == "timing":
        part_timing()
        return
    if part == "timing2":
        part_timing2()
        return
    choice()
    J = sorted(jobs_for(part), key=lambda k: -_cost(part, k))
    print(f"[{time.strftime('%H:%M:%S')}] {part}：{len(J)} 个作业", flush=True)
    for _ in pmap(job, [(part, k) for k in J], workers):
        pass
    print(f"[{time.strftime('%H:%M:%S')}] {part} 完成", flush=True)


def part_timing():
    from unrolled_em.experiments import run as R
    r, a = choice()
    _set_smax(10.0)
    orig = R.algo_for

    def af(name, sys, y, lam, A0, models, log):
        if name == P:
            EH.VARIANTS["_cur"] = {"r": r, "amax0": a, "safe": True}
            return lambda: EH.run_hand(models["learned"][:cm.L_MAIN], sys, y, lam, A0, "_cur", log=log)
        return orig(name, sys, y, lam, A0, models, log)
    R.algo_for = af
    items = [("testII", c, j) for c in ("F32_500", "F64_500", "F128_500") for j in range(cm.N_TEST_II)]
    items += [("C", "F256_500", j) for j in range(cm.N_C)]
    if CFG["smoke"]:
        items = items[:1]
    bad = []
    for src, c, j in items:
        fn = f"{c}_{j:02d}.json"
        new_f = os.path.join(out_root(), src, fn)
        if _done("timing", fn) or not os.path.exists(new_f):
            continue
        rec = cm.load(os.path.join(V2, src, fn))
        hits = cm.load(new_f)["arms"][P]["hits"]
        A0 = np.load(os.path.join(V2, "refs", src, fn[:-5] + ".npz"))["A0"]
        cell = rec["cell"]
        sys, y, yv, _ = cm.setup(cell, rec["split"], rec["j"])
        layers = _learned(os.path.join(V2, "models", f"{cell['source']}_learned.pt"))
        metric = cm.Recorder(sys, yv, 0.0, cell.get("metric", "exact"))
        out = {"cell": cell["name"], "n": cell["n"], "j": j, P: {}}
        for m in R.TIMING_TARGETS:
            i = cm.MULTS.index(m)
            k = hits[i]
            if k is None:
                out[P][str(m)] = {"h": None, "t": None}
                continue
            ts = []
            for _ in range(R.REPEATS.get(cell["n"], 1)):
                x, t, cnt = R.replay(P, sys, y, rec["lam"], A0, {"learned": layers}, k)
                ts.append(t)
            assert cnt == k, (fn, m, cnt, k)
            ok = metric.metric(x) <= rec["taus"][i]
            if not ok:
                bad.append((fn, m))
            out[P][str(m)] = {"h": k, "t": float(np.median(ts)), "ts": ts, "reached": bool(ok)}
        _save("timing", fn, out)
        print(f"  计时 {fn[:-5]}: " + "/".join("-" if v["t"] is None else f"{v['t']:.3f}s" for v in out[P].values()), flush=True)
    R.algo_for = orig
    print(f"回放点未达标 {len(bad)} 项 {bad[:5]}", flush=True)


def part_timing2():
    from unrolled_em.experiments import run as R
    from unrolled_em.experiments import exp_sqbt as SB
    r, a = choice()
    _set_smax(10.0)
    orig = R.algo_for

    def af(name, sys, y, lam, A0, models, log):
        if name == P:
            EH.VARIANTS["_cur"] = {"r": r, "amax0": a, "safe": True}
            return lambda: EH.run_hand(models["learned"][:cm.L_MAIN], sys, y, lam, A0, "_cur", log=log)
        if name == "SQpkg_ss":
            return SB.algo(name, sys, y, lam, A0, log)
        return orig(name, sys, y, lam, A0, models, log)
    R.algo_for = af
    items = [("testII", c, j) for c in ("F32_500", "F64_500", "F128_500") for j in range(cm.N_TEST_II)]
    items += [("C", "F256_500", j) for j in range(cm.N_C)]
    if CFG["smoke"]:
        items = items[:1]
    arms = [P, "SQpkg_ss", "DAAREM_tun", "SQ_ss_K50"]
    bad = []
    for src, c, j in items:
        fn = f"{c}_{j:02d}.json"
        if _done("timing2", fn):
            continue
        rec = cm.load(os.path.join(V2, src, fn))
        hits = {P: cm.load(os.path.join(out_root(), src, fn))["arms"][P]["hits"],
                "SQpkg_ss": (cm.load(os.path.join(V2, "sqbt", "testII", fn))["arms"]["SQpkg_ss"]["hits"] if src == "testII"
                             else cm.load(os.path.join(V2, "sqpkg_rest", "C", fn))["hits"]),
                "DAAREM_tun": rec["arms"]["DAAREM_tun"]["hits"], "SQ_ss_K50": rec["arms"]["SQ_ss_K50"]["hits"]}
        A0 = np.load(os.path.join(V2, "refs", src, fn[:-5] + ".npz"))["A0"]
        cell = rec["cell"]
        sys, y, yv, _ = cm.setup(cell, rec["split"], rec["j"])
        models = R.load_models(cell["source"])
        models["learned"] = models["learned"][:cm.L_MAIN]
        metric = cm.Recorder(sys, yv, 0.0, cell.get("metric", "exact"))
        out = {"cell": cell["name"], "n": cell["n"], "j": j, "arms": {x: {} for x in arms}}
        for m in R.TIMING_TARGETS:
            i = cm.MULTS.index(m)
            for rep in range(R.REPEATS.get(cell["n"], 1)):
                for arm in arms:
                    k = hits[arm][i]
                    slot = out["arms"][arm].setdefault(str(m), {"h": k, "ts": []})
                    if k is None:
                        continue
                    x, t, cnt = R.replay(arm, sys, y, rec["lam"], A0, models, k)
                    assert cnt == k, (fn, arm, m, cnt, k)
                    slot["ts"].append(t)
                    if rep == 0 and metric.metric(x) > rec["taus"][i]:
                        bad.append((fn, arm, m))
            for arm in arms:
                sl = out["arms"][arm][str(m)]
                sl["t"] = float(np.median(sl["ts"])) if sl["ts"] else None
        _save("timing2", fn, out)
        print(f"  计时2 {fn[:-5]}: " + "  ".join(f"{x} " + "/".join("-" if v["t"] is None else f"{v['t']:.3f}s"
                                                                  for v in out["arms"][x].values()) for x in arms), flush=True)
    R.algo_for = orig
    print(f"回放点未达标 {len(bad)} 项 {bad[:5]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=PARTS + EXTRA_PARTS + ["all", "after_valsel"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    CFG.update(smoke=a.smoke)
    parts = PARTS if a.part == "all" else (PARTS[1:] if a.part == "after_valsel" else [a.part])
    for p in parts:
        part_run(p, a.workers)


if __name__ == "__main__":
    main()
