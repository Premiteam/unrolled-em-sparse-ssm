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
from unrolled_em.experiments import exp_sqbt as SB

V2 = os.path.join(cm.HERE, "results", "v2")
CFG = {"tag": ""}
ARM = "SQpkg_ss"
P = "learned_tied_L4"
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)


def out_dir():
    return os.path.join(V2, "sqpkg_rest" + (f"_{CFG['tag']}" if CFG["tag"] else ""))


def _run_pkg(sys, y, yv, lam, A0, taus, kind, rich=None):
    rec = cm.Recorder(sys, yv, taus[-1], kind, taus=taus, rich=rich,
                      stop_dist=0.01 if rich else None, stop_gap=1e-5 if rich else None)
    log = []
    t = time.time()
    at.run(SB.algo(ARM, sys, y, lam, A0, log), sys, lam, SB.K_M, rec, A0)
    return {"hits": cm.hits_from_trace(rec.trace, taus), "q": rec.q, "extra": rec.extra or None, "sec": time.time() - t,
            "extrap_rate": float(np.mean([e["extrap"] for e in log])) if log else None}


def _done(part, name):
    return os.path.exists(os.path.join(out_dir(), part, name))


def _save(part, name, obj):
    cm.dump(obj, os.path.join(out_dir(), part, name))


def job(args):
    part, key = args
    t0 = time.time()
    if part == "B":
        src, shift, j = key
        name = f"{src}_{shift}_{j:02d}.json"
        if _done(part, name):
            return
        old = cm.load(os.path.join(V2, "B", name))
        z = np.load(os.path.join(V2, "refs", "B", name[:-5] + ".npz"))
        sys, y, yv, _ = cm.setup(cm.shift_cell(src, shift), f"g_{shift}", j)
        res = _run_pkg(sys, y, yv, float(z["lam"]), z["A0"], old["taus"], "exact")
    elif part == "C":
        j = key
        cell = cm.big_cell(256)
        name = f"{cell['name']}_{j:02d}.json"
        if _done(part, name):
            return
        old = cm.load(os.path.join(V2, "C", name))
        z = np.load(os.path.join(V2, "refs", "C", name[:-5] + ".npz"))
        sys, y, yv, _ = cm.setup(cell, "test1g", j)
        res = _run_pkg(sys, y, yv, float(z["lam"]), z["A0"], old["taus"], "ss")
    elif part == "ge":
        from unrolled_em.experiments import run as R
        j = key
        name = f"test_{j:02d}.json"
        if _done(part, name):
            return
        old = cm.load(os.path.join(V2, "ge", name))
        z = np.load(os.path.join(V2, "refs", "ge", name[:-5] + ".npz"))
        sys, y, yv, _ = R.ge_setup("test", j)
        res = _run_pkg(sys, y, yv, float(z["lam"]), z["A0"], old["taus"], "exact")
    elif part == "heat":
        from unrolled_em.experiments import group3 as G3
        from unrolled_em.experiments import run as R
        hname, j = key
        name = f"{hname}_{j:02d}.json"
        if _done(part, name):
            return
        old = cm.load(os.path.join(V2, "g3", "H_A7", name))
        z = np.load(os.path.join(V2, "g3", "refs", f"H_{hname}_{j:02d}.npz"))
        cell = G3.heat_cell(hname)
        sys, y, yv, _ = G3.heat_setup(cell, "test", j)
        lam, A0 = float(z["lam"]), z["A0"]
        A_Rss, nE_Rss, _ = cm.reference(sys, y, A0, lam, kind="ss", max_E=R.REF_SS_MAX_E, rel_tol=R.REF_SS_TOL)
        F_Rss = -cm.loglik_ss(A_Rss, sys.C, sys.q, sys.r, y) + lam * float(np.sum(np.abs(A_Rss)))
        rich = {"y": y, "lam": lam, "refs": {"exact": (z["A_R"], old["F_R"]), "ss": (A_Rss, F_Rss)}, "own": "ss"}
        res = _run_pkg(sys, y, yv, lam, A0, old["taus"], G3.kind_for(cell), rich=rich)
        res["F_Rss_check"] = abs(F_Rss - old["F_Rss"]) / abs(old["F_Rss"])
    else:
        raise ValueError(part)
    res["sec_job"] = time.time() - t0
    _save(part, name, res)
    h = res["hits"]
    print(f"  {part} {name[:-5]}: SQpkg {h[I11]}/{h[I101]} | {res['sec_job']:.0f}s", flush=True)


def jobs_all(smoke=False):
    J = []
    for src in cm.SOURCES_B:
        for shift in cm.SHIFTS:
            for j in ([0] if smoke else range(cm.N_B)):
                if os.path.exists(os.path.join(V2, "B", f"{src}_{shift}_{j:02d}.json")):
                    J.append(("B", (src, shift, j)))
                if smoke:
                    break
            if smoke:
                break
    J += [("C", j) for j in ([0] if smoke else range(cm.N_C))]
    J += [("ge", j) for j in ([0] if smoke else range(50))]
    for hn in ("HF64_500", "HF128_500"):
        J += [("heat", (hn, j)) for j in ([0] if smoke else range(30))]
    cost = {"C": 6, "heat": 2, "B": 1, "ge": 0}
    J.sort(key=lambda x: -cost[x[0]])
    return J


def _init(cfg):
    CFG.update(cfg)


def pmap(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init, initargs=(dict(CFG),)) as pool:
            yield from pool.imap_unordered(fn, jobs)
    else:
        for jb in jobs:
            yield fn(jb)


def part_runs(workers, smoke=False):
    J = jobs_all(smoke)
    print(f"补跑 SQpkg：{len(J)} 个作业", flush=True)
    for _ in pmap(job, J, workers):
        pass


def part_timing(smoke=False):
    from unrolled_em.experiments import run as R
    orig = R.algo_for

    def af(name, sys, y, lam, A0, models, log):
        if name == ARM:
            return SB.algo(ARM, sys, y, lam, A0, log)
        return orig(name, sys, y, lam, A0, models, log)
    R.algo_for = af
    items = []
    for c in ("F32_500", "F64_500", "F128_500"):
        for j in ([0] if smoke else range(cm.N_TEST_II)):
            items.append(("II", c, j))
    for j in ([0] if smoke else range(cm.N_C)):
        items.append(("C", "F256_500", j))
    bad = []
    for src, c, j in items:
        name = f"{c}_{j:02d}.json"
        if _done("timing", name):
            continue
        need = os.path.join(V2, "sqbt", "testII", name) if src == "II" else os.path.join(out_dir(), "C", name)
        if not os.path.exists(need):
            print(f"  缺少 {need}，跳过", flush=True)
            continue
        if src == "II":
            rec = cm.load(os.path.join(V2, "testII", name))
            hits = cm.load(os.path.join(V2, "sqbt", "testII", name))["arms"][ARM]["hits"]
            A0 = np.load(os.path.join(V2, "refs", "testII", name[:-5] + ".npz"))["A0"]
        else:
            rec = cm.load(os.path.join(V2, "C", name))
            hits = cm.load(os.path.join(out_dir(), "C", name))["hits"]
            A0 = np.load(os.path.join(V2, "refs", "C", name[:-5] + ".npz"))["A0"]
        cell = rec["cell"]
        sys, y, yv, _ = cm.setup(cell, rec["split"], rec["j"])
        metric = cm.Recorder(sys, yv, 0.0, cell.get("metric", "exact"))
        out = {"cell": cell["name"], "n": cell["n"], "j": j, ARM: {}}
        for m in R.TIMING_TARGETS:
            i = cm.MULTS.index(m)
            k = hits[i]
            if k is None:
                out[ARM][str(m)] = {"h": None, "t": None}
                continue
            ts = []
            for _ in range(R.REPEATS.get(cell["n"], 1)):
                x, t, cnt = R.replay(ARM, sys, y, rec["lam"], A0, {}, k)
                ts.append(t)
            assert cnt == k, (name, m, cnt, k)
            ok = metric.metric(x) <= rec["taus"][i]
            if not ok:
                bad.append((name, m))
            out[ARM][str(m)] = {"h": k, "t": float(np.median(ts)), "ts": ts, "reached": bool(ok)}
        _save("timing", name, out)
        print(f"  计时 {name[:-5]}: " + "/".join("-" if v["t"] is None else f"{v['t']:.3f}s" for v in out[ARM].values()), flush=True)
    R.algo_for = orig
    print(f"回放点未达标 {len(bad)} 项 {bad[:5]}", flush=True)


def _G(hc, hp):
    return cm.ratio_stats(hc, hp)


def part_summary():
    res = {}
    B = {}
    for src in cm.SOURCES_B:
        for shift in cm.SHIFTS:
            fs = sorted(glob.glob(os.path.join(out_dir(), "B", f"{src}_{shift}_*.json")))
            if not fs:
                continue
            new = [cm.load(f) for f in fs]
            old = [cm.load(os.path.join(V2, "B", os.path.basename(f))) for f in fs]
            e = {"n": len(new)}
            for m in (1.1, 1.01):
                i = cm.MULTS.index(m)
                hp = [o["arms"][P]["hits"][i] for o in old]
                pool = {a: [o["arms"][a]["hits"][i] for o in old] for a in ("SQ_ss_K50", "DAAREM_tun")}
                pool[ARM] = [d["hits"][i] for d in new]
                st = {a: _G(h, hp) for a, h in pool.items()}
                o_old = min(("SQ_ss_K50", "DAAREM_tun"), key=lambda a: st[a]["G"])
                o_new = min(st, key=lambda a: st[a]["G"])
                e[str(m)] = {"old": {"arm": o_old, **st[o_old]}, "new": {"arm": o_new, **st[o_new]},
                             "median": {a: cm.med(h) for a, h in {**pool, P: hp}.items()}}
            B[f"{src}_{shift}"] = e
    res["B"] = B
    fs = sorted(glob.glob(os.path.join(out_dir(), "C", "*.json")))
    if fs:
        new = [cm.load(f) for f in fs]
        old = [cm.load(os.path.join(V2, "C", os.path.basename(f))) for f in fs]
        e = {"n": len(new)}
        for m in (1.1, 1.01):
            i = cm.MULTS.index(m)
            hp = [o["arms"][P]["hits"][i] for o in old]
            pool = {a: [o["arms"][a]["hits"][i] for o in old] for a in ("SQ_ss_K50", "DAAREM_tun")}
            pool[ARM] = [d["hits"][i] for d in new]
            st = {a: _G(h, hp) for a, h in pool.items()}
            o_new = min(st, key=lambda a: st[a]["G"])
            e[str(m)] = {"new": {"arm": o_new, **st[o_new]}, "all": st, "median": {a: cm.med(h) for a, h in {**pool, P: hp}.items()}}
            hn = []
            for d0 in old:
                fn = os.path.join(V2, "g3", "C_native", f"F256_{d0['j']:02d}.json")
                hn.append(cm.load(fn)["arms"]["native_F256"]["hits"][i] if os.path.exists(fn) else "missing")
            if "missing" not in hn:
                e[str(m)]["native_new"] = _G(pool[o_new], hn)
                e[str(m)]["transfer_vs_native"] = _G(hp, hn)
        res["C"] = e
    fs = sorted(glob.glob(os.path.join(out_dir(), "ge", "*.json")))
    if fs:
        new = [cm.load(f) for f in fs]
        old = [cm.load(os.path.join(V2, "ge", os.path.basename(f))) for f in fs]
        e = {"n": len(new)}
        cls = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
        for m in (1.1, 1.01):
            i = cm.MULTS.index(m)
            hp = [o["arms"][P]["hits"][i] for o in old]
            pool = {a: [o["arms"][a]["hits"][i] for o in old] for a in cls}
            pool[ARM] = [d["hits"][i] for d in new]
            st = {a: _G(h, hp) for a, h in pool.items()}
            o_old = min(cls, key=lambda a: st[a]["G"])
            o_new = min(st, key=lambda a: st[a]["G"])
            e[str(m)] = {"old": {"arm": o_old, **st[o_old]}, "new": {"arm": o_new, **st[o_new]},
                         "median": {a: cm.med(h) for a, h in {**pool, P: hp}.items()}}
        res["ge"] = e
    H = {}
    for hn in ("HF64_500", "HF128_500"):
        fs = sorted(glob.glob(os.path.join(out_dir(), "heat", f"{hn}_*.json")))
        if not fs:
            continue
        new = [cm.load(f) for f in fs]
        old = [cm.load(os.path.join(V2, "g3", "H_A7", os.path.basename(f))) for f in fs]
        cls = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
        src = [a for a in old[0]["arms"] if a.startswith("from_")][0]
        e = {"n": len(new), "F_Rss_check_max": max(d["F_Rss_check"] for d in new), "src": src}

        def xh(ex, col, thr):
            h = next((x[0] for x in (ex or []) if x[col] <= thr), None)
            return h if (h is not None and h <= cm.MAX_E) else None
        for key, col, thr in (("dist5", 1, 0.05), ("dist1", 1, 0.01), ("gap1e-4", 3, 1e-4), ("gap1e-5", 3, 1e-5)):
            pool = {a: [xh(o["arms"][a].get("extra"), col, thr) for o in old] for a in cls}
            pool[ARM] = [xh(d["extra"], col, thr) for d in new]
            ent = {}
            for lab, prim in (("native", P), ("transfer", src)):
                hp = [xh(o["arms"][prim].get("extra"), col, thr) for o in old]
                st = {a: _G(h, hp) for a, h in pool.items()}
                o_old = min(cls, key=lambda a: st[a]["G"])
                o_new = min(st, key=lambda a: st[a]["G"])
                ent[lab] = {"old": {"arm": o_old, **st[o_old]}, "new": {"arm": o_new, **st[o_new]}}
            ent["median"] = {a: cm.med(h) for a, h in pool.items()}
            ent["median"][P] = cm.med([xh(o["arms"][P].get("extra"), col, thr) for o in old])
            e[key] = ent
        for m in (1.1, 1.01):
            i = cm.MULTS.index(m)
            e[f"median_alpha{m}"] = {ARM: cm.med([d["hits"][i] for d in new]),
                                    **{a: cm.med([o["arms"][a]["hits"][i] for o in old]) for a in cls + [P]}}
        H[hn] = e
    res["heat"] = H
    fs = sorted(glob.glob(os.path.join(out_dir(), "timing", "*.json")))
    if fs:
        T = {}
        for f in fs:
            d = cm.load(f)
            fo = os.path.join(V2, "timing", os.path.basename(f))
            if not os.path.exists(fo):
                continue
            o = cm.load(fo)
            T.setdefault(d["cell"], []).append((d, o))
        tt = {}
        for c, L in T.items():
            ent = {}
            for m in ("1.1", "1.01"):
                tp = [o["arms"][P][m]["t"] for d, o in L]
                tq = [d[ARM][m]["t"] for d, o in L]
                tda = [o["arms"]["DAAREM_tun"][m]["t"] for d, o in L]
                tsq = [o["arms"]["SQ_ss_K50"][m]["t"] for d, o in L]
                ok = [k for k in range(len(L)) if None not in (tp[k], tq[k], tda[k])]
                fastest = [min(tq[k], tda[k]) for k in ok]
                ratio = [fastest[n] / tp[k] for n, k in enumerate(ok)]
                ent[m] = {"n": len(ok), "G_time_vs_faster_of_SQpkg_DAAREM": float(np.exp(np.mean(np.log(ratio)))) if ok else None,
                          "median_saving_s": float(np.median([fastest[n] - tp[k] for n, k in enumerate(ok)])) if ok else None,
                          "median_t": {"prop": float(np.median([tp[k] for k in ok])), "SQpkg": float(np.median([tq[k] for k in ok])),
                                       "DAAREM": float(np.median([tda[k] for k in ok]))}}
            tt[c] = ent
        res["timing"] = tt
    cm.dump(res, os.path.join(out_dir(), "summary.json"))
    for k, e in B.items():
        print(f"B {k}: α=1.1 {e['1.1']['old']['G']:.2f} → {e['1.1']['new']['arm']} {e['1.1']['new']['G']:.2f}  "
              f"α=1.01 {e['1.01']['old']['G']:.2f} → {e['1.01']['new']['arm']} {e['1.01']['new']['G']:.2f}")
    for k in ("C", "ge"):
        if k in res:
            print(k, {m: (res[k][m]["new"]["arm"], round(res[k][m]["new"]["G"], 2)) for m in ("1.1", "1.01")})
    for hn, e in H.items():
        print(hn, {key: (round(e[key]["native"]["old"]["G"], 2), e[key]["native"]["new"]["arm"], round(e[key]["native"]["new"]["G"], 2),
                         round(e[key]["transfer"]["new"]["G"], 2)) for key in ("dist5", "dist1", "gap1e-4", "gap1e-5")})
    if "timing" in res:
        print("timing", res["timing"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="runs", choices=["runs", "timing", "summary", "smoke"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    CFG.update(tag=a.tag)
    if a.part == "smoke":
        CFG["tag"] = a.tag or "smoke"
        part_runs(a.workers, smoke=True)
        part_timing(smoke=True)
        part_summary()
        return
    if a.part == "runs":
        part_runs(a.workers)
    elif a.part == "timing":
        part_timing()
    part_summary()


if __name__ == "__main__":
    main()
