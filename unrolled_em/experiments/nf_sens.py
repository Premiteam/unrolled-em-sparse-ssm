from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import time

import numpy as np

from unrolled_em import core as ls
from unrolled_em import anytime as at
from unrolled_em.experiments import common as cm
from unrolled_em.experiments import run as rn

NFS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
ARMS = ["EM", "SQUAREM", "DAAREM", "learned"]
CELLS = ["F32_500", "F64_500", "F128_500", "P64_500"]
ARGS = {"tag": ""}


def out_dir():
    return os.path.join(cm.OUT, ("sqdiag_" + ARGS["tag"]) if ARGS["tag"] else "sqdiag", "nf")


def nf_name(K):
    return "full" if K is None else str(K)


def load_system(name, j):
    old = cm.load(os.path.join(cm.OUT, "testII", f"{name}_{j:02d}.json"))
    z = np.load(rn.p_ref("testII", f"{name}_{j:02d}"))
    sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
    lam = float(z["lam"])
    assert abs(lam - old["lam"]) <= 1e-9 * abs(lam), "λ 与已存结果不一致"
    return sys, y, yv, lam, z["A0"], old


class Timer:
    def __init__(self):
        self.tE, self.tM = [], []
        self.o_ss, self.o_m = ls.estep_ss, ls.mstep

    def install(self):
        o_ss, o_m = self.o_ss, self.o_m

        def es(*a, **k):
            t = time.perf_counter(); r = o_ss(*a, **k); self.tE.append(time.perf_counter() - t); return r

        def ms(*a, **k):
            t = time.perf_counter(); r = o_m(*a, **k); self.tM.append(time.perf_counter() - t); return r
        ls.estep_ss, ls.mstep = es, ms

    def remove(self):
        ls.estep_ss, ls.mstep = self.o_ss, self.o_m


SQ_LOG = []


def logged_squarem_step(a0, F, obj):
    F1 = F(a0)
    F2 = F(F1)
    r = F1 - a0
    v = F2 - 2 * F1 + a0
    nr, nv = np.linalg.norm(r), np.linalg.norm(v)
    alpha = min(-nr / nv, -1.0) if nv > 0 else -1.0
    ap = a0 - 2 * alpha * r + alpha ** 2 * v
    Fa = F(ap)
    o_ap, o_F1 = obj(ap), obj(F1)
    fb = (not (np.isfinite(o_ap) and o_ap <= o_F1)) or (not np.all(np.isfinite(Fa)))
    SQ_LOG.append(bool(fb))
    if fb:
        return F2, {"fallback": True, "alpha": alpha}
    return Fa, {"fallback": False, "alpha": alpha}


def algo_for(arm, K, sys, y, lam, A0, name):
    mx = cm.MAX_E
    if arm == "EM":
        return lambda: ls.em_run(sys, y, A0, lam, estep="ss", K=K, max_E=mx, stop_fn=rn.NEVER)
    if arm == "SQUAREM":
        return lambda: ls.squarem_run(sys, y, A0, lam, K=K, max_E=mx, stop_fn=rn.NEVER, estep="ss")
    if arm == "DAAREM":
        from unrolled_em import daarem as da
        o, k = tuple(cm.load(os.path.join(cm.OUT, "tune_choice.json"))[name]["config"])
        return lambda: da.daarem_run(sys, y, A0, lam, K=K, max_E=mx, estep="ss", order=o, kappa=k)
    if arm == "learned":
        from unrolled_em.experiments import exp_ab as ea
        from unrolled_em.experiments import method as me
        layers = me.load_layers(rn.p_model(name, "learned"))[:cm.L_MAIN]
        return lambda: ea.run_unrolled(layers, sys, y, lam, A0, K, "ss", max_E=mx)
    raise ValueError(arm)


def job(args):
    name, j, arm = args
    f = os.path.join(out_dir(), f"{name}_{j:02d}_{arm}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    ls.squarem_step = logged_squarem_step
    sys, y, yv, lam, A0, old = load_system(name, j)
    taus = old["taus"]
    out = {"cell": name, "j": j, "arm": arm, "taus": taus, "nf": {}}
    for K in NF_LIST:
        SQ_LOG.clear()
        rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus)
        tm = Timer(); tm.install()
        t = time.time()
        try:
            at.run(algo_for(arm, K, sys, y, lam, A0, name), sys, lam, K, rec, A0)
        finally:
            tm.remove()
        out["nf"][nf_name(K)] = {"hits": cm.hits_from_trace(rec.trace, rec.taus), "sec": time.time() - t,
                                 "tE_med": float(np.median(tm.tE)) if tm.tE else None,
                                 "tM_med": float(np.median(tm.tM)) if tm.tM else None,
                                 "n_E_calls": len(tm.tE), "n_M_calls": len(tm.tM),
                                 "sq_fallback": float(np.mean(SQ_LOG)) if SQ_LOG else None}
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    i11, i101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
    print(f"{name} {j:02d} {arm:8s}: " + " ".join(f"{k}:{v['hits'][i11]}/{v['hits'][i101]}" for k, v in out["nf"].items())
          + f" | {out['sec']:.0f}s", flush=True)
    return f


def summary():
    import csv
    import glob
    i11, i101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
    files = sorted(glob.glob(os.path.join(out_dir(), "*.json")))
    data = [cm.load(f) for f in files if not f.endswith("summary.json")]
    res, rows = {}, []
    for c in sorted({d["cell"] for d in data}, key=lambda c: (c[0], cm.base_cell(c)["n"])):
        res[c] = {}
        print(f"\n=== {c} ===  中位 E 步数 alpha=1.1 / 1.01；括号内：相对“解满”的 E 步比（几何平均，>1 表示更慢）；"
              f"M/E：单次 M 步与单次 E 步的耗时比（中位）")
        for arm in ARMS:
            L = [d for d in data if d["cell"] == c and d["arm"] == arm]
            if not L:
                continue
            res[c][arm] = {}
            keys = list(L[0]["nf"].keys())
            line = []
            for k in keys:
                h = [[d["nf"][k]["hits"][i] for d in L] for i in (i11, i101)]
                med = [cm.med(x) for x in h]
                G = [cm.ratio_stats([d["nf"][k]["hits"][i] for d in L],
                                    [d["nf"]["full"]["hits"][i] for d in L])["G"] if "full" in keys else None
                     for i in (i11, i101)]
                me_ = [d["nf"][k]["tM_med"] / d["nf"][k]["tE_med"] for d in L
                       if d["nf"][k]["tM_med"] and d["nf"][k]["tE_med"]]
                fb = [d["nf"][k]["sq_fallback"] for d in L if d["nf"][k]["sq_fallback"] is not None]
                r = {"n_sys": len(L), "med": med, "G_vs_full": G, "M_over_E": float(np.median(me_)) if me_ else None,
                     "sq_fallback": float(np.mean(fb)) if fb else None}
                res[c][arm][k] = r
                rows.append({"cell": c, "arm": arm, "nf": k, "med11": med[0], "med101": med[1],
                             "G11": G[0], "G101": G[1], "M_over_E": r["M_over_E"], "sq_fallback": r["sq_fallback"]})
                g = "" if G[0] is None else f" ({G[0]:.2f}/{G[1]:.2f})"
                line.append(f"{k:>4s}: {med[0]:5.1f}/{med[1]:5.1f}{g} M/E {r['M_over_E'] or 0:.2f}")
            print(f"  {arm} ({len(L)} 个系统)")
            for s in line:
                print("    " + s)
    cm.dump(res, os.path.join(out_dir(), "summary.json"))
    with open(os.path.join(out_dir(), "nf_sensitivity.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n已写 {os.path.join(out_dir(), 'summary.json')} 与 nf_sensitivity.csv")


NF_LIST = NFS


def main():
    global NF_LIST
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--js", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--nfs", default=None, help="逗号分隔，full 表示解满，如 10,50,full")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ARGS["tag"] = a.tag
    if a.nfs:
        NF_LIST = [None if x == "full" else int(x) for x in a.nfs.split(",")]
    if a.part == "summary":
        summary()
        return
    cells = a.cells.split(",") if a.cells else CELLS
    js = [int(x) for x in a.js.split(",")] if a.js else list(range(10))
    arms = a.arms.split(",") if a.arms else ARMS
    jobs = [(c, j, arm) for c in cells for j in js for arm in arms]
    jobs.sort(key=lambda x: (-cm.base_cell(x[0])["n"], x[2] == "learned"))
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(out_dir(), f"{jb[0]}_{jb[1]:02d}_{jb[2]}.json"))]
    print(f"N_F 敏感度：{len(jobs)} 个作业，待跑 {len(todo)}，{a.workers} 个进程，N_F = "
          f"{[nf_name(k) for k in NF_LIST]} → {out_dir()}", flush=True)
    if a.workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers, initializer=_init_worker, initargs=(NF_LIST, a.tag)) as pool:
            for _ in pool.imap_unordered(job, todo):
                pass
    else:
        for jb in todo:
            job(jb)


def _init_worker(nfs, tag):
    global NF_LIST
    NF_LIST = nfs
    ARGS["tag"] = tag


if __name__ == "__main__":
    main()
