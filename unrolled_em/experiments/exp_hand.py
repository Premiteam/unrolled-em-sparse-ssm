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

ROOTS = {"pred": os.path.join(cm.HERE, "results", "v2"), "supp": os.path.join(cm.HERE, "results", "v2_supp")}
CFG = {"tag": "", "lt": "pred"}
VARIANTS = {"H0": dict(r=0, amax0=1.0, safe=True), "R5": dict(r=5, amax0=1.0, safe=True),
            "R5s16": dict(r=5, amax0=16.0, safe=True), "R5u": dict(r=5, amax0=1.0, safe=False)}
P = "learned_tied_L4"
NEW_SQ = ["SQbt3_ss", "SQbt1_ss", "SQbt2_ss", "SQpkg_ss", "SQbt3_exact"]
OLD_CLS = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
POOL = OLD_CLS + NEW_SQ
POOL_SS = ["EM_ss_K50", "SQ_ss_K50", "DAAREM_tun", "SQbt3_ss", "SQbt1_ss", "SQbt2_ss", "SQpkg_ss"]
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)


def root():
    return ROOTS[CFG["lt"]]


def out_dir():
    return os.path.join(root(), "hand" + (f"_{CFG['tag']}" if CFG["tag"] else ""))


def _ok(X):
    return bool(np.all(np.isfinite(X))) and float(np.max(np.abs(np.linalg.eigvals(X)))) < 1.0


def _estep(me, X, sys, y):
    try:
        s = me.estep(X, sys, y)
        return s if np.isfinite(s["ll"]) else None
    except np.linalg.LinAlgError:
        return None


def guarded_squarem(A, st, sys, y, lam, n, max_E, amax0, safe, log):
    from unrolled_em.experiments import method as me
    amax = amax0
    while n < max_E:
        F_A = me.fobj(A, st, lam)
        B1 = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
        dA = max(me.surrogate(A, st["Phi"], st["Psi"], sys.q, lam) - me.surrogate(B1, st["Phi"], st["Psi"], sys.q, lam), 0.0)
        st1 = _estep(me, B1, sys, y); n += 1
        if st1 is None or n >= max_E:
            return
        B2 = ls.mstep(B1, st1["Phi"], st1["Psi"], sys.q, lam, K=50)
        r, v = B1 - A, B2 - 2 * B1 + A
        rr, vv = float(np.sum(r * r)), float(np.sum(v * v))
        a = float(np.sqrt(rr / vv)) if vv > 0 else 1.0
        a = max(1.0, min(amax, a))
        acc, why = False, "em"
        if abs(a - 1.0) > 0.01:
            X = A + 2 * a * r + a * a * v
            why = "rho"
            if _ok(X):
                sX = _estep(me, X, sys, y); n += 1
                why = "obj"
                if sX is not None and n < max_E:
                    Xt = ls.mstep(X, sX["Phi"], sX["Psi"], sys.q, lam, K=50)
                    if _ok(Xt):
                        sXt = _estep(me, Xt, sys, y); n += 1
                        if sXt is not None:
                            F_Xt = me.fobj(Xt, sXt, lam)
                            acc = F_Xt <= (F_A - me.ETA * dA if safe else F_A + 1.0)
        if acc:
            A, st = Xt, sXt
            why = "ok"
        else:
            if n >= max_E:
                return
            s2 = _estep(me, B2, sys, y); n += 1
            if s2 is None:
                return
            A, st = B2, s2
            if why != "em":
                if a == amax:
                    amax = max(amax0, amax / 4.0)
                a = 1.0
        if a == amax:
            amax *= 4.0
        log.append((f"sq_{why}", float(a), float(amax)))
        if rr == 0.0:
            return


def run_hand(layers, sys, y, lam, A0, variant, max_E=cm.MAX_E, log=None):
    from unrolled_em.experiments import method as me
    log = [] if log is None else log
    A = A0.copy()
    st = me.estep(A, sys, y)
    n = 1
    Rp = np.zeros_like(A)
    L = len(layers)
    cfg = VARIANTS.get(variant)
    k, consec = 0, 0
    while n + 1 <= max_E:
        if cfg is not None and k >= L and consec >= cfg["r"]:
            log.append(("handover", None, None))
            guarded_squarem(A, st, sys, y, lam, n, max_E, cfg["amax0"], cfg["safe"], log)
            return
        o = me.layer_step(layers[min(k, L - 1)], A, st, Rp, sys, y, lam)
        n += o["nE"]
        A, st, Rp = o["A"], o["st"], o["Rp"]
        log.append((o["reject"] or "ok", None, None))
        if k >= L:
            consec = consec + 1 if o["reject"] else 0
        k += 1


def test_job(args):
    name, j = args
    f = os.path.join(out_dir(), "testII", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    from unrolled_em.experiments import method as me
    t0 = time.time()
    old = cm.load(os.path.join(root(), "testII", f"{name}_{j:02d}.json"))
    z = np.load(os.path.join(root(), "refs", "testII", f"{name}_{j:02d}.npz"))
    sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
    lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
    assert abs(lam - old["lam"]) <= 1e-9 * abs(lam)
    rich = {"y": y, "lam": lam, "refs": {"exact": (z["A_R"], old["F_R"]), "ss": (z["A_Rss"], old["F_Rss"])}, "own": "ss"}
    layers = me.load_layers(os.path.join(root(), "models", f"{name}_learned.pt"))[:cm.L_MAIN]
    V = list(VARIANTS) + (["base"] if j == 0 else [])
    out = {"cell": name, "j": j, "taus": taus, "arms": {}}
    for v in V:
        rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus, rich=rich, stop_dist=0.01, stop_gap=1e-5)
        log = []
        t = time.time()
        at.run(lambda: run_hand(layers, sys, y, lam, A0, v, log=log), sys, lam, 50, rec, A0)
        out["arms"][v] = {"hits": cm.hits_from_trace(rec.trace, taus), "q": rec.q, "extra": rec.extra, "sec": time.time() - t,
                          "log": [x[0] for x in log]}
    if "base" in out["arms"]:
        out["repro_base"] = out["arms"]["base"]["hits"] == old["arms"][P]["hits"]
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    h = lambda r: f"{r['hits'][I11]}/{r['hits'][I101]}"
    print(f"  {name} {j:02d}: " + "  ".join(f"{v} {h(out['arms'][v])}" for v in V)
          + f" | 表中本方法 {h(old['arms'][P])}" + ("" if "repro_base" not in out else f" | 复现 {out['repro_base']}")
          + f" | {out['sec']:.0f}s", flush=True)
    return f


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


def part_test(workers, cells, js):
    jobs = [(c, j) for c in cells for j in js if os.path.exists(os.path.join(root(), "testII", f"{c}_{j:02d}.json"))]
    jobs.sort(key=lambda x: (-cm.base_cell(x[0])["n"], x[1]))
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(out_dir(), "testII", f"{jb[0]}_{jb[1]:02d}.json"))]
    print(f"[{time.strftime('%H:%M:%S')}] {CFG['lt']} 测试集 II：{len(jobs)} 个系统，待跑 {len(todo)}", flush=True)
    for _ in pmap(test_job, todo, workers):
        pass


def _xh(arm, col, thr):
    ex = arm.get("extra") or []
    h = next((e[0] for e in ex if e[col] <= thr), None)
    return h if (h is not None and h <= cm.MAX_E) else None


def part_summary(cells):
    res = {"variants": list(VARIANTS), "cells": {}}
    for c in cells:
        fs = sorted(glob.glob(os.path.join(out_dir(), "testII", f"{c}_*.json")))
        if not fs:
            continue
        rows = []
        for f in fs:
            d = cm.load(f)
            old = cm.load(os.path.join(root(), "testII", os.path.basename(f)))
            sq = cm.load(os.path.join(root(), "sqbt", "testII", os.path.basename(f)))
            arms = dict(old["arms"]); arms.update({a: sq["arms"][a] for a in NEW_SQ}); arms.update(d["arms"])
            rows.append(arms)
        e = {"n": len(rows)}
        mets = [("a11", None), ("a101", None), ("dist5", (1, 0.05)), ("dist1", (1, 0.01)), ("gap4", (3, 1e-4)), ("gap5", (3, 1e-5))]
        for met, spec in mets:
            if spec is None:
                i = I11 if met == "a11" else I101
                H = {a: [r[a]["hits"][i] for r in rows] for a in [P] + POOL + list(VARIANTS)}
                pool = POOL
            else:
                H = {a: [_xh(r[a], *spec) for r in rows] for a in [P] + POOL_SS + list(VARIANTS)}
                pool = POOL_SS
            ent = {}
            for v in [P] + list(VARIANTS):
                st = {a: cm.ratio_stats(H[a], H[v]) for a in pool}
                o = min(pool, key=lambda a: st[a]["G"])
                ent[v] = {"G": st[o]["G"], "ci": st[o]["ci"], "vs": o, "median": cm.med(H[v]), "fail": sum(x is None for x in H[v])}
                if v != P:
                    ent[v]["paper_over_var"] = cm.ratio_stats(H[P], H[v])
            e[met] = ent
        res["cells"][c] = e
        print(f"\n=== {CFG['lt']} {c}（{len(rows)} 个系统）===")
        for met, _ in mets:
            print(f"  {met:5s}: " + "  ".join(
                f"{v} {e[met][v]['G']:.2f}[{e[met][v]['ci'][0]:.2f},{e[met][v]['ci'][1]:.2f}] 中位{e[met][v]['median']} 败{e[met][v]['fail']}"
                + ("" if v == P else f" (论文/它 {e[met][v]['paper_over_var']['G']:.2f})") for v in [P] + list(VARIANTS)), flush=True)
    cm.dump(res, os.path.join(out_dir(), "summary.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["test", "summary", "all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=",".join(cm.CELLS))
    ap.add_argument("--js", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lt", default="pred", choices=["pred", "supp"])
    a = ap.parse_args()
    CFG.update(tag=a.tag, lt=a.lt)
    cells = a.cells.split(",")
    js = [int(x) for x in a.js.split(",")] if a.js else list(range(cm.N_TEST_II))
    if a.part in ("test", "all"):
        part_test(a.workers, cells, js)
    if a.part in ("summary", "all"):
        part_summary(cells)


if __name__ == "__main__":
    main()
