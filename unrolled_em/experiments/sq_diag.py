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

ARGS = {"tag": ""}


def out_dir():
    return os.path.join(cm.OUT, "sqdiag" + (f"_{ARGS['tag']}" if ARGS["tag"] else ""))


def load_system(name, j):
    old = cm.load(os.path.join(cm.OUT, "testII", f"{name}_{j:02d}.json"))
    z = np.load(rn.p_ref("testII", f"{name}_{j:02d}"))
    sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
    lam = float(z["lam"])
    assert abs(lam - old["lam"]) <= 1e-9 * abs(lam), "λ 与已存结果不一致"
    return sys, y, yv, lam, z["A0"], z["A_R"], z["A_Rss"], old


M_TOL, M_MAXIT = 1e-12, 20000


def em_map_full(A, sys, y, lam):
    st = ls.estep_ss(A, sys.C, sys.q, sys.r, sys.P1, y)
    return ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=None, tol=M_TOL, maxit=M_MAXIT)


def jac_job(job):
    name, j = job
    f = os.path.join(out_dir(), "jac", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    sys, y, yv, lam, A0, A_R, A_Rss, old = load_system(name, j)
    A = A_Rss.copy()
    for _ in range(20):
        A = em_map_full(A, sys, y, lam)
    G0 = em_map_full(A, sys, y, lam)
    fp_res = float(np.linalg.norm(G0 - A) / np.linalg.norm(A))
    S = np.flatnonzero(np.abs(A).ravel() > 1e-10)
    h = 1e-5 * float(np.max(np.abs(A)))
    J = np.empty((S.size, S.size))
    kinks = 0
    for c, s in enumerate(S):
        E = np.zeros(A.size); E[s] = h; E = E.reshape(A.shape)
        Gp = em_map_full(A + E, sys, y, lam).ravel()
        Gm = em_map_full(A - E, sys, y, lam).ravel()
        off = np.ones(A.size, bool); off[S] = False
        kinks += int(np.any(np.abs(Gp[off]) > 1e-10) or np.any(np.abs(Gm[off]) > 1e-10))
        J[:, c] = (Gp[S] - Gm[S]) / (2 * h)
    ev = np.linalg.eigvals(J)
    mag = np.sort(np.abs(ev))[::-1]
    l1 = float(mag[0])
    out = {"cell": name, "j": j, "n": int(A.shape[0]), "support": int(S.size), "fp_res": fp_res, "h": h, "kinks": kinks,
           "top": mag[:12].tolist(), "lambda1": l1, "ratio2": float(mag[1] / l1),
           "n_within_0.9": int(np.sum(mag >= 0.9 * l1)), "n_within_0.95": int(np.sum(mag >= 0.95 * l1)),
           "n_above_0.9abs": int(np.sum(mag >= 0.9)), "complex_top": bool(abs(np.imag(ev[np.argmax(np.abs(ev))])) > 1e-8),
           "sec": time.time() - t0}
    cm.dump(out, f)
    print(f"jac {name} {j:02d}: |S|={S.size} 不动点残差 {fp_res:.1e} | |λ| 前 5 = {np.round(mag[:5], 4)} | "
          f"≥0.9|λ1| 的个数 {out['n_within_0.9']} | 非光滑 {kinks} | {out['sec']:.0f}s", flush=True)
    return f


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
    SQ_LOG.append((float(alpha), bool(fb), float(nv / nr) if nr > 0 else None))
    if fb:
        return F2, {"fallback": True, "alpha": alpha}
    return Fa, {"fallback": False, "alpha": alpha}


ARMS = ["SQ_ss_K50", "SQ_ss_full", "EM_ss_full", "DAAREM_full"]


def arm_algo(arm, sys, y, lam, A0, daarem_cfg):
    from unrolled_em import daarem as da
    mx = cm.MAX_E
    if arm == "SQ_ss_K50":
        return (lambda: ls.squarem_run(sys, y, A0, lam, K=50, max_E=mx, stop_fn=rn.NEVER, estep="ss")), 50
    if arm == "SQ_ss_full":
        return (lambda: ls.squarem_run(sys, y, A0, lam, K=None, max_E=mx, stop_fn=rn.NEVER, estep="ss")), None
    if arm == "EM_ss_full":
        return (lambda: ls.em_run(sys, y, A0, lam, estep="ss", K=None, max_E=mx, stop_fn=rn.NEVER)), None
    if arm == "DAAREM_full":
        o, k = daarem_cfg
        return (lambda: da.daarem_run(sys, y, A0, lam, K=None, max_E=mx, estep="ss", order=o, kappa=k)), None
    raise ValueError(arm)


def arms_job(job):
    name, j = job
    f = os.path.join(out_dir(), "arms", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    ls.squarem_step = logged_squarem_step
    sys, y, yv, lam, A0, A_R, A_Rss, old = load_system(name, j)
    taus = old["taus"]
    daarem_cfg = tuple(cm.load(os.path.join(cm.OUT, "tune_choice.json"))[name]["config"])
    out = {"cell": name, "j": j, "taus": taus, "daarem_cfg": daarem_cfg, "arms": {},
           "stored": {k: old["arms"][k]["hits"] for k in ("EM_ss_K50", "SQ_ss_K50", "DAAREM_tun", "learned_tied_L4")}}
    for arm in ARMS:
        SQ_LOG.clear()
        rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus)
        algo, K = arm_algo(arm, sys, y, lam, A0, daarem_cfg)
        t = time.time()
        at.run(algo, sys, lam, K, rec, A0)
        out["arms"][arm] = {"hits": cm.hits_from_trace(rec.trace, rec.taus), "sec": time.time() - t,
                            "sq_log": list(SQ_LOG) if arm.startswith("SQ") else None}
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    i11, i101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
    h = lambda v: f"{v[i11]}/{v[i101]}"
    print(f"arms {name} {j:02d}: " + "  ".join(f"{a} {h(r['hits'])}" for a, r in out["arms"].items())
          + f" | 原 SQ {h(out['stored']['SQ_ss_K50'])} 原 EM {h(out['stored']['EM_ss_K50'])}"
          + f" 原 DA {h(out['stored']['DAAREM_tun'])} | {out['sec']:.0f}s", flush=True)
    return f


def summary():
    import glob
    i11, i101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
    res = {"jac": {}, "arms": {}}
    print("\n=== H1：稳态 EM 映射（M 步解满）在不动点处的雅可比特征值 ===")
    print(f"{'格':8s} {'系统':>4s} {'|λ1|':>7s} {'|λ2|/|λ1|':>10s} {'≥0.9|λ1|':>9s} {'≥0.95|λ1|':>10s} {'非光滑':>6s}")
    for f in sorted(glob.glob(os.path.join(out_dir(), "jac", "*.json"))):
        d = cm.load(f)
        res["jac"].setdefault(d["cell"], []).append(d)
        print(f"{d['cell']:8s} {d['j']:4d} {d['lambda1']:7.4f} {d['ratio2']:10.3f} {d['n_within_0.9']:9d} "
              f"{d['n_within_0.95']:10d} {d['kinks']:6d}")
    for c, L in res["jac"].items():
        res["jac"][c] = {"n_sys": len(L), "lambda1_med": cm.med([d["lambda1"] for d in L]),
                         "ratio2_med": cm.med([d["ratio2"] for d in L]),
                         "n_within_0.9_med": cm.med([d["n_within_0.9"] for d in L])}
    print("\n=== H2：M 步解满与否（中位 E 步数，alpha = 1.1 / 1.01；括号内为同一批系统的原表结果）===")
    for c in sorted({cm.load(f)["cell"] for f in glob.glob(os.path.join(out_dir(), "arms", "*.json"))}):
        L = [cm.load(f) for f in sorted(glob.glob(os.path.join(out_dir(), "arms", f"{c}_*.json")))]
        row = {"n_sys": len(L)}
        for a in ARMS:
            row[a] = [cm.med([d["arms"][a]["hits"][i] for d in L]) for i in (i11, i101)]
        for a in ("EM_ss_K50", "SQ_ss_K50", "DAAREM_tun", "learned_tied_L4"):
            row["stored_" + a] = [cm.med([d["stored"][a][i] for d in L]) for i in (i11, i101)]
        row["repro_SQ_ss_K50"] = sum(d["arms"]["SQ_ss_K50"]["hits"] == d["stored"]["SQ_ss_K50"] for d in L)
        row["ratio_SQK50_over_SQfull"] = [cm.ratio_stats([d["arms"]["SQ_ss_K50"]["hits"][i] for d in L],
                                                         [d["arms"]["SQ_ss_full"]["hits"][i] for d in L])["G"]
                                          for i in (i11, i101)]
        for a in ("SQ_ss_K50", "SQ_ss_full"):
            lg = [x for d in L for x in d["arms"][a]["sq_log"]]
            row[a + "_fallback_rate"] = float(np.mean([x[1] for x in lg])) if lg else None
            row[a + "_gamma_med"] = float(np.median([x[0] for x in lg])) if lg else None
            row[a + "_d2d1_med"] = float(np.median([x[2] for x in lg if x[2] is not None])) if lg else None
        res["arms"][c] = row
        print(f"\n{c}（{len(L)} 个系统；SQ_ss_K50 复现 {row['repro_SQ_ss_K50']}/{len(L)}）")
        for a in ARMS:
            print(f"  {a:12s} {row[a][0]:6.1f} / {row[a][1]:6.1f}")
        for a in ("EM_ss_K50", "SQ_ss_K50", "DAAREM_tun", "learned_tied_L4"):
            print(f"  ({a:15s} {row['stored_' + a][0]:6.1f} / {row['stored_' + a][1]:6.1f})")
        print(f"  SQ_K50 / SQ_full 的 E 步比（几何平均）: {row['ratio_SQK50_over_SQfull'][0]:.2f} / "
              f"{row['ratio_SQK50_over_SQfull'][1]:.2f}")
        for a in ("SQ_ss_K50", "SQ_ss_full"):
            if row[a + "_fallback_rate"] is not None:
                print(f"  {a}: 回退率 {row[a + '_fallback_rate']:.2f}，gamma 中位 {row[a + '_gamma_med']:.2f}，"
                      f"||D2||/||D1|| 中位 {row[a + '_d2d1_med']:.3f}")
    print("\n读法：H1 成立 ⇔ |λ2|/|λ1| 接近 1、≥0.9|λ1| 的特征值不止一个；"
          "H2 成立 ⇔ SQ_ss_full 明显少于 SQ_ss_K50（比值 > 1.2 左右）且 K50 的回退率更高。")
    cm.dump(res, os.path.join(out_dir(), "summary.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["jac", "arms", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--js", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ARGS["tag"] = a.tag
    if a.part == "summary":
        summary()
        return
    if a.part == "jac":
        cells = a.cells.split(",") if a.cells else ["F32_500", "F64_500", "P32_500"]
        js = [int(x) for x in a.js.split(",")] if a.js else list(range(5))
        fn, sub = jac_job, "jac"
    else:
        cells = a.cells.split(",") if a.cells else ["F32_500", "F64_500", "F128_500"]
        js = [int(x) for x in a.js.split(",")] if a.js else list(range(10))
        fn, sub = arms_job, "arms"
    jobs = sorted([(c, j) for c in cells for j in js], key=lambda x: -cm.base_cell(x[0])["n"])
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(out_dir(), sub, f"{jb[0]}_{jb[1]:02d}.json"))]
    print(f"{sub}：{len(jobs)} 个系统，待跑 {len(todo)}，{a.workers} 个进程 → {out_dir()}", flush=True)
    for _ in cm.pmap(fn, todo, a.workers):
        pass


if __name__ == "__main__":
    main()
