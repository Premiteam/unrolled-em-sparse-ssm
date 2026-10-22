from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("V2_TAG", "v2_supp")

import argparse
import glob
import time

import numpy as np

from unrolled_em.experiments import common as cm
from unrolled_em.experiments import run as rn

CELLS = ["F32_500", "F64_500", "F128_500"]
C_GRID = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
N_DEV = 4
ARMS = ["learned_tied_L4", "EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
CLASSICAL = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
V2 = os.path.join(cm.HERE, "results", "v2")
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
OUT = cm.OUT


def dev_job(args):
    from unrolled_em import core as ls
    name, j = args
    f = os.path.join(OUT, "lamdev", f"{name}_{j}.json")
    if os.path.exists(f):
        return f
    cell = cm.base_cell(name)
    sys, y, yv, info = cm.setup(cell, "dev", j)
    res = {}
    for c in C_GRID:
        lam = cm.lam_of(c, cell, sys)
        A0, _ = cm.init(cell, sys, y, lam)
        A, nE, _ = cm.reference(sys, y, A0, lam, max_E=cm.LAMBDA_REF_MAX_E)
        res[str(c)] = {"M": ls.excess_nll(A, sys, yv), "nE": nE, **cm.quality(A, sys.A)}
    cm.dump({"cell": name, "j": j, "res": res, **info}, f)
    print(f"  dev {name} {j}: " + "  ".join(f"c={c}: F1 {v['f1']:.3f} M {v['M']:.4f}" for c, v in res.items()), flush=True)
    return f


def part_dev(workers):
    jobs = [(c, j) for c in CELLS for j in range(N_DEV)]
    jobs.sort(key=lambda x: -cm.base_cell(x[0])["n"])
    for _ in cm.pmap(dev_job, [jb for jb in jobs if not os.path.exists(os.path.join(OUT, "lamdev", f"{jb[0]}_{jb[1]}.json"))],
                     workers):
        pass
    f = os.path.join(OUT, "lambda.json")
    sel = cm.load(f) if os.path.exists(f) else {}
    pred = cm.load(os.path.join(V2, "lambda.json"))
    for name in CELLS:
        if name in sel:
            continue
        rs = [cm.load(os.path.join(OUT, "lamdev", f"{name}_{j}.json")) for j in range(N_DEV)]
        mean = {str(c): {k: float(np.mean([r["res"][str(c)][k] for r in rs])) for k in ("f1", "prec", "rec", "M", "delta", "nE")}
                for c in C_GRID}
        best = max(C_GRID, key=lambda c: mean[str(c)]["f1"])
        sel[name] = {"c": best, "c_pred": pred[name]["c"], "dev_mean": mean, "rule": "max mean F1 of the reference on dev0-3"}
        print(f"{name}: c = {best}（预测选法 c = {pred[name]['c']}）  " +
              "  ".join(f"{c}: F1 {m['f1']:.3f}/P {m['prec']:.2f}/R {m['rec']:.2f}/M {m['M']:.4f}" for c, m in mean.items()),
              flush=True)
    cm.dump(sel, f)


def part_prep(workers):
    jobs = [(nm, s, j) for nm in CELLS for s, k in zip(("train", "val"), cm.N_TRVA[nm]) for j in range(k)]
    jobs.sort(key=lambda x: -cm.base_cell(x[0])["n"])
    todo = [jb for jb in jobs if not os.path.exists(rn.p_data(*jb))]
    print(f"参照解：{len(jobs)} 个，待算 {len(todo)}", flush=True)
    for _ in cm.pmap(rn._prep_job, todo, workers):
        pass


def part_train(workers):
    jobs = [(c, "learned", cm.L_MAIN) for c in CELLS if not os.path.exists(rn.p_model(c, "learned"))]
    jobs.sort(key=lambda s: -cm.base_cell(s[0])["n"])
    print(f"训练：{len(jobs)} 条链", flush=True)
    for _ in cm.pmap(rn._train_job, jobs, workers):
        pass


def part_tune(workers):
    f = os.path.join(OUT, "tune_choice.json")
    choice = cm.load(f) if os.path.exists(f) else {}
    todo_cells = [c for c in CELLS if c not in choice]
    jobs = [(nm, j) for nm in todo_cells for j in range(cm.N_TRVA[nm][1])]
    jobs.sort(key=lambda x: -cm.base_cell(x[0])["n"])
    for _ in cm.pmap(rn._tune_job, [jb for jb in jobs if not os.path.exists(os.path.join(OUT, "tune", f"{jb[0]}_{jb[1]:02d}.json"))],
                     workers):
        pass
    for nm in todo_cells:
        rs = [cm.load(os.path.join(OUT, "tune", f"{nm}_{j:02d}.json")) for j in range(cm.N_TRVA[nm][1])]
        gm = {g: rn._gm([r["hits"][f"{g[0]},{g[1]}"] for r in rs]) for g in rn.GRID_DA}
        best = min(gm.values())
        tied = [g for g in gm if gm[g] <= best * (1 + 1e-12)]
        cfg = rn.DEFAULT_DA if rn.DEFAULT_DA in tied else sorted(tied)[0]
        choice[nm] = {"config": list(cfg), "gm": {f"{g[0]},{g[1]}": x for g, x in gm.items()},
                      "gm_SQ_ss": rn._gm([r["SQ_ss_K50"] for r in rs])}
        print(f"  DAAREM {nm}: (order, κ) = {cfg}", flush=True)
    cm.dump(choice, f)


def test_job(args):
    name, j = args
    f = os.path.join(OUT, "testII", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    cell = cm.base_cell(name)
    t0 = time.time()
    sys, y, yv, lam, A0, A_R, F_R, taus, head, rich = rn._test_common(cell, "test1g", j, "testII", f"{name}_{j:02d}",
                                                                     need_ss=True)
    models = rn.load_models(name)
    out = {**head, "arms": rn._run_arms(ARMS, sys, y, yv, lam, A0, taus, models, rich=rich)}
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    rn._print_job(f"II(c={cm.c_of(name)}) {name} {j:02d}", out)
    return f


def part_test(workers, js=None):
    js = js if js is not None else list(range(cm.N_TEST_II))
    jobs = sorted([(c, j) for c in CELLS for j in js], key=lambda x: -cm.base_cell(x[0])["n"])
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(OUT, "testII", f"{jb[0]}_{jb[1]:02d}.json"))]
    print(f"测试集 II（支撑集 λ）：{len(jobs)} 个系统，待跑 {len(todo)}", flush=True)
    for _ in cm.pmap(test_job, todo, workers):
        pass


def _first(extra, dist=None, gap=None):
    for c, d_own, _, g, _ in extra:
        if (dist is None or d_own <= dist) and (gap is None or g <= gap):
            return c if c <= cm.MAX_E else None
    return None


def _cell_stats(L, label):
    v = lambda x: cm.FAIL if x is None else x
    H = lambda a, i: [d["arms"][a]["hits"][i] for d in L]
    P = "learned_tied_L4"
    r = {"n_sys": len(L), "label": label,
         "ref": {k: float(np.median([d["q_R"][k] for d in L])) for k in ("f1", "prec", "rec", "delta", "nnz")},
         "nE_R_cap": sum(d["nE_R"] >= cm.REF_MAX_E for d in L)}
    for i, a in ((I11, "1.1"), (I101, "1.01")):
        cand = {k: cm.ratio_stats(H(k, i), H(P, i)) for k in CLASSICAL if k in L[0]["arms"]}
        ks = min(cand, key=lambda k: cand[k]["G"])
        r[a] = {"vs": ks, "G": cand[ks]["G"], "ci": cand[ks]["ci"],
                "median": {k: cm.med(H(k, i)) for k in [P] + list(cand)},
                "fail": {k: sum(x is None for x in H(k, i)) for k in [P] + list(cand)},
                "q_at_hit": {k: {q: float(np.median([d["arms"][k]["q"][i][q] for d in L if d["arms"][k]["q"][i]]))
                                 for q in ("f1", "delta")} for k in (P, ks)}}
    for key, kw in (("dist5", {"dist": 0.05}), ("dist1", {"dist": 0.01}), ("gap4", {"gap": 1e-4}), ("gap5", {"gap": 1e-5})):
        hp = [_first(d["arms"][P]["extra"], **kw) for d in L]
        cand = {k: cm.ratio_stats([_first(d["arms"][k]["extra"], **kw) for d in L], hp) for k in CLASSICAL if k in L[0]["arms"]}
        ks = min(cand, key=lambda k: cand[k]["G"])
        r[key] = {"vs": ks, "G": cand[ks]["G"], "ci": cand[ks]["ci"], "median_learned": cm.med(hp),
                  "median_vs": cm.med([_first(d["arms"][ks]["extra"], **kw) for d in L])}
    return r


def summary():
    res = {}
    lamsel = cm.load(os.path.join(OUT, "lambda.json"))
    for name in CELLS:
        Ls = [cm.load(f) for f in sorted(glob.glob(os.path.join(OUT, "testII", f"{name}_*.json")))]
        if not Ls:
            continue
        js = [d["j"] for d in Ls]
        Lp = [cm.load(os.path.join(V2, "testII", f"{name}_{j:02d}.json")) for j in js]
        res[name] = {"c_supp": lamsel[name]["c"], "c_pred": lamsel[name]["c_pred"], "dev": lamsel[name]["dev_mean"],
                     "supp": _cell_stats(Ls, "support"), "pred": _cell_stats(Lp, "prediction (paper)")}
        print(f"\n=== {name}：{len(Ls)} 个测试系统（与论文同一组）===")
        for key in ("pred", "supp"):
            s = res[name][key]
            c = res[name]["c_pred" if key == "pred" else "c_supp"]
            print(f"  c = {c:<5}（{s['label']}）参照解 F1 {s['ref']['f1']:.3f} P {s['ref']['prec']:.2f} R {s['ref']['rec']:.2f} "
                  f"δ {s['ref']['delta']:.3f} nnz {s['ref']['nnz']:.0f}；参照解撞上限 {s['nE_R_cap']}")
            for a in ("1.1", "1.01"):
                q = s[a]
                print(f"     α={a}: G = {q['G']:.2f} [{q['ci'][0]:.2f}, {q['ci'][1]:.2f}] 对 {q['vs']}；中位 " +
                      ", ".join(f"{k} {m}" for k, m in q["median"].items()) + f"；失败 {q['fail']}")
            print("     参数空间：" + "  ".join(f"{k} {s[k]['G']:.2f} [{s[k]['ci'][0]:.2f}, {s[k]['ci'][1]:.2f}]"
                                         f"（{s[k]['median_learned']} vs {s[k]['median_vs']} {s[k]['vs']}）"
                                         for k in ("dist5", "dist1", "gap4", "gap5")))
    cm.dump(res, os.path.join(OUT, "lam_summary.json"))
    print(f"\n已写 {os.path.join(OUT, 'lam_summary.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["dev", "prep", "train", "tune", "test", "summary", "all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--js", default=None)
    a = ap.parse_args()
    if a.cells:
        CELLS[:] = a.cells.split(",")
    js = [int(x) for x in a.js.split(",")] if a.js else None
    parts = ["dev", "prep", "train", "tune", "test", "summary"] if a.part == "all" else [a.part]
    for p in parts:
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {p} 开始 → {OUT}", flush=True)
        if p == "dev":
            part_dev(a.workers)
        elif p == "prep":
            part_prep(a.workers)
        elif p == "train":
            part_train(a.workers)
        elif p == "tune":
            part_tune(a.workers)
        elif p == "test":
            part_test(a.workers, js)
        else:
            summary()
        print(f"[{time.strftime('%H:%M:%S')}] {p} 完成（{time.time() - t:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
