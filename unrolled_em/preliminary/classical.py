from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(HERE), "results", "preliminary")


def _load(p):
    return json.load(open(p, encoding="utf-8"))


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path + ".tmp", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def _seed(root, *parts):
    h = hashlib.sha256("|".join([str(root)] + [str(p) for p in parts]).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63)


def _setup(pp, cell, split, j):
    from unrolled_em import core as ls
    sys = ls.make_system(cell["n"], _seed(pp["seed_root"], "sys", cell["name"], split, j),
                         cell["snr_db"], obs=cell["obs"], q=pp["q"])
    _, y = ls.simulate(sys, cell["T"], np.random.default_rng(_seed(pp["seed_root"], "y", cell["name"], split, j)))
    _, yv = ls.simulate(sys, pp["T_val"], np.random.default_rng(_seed(pp["seed_root"], "yv", cell["name"], split, j)))
    return sys, y, yv


def _lam(cell, sys, c):
    return float(c) * (cell["T"] - 1) * sys.sigbar2 / sys.q


def _project(A, rmax=0.95):
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    return A * min(1.0, rmax / rho) if rho > 0 else A


def _init(cell, sys, y, lam, return_all=False):
    from unrolled_em import core as ls
    n = sys.A.shape[0]
    if cell["obs"] == "full":
        cand = {"EIV": _project(ls.eiv(y, sys.r, sys.q, lam)), "LS": _project(ls.eiv(y, 0.0, sys.q, lam)),
                "half": 0.5 * np.eye(n)}
    else:
        cand = {"half": 0.5 * np.eye(n), "zero": np.zeros((n, n))}
    objs = {k: ls.objective(A, sys, y, lam) for k, A in cand.items()}
    best = min(objs, key=objs.get)
    if return_all:
        return cand[best], best, objs
    return cand[best]


def reference(sys, y, A0, lam, max_E, rel_tol):
    from unrolled_em import core as ls
    A = A0.copy()
    prev = None
    nE = 0
    for k in range(1, max_E + 1):
        st = ls.estep_exact(A, sys.C, sys.q, sys.r, sys.P1, y)
        obj = -st["ll"] + lam * float(np.sum(np.abs(A)))
        A = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=None)
        nE = k
        if prev is not None and abs(prev - obj) <= rel_tol * abs(prev):
            break
        prev = obj
    return A, nE


def _lambda_job(args):
    pp, ci, j = args
    from unrolled_em import core as ls
    cell = pp["cells"][ci]
    sys, y, yv = _setup(pp, cell, "dev", j)
    out = {}
    for c in pp["lambda_c"]:
        lam = _lam(cell, sys, c)
        A, nE = reference(sys, y, _init(cell, sys, y, lam), lam, pp["lambda_ref_max_E"], pp["ref"]["rel_tol"])
        out[str(c)] = ls.excess_nll(A, sys, yv)
    return cell["name"], j, out


def part_lambda(pp, workers):
    f = os.path.join(OUT, "lambda.json")
    if os.path.exists(f):
        print("lambda.json 已存在（已冻结），跳过"); return
    jobs = [(pp, ci, j) for ci in range(len(pp["cells"])) for j in range(pp["n_dev"])]
    jobs.sort(key=lambda x: -pp["cells"][x[1]]["n"] * pp["cells"][x[1]]["T"])
    res = {}
    for name, j, out in _pmap(_lambda_job, jobs, workers):
        res.setdefault(name, {})[j] = out
        print(f"  λ {name} dev{j}: " + "  ".join(f"c={c}: {v:.5f}" for c, v in out.items()), flush=True)
    sel = {}
    for name, d in res.items():
        cs = pp["lambda_c"]
        mean = {str(c): float(np.mean([d[j][str(c)] for j in d])) for c in cs}
        best = min(cs, key=lambda c: mean[str(c)])
        sel[name] = {"c": best, "dev_mean_M": mean}
    _dump({"frozen": time.strftime("%Y-%m-%d %H:%M"), "cells": sel}, f)
    print("λ 已冻结：" + ", ".join(f"{k}: c={v['c']}" for k, v in sel.items()))


def _main_job(args):
    pp, ci, j = args
    from unrolled_em import core as ls
    cell = pp["cells"][ci]
    f = os.path.join(OUT, cell["name"], f"test_{j:02d}.json")
    if os.path.exists(f):
        return f
    c = _load(os.path.join(OUT, "lambda.json"))["cells"][cell["name"]]["c"]
    sys, y, yv = _setup(pp, cell, "test", j)
    lam = _lam(cell, sys, c)
    A0, init_name, init_objs = _init(cell, sys, y, lam, return_all=True)
    t0 = time.time()
    A_R, nE_R = reference(sys, y, A0, lam, pp["ref"]["max_E"], pp["ref"]["rel_tol"])
    M_R = ls.excess_nll(A_R, sys, yv)
    tau = pp["tau_mult"] * M_R + pp["tau_abs"]
    out = {"cell": cell["name"], "j": j, "c": c, "lam": lam, "M_R": M_R, "nE_R": nE_R, "tau": tau,
           "init": init_name, "init_objs": init_objs,
           "sec_R": time.time() - t0, "M_init": ls.excess_nll(A0, sys, yv), "arms": {}}
    out["ONESHOT"] = {"M": out["M_init"], "hit": bool(out["M_init"] <= tau)}
    if cell["obs"] == "full":
        out["EIV"] = {"M": out["M_init"], "hit": bool(out["M_init"] <= tau)}
        nrm = float(np.sum(sys.A ** 2))
        out["EIV"]["nmse"] = float(np.sum((A0 - sys.A) ** 2) / nrm)
        out["nmse_R"] = float(np.sum((A_R - sys.A) ** 2) / nrm)
    l_true = ls.loglik(sys.A, sys.C, sys.q, sys.r, sys.P1, yv)
    Tm = yv.shape[0] * yv.shape[1]
    for arm in pp["arms"]:
        curve = []

        def stop(A, k):
            m = float((l_true - ls.loglik(A, sys.C, sys.q, sys.r, sys.P1, yv)) / Tm)
            curve.append((k, m))
            return m <= tau
        t1 = time.time()
        if arm["estep"] == "squarem":
            tr = ls.squarem_run(sys, y, A0, lam, K=arm["K"], max_E=pp["arm_max_E"], stop_fn=stop)
            tE, tM = None, None
        else:
            tr = ls.em_run(sys, y, A0, lam, estep=arm["estep"], K=arm["K"], max_E=pp["arm_max_E"], stop_fn=stop)
            tE, tM = float(np.sum(tr["t_E"])), float(np.sum(tr["t_M"]))
        hit = ls.first_hit([m for _, m in curve], [k for k, _ in curve], tau)
        out["arms"][arm["name"]] = {"hit_E": hit, "curve": curve, "best_M": float(min(m for _, m in curve)),
                                   "sec": time.time() - t1, "t_E": tE, "t_M": tM,
                                   "n_E_run": int(tr["n_E"][-1]),
                                   "fallbacks": int(np.sum(tr.get("fallback", [])))}
    out["sec_total"] = time.time() - t0
    _dump(out, f)
    hits = {k: v["hit_E"] for k, v in out["arms"].items()}
    print(f"  {cell['name']} test{j}: M_R {M_R:.5f} (R {nE_R} E 步) EIV命中 {out.get('EIV', {}).get('hit')} | {hits} "
          f"| {out['sec_total']:.0f}s", flush=True)
    return f


def part_main(pp, workers, cells):
    if not os.path.exists(os.path.join(OUT, "lambda.json")):
        raise SystemExit("先跑 --part lambda")
    idx = [i for i, c in enumerate(pp["cells"]) if cells is None or c["name"] in cells]
    jobs = [(pp, ci, j) for ci in idx for j in range(pp["n_test"])]
    jobs.sort(key=lambda x: -pp["cells"][x[1]]["n"] * pp["cells"][x[1]]["T"])
    print(f"主运行：{len(jobs)} 个 (格, 系统)，{workers} 个进程", flush=True)
    for _ in _pmap(_main_job, jobs, workers):
        pass


def part_verdict(pp):
    vd = pp["verdict"]
    cells = {}
    for cell in pp["cells"]:
        rows = []
        for j in range(pp["n_test"]):
            f = os.path.join(OUT, cell["name"], f"test_{j:02d}.json")
            if os.path.exists(f):
                rows.append(_load(f))
        if not rows:
            continue
        med = {}
        for arm in pp["arms"]:
            h = [r["arms"][arm["name"]]["hit_E"] for r in rows]
            med[arm["name"]] = float(np.median([101 if x is None else x for x in h]))
        best_arm = min(med, key=med.get)
        k1 = med[best_arm] <= vd["k1_max_median_E"]
        k2 = sum(r["ONESHOT"]["hit"] for r in rows) >= vd["k2_min_systems"]
        ex = [r["arms"]["EM_exact_K50"] for r in rows]
        e_frac = float(np.sum([a["t_E"] for a in ex]) / np.sum([a["t_E"] + a["t_M"] for a in ex]))
        ss = [r["arms"]["EM_ss_K50"] for r in rows]
        per_E = lambda arms: float(np.sum([a["t_E"] for a in arms]) / max(np.sum([a["n_E_run"] for a in arms]), 1))
        cells[cell["name"]] = {
            "n": len(rows), "median_hit_E": med, "best_arm": best_arm, "K1": bool(k1), "K2": k2,
            "dead": bool(k1 or bool(k2)), "E_time_frac_exact_K50": e_frac,
            "ss_speedup_per_E": per_E(ex) / max(per_E(ss), 1e-12),
            "M_R_median": float(np.median([r["M_R"] for r in rows])),
            "nE_R_median": float(np.median([r["nE_R"] for r in rows])),
            "oneshot_hits": sum(r["ONESHOT"]["hit"] for r in rows),
            "oneshot_over_R_median": float(np.median([r["ONESHOT"]["M"] / max(r["M_R"], 1e-12) for r in rows])),
            "init_choice": {k: sum(r["init"] == k for r in rows) for k in set(r["init"] for r in rows)},
            "R_hit_cap": sum(r["nE_R"] >= 200 for r in rows),
            "ss_best_over_R_median": float(np.median([r["arms"]["EM_ss_K50"]["best_M"] / max(r["M_R"], 1e-12)
                                                      for r in rows])),
            "c": rows[0]["c"]}
    big = [c for c in vd["big_cells"] if c in cells]
    n_dead = sum(cells[c]["dead"] for c in big)
    n_alive = len(big) - n_dead
    if len(big) < len(vd["big_cells"]):
        verdict = f"格没跑全（{len(big)}/{len(vd['big_cells'])}），不裁决"
    elif all(not cells[c]["dead"] for c in vd["n128_cells"]) and n_alive >= 4:
        verdict = "活：进入第 1 步"
    elif n_dead >= 4:
        verdict = "死：本方向判负"
    else:
        verdict = "灰区"
    _dump({"cells": cells, "big_alive": n_alive, "big_dead": n_dead, "verdict": verdict},
          os.path.join(OUT, "verdict.json"))
    for k, v in cells.items():
        print(f"{k:11} c={v['c']:<6} 最短链 {v['best_arm']} 中位 {v['median_hit_E']} | K1 {v['K1']} K2 {v['K2']} "
              f"→ {'死' if v['dead'] else '活'} | E步耗时占比 {v['E_time_frac_exact_K50']:.2f} ss 单步加速 "
              f"{v['ss_speedup_per_E']:.1f}× | R 中位 {v['nE_R_median']:.0f} E 步 | 初值/R {v['oneshot_over_R_median']:.2f} "
              f"| 初值 {v['init_choice']}")
    print(f"n≥64 六格：活 {n_alive} 死 {n_dead} → 裁决：{verdict}")


def _pmap(fn, jobs, workers):
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            yield from pool.imap_unordered(fn, jobs)
    else:
        for j in jobs:
            yield fn(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=os.path.join(HERE, "protocol.json"))
    ap.add_argument("--part", choices=["lambda", "main", "verdict"], required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = a.out
    pp = _load(a.protocol)
    if a.part == "lambda":
        part_lambda(pp, a.workers)
    elif a.part == "main":
        part_main(pp, a.workers, a.cells.split(",") if a.cells else None)
    else:
        part_verdict(pp)


if __name__ == "__main__":
    main()
