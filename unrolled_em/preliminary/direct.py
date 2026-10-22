from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import time

import numpy as np

from unrolled_em.preliminary import classical as r0

CELLS = ["F32_500", "F64_500", "F128_500", "P32_500", "P64_500"]
NEW_ARMS = ["SQ_ss_K50", "AA_exact_K50", "OWLQN", "SpaRSA"]


def _job(args):
    pp, name, j = args
    from unrolled_em import accel as ac
    from unrolled_em import core as ls
    f_out = os.path.join(r0.OUT, name, f"test_{j:02d}_0b.json")
    if os.path.exists(f_out):
        return f_out
    cell = next(c for c in pp["cells"] if c["name"] == name)
    old = r0._load(os.path.join(r0.OUT, name, f"test_{j:02d}.json"))
    sys, y, yv = r0._setup(pp, cell, "test", j)
    lam = r0._lam(cell, sys, old["c"])
    assert abs(lam - old["lam"]) <= 1e-9 * max(abs(lam), 1.0), "λ 与第 0 步不一致"
    A0, init_name, _ = r0._init(cell, sys, y, lam, return_all=True)
    assert init_name == old["init"], "初值与第 0 步不一致"
    tau = old["tau"]
    l_true = ls.loglik(sys.A, sys.C, sys.q, sys.r, sys.P1, yv)
    Tm = yv.shape[0] * yv.shape[1]
    maxE = pp["arm_max_E"]
    out = {"cell": name, "j": j, "tau": tau, "M_R": old["M_R"], "arms": {}}
    for arm in NEW_ARMS:
        curve = []

        def stop(A, k):
            m = float((l_true - ls.loglik(A, sys.C, sys.q, sys.r, sys.P1, yv)) / Tm)
            curve.append((k, m))
            return m <= tau
        t0 = time.time()
        extra = {}
        if arm == "SQ_ss_K50":
            tr = ls.squarem_run(sys, y, A0, lam, K=50, max_E=maxE, stop_fn=stop, estep="ss")
            extra["fallbacks"] = int(np.sum(tr["fallback"]))
        elif arm == "AA_exact_K50":
            tr = ac.anderson(ac.make_em_map(sys, y, lam, K=50), A0, max_evals=maxE, m=5, stop_fn=stop)
            extra.update(rejects=tr["rejects"], memory_resets=tr["memory_resets"])
        elif arm == "OWLQN":
            tr = ac.owlqn(ac.make_fg(sys, y), A0, lam, max_evals=maxE, m=10, stop_fn=stop)
        else:
            tr = ac.sparsa(ac.make_fg(sys, y), A0, lam, max_evals=maxE, stop_fn=stop)
        hit = ls.first_hit([m for _, m in curve], [k for k, _ in curve], tau)
        out["arms"][arm] = {"hit_E": hit, "curve": curve, "best_M": float(min(m for _, m in curve)),
                            "n_E_run": int(tr["n_E"][-1]), "sec": time.time() - t0, **extra}
    r0._dump(out, f_out)
    print(f"  {name} test{j}: " + "  ".join(f"{k} {v['hit_E']}" for k, v in out["arms"].items())
          + f"  | 第0步 SQ {old['arms']['SQ_exact_K50']['hit_E']} EM50 {old['arms']['EM_exact_K50']['hit_E']}", flush=True)
    return f_out


def part_verdict(pp):
    rows_out = {}
    for name in CELLS:
        med = {}
        for j in range(pp["n_test"]):
            pass
        old = [r0._load(os.path.join(r0.OUT, name, f"test_{j:02d}.json")) for j in range(pp["n_test"])]
        new = [r0._load(os.path.join(r0.OUT, name, f"test_{j:02d}_0b.json")) for j in range(pp["n_test"])
               if os.path.exists(os.path.join(r0.OUT, name, f"test_{j:02d}_0b.json"))]
        if len(new) < pp["n_test"]:
            print(f"{name}: 0b 只有 {len(new)}/{pp['n_test']}，不裁决"); return
        for arm in [a["name"] for a in pp["arms"]]:
            med[arm] = float(np.median([101 if r["arms"][arm]["hit_E"] is None else r["arms"][arm]["hit_E"] for r in old]))
        for arm in NEW_ARMS:
            med[arm] = float(np.median([101 if r["arms"][arm]["hit_E"] is None else r["arms"][arm]["hit_E"] for r in new]))
        best = min(med, key=med.get)
        per_sys_best = [min([101 if r["arms"][a]["hit_E"] is None else r["arms"][a]["hit_E"] for a in o["arms"]]
                            + [101 if r2["arms"][a]["hit_E"] is None else r2["arms"][a]["hit_E"] for a in NEW_ARMS])
                        for o, r2 in zip(old, new) for r in [o]]
        rows_out[name] = {"median_hit_E": med, "best_arm": best, "shortest": med[best],
                          "per_system_oracle_best": per_sys_best,
                          "aa_rejects": [r["arms"]["AA_exact_K50"]["rejects"] for r in new],
                          "sq_ss_fallbacks": [r["arms"]["SQ_ss_K50"]["fallbacks"] for r in new]}
    n_ok = sum(v["shortest"] >= 8 for v in rows_out.values())
    go = rows_out["F128_500"]["shortest"] >= 8 and n_ok >= 3
    verdict = "进入第 1 步" if go else "本方向判负"
    r0._dump({"cells": rows_out, "n_cells_ge8": n_ok, "verdict": verdict}, os.path.join(r0.OUT, "verdict_0b.json"))
    for k, v in rows_out.items():
        print(f"{k:9} 最短链 {v['shortest']:.1f}（{v['best_arm']}）| " +
              " ".join(f"{a}:{m:g}" for a, m in v["median_hit_E"].items()) +
              f" | 逐系统最好（事后挑，不作判据）{v['per_system_oracle_best']}")
    print(f"最短链 ≥ 8 的格：{n_ok}/5；F128_500 {rows_out['F128_500']['shortest']} → 裁决（design 14.3）：{verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "verdict"], required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    pp = r0._load(os.path.join(r0.HERE, "protocol.json"))
    if a.part == "main":
        jobs = [(pp, name, j) for name in CELLS for j in range(pp["n_test"])]
        jobs.sort(key=lambda x: -next(c for c in pp["cells"] if c["name"] == x[1])["n"])
        print(f"0b：{len(jobs)} 个 (格, 系统)，{a.workers} 个进程", flush=True)
        for _ in r0._pmap(_job, jobs, a.workers):
            pass
    else:
        part_verdict(pp)


if __name__ == "__main__":
    main()
