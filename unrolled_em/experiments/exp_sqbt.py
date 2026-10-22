# The function sq_pkg is a line-by-line Python translation of the function squarem1 of the R package
# SQUAREM (version on CRAN in October 2026) by Ravi Varadhan, distributed under the GNU General Public
# License version 2 or later. Its default arguments are the default control settings of squarem1
# (method 3, step.min0 = 1, step.max0 = 1, mstep = 4, objfn.inc = 1).
#
# Modified by Ruiqi Li in October 2026: translated from R to Python; a candidate that is not finite or
# not stable is rejected without evaluating the objective; the iteration stops when no new E-step is made.
# All other functions in this file are original work of the authors of this repository.
#
# This file is distributed under the GNU General Public License version 2. See LICENSE and NOTICE.

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
CFG = {"tag": "", "check_all": False, "lt": "pred"}
ARMS = ["SQbt3_ss", "SQbt1_ss", "SQbt2_ss", "SQpkg_ss", "SQbt3_exact"]
OLD_CLS = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
OLD_SS = ["EM_ss_K50", "SQ_ss_K50", "DAAREM_tun"]
P = "learned_tied_L4"
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
ALPHA_EPS = 0.01
K_M = 50


def root():
    return ROOTS[CFG["lt"]]


def out_dir():
    return os.path.join(root(), "sqbt" + (f"_{CFG['tag']}" if CFG["tag"] else ""))


class _E:
    def __init__(self, sys, y, lam, kind):
        self.sys, self.y, self.lam, self.kind = sys, y, lam, kind
        self.cache, self.keys = {}, []
        self.n = 0

    def st(self, X):
        key = X.tobytes()
        if key in self.cache:
            return self.cache[key]
        self.n += 1
        fn = ls.estep_ss if self.kind == "ss" else ls.estep_exact
        try:
            s = fn(X, self.sys.C, self.sys.q, self.sys.r, self.sys.P1, self.y)
            if not np.isfinite(s["ll"]):
                s = None
        except np.linalg.LinAlgError:
            s = None
        self.cache[key] = s
        self.keys.append(key)
        while len(self.keys) > 8:
            self.cache.pop(self.keys.pop(0), None)
        return s

    def F(self, X):
        s = self.st(X)
        return None if s is None else ls.mstep(X, s["Phi"], s["Psi"], self.sys.q, self.lam, K=K_M)

    def obj(self, X):
        s = self.st(X)
        return np.inf if s is None else -s["ll"] + self.lam * float(np.sum(np.abs(X)))


def _ok(X):
    return bool(np.all(np.isfinite(X))) and float(np.max(np.abs(np.linalg.eigvals(X)))) < 1.0


def _alpha(r, v, scheme):
    rr, rv, vv = float(np.sum(r * r)), float(np.sum(r * v)), float(np.sum(v * v))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = {1: rv / vv, 2: rr / rv, 3: -np.sqrt(rr / vv)}[scheme]
    return -1.0 if not np.isfinite(a) else min(float(a), -1.0)


def sq_backtrack(sys, y, A0, lam, scheme=3, kind="ss", max_E=cm.MAX_E, log=None):
    E = _E(sys, y, lam, kind)
    x = A0.copy()
    while E.n < max_E:
        n_start = E.n
        F0 = E.obj(x)
        x1 = E.F(x)
        if x1 is None or E.n >= max_E:
            return x
        x2 = E.F(x1)
        if x2 is None:
            return x1
        r, v = x1 - x, x2 - 2 * x1 + x
        a0 = a = _alpha(r, v, scheme)
        nb, n_free = 0, 0
        while True:
            if a >= -1.0 - ALPHA_EPS:
                a, xp = -1.0, x2
                break
            xp = x - 2 * a * r + a * a * v
            if _ok(xp):
                if E.n >= max_E:
                    return x2
                if E.obj(xp) <= F0:
                    break
            else:
                n_free += 1
            a = (a - 1.0) / 2.0
            nb += 1
        if log is not None:
            log.append({"a0": a0, "a": a, "nb": nb, "free": n_free})
        if a == -1.0:
            x = x2
            if E.n == n_start:
                return x
            continue
        xn = E.F(xp)
        if xn is None:
            return xp
        if E.n == n_start:
            return xn
        x = xn
    return x


def sq_pkg(sys, y, A0, lam, kind="ss", max_E=cm.MAX_E, step_min0=1.0, step_max0=1.0, mstep=4.0, objfn_inc=1.0, log=None):
    E = _E(sys, y, lam, kind)
    p = A0.copy()
    lold = E.obj(p)
    step_min, step_max = step_min0, step_max0
    while E.n < max_E:
        n_start = E.n
        p1 = E.F(p)
        if p1 is None:
            break
        q1 = p1 - p
        p2 = E.F(p1)
        if p2 is None:
            break
        q2 = p2 - p1
        sr2 = float(np.sum(q1 * q1)); d = q2 - q1; sv2 = float(np.sum(d * d))
        alpha = np.sqrt(sr2 / sv2) if sv2 > 0 else 1.0
        alpha = max(step_min, min(step_max, alpha))
        pnew = p + 2 * alpha * q1 + alpha ** 2 * d
        extrap = True
        if abs(alpha - 1.0) > 0.01:
            pnew = E.F(pnew) if _ok(pnew) else None
        if pnew is None or not np.all(np.isfinite(pnew)) or not _ok(pnew):
            fail = True
        else:
            lnew = E.obj(pnew)
            fail = (not np.isfinite(lnew)) or lnew > lold + objfn_inc
        if fail:
            pnew = p2
            lnew = E.obj(p2)
            if alpha == step_max:
                step_max = max(step_max0, step_max / mstep)
            alpha = 1.0
            extrap = False
        if alpha == step_max:
            step_max = mstep * step_max
        if step_min < 0 and alpha == step_min:
            step_min = mstep * step_min
        if log is not None:
            log.append({"alpha": float(alpha), "extrap": extrap, "step_max": float(step_max)})
        p = pnew
        if np.isfinite(lnew):
            lold = lnew
        if E.n == n_start:
            break
    return p


def algo(arm, sys, y, lam, A0, log):
    never = lambda A, k: False
    if arm == "SQ_ss_K50":
        return lambda: ls.squarem_run(sys, y, A0, lam, K=K_M, max_E=cm.MAX_E, stop_fn=never, estep="ss")
    if arm.startswith("SQbt"):
        s, kind = int(arm[4]), arm.split("_")[1]
        return lambda: sq_backtrack(sys, y, A0, lam, scheme=s, kind=kind, log=log)
    if arm == "SQpkg_ss":
        return lambda: sq_pkg(sys, y, A0, lam, kind="ss", log=log)
    raise ValueError(arm)


def test_job(args):
    name, j = args
    f = os.path.join(out_dir(), "testII", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    old = cm.load(os.path.join(root(), "testII", f"{name}_{j:02d}.json"))
    z = np.load(os.path.join(root(), "refs", "testII", f"{name}_{j:02d}.npz"))
    sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
    lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
    assert abs(lam - old["lam"]) <= 1e-9 * abs(lam)
    rich = {"y": y, "lam": lam, "refs": {"exact": (z["A_R"], old["F_R"]), "ss": (z["A_Rss"], old["F_Rss"])}}
    arms = list(ARMS) + (["SQ_ss_K50"] if (j == 0 or CFG["check_all"]) else [])
    out = {"cell": name, "j": j, "taus": taus, "arms": {}}
    for arm in arms:
        rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus, rich={**rich, "own": "exact" if "exact" in arm else "ss"},
                          stop_dist=0.01, stop_gap=1e-5)
        log = []
        t = time.time()
        at.run(algo(arm, sys, y, lam, A0, log), sys, lam, K_M, rec, A0)
        out["arms"][arm] = {"hits": cm.hits_from_trace(rec.trace, taus), "q": rec.q, "extra": rec.extra,
                            "sec": time.time() - t, "log": log}
    if "SQ_ss_K50" in out["arms"]:
        out["repro_SQ_ss"] = out["arms"]["SQ_ss_K50"]["hits"] == old["arms"]["SQ_ss_K50"]["hits"]
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    h = lambda r: f"{r['hits'][I11]}/{r['hits'][I101]}"
    print(f"  {name} {j:02d}: " + "  ".join(f"{a} {h(out['arms'][a])}" for a in arms)
          + f" | 表中 SQ_ss {h(old['arms']['SQ_ss_K50'])}, DAAREM {h(old['arms']['DAAREM_tun'])}, 本方法 {h(old['arms'][P])}"
          + ("" if "repro_SQ_ss" not in out else f" | 复现 {out['repro_SQ_ss']}") + f" | {out['sec']:.0f}s", flush=True)
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
    print(f"测试集 II：{len(jobs)} 个系统，待跑 {len(todo)}", flush=True)
    for _ in pmap(test_job, todo, workers):
        pass


def _hits_extra(arm_rec, col, thr):
    ex = arm_rec.get("extra") or []
    h = next((c for c, *rest in ex if rest[col - 1] <= thr), None)
    return h if (h is not None and h <= cm.MAX_E) else None


def part_summary(cells):
    res = {"arms": ARMS, "cells": {}}
    for c in cells:
        fs = sorted(glob.glob(os.path.join(out_dir(), "testII", f"{c}_*.json")))
        if not fs:
            continue
        new = [cm.load(f) for f in fs]
        old = [cm.load(os.path.join(root(), "testII", f"{c}_{d['j']:02d}.json")) for d in new]
        merged = [{**o["arms"], **{a: d["arms"][a] for a in ARMS}} for o, d in zip(old, new)]
        Hm = lambda a, i: [m[a]["hits"][i] for m in merged]
        ent = {"n": len(new), "repro_SQ_ss": [d.get("repro_SQ_ss") for d in new if "repro_SQ_ss" in d], "targets": {}}
        for m_ in (1.1, 1.01):
            i = cm.MULTS.index(m_)
            vs = {a: cm.ratio_stats(Hm(a, i), Hm(P, i)) for a in OLD_CLS + ARMS}
            o_old = min(OLD_CLS, key=lambda a: vs[a]["G"])
            o_new = min(OLD_CLS + ARMS, key=lambda a: vs[a]["G"])
            sq_all = ["SQ_ss_K50", "SQ_exact_K50"] + ARMS
            o_sq = min(sq_all, key=lambda a: vs[a]["G"])
            ent["targets"][str(m_)] = {
                "median": {a: cm.med(Hm(a, i)) for a in [P] + OLD_CLS + ARMS},
                "n_fail": {a: sum(x is None for x in Hm(a, i)) for a in [P] + OLD_CLS + ARMS},
                "G_vs_proposed": {a: vs[a] for a in OLD_CLS + ARMS},
                "strongest_old": {"arm": o_old, **vs[o_old]}, "strongest_new": {"arm": o_new, **vs[o_new]},
                "best_squarem": {"arm": o_sq, **vs[o_sq]},
                "new_over_oldSQss": {a: cm.ratio_stats(Hm(a, i), Hm("SQ_ss_K50", i))["G"] for a in ARMS}}
        tail = {}
        ss_new = [a for a in ARMS if a.endswith("_ss")]
        for nm, col, thr in (("dist5", 1, 0.05), ("dist1", 1, 0.01), ("gap1e-4", 3, 1e-4), ("gap1e-5", 3, 1e-5)):
            Hx = {a: [_hits_extra(m[a], col, thr) for m in merged] for a in [P] + OLD_SS + ss_new + ["SQbt3_exact"]}
            vs = {a: cm.ratio_stats(Hx[a], Hx[P]) for a in Hx if a != P}
            o_old = min(OLD_SS, key=lambda a: vs[a]["G"])
            o_new = min(OLD_SS + ss_new, key=lambda a: vs[a]["G"])
            tail[nm] = {"strongest_old": {"arm": o_old, **vs[o_old]}, "strongest_new": {"arm": o_new, **vs[o_new]},
                        "G_vs_proposed": vs, "median": {a: cm.med(x) for a, x in Hx.items()},
                        "n_fail": {a: sum(x is None for x in x_) for a, x_ in Hx.items()}}
        ent["tail"] = tail
        st = {}
        for a in ARMS:
            L = [e for d in new for e in d["arms"][a]["log"]]
            if a.startswith("SQbt"):
                st[a] = {"cycles": len(L), "accepted_first": float(np.mean([e["nb"] == 0 and e["a"] != -1.0 for e in L])) if L else None,
                         "ended_em": float(np.mean([e["a"] == -1.0 for e in L])) if L else None,
                         "mean_backtracks": float(np.mean([e["nb"] for e in L])) if L else None,
                         "mean_paid_backtracks": float(np.mean([e["nb"] - e["free"] for e in L])) if L else None,
                         "median_a0": float(np.median([e["a0"] for e in L])) if L else None}
            else:
                st[a] = {"cycles": len(L), "extrap_rate": float(np.mean([e["extrap"] for e in L])) if L else None,
                         "median_alpha": float(np.median([e["alpha"] for e in L])) if L else None,
                         "max_step_max": float(max(e["step_max"] for e in L)) if L else None}
        ent["stats"] = st
        res["cells"][c] = ent
    cm.dump(res, os.path.join(out_dir(), "summary.json"))
    for c, e in res["cells"].items():
        for m_ in ("1.1", "1.01"):
            t = e["targets"][m_]
            print(f"{cm.SHORT[c]} α={m_}: 中位 " + " ".join(f"{a} {t['median'][a]}" for a in [P, 'SQ_ss_K50', 'DAAREM_tun'] + ARMS)
                  + f" | 最强 旧 {t['strongest_old']['arm']} {t['strongest_old']['G']:.2f} → 新 {t['strongest_new']['arm']} "
                  f"{t['strongest_new']['G']:.2f} [{t['strongest_new']['ci'][0]:.2f}, {t['strongest_new']['ci'][1]:.2f}]")
        for nm in ("dist1", "gap1e-5"):
            t = e["tail"][nm]
            print(f"   尾部 {nm}: 旧 {t['strongest_old']['arm']} {t['strongest_old']['G']:.2f} → 新 {t['strongest_new']['arm']} {t['strongest_new']['G']:.2f}")
        print("   " + "  ".join(f"{a} {v}" for a, v in e["stats"].items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["test", "summary", "all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=",".join(cm.CELLS))
    ap.add_argument("--js", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--check", action="store_true", help="每个系统都另跑旧 SQ_ss_K50 核对复现")
    ap.add_argument("--lt", default="pred", choices=["pred", "supp"], help="supp：支撑集 λ 的测试集 II（results/v2_supp，F32、F64）")
    a = ap.parse_args()
    CFG.update(tag=a.tag, check_all=a.check, lt=a.lt)
    cells = a.cells.split(",")
    js = [int(x) for x in a.js.split(",")] if a.js else list(range(cm.N_TEST_II))
    if a.part in ("test", "all"):
        part_test(a.workers, cells, js)
    if a.part in ("summary", "all"):
        part_summary(cells)


if __name__ == "__main__":
    main()
