from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import glob
import json
import time

import numpy as np

from unrolled_em import anytime as at
from unrolled_em.experiments import common as cm

V2 = os.path.join(cm.HERE, "results", "v2")
V2C = os.path.join(cm.HERE, "results", "v2c")
G3 = os.path.join(V2, "g3")
RB = os.path.join(cm.HERE, "results", "review_b")
P = "learned_tied_L4"
A16 = "SQpkg16_ss"
ARMS = [P, "SQpkg_ss", A16]
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
CFG = {"smoke": False}


def out_dir():
    return os.path.join(RB, "time16_smoke" if CFG["smoke"] else "time16")


def _f(part, name):
    return os.path.join(out_dir(), part, name)


def _choice():
    c = cm.load(os.path.join(V2C, "choice.json"))
    return int(c["r"]), float(c["amax0"])


def _hits(sys, y, yv, lam, A0, taus, max_E):
    from unrolled_em.experiments import exp_review as X
    rec = cm.Recorder(sys, yv, taus[-1], "ss", taus=taus)
    t = time.time()
    at.run(X._new_arm(A16, sys, y, lam, A0, max_E=max_E), sys, lam, 50, rec, A0)
    return {"hits": cm.hits_from_trace(rec.trace, taus), "sec": time.time() - t}


def job_hits(key):
    part, j = key
    n = 256 if part == "hits256" else 512
    cell = cm.big_cell(n)
    fn = f"{cell['name']}_{j:02d}.json"
    if os.path.exists(_f(part, fn)):
        return
    if n == 256:
        old = cm.load(os.path.join(V2, "C", fn))
        A0 = np.load(os.path.join(V2, "refs", "C", fn[:-5] + ".npz"))["A0"]
        lam, taus = float(old["lam"]), old["taus"]
    else:
        old = cm.load(os.path.join(G3, "D", fn))
        z = np.load(os.path.join(G3, "refs", f"D_{fn[:-5]}.npz"))
        A0, lam, taus = z["A0"], float(z["lam"]), old["taus"]
    sys, y, yv, _ = cm.setup(cell, "test1g", j)
    cap = 6 if CFG["smoke"] else cm.MAX_E
    out = _hits(sys, y, yv, lam, A0, taus, cap)
    out.update(cell=cell["name"], j=j)
    cm.dump(out, _f(part, fn))
    h = out["hits"]
    print(f"  {part} {j:02d}: {A16} {h[I11]}/{h[I101]}  {out['sec']:.0f}s", flush=True)


def _patch_algo_for(layers_override=None):
    from unrolled_em.experiments import run as R
    from unrolled_em.experiments import exp_hand as EH
    from unrolled_em.experiments import exp_sqbt as SB
    from unrolled_em.experiments import exp_review as X
    r, a = _choice()
    orig = R.algo_for

    def af(name, sys, y, lam, A0, models, log):
        if name == P:
            EH.VARIANTS["_cur"] = {"r": r, "amax0": a, "safe": True}
            lay = layers_override if layers_override is not None else models["learned"][:cm.L_MAIN]
            return lambda: EH.run_hand(lay, sys, y, lam, A0, "_cur", log=log)
        if name == "SQpkg_ss":
            return SB.algo(name, sys, y, lam, A0, log)
        if name == A16:
            return X._new_arm(A16, sys, y, lam, A0)
        return orig(name, sys, y, lam, A0, models, log)
    R.algo_for = af
    return orig


def part_timing():
    from unrolled_em.experiments import run as R
    from unrolled_em.experiments import exp_cont as EC
    EC.CFG["smoke"] = False
    EC._set_smax(10.0)
    orig = _patch_algo_for()
    items = [("testII", c, j) for c in ("F32_500", "F64_500", "F128_500") for j in range(cm.N_TEST_II)]
    items += [("C", "F256_500", j) for j in range(cm.N_C)]
    if CFG["smoke"]:
        items = items[:1]
    bad = []
    warmed = False
    try:
        for src, c, j in items:
            fn = f"{c}_{j:02d}.json"
            if os.path.exists(_f("timing", fn)):
                continue
            if src == "C" and not os.path.exists(_f("hits256", fn)):
                print(f"  跳过 {fn}：先跑 --part hits256", flush=True)
                continue
            rec = cm.load(os.path.join(V2, src, fn))
            hits = {P: cm.load(os.path.join(V2C, src, fn))["arms"][P]["hits"],
                    "SQpkg_ss": (cm.load(os.path.join(V2, "sqbt", "testII", fn))["arms"]["SQpkg_ss"]["hits"] if src == "testII"
                                 else cm.load(os.path.join(V2, "sqpkg_rest", "C", fn))["hits"]),
                    A16: (cm.load(os.path.join(RB, "sq16", fn))["arms"][A16]["hits"] if src == "testII"
                          else cm.load(_f("hits256", fn))["hits"])}
            A0 = np.load(os.path.join(V2, "refs", src, fn[:-5] + ".npz"))["A0"]
            cell = rec["cell"]
            sys, y, yv, _ = cm.setup(cell, rec["split"], rec["j"])
            models = R.load_models(cell["source"])
            models["learned"] = models["learned"][:cm.L_MAIN]
            metric = cm.Recorder(sys, yv, 0.0, cell.get("metric", "exact"))
            out = {"cell": cell["name"], "n": cell["n"], "j": j, "arms": {x: {} for x in ARMS}}
            if not warmed:
                for arm in ARMS:
                    k = hits[arm][I11]
                    if k is not None:
                        R.replay(arm, sys, y, rec["lam"], A0, models, k)
                warmed = True
            for m in R.TIMING_TARGETS:
                i = cm.MULTS.index(m)
                reps = 1 if CFG["smoke"] else R.REPEATS.get(cell["n"], 1)
                for rep in range(reps):
                    for arm in ARMS:
                        k = hits[arm][i]
                        slot = out["arms"][arm].setdefault(str(m), {"h": k, "ts": []})
                        if k is None:
                            continue
                        x, t, cnt = R.replay(arm, sys, y, rec["lam"], A0, models, k)
                        assert cnt == k, (fn, arm, m, cnt, k)
                        slot["ts"].append(t)
                        if rep == 0 and metric.metric(x) > rec["taus"][i]:
                            bad.append((fn, arm, m))
                for arm in ARMS:
                    sl = out["arms"][arm][str(m)]
                    sl["t"] = float(np.median(sl["ts"])) if sl["ts"] else None
            cm.dump(out, _f("timing", fn))
            print(f"  计时 {fn[:-5]}: " + "  ".join(f"{x} " + "/".join("-" if v["t"] is None else f"{v['t']:.3f}s"
                                                                     for v in out["arms"][x].values()) for x in ARMS), flush=True)
    finally:
        R.algo_for = orig
    print(f"回放点未达标 {len(bad)} 项 {bad[:5]}", flush=True)


def part_t512():
    from unrolled_em.experiments import run as R
    from unrolled_em.experiments import method as me
    from unrolled_em.experiments import exp_cont as EC
    EC.CFG["smoke"] = False
    lay = me.load_layers(os.path.join(G3, "models", "F256_500_learned.pt"))[:cm.L_MAIN]
    da_cfg = R.load_models("F128_500")["daarem"]
    orig = _patch_algo_for(layers_override=lay)
    cell = cm.big_cell(512)
    js = [0] if CFG["smoke"] else list(range(cm.N_D))
    try:
        for j in js:
            fn = f"{cell['name']}_{j:02d}.json"
            if os.path.exists(_f("t512", fn)) or not os.path.exists(_f("hits512", fn)):
                continue
            hits = {P: cm.load(os.path.join(V2C, "D", fn))["arms"]["from_F256"]["hits"],
                    "SQpkg_ss": cm.load(os.path.join(RB, "d512", fn))["arms"]["SQpkg_ss"]["hits"],
                    A16: cm.load(_f("hits512", fn))["hits"]}
            z = np.load(os.path.join(G3, "refs", f"D_{fn[:-5]}.npz"))
            sys, y, yv, _ = cm.setup(cell, "test1g", j)
            lam, A0 = float(z["lam"]), z["A0"]
            out = {"cell": cell["name"], "n": 512, "j": j, "arms": {x: {} for x in ARMS}}
            for m in ((1.1,) if CFG["smoke"] else (1.1, 1.01)):
                i = cm.MULTS.index(m)
                for arm in ARMS:
                    k = hits[arm][i]
                    if k is None:
                        out["arms"][arm][str(m)] = {"h": None, "t": None}
                        continue
                    _, tt, cnt = R.replay(arm, sys, y, lam, A0, {"learned": lay, "daarem": da_cfg}, k)
                    out["arms"][arm][str(m)] = {"h": k, "t": tt, "cnt": cnt}
            cm.dump(out, _f("t512", fn))
            print(f"  t512 {j:02d}: " + "  ".join(f"{x} " + "/".join("-" if v["t"] is None else f"{v['t']:.1f}s" for v in o.values())
                                                for x, o in out["arms"].items()), flush=True)
    finally:
        R.algo_for = orig


def _gm(x):
    return float(np.exp(np.mean(np.log(x)))) if x else None


def _ratios(rows, m, other):
    ok, tr, er, save = 0, [], [], []
    for x in rows:
        tp = x[P].get(m, {}).get("t")
        if other == "faster":
            cand = [(x[a][m]["t"], x[a][m]["h"]) for a in ("SQpkg_ss", A16) if x[a].get(m, {}).get("t")]
            if not cand or not tp:
                continue
            to, ho = min(cand)
        else:
            to, ho = x[other].get(m, {}).get("t"), x[other].get(m, {}).get("h")
            if not (tp and to):
                continue
        ok += 1
        tr.append(to / tp)
        er.append(ho / x[P][m]["h"])
        save.append(to - tp)
    return {"G_T": _gm(tr), "G_E": _gm(er), "save_mean": float(np.mean(save)) if save else None, "n": ok,
            "t_prop_med": float(np.median([x[P][m]["t"] for x in rows if x[P].get(m, {}).get("t")])) if rows else None}


def summary():
    res = {"timing": {}, "consistency_with_timing2": {}}
    t2 = cm.load(os.path.join(V2C, "timing2_summary.json"))["timing"]
    for n, c in ((32, "F32_500"), (64, "F64_500"), (128, "F128_500"), (256, "F256_500"), (512, "F512_500")):
        part = "t512" if n == 512 else "timing"
        fs = sorted(glob.glob(_f(part, f"{c}_*.json")))
        if not fs:
            continue
        rows = [cm.load(f)["arms"] for f in fs]
        for m in ("1.1", "1.01"):
            e = {o: _ratios(rows, m, o) for o in ("SQpkg_ss", A16, "faster")}
            e["fastest_by_GT"] = min(("SQpkg_ss", A16), key=lambda o: e[o]["G_T"] or np.inf)
            e["median_t"] = {a: (float(np.median([x[a][m]["t"] for x in rows if x[a].get(m, {}).get("t")]))
                                 if any(x[a].get(m, {}).get("t") for x in rows) else None) for a in ARMS}
            res["timing"][f"{n}/{m}"] = e
            k = f"{n}/{m}"
            if k in t2:
                res["consistency_with_timing2"][k] = {"G_T_vs_SQpkg_timing2": t2[k]["all"].get("SQpkg"),
                                                      "G_T_vs_SQpkg_now": e["SQpkg_ss"]["G_T"]}
    v = cm.load(os.path.join(V2, "verdict.json"))["cost"]
    cost128 = v["F128_500"]["train_sec_L4"] + v["F128_500"]["ref_sec_total"]
    cost64 = v["F64_500"]["train_sec_L4"] + v["F64_500"]["ref_sec_total"]
    pb, pbt = {}, {}
    fs = sorted(glob.glob(_f("timing", "F128_500_*.json")))
    for m in ("1.1", "1.01"):
        key = f"128/{m}"
        if key not in res["timing"]:
            continue
        o = res["timing"][key]["fastest_by_GT"]
        pb[m] = cost128 / res["timing"][key][o]["save_mean"]
        i = I11 if m == "1.1" else I101
        sv = []
        for f in fs:
            x = cm.load(f)["arms"]
            tII = cm.load(os.path.join(V2C, "testII", os.path.basename(f)))["arms"]
            htr, hp = tII["tied_from64_L4"]["hits"][i], tII[P]["hits"][i]
            tp, to = x[P][m]["t"], x[o][m]["t"]
            if None in (htr, hp, tp, to):
                continue
            sv.append(to - tp * htr / hp)
        pbt[m] = cost64 / float(np.mean(sv)) if sv else None
    res["payback_vs_fastest"] = pb
    res["payback_transfer_vs_fastest"] = pbt
    res["payback_timing2"] = cm.load(os.path.join(V2C, "timing2_summary.json")).get("payback")
    cm.dump(res, os.path.join(out_dir(), "summary.json"))
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3))[:12000])
    print("\n本方法对两版 SQUAREM 的时间比 G_T（括号内为 E 步比 G_E）：")
    for k, e in res["timing"].items():
        print(f"  n/α = {k:9s} 默认界 {e['SQpkg_ss']['G_T']:.2f} ({e['SQpkg_ss']['G_E']:.2f})   "
              f"γ0=16 {e[A16]['G_T']:.2f} ({e[A16]['G_E']:.2f})   较快者 {e['faster']['G_T']:.2f}   按 G_T 最快：{e['fastest_by_GT']}")


def _work(args):
    key, smoke = args
    CFG["smoke"] = smoke
    job_hits(key)
    return key


def _init_worker(cfg):
    CFG.update(cfg)


def run_hits(part, workers):
    os.makedirs(os.path.join(out_dir(), part), exist_ok=True)
    N = cm.N_C if part == "hits256" else cm.N_D
    jobs = [((part, j), CFG["smoke"]) for j in ([0] if CFG["smoke"] else range(N))]
    print(f"[{time.strftime('%H:%M:%S')}] {part}：{len(jobs)} 个作业", flush=True)
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init_worker, initargs=(dict(CFG),)) as pool:
            for _ in pool.imap_unordered(_work, jobs):
                pass
    else:
        for jb in jobs:
            _work(jb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["hits256", "timing", "hits512", "t512", "summary", "smoke"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.part == "smoke":
        CFG["smoke"] = True
        run_hits("hits256", 1)
        os.makedirs(os.path.join(out_dir(), "timing"), exist_ok=True)
        part_timing()
        summary()
        return
    if a.part in ("hits256", "hits512"):
        run_hits(a.part, a.workers)
    elif a.part == "timing":
        os.makedirs(os.path.join(out_dir(), "timing"), exist_ok=True)
        part_timing()
    elif a.part == "t512":
        os.makedirs(os.path.join(out_dir(), "t512"), exist_ok=True)
        part_t512()
    else:
        summary()


if __name__ == "__main__":
    main()
