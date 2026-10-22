from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import math
import time

import numpy as np

from unrolled_em import core as ls
from unrolled_em import anytime as at
from unrolled_em import unroll as u2
from unrolled_em.experiments import common as cm
from unrolled_em.experiments import run as rn

CELLS = ["P32_500", "P64_500", "F64_500"]
CLASSICAL = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
CFG = {"smax": 20.0, "tag": "", "steps": cm.TRAIN_STEPS}


def set_smax(v):
    u2.S_MAX = float(v)
    u2.B0 = math.log(1.0 / (float(v) - 1.0))


def out_dir():
    base = f"smax{int(CFG['smax'])}" + (f"_{CFG['tag']}" if CFG["tag"] else "")
    return os.path.join(cm.OUT, base)


def p_new_model(name):
    return os.path.join(out_dir(), "models", f"{name}_learned.pt")


def train_job(name):
    from unrolled_em.experiments import method as me
    f = p_new_model(name)
    if os.path.exists(f):
        return f
    set_smax(CFG["smax"])
    t = time.time()
    layers, log = me.train_chain(rn.states(name, "train"), rn.states(name, "val"), "learned", cm.L_MAIN,
                                 CFG["steps"], cm.TRAIN_LR, tag=f"{name}/s_max={CFG['smax']:g}")
    me.save_layers(f, "learned", layers)
    cm.dump({"smax": CFG["smax"], "steps": CFG["steps"], "sec": time.time() - t, "log": log},
            f.replace(".pt", "_log.json"))
    print(f"训练完成 {name}（s_max = {CFG['smax']:g}）：{time.time() - t:.0f}s", flush=True)
    return f


def saturation(layer, A0, sys, y, lam, smax):
    import torch
    from unrolled_em.experiments import method as me
    st = ls.estep_ss(A0, sys.C, sys.q, sys.r, sys.P1, y)
    B = ls.mstep(A0, st["Phi"], st["Psi"], sys.q, lam, K=50)
    F = me.features(A0, B, np.zeros_like(A0), st, sys, y.shape[0])
    with torch.no_grad():
        s, _ = layer.step_params(torch.tensor(F))
    s = s.numpy()
    return {"sat": float(np.mean(s >= 0.95 * smax)), "S_med": float(np.median(s)), "S_q90": float(np.quantile(s, 0.9))}


def run_learned(layers, sys, y, yv, lam, A0, taus):
    from unrolled_em.experiments import method as me
    rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus)
    log = []
    at.run(lambda: me.run_unrolled(layers, sys, y, lam, A0, max_E=cm.MAX_E, cont="tied", log=log), sys, lam, 50, rec, A0)
    rej = [x[0] for x in log]
    return {"hits": cm.hits_from_trace(rec.trace, rec.taus),
            "reject_rate": float(np.mean([r != "ok" for r in rej])) if rej else None, "n_layers": len(rej)}


def test_job(args):
    from unrolled_em.experiments import method as me
    name, j = args
    f = os.path.join(out_dir(), "testII", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    old = cm.load(os.path.join(cm.OUT, "testII", f"{name}_{j:02d}.json"))
    z = np.load(rn.p_ref("testII", f"{name}_{j:02d}"))
    sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
    lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
    assert abs(lam - old["lam"]) <= 1e-9 * abs(lam)
    out = {"cell": name, "j": j, "taus": taus, "smax_new": CFG["smax"],
           "stored": {k: old["arms"][k]["hits"] for k in CLASSICAL + ["learned_tied_L4"]}}
    set_smax(10.0)
    lay_old = me.load_layers(rn.p_model(name, "learned"))[:cm.L_MAIN]
    out["old"] = run_learned(lay_old, sys, y, yv, lam, A0, taus)
    out["old"].update(saturation(lay_old[0], A0, sys, y, lam, 10.0))
    set_smax(CFG["smax"])
    lay_new = me.load_layers(p_new_model(name))[:cm.L_MAIN]
    out["new"] = run_learned(lay_new, sys, y, yv, lam, A0, taus)
    out["new"].update(saturation(lay_new[0], A0, sys, y, lam, CFG["smax"]))
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    i11, i101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
    h = lambda v: f"{v[i11]}/{v[i101]}"
    print(f"{name} {j:02d}: 原 {h(out['old']['hits'])}（表 {h(out['stored']['learned_tied_L4'])}，贴边 {out['old']['sat']:.2f}）"
          f"  新 {h(out['new']['hits'])}（贴边 {out['new']['sat']:.2f}，中位 S {out['new']['S_med']:.2f}） | {out['sec']:.0f}s",
          flush=True)
    return f


def summary():
    import glob
    i11, i101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
    res = {}
    for c in CELLS:
        L = [cm.load(f) for f in sorted(glob.glob(os.path.join(out_dir(), "testII", f"{c}_*.json")))]
        if not L:
            continue
        r = {"n_sys": len(L)}
        for key in ("old", "new"):
            r[key] = {"med": [cm.med([d[key]["hits"][i] for d in L]) for i in (i11, i101)],
                      "sat": float(np.mean([d[key]["sat"] for d in L])),
                      "S_med": float(np.median([d[key]["S_med"] for d in L])),
                      "reject_rate": float(np.mean([d[key]["reject_rate"] for d in L if d[key]["reject_rate"] is not None]))}
            Gs = {}
            for i, a in ((i11, "1.1"), (i101, "1.01")):
                cand = {k: cm.ratio_stats([d["stored"][k][i] for d in L], [d[key]["hits"][i] for d in L]) for k in CLASSICAL}
                best = min(cand, key=lambda k: cand[k]["G"])
                Gs[a] = {"vs": best, **cand[best]}
            r[key]["G_strongest"] = Gs
        r["repro_old"] = sum(d["old"]["hits"] == d["stored"]["learned_tied_L4"] for d in L)
        r["old_over_new"] = {a: cm.ratio_stats([d["old"]["hits"][i] for d in L], [d["new"]["hits"][i] for d in L])
                             for i, a in ((i11, "1.1"), (i101, "1.01"))}
        r["faster_slower"] = {a: (sum((d["new"]["hits"][i] or 101) < (d["old"]["hits"][i] or 101) for d in L),
                                  sum((d["new"]["hits"][i] or 101) > (d["old"]["hits"][i] or 101) for d in L))
                              for i, a in ((i11, "1.1"), (i101, "1.01"))}
        res[c] = r
        print(f"\n=== {c}（{len(L)} 个系统；原模型复现 {r['repro_old']}/{len(L)}）===")
        for key, lab in (("old", "s_max = 10"), ("new", f"s_max = {CFG['smax']:g}")):
            q = r[key]; g = q["G_strongest"]
            print(f"  {lab:12s} 中位 E 步 {q['med'][0]:4.1f}/{q['med'][1]:4.1f} | 对最强经典 G = {g['1.1']['G']:.2f} "
                  f"[{g['1.1']['ci'][0]:.2f}, {g['1.1']['ci'][1]:.2f}] / {g['1.01']['G']:.2f} "
                  f"[{g['1.01']['ci'][0]:.2f}, {g['1.01']['ci'][1]:.2f}] | 第一层贴边 {q['sat']:.3f}，中位 S {q['S_med']:.2f}"
                  f" | 拒绝率 {q['reject_rate']:.2f}")
        oo = r["old_over_new"]; fs = r["faster_slower"]
        print(f"  原/新 E 步比：{oo['1.1']['G']:.2f} [{oo['1.1']['ci'][0]:.2f}, {oo['1.1']['ci'][1]:.2f}] / "
              f"{oo['1.01']['G']:.2f} [{oo['1.01']['ci'][0]:.2f}, {oo['1.01']['ci'][1]:.2f}]（>1 表示新模型更快）；"
              f"新更快/更慢的系统数 {fs['1.1']} / {fs['1.01']}")
    cm.dump(res, os.path.join(out_dir(), "summary.json"))
    print(f"\n已写 {os.path.join(out_dir(), 'summary.json')}")


def _init(cfg):
    CFG.update(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["train", "test", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--js", default=None)
    ap.add_argument("--smax", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=cm.TRAIN_STEPS)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    CFG.update(smax=a.smax, tag=a.tag, steps=a.steps)
    if a.cells:
        CELLS[:] = a.cells.split(",")
    if a.part == "summary":
        summary()
        return
    if a.part == "train":
        jobs = [c for c in CELLS if not os.path.exists(p_new_model(c))]
        fn = train_job
    else:
        js = [int(x) for x in a.js.split(",")] if a.js else list(range(cm.N_TEST_II))
        jobs = [(c, j) for c in CELLS for j in js
                if not os.path.exists(os.path.join(out_dir(), "testII", f"{c}_{j:02d}.json"))]
        missing = [c for c in CELLS if not os.path.exists(p_new_model(c))]
        if missing:
            raise SystemExit(f"先训练：{missing}")
        fn = test_job
    print(f"{a.part}：待跑 {len(jobs)}，{a.workers} 个进程，s_max = {CFG['smax']:g} → {out_dir()}", flush=True)
    if a.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers, initializer=_init, initargs=(dict(CFG),)) as pool:
            for _ in pool.imap_unordered(fn, jobs):
                pass
    else:
        for jb in jobs:
            fn(jb)


if __name__ == "__main__":
    main()
