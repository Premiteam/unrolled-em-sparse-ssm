from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys as _sys
import time

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from unrolled_em import core as ls
from unrolled_em import anytime as at
from unrolled_em.experiments import common as cm
from unrolled_em.experiments import method as me
from unrolled_em.experiments import run as rn

OUT = os.path.join(cm.OUT, "advisor")
os.makedirs(OUT, exist_ok=True)


_orig_exact = ls.estep_exact


def estep_exact_ps(A, C, q, r, P1, y):
    ll, XP, PP, XF, PF = ls._forward(A, C, q, r, P1, y)
    T, n = XF.shape
    xs = XF[-1].copy(); Ps = PF[-1].copy()
    Phi = np.zeros((n, n)); Psi = np.zeros((n, n)); Psum = np.zeros((n, n))
    for t in range(T - 2, -1, -1):
        cf = cho_factor(PP[t + 1], lower=True)
        J = cho_solve(cf, A @ PF[t]).T
        xs_t = XF[t] + J @ (xs - XP[t + 1])
        Ps_t = PF[t] + J @ (Ps - PP[t + 1]) @ J.T
        Ps_t = 0.5 * (Ps_t + Ps_t.T)
        Psi += Ps @ J.T + np.outer(xs, xs_t)
        Phi += Ps_t + np.outer(xs_t, xs_t)
        Psum += Ps_t
        xs, Ps = xs_t, Ps_t
    return {"Phi": Phi, "Psi": Psi, "ll": float(ll), "Ps": Psum / (T - 1)}


ls.estep_exact = estep_exact_ps


def es(kind):
    if kind == "exact":
        return lambda A, s, y: ls.estep_exact(A, s.C, s.q, s.r, s.P1, y)
    return lambda A, s, y: ls.estep_ss(A, s.C, s.q, s.r, s.P1, y)


def layer_step(layer, A, st, Rp, sys, y, lam, K, estep, eta=me.ETA):
    B = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=K)
    delta = max(me.surrogate(A, st["Phi"], st["Psi"], sys.q, lam) - me.surrogate(B, st["Phi"], st["Psi"], sys.q, lam), 0.0)
    R = B - A
    F = me.features(A, B, Rp, st, sys, y.shape[0])
    with torch.no_grad():
        cand = layer(torch.tensor(A), torch.tensor(R), torch.tensor(Rp), torch.tensor(F)).numpy().copy()
    F_old = me.fobj(A, st, lam)
    out = {"Rp": R, "F_old": F_old, "delta": delta}
    if not np.all(np.isfinite(cand)) or float(np.max(np.abs(np.linalg.eigvals(cand)))) >= 1.0:
        stB = estep(B, sys, y)
        out.update(A=B, st=stB, nE=1, reject="rho", F_new=me.fobj(B, stB, lam))
        return out
    reason = "obj"
    try:
        st_c = estep(cand, sys, y)
        F_c = me.fobj(cand, st_c, lam)
        ok = bool(np.isfinite(st_c["ll"]) and F_c <= F_old - eta * delta)
        if not ok and np.isfinite(F_c) and F_c <= F_old:
            reason = "eta"
    except np.linalg.LinAlgError:
        ok = False
    if ok:
        out.update(A=cand, st=st_c, nE=1, reject=None, F_new=F_c)
    else:
        stB = estep(B, sys, y)
        out.update(A=B, st=stB, nE=2, reject=reason, F_new=me.fobj(B, stB, lam))
    return out


TRACKER = {}


def run_unrolled(layers, sys, y, lam, A0, K, kind, switch=False, max_E=cm.MAX_E, log=None, info=None):
    log = [] if log is None else log
    estep = es(kind)
    nE = 0
    A = A0.copy(); st = estep(A, sys, y); nE += 1
    Rp = np.zeros_like(A)
    k = 0
    switched_at = None
    while nE + 1 <= max_E:
        layer = layers[min(k, len(layers) - 1)]
        o = layer_step(layer, A, st, Rp, sys, y, lam, K, estep)
        nE += o["nE"]
        A, st, Rp = o["A"], o["st"], o["Rp"]
        log.append((o["reject"] or "ok", o["F_old"], o["F_new"], o["delta"], kind if switched_at is None else "exact"))
        k += 1
        if switch and switched_at is None and o["reject"] is not None and o["F_new"] > o["F_old"]:
            switched_at = nE
            if info is not None:
                info["switched_at"] = nE
            estep = es("exact")
            TRACKER["t"].best = None
            if nE + 1 > max_E:
                break
            st = estep(A, sys, y); nE += 1
    return {"A": A, "n_E": nE, "log": log, "switched_at": switched_at}


def anytime_run(algo, sys, lam, K, rec, A0):
    tr = at.Tracker(sys, lam, K, rec)
    TRACKER["t"] = tr
    o_ex, o_ss = ls.estep_exact, ls.estep_ss

    def wrap(fn):
        def w(A, *a, **kw):
            try:
                s = fn(A, *a, **kw)
            except np.linalg.LinAlgError:
                tr.on_fail()
                raise
            tr.on_estep(A, s)
            return s
        return w
    ls.estep_exact, ls.estep_ss = wrap(o_ex), wrap(o_ss)
    try:
        tr.start(A0)
        algo()
    except at.Hit:
        pass
    finally:
        ls.estep_exact, ls.estep_ss = o_ex, o_ss
    return tr.hit, tr


def run_arm(name, sys, y, lam, A0, rec, layers):
    log, info = [], {}
    if name == "EM_ss_K50":
        algo, K = (lambda: ls.em_run(sys, y, A0, lam, estep="ss", K=50, max_E=cm.MAX_E, stop_fn=rn.NEVER)), 50
    elif name == "EM_exact_K50":
        algo, K = (lambda: ls.em_run(sys, y, A0, lam, estep="exact", K=50, max_E=cm.MAX_E, stop_fn=rn.NEVER)), 50
    elif name == "EM_ss_full":
        algo, K = (lambda: ls.em_run(sys, y, A0, lam, estep="ss", K=None, max_E=cm.MAX_E, stop_fn=rn.NEVER)), None
    elif name == "EM_exact_full":
        algo, K = (lambda: ls.em_run(sys, y, A0, lam, estep="exact", K=None, max_E=cm.MAX_E, stop_fn=rn.NEVER)), None
    else:
        K = None if name.endswith("_full") else 50
        kind = "exact" if "_exact" in name else "ss"
        switch = name.startswith("switch")

        info["switched_at"] = None

        def algo():
            info.update(run_unrolled(layers, sys, y, lam, A0, K, kind, switch=switch, log=log, info=info))
    t = time.time()
    anytime_run(algo, sys, lam, K, rec, A0)
    out = {"sec": time.time() - t, "hits": cm.hits_from_trace(rec.trace, rec.taus), "q": rec.q, "trace": rec.trace}
    if rec.extra:
        out["extra"] = rec.extra
    if log:
        out["log"] = log
    if "switched_at" in info:
        out["switched_at"] = info["switched_at"]
    return out


ARMS_A = ["EM_ss_K50", "learned_tied_L4", "EM_exact_K50", "EM_ss_full", "EM_exact_full", "learned_ss_full",
          "learned_exact_full"]


def part_A(js=None):
    layers = me.load_layers(rn.p_model(rn.GE["name"], "learned"))[:cm.L_MAIN]
    js = range(rn.GE["n_test"]) if js is None else js
    res = {}
    for j in js:
        f = os.path.join(OUT, "A", f"test_{j:02d}.json")
        if os.path.exists(f):
            res[j] = cm.load(f); continue
        old = cm.load(os.path.join(cm.OUT, "ge", f"test_{j:02d}.json"))
        z = np.load(rn.p_ref("ge", f"test_{j:02d}"))
        sys, y, yv, _ = rn.ge_setup("test", j)
        lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
        assert abs(lam - old["lam"]) <= 1e-9 * abs(lam)
        out = {"j": j, "taus": taus, "arms": {}, "old": {k: v["hits"] for k, v in old["arms"].items()}}
        for name in ARMS_A:
            rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus)
            r = run_arm(name, sys, y, lam, A0, rec, layers)
            r.pop("trace"); r.pop("log", None)
            out["arms"][name] = r
        os.makedirs(os.path.dirname(f), exist_ok=True)
        cm.dump(out, f)
        i11 = cm.MULTS.index(1.1)
        print(f"A {j:02d}: " + "  ".join(f"{k} {v['hits'][i11]}/{v['hits'][-1]}" for k, v in out["arms"].items())
              + f" | 原 EM {out['old']['EM_ss_K50'][i11]}/{out['old']['EM_ss_K50'][-1]}"
              + f" 原学 {out['old']['learned_tied_L4'][i11]}/{out['old']['learned_tied_L4'][-1]}", flush=True)
        res[j] = out
    return res


B_SYS = [6, 16, 27]
ARMS_B = ["learned_tied_L4", "switch_L4"]


def part_B():
    layers = me.load_layers(rn.p_model("F32_500", "learned"))[:cm.L_MAIN]
    cell = cm.base_cell("F32_500")
    for j in B_SYS:
        f = os.path.join(OUT, "B", f"F32_500_{j:02d}.json")
        if os.path.exists(f):
            continue
        old = cm.load(os.path.join(cm.OUT, "testII", f"F32_500_{j:02d}.json"))
        z = np.load(rn.p_ref("testII", f"F32_500_{j:02d}"))
        sys, y, yv, _ = cm.setup(cell, "test1g", j)
        lam, A0, A_R, A_Rss = float(z["lam"]), z["A0"], z["A_R"], z["A_Rss"]
        taus, F_R, F_Rss = old["taus"], old["F_R"], old["F_Rss"]
        out = {"j": j, "taus": taus, "arms": {}, "old": {"hits": old["arms"]["learned_tied_L4"]["hits"],
                                                       "extra": old["arms"]["learned_tied_L4"]["extra"]}}
        for name in ARMS_B:
            own = "exact" if name.startswith("switch") else "ss"
            rich = {"y": y, "lam": lam, "refs": {"exact": (A_R, F_R), "ss": (A_Rss, F_Rss)}, "own": own}
            rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus, rich=rich, stop_dist=rn.STOP_DIST,
                              stop_gap=rn.STOP_GAP)
            r = run_arm(name, sys, y, lam, A0, rec, layers)
            r.pop("trace")
            out["arms"][name] = r
            print(f"B {j:02d} {name}: hits {r['hits']} 切换 {r.get('switched_at')} 末点 {r['extra'][-1]}", flush=True)
        print(f"   原 learned_tied_L4: hits {out['old']['hits']} 末点 {out['old']['extra'][-1]}", flush=True)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        cm.dump(out, f)


ARMS_C = ["EM_ss_K50", "EM_ss_full", "EM_exact_full", "learned_tied_L4", "learned_ss_full"]


def mstep_gap(sys, y, lam, A0, k_max=20):
    A = A0.copy(); out = []
    for _ in range(k_max):
        st = ls.estep_ss(A, sys.C, sys.q, sys.r, sys.P1, y)
        Bf = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=None)
        B5 = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
        out.append(float(np.linalg.norm(B5 - Bf) / max(np.linalg.norm(Bf - A), 1e-300)))
        A = Bf
    return out


def part_C(name, js):
    layers = me.load_layers(rn.p_model(name, "learned"))[:cm.L_MAIN]
    cell = cm.base_cell(name)
    for j in js:
        f = os.path.join(OUT, "C", f"{name}_{j:02d}.json")
        if os.path.exists(f):
            continue
        t0 = time.time()
        old = cm.load(os.path.join(cm.OUT, "testII", f"{name}_{j:02d}.json"))
        z = np.load(rn.p_ref("testII", f"{name}_{j:02d}"))
        sys, y, yv, _ = cm.setup(cell, "test1g", j)
        lam, A0, A_R, A_Rss = float(z["lam"]), z["A0"], z["A_R"], z["A_Rss"]
        assert abs(lam - old["lam"]) <= 1e-9 * abs(lam)
        taus, F_R, F_Rss = old["taus"], old["F_R"], old["F_Rss"]
        out = {"j": j, "taus": taus, "arms": {},
               "old": {k: {"hits": old["arms"][k]["hits"], "extra_last": old["arms"][k]["extra"][-1]}
                       for k in ("EM_ss_K50", "learned_tied_L4")},
               "mstep_gap": mstep_gap(sys, y, lam, A0)}
        for arm in ARMS_C:
            own = "exact" if "_exact" in arm else "ss"
            rich = {"y": y, "lam": lam, "refs": {"exact": (A_R, F_R), "ss": (A_Rss, F_Rss)}, "own": own}
            rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus, rich=rich, stop_dist=rn.STOP_DIST,
                              stop_gap=rn.STOP_GAP)
            r = run_arm(arm, sys, y, lam, A0, rec, layers)
            r.pop("trace"); r.pop("log", None)
            out["arms"][arm] = r
        out["sec"] = time.time() - t0
        os.makedirs(os.path.dirname(f), exist_ok=True)
        cm.dump(out, f)
        i11 = cm.MULTS.index(1.1)
        print(f"C {name} {j:02d}: " + "  ".join(f"{k} {v['hits'][i11]}/{v['hits'][-1]}/{v['extra'][-1][0]}"
                                               for k, v in out["arms"].items())
              + f" | 原 EM {out['old']['EM_ss_K50']['hits'][i11]}/{out['old']['EM_ss_K50']['hits'][-1]}"
              + f" 原学 {out['old']['learned_tied_L4']['hits'][i11]}/{out['old']['learned_tied_L4']['hits'][-1]}"
              + f" | M50 偏差中位 {np.median(out['mstep_gap']):.2e} | {out['sec']:.0f}s", flush=True)


def _c_job(job):
    part_C(job[0], [job[1]])
    return job


if __name__ == "__main__":
    which = _sys.argv[1]
    if which == "A":
        part_A([int(x) for x in _sys.argv[2:]] or None)
    elif which == "Call":
        workers = int(_sys.argv[2]) if len(_sys.argv) > 2 else 1
        jobs = [(nm, j) for nm in ("F128_500", "F64_500") for j in range(cm.N_TEST_II)
                if not os.path.exists(os.path.join(OUT, "C", f"{nm}_{j:02d}.json"))]
        print(f"C：待跑 {len(jobs)} 个系统，{workers} 个进程", flush=True)
        for _ in cm.pmap(_c_job, jobs, workers):
            pass
    elif which == "C":
        part_C(_sys.argv[2], [int(x) for x in _sys.argv[3:]] or range(cm.N_TEST_II))
    else:
        part_B()
