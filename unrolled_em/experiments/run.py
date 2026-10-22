from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import glob
import re
import time

import numpy as np

from unrolled_em.experiments import common as cm

OUT = cm.OUT
GRID_DA = [(o, k) for o in (3, 5, 10) for k in (0, 5, 10, 25)]
DEFAULT_DA = (5, 25)
NEVER = lambda A, k: False
MU_VARIANTS = ("j", "max", "or")
ABL_ARMS = ["scalar_tied", "sec0_tied", "ait0_tied", "mu0_tied", "ait_t_tied", "mu_t_tied"]
SENS_ARMS = [f"learned_tied_L{l}" for l in range(1, cm.L_MAX + 1)]
CLS_ARMS_I = ["SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]


GE = {"name": "GE16_1000", "n": 16, "T": 1000, "obs": "full", "blocks": [3, 5, 5, 3], "specnorm": 0.99, "sigma": 0.1,
      "sigP": 1e-4, "n_train": 48, "n_val": 12, "n_test": 50, "source": "GE16_1000"}
if cm.SMOKE:
    GE.update(n=5, T=150, blocks=[2, 3], n_train=3, n_val=2, n_test=2)


def ge_setup(split, j):
    from scipy.linalg import solve_discrete_lyapunov
    from unrolled_em import core as ls
    rng = np.random.default_rng(cm.seed("sys", GE["name"], split, j))
    n = GE["n"]
    A = np.zeros((n, n))
    i = 0
    for b in GE["blocks"]:
        B = rng.standard_normal((b, b))
        B *= GE["specnorm"] / np.linalg.norm(B, 2)
        A[i:i + b, i:i + b] = B
        i += b
    q = r = GE["sigma"] ** 2
    Sig = solve_discrete_lyapunov(A, q * np.eye(n))
    true = ls.Sys(A=A, C=np.eye(n), q=float(q), r=float(r), P1=GE["sigP"] ** 2 * np.eye(n), sigbar2=float(np.trace(Sig) / n))
    _, y = ls.simulate(true, GE["T"], np.random.default_rng(cm.seed("y", GE["name"], split, j)))
    _, yv = ls.simulate(true, cm.T_VAL if not cm.SMOKE else 100, np.random.default_rng(cm.seed("yv", GE["name"], split, j)))
    s2 = cm.scale_hat(y, true.C, true.r)
    sys = ls.Sys(A=A, C=true.C, q=true.q, r=true.r, P1=true.P1, sigbar2=s2)
    return sys, y, yv, {"sigbar2_true": true.sigbar2, "s2_hat": s2}


def any_setup(cell, split, j):
    return ge_setup(split, j) if cell["name"] == GE["name"] else cm.setup(cell, split, j)


def cell_by_name(name):
    return dict(GE) if name == GE["name"] else cm.base_cell(name)


TRAIN_CELLS = cm.CELLS + [GE["name"]]


def n_trva(name):
    return (GE["n_train"], GE["n_val"]) if name == GE["name"] else cm.N_TRVA[name]


def p_data(name, split, j):
    return os.path.join(OUT, "data", f"{name}_{split}_{j:02d}.npz")


def p_model(name, arm):
    return os.path.join(OUT, "models", f"{name}_{arm}.pt")


def p_ref(part, tag):
    return os.path.join(OUT, "refs", part, f"{tag}.npz")


def _lambda_job(args):
    from unrolled_em import core as ls
    name, j = args
    f = os.path.join(OUT, "lambda", f"{name}_{j}.json")
    if os.path.exists(f):
        return f
    cell = cell_by_name(name)
    sys, y, yv, info = any_setup(cell, "dev", j)
    res = {}
    for c in cm.LAMBDA_C:
        lam = cm.lam_of(c, cell, sys)
        A0, _ = cm.init(cell, sys, y, lam)
        A, _, _ = cm.reference(sys, y, A0, lam, max_E=cm.LAMBDA_REF_MAX_E)
        res[str(c)] = ls.excess_nll(A, sys, yv)
    cm.dump({"cell": name, "j": j, "M": res, **info}, f)
    print(f"  λ {name} dev{j}: " + "  ".join(f"c={c}: {v:.5f}" for c, v in res.items()), flush=True)
    return f


def part_lambda(workers):
    f = os.path.join(OUT, "lambda.json")
    if os.path.exists(f):
        print("lambda.json 已存在，跳过"); return
    jobs = [(nm, j) for nm in TRAIN_CELLS for j in range(cm.N_DEV)]
    jobs.sort(key=lambda x: -cell_by_name(x[0])["n"])
    for _ in cm.pmap(_lambda_job, jobs, workers):
        pass
    sel = {}
    for nm in TRAIN_CELLS:
        rs = [cm.load(os.path.join(OUT, "lambda", f"{nm}_{j}.json")) for j in range(cm.N_DEV)]
        mean = {str(c): float(np.mean([r["M"][str(c)] for r in rs])) for c in cm.LAMBDA_C}
        best = min(cm.LAMBDA_C, key=lambda c: mean[str(c)])
        sel[nm] = {"c": best, "dev_mean_M": mean}
    cm.dump(sel, f)
    print("λ 冻结：" + ", ".join(f"{k}: c={v['c']}" for k, v in sel.items()), flush=True)


def _prep_job(args):
    name, split, j = args
    f = p_data(name, split, j)
    if os.path.exists(f):
        return f
    cell = cell_by_name(name)
    sys, y, _, info = any_setup(cell, split, j)
    lam = cm.lam_of(cm.c_of(name), cell, sys)
    A0, init_name = cm.init(cell, sys, y, lam)
    t = time.time()
    A_R, nE, _ = cm.reference(sys, y, A0, lam)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, A_R=A_R, A0=A0, lam=lam, nE=nE, s2_hat=info["s2_hat"], sigbar2_true=info["sigbar2_true"], sec=time.time() - t)
    print(f"  参照解 {name} {split}{j:02d}: {nE} E 步 {time.time() - t:.0f}s", flush=True)
    return f


def part_prep(workers):
    jobs = [(nm, s, j) for nm in TRAIN_CELLS for s, k in zip(("train", "val"), n_trva(nm)) for j in range(k)]
    jobs.sort(key=lambda x: -cell_by_name(x[0])["n"] ** 3 * cell_by_name(x[0])["T"])
    todo = [jb for jb in jobs if not os.path.exists(p_data(*jb))]
    print(f"参照解：{len(jobs)} 个，待算 {len(todo)}", flush=True)
    for _ in cm.pmap(_prep_job, todo, workers):
        pass


def states(name, split):
    from unrolled_em.experiments import method as me
    cell = cell_by_name(name)
    out = []
    for j in range(n_trva(name)[0 if split == "train" else 1]):
        z = np.load(p_data(name, split, j))
        sys, y, _, _ = any_setup(cell, split, j)
        lam = float(z["lam"])
        cands = cm.init_candidates(cell, sys, y, lam)
        assert any(np.array_equal(z["A0"], A) for A in cands.values())
        out += me.make_states(sys, y, lam, z["A_R"], cands)
    return out


def _musel_job(args):
    from unrolled_em.experiments import method as me
    name, variant = args
    f = os.path.join(OUT, "musel", f"{name}_{variant}.json")
    if os.path.exists(f):
        return f
    _, log = me.train_chain(states(name, "train"), states(name, "val"), "mu_t", 1, cm.TRAIN_STEPS, cm.TRAIN_LR,
                            mu_variant=variant, tag=f"{name}/μ̄={variant}")
    cm.dump({"cell": name, "variant": variant, "ratio": log[0]["ratio"]}, f)
    return f


def part_musel(workers):
    f = os.path.join(OUT, "mu_choice.json")
    if os.path.exists(f):
        return
    jobs = sorted([(c, v) for c in cm.CELLS for v in MU_VARIANTS], key=lambda x: -cm.base_cell(x[0])["n"])
    for _ in cm.pmap(_musel_job, jobs, workers):
        pass
    ratios = {v: [cm.load(os.path.join(OUT, "musel", f"{c}_{v}.json"))["ratio"] for c in cm.CELLS] for v in MU_VARIANTS}
    mean = {v: float(np.mean(x)) for v, x in ratios.items()}
    best = min(mean, key=mean.get)
    if mean["max"] <= mean[best] * 1.01:
        best = "max"
    cm.dump({"ratios": ratios, "mean": mean, "choice": best}, f)
    print(f"μ̄ 变体：{mean} → {best}", flush=True)


def mu_variant():
    return cm.load(os.path.join(OUT, "mu_choice.json"))["choice"]


def train_specs():
    sp = []
    for c in cm.CELLS:
        sp += [(c, "learned", cm.L_MAX), (c, "scalar", cm.L_ABL), (c, "ait_t", cm.L_ABL), (c, "mu_t", cm.L_ABL)]
    if GE["name"] in TRAIN_CELLS:
        sp += [(GE["name"], "learned", cm.L_MAIN), (GE["name"], "scalar", cm.L_MAIN)]
    return sp


def _train_job(args):
    from unrolled_em.experiments import method as me
    name, arm, L = args
    f = p_model(name, arm)
    if os.path.exists(f):
        return f
    mv = mu_variant() if arm == "mu_t" else None
    layers, log = me.train_chain(states(name, "train"), states(name, "val"), arm, L, cm.TRAIN_STEPS, cm.TRAIN_LR,
                                 mu_variant=mv, tag=name)
    me.save_layers(f, arm, layers, mv)
    cm.dump(log, f.replace(".pt", "_log.json"))
    return f


def part_train(workers):
    jobs = [s for s in train_specs() if not os.path.exists(p_model(s[0], s[1]))]
    jobs.sort(key=lambda s: -cell_by_name(s[0])["n"] ** 3 * s[2])
    print(f"训练：{len(jobs)} 条链", flush=True)
    for _ in cm.pmap(_train_job, jobs, workers):
        pass


def load_models(name):
    from unrolled_em.experiments import method as me
    out = {}
    for arm in ("learned", "scalar", "ait_t", "mu_t"):
        f = p_model(name, arm)
        if os.path.exists(f):
            out[arm] = me.load_layers(f)
    out["mu_variant"] = mu_variant() if os.path.exists(os.path.join(OUT, "mu_choice.json")) else None
    tc = os.path.join(OUT, "tune_choice.json")
    if os.path.exists(tc):
        out["daarem"] = tuple(cm.load(tc)[name]["config"])
    return out


def _gm(hs):
    v = [1 if h == 0 else (cm.FAIL if h is None else h) for h in hs]
    return float(np.exp(np.mean(np.log(v))))


def _tune_job(args):
    from unrolled_em import core as ls
    name, j = args
    f = os.path.join(OUT, "tune", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    cell = cell_by_name(name)
    z = np.load(p_data(name, "val", j))
    sys, y, yv, _ = any_setup(cell, "val", j)
    lam = float(z["lam"])
    A0 = z["A0"]
    tau = cm.tau_of(ls.excess_nll(z["A_R"], sys, yv), 1.1)
    out = {"cell": name, "j": j, "tau": tau, "hits": {}}
    for o, k in GRID_DA:
        rec = cm.Recorder(sys, yv, tau)
        out["hits"][f"{o},{k}"] = run_arm("DAAREM", sys, y, lam, A0, rec, {"daarem": (o, k)})["hits_single"]
    rec = cm.Recorder(sys, yv, tau)
    out["SQ_ss_K50"] = run_arm("SQ_ss_K50", sys, y, lam, A0, rec, {})["hits_single"]
    cm.dump(out, f)
    print(f"  调参 {name} val{j:02d}: " + "  ".join(f"{g} {h}" for g, h in out["hits"].items()) + f" | SQ_ss {out['SQ_ss_K50']}",
          flush=True)
    return f


def part_tune(workers):
    f = os.path.join(OUT, "tune_choice.json")
    if os.path.exists(f):
        return
    jobs = sorted([(nm, j) for nm in TRAIN_CELLS for j in range(n_trva(nm)[1])], key=lambda x: -cell_by_name(x[0])["n"])
    for _ in cm.pmap(_tune_job, jobs, workers):
        pass
    choice = {}
    for nm in TRAIN_CELLS:
        rs = [cm.load(os.path.join(OUT, "tune", f"{nm}_{j:02d}.json")) for j in range(n_trva(nm)[1])]
        gm = {g: _gm([r["hits"][f"{g[0]},{g[1]}"] for r in rs]) for g in GRID_DA}
        best = min(gm.values())
        tied = [g for g in gm if gm[g] <= best * (1 + 1e-12)]
        cfg = DEFAULT_DA if DEFAULT_DA in tied else sorted(tied)[0]
        choice[nm] = {"config": list(cfg), "gm": {f"{g[0]},{g[1]}": x for g, x in gm.items()},
                      "gm_SQ_ss": _gm([r["SQ_ss_K50"] for r in rs])}
        print(f"  DAAREM {nm}: (order, κ) = {cfg}，几何平均 {gm[cfg]:.2f}（SQ_ss {choice[nm]['gm_SQ_ss']:.2f}）", flush=True)
    cm.dump(choice, f)


def algo_for(name, sys, y, lam, A0, models, log):
    from unrolled_em import core as ls
    from unrolled_em import daarem as da
    from unrolled_em.experiments import method as me
    mx = cm.MAX_E
    if name == "EM_ss_K50":
        return lambda: ls.em_run(sys, y, A0, lam, estep="ss", K=50, max_E=mx, stop_fn=NEVER)
    if name in ("SQ_ss_K50", "SQ_exact_K50"):
        est = "exact" if "exact" in name else "ss"
        return lambda: ls.squarem_run(sys, y, A0, lam, K=50, max_E=mx, stop_fn=NEVER, estep=est)
    if name in ("DAAREM", "DAAREM_tun"):
        o, k = models["daarem"]
        return lambda: da.daarem_run(sys, y, A0, lam, K=50, max_E=mx, estep="ss", order=o, kappa=k)
    m = re.fullmatch(r"learned_tied_L(\d)", name)
    if m:
        lay, cont = models["learned"][:int(m.group(1))], "tied"
    elif name == "learned_L4_sq":
        lay, cont = models["learned"][:cm.L_MAIN], "sq"
    elif name in ("tied_from32_L4", "tied_from64_L4"):
        lay, cont = models["learned_from"][:cm.L_MAIN], "tied"
    elif name == "scalar_tied":
        lay, cont = models["scalar"][:cm.L_ABL], "tied"
    elif name in ("ait_t_tied", "mu_t_tied"):
        lay, cont = models[name[:-5]][:cm.L_ABL], "tied"
    elif name in ("sec0_tied", "ait0_tied", "mu0_tied"):
        lay, cont = me.zero_layers(name[:-6], cm.L_ABL, models.get("mu_variant")), "tied"
    else:
        raise ValueError(name)
    return lambda: me.run_unrolled(lay, sys, y, lam, A0, max_E=mx, cont=cont, log=log)


def run_arm(name, sys, y, lam, A0, rec, models):
    from unrolled_em import anytime as at
    log = []
    t = time.time()
    hit, _ = at.run(algo_for(name, sys, y, lam, A0, models, log), sys, lam, 50, rec, A0)
    out = {"sec": time.time() - t, "trace": rec.trace, "hits_single": hit if (hit is not None and hit <= cm.MAX_E) else None}
    if rec.taus:
        out["hits"] = cm.hits_from_trace(rec.trace, rec.taus)
        out["q"] = rec.q
    if rec.extra:
        out["extra"] = rec.extra
    if log:
        out["log"] = [x[0] for x in log]
        out["F"] = [(x[1], x[2], x[3]) for x in log]
    return out


def _test_common(cell, split, j, part, tag, kind="exact", need_ss=False):
    from unrolled_em import core as ls
    sys, y, yv, info = any_setup(cell, split, j)
    lam = cm.lam_of(cm.c_of(cell["source"]), cell, sys)
    A0, init_name = cm.init(cell, sys, y, lam)
    t = time.time()
    A_R, nE_R, rinfo = cm.reference(sys, y, A0, lam, kind=kind)
    sec_R = time.time() - t
    f = p_ref(part, tag)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    extra_ref = {}
    if need_ss:
        A_Rss, nE_Rss, _ = cm.reference(sys, y, A0, lam, kind="ss", max_E=REF_SS_MAX_E, rel_tol=REF_SS_TOL)
        F_Rss = -cm.loglik_ss(A_Rss, sys.C, sys.q, sys.r, y) + lam * float(np.sum(np.abs(A_Rss)))
        extra_ref = {"A_Rss": A_Rss}
    np.savez(f, A_R=A_R, A0=A0, lam=lam, **extra_ref)
    if kind == "exact":
        M_R = ls.excess_nll(A_R, sys, yv)
        F_R = -ls.loglik(A_R, sys.C, sys.q, sys.r, sys.P1, y) + lam * float(np.sum(np.abs(A_R)))
    else:
        rec0 = cm.Recorder(sys, yv, 0.0, "ss")
        M_R = rec0.metric(A_R)
        F_R = -cm.loglik_ss(A_R, sys.C, sys.q, sys.r, y) + lam * float(np.sum(np.abs(A_R)))
    taus = [cm.tau_of(M_R, m) for m in cm.MULTS]
    M_m50 = None
    if rinfo["A_m50"] is not None:
        M_m50 = ls.excess_nll(rinfo["A_m50"], sys, yv) if kind == "exact" else cm.Recorder(sys, yv, 0.0, "ss").metric(rinfo["A_m50"])
    head = {"M_R_m50": M_m50, "cell": cell, "split": split, "j": j, "init": init_name, "lam": lam, "M_R": M_R, "F_R": F_R, "nE_R": nE_R,
            "ref_last_rel_change": rinfo["last_rel_change"], "sec_R": sec_R, "taus": taus,
            "q_R": cm.quality(A_R, sys.A), "q_A0": cm.quality(A0, sys.A), **info}
    if need_ss:
        head.update(nE_Rss=nE_Rss, F_Rss=F_Rss, q_Rss=cm.quality(A_Rss, sys.A),
                    dist_Rss_R=float(np.linalg.norm(A_Rss - A_R) / np.linalg.norm(A_R)))
        rich = {"y": y, "lam": lam, "refs": {"exact": (A_R, F_R), "ss": (A_Rss, F_Rss)}}
        return sys, y, yv, lam, A0, A_R, F_R, taus, head, rich
    return sys, y, yv, lam, A0, A_R, F_R, taus, head


REF_SS_MAX_E, REF_SS_TOL = 2000, 1e-11
STOP_DIST, STOP_GAP = 0.01, 1e-5


def _run_arms(arms, sys, y, yv, lam, A0, taus, models, kind="exact", rich=None, keep_trace=True):
    out = {}
    for name in arms:
        rr = None if rich is None else {**rich, "own": "exact" if "exact" in name else "ss"}
        rec = cm.Recorder(sys, yv, taus[-1], kind, taus=taus, rich=rr,
                          stop_dist=STOP_DIST if rich else None, stop_gap=STOP_GAP if rich else None)
        res = run_arm(name, sys, y, lam, A0, rec, models)
        res.pop("hits_single", None)
        if not keep_trace:
            res.pop("trace", None)
        out[name] = res
    return out


def _print_job(label, out):
    i11 = cm.MULTS.index(1.1)
    print(f"  {label}: " + "  ".join(f"{k} {v['hits'][i11]}/{v['hits'][-1]}" for k, v in out["arms"].items())
          + f" | {out['sec']:.0f}s", flush=True)


def _testI_job(args):
    name, j = args
    f = os.path.join(OUT, "testI", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    cell = cm.base_cell(name)
    t0 = time.time()
    sys, y, yv, lam, A0, A_R, F_R, taus, head = _test_common(cell, "test1c", j, "testI", f"{name}_{j:02d}")
    models = load_models(name)
    arms = SENS_ARMS + ABL_ARMS + CLS_ARMS_I
    out = {**head, "arms": _run_arms(arms, sys, y, yv, lam, A0, taus, models)}
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    _print_job(f"I {name} {j:02d}", out)
    return f


def part_testI(workers):
    jobs = sorted([(c, j) for c in cm.CELLS for j in range(cm.N_TEST_I)], key=lambda x: -cm.base_cell(x[0])["n"])
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(OUT, "testI", f"{jb[0]}_{jb[1]:02d}.json"))]
    print(f"测试集 I：{len(jobs)} 个系统，待跑 {len(todo)}", flush=True)
    for _ in cm.pmap(_testI_job, todo, workers):
        pass


def arms_II(name):
    a = ["learned_tied_L4", "learned_tied_L3", "EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
    if name == "F64_500":
        a.append("tied_from32_L4")
    if name == "F128_500":
        a += ["tied_from64_L4", "learned_L4_sq"]
    return a


def _testII_job(args):
    name, j = args
    f = os.path.join(OUT, "testII", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    cell = cm.base_cell(name)
    t0 = time.time()
    sys, y, yv, lam, A0, A_R, F_R, taus, head, rich = _test_common(cell, "test1g", j, "testII", f"{name}_{j:02d}",
                                                                  need_ss=True)
    models = load_models(name)
    src = {"F64_500": "F32_500", "F128_500": "F64_500"}.get(name)
    if src:
        models["learned_from"] = load_models(src)["learned"]
    out = {**head, "arms": _run_arms(arms_II(name), sys, y, yv, lam, A0, taus, models, rich=rich)}
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    _print_job(f"II {name} {j:02d}", out)
    return f


def part_testII(workers):
    jobs = sorted([(c, j) for c in cm.CELLS for j in range(cm.N_TEST_II)], key=lambda x: -cm.base_cell(x[0])["n"])
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(OUT, "testII", f"{jb[0]}_{jb[1]:02d}.json"))]
    print(f"测试集 II：{len(jobs)} 个系统，待跑 {len(todo)}", flush=True)
    for _ in cm.pmap(_testII_job, todo, workers):
        pass


ARMS_BC = ["learned_tied_L4", "SQ_ss_K50", "DAAREM_tun"]


def _B_job(args):
    src, shift, j = args
    f = os.path.join(OUT, "B", f"{src}_{shift}_{j:02d}.json")
    if os.path.exists(f):
        return f
    cell = cm.shift_cell(src, shift)
    t0 = time.time()
    sys, y, yv, lam, A0, A_R, F_R, taus, head = _test_common(cell, f"g_{shift}", j, "B", f"{src}_{shift}_{j:02d}")
    out = {**head, "arms": _run_arms(ARMS_BC, sys, y, yv, lam, A0, taus, load_models(src), keep_trace=False)}
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    _print_job(f"B {src} {shift} {j:02d}", out)
    return f


def part_B(workers):
    jobs = [(s, c, j) for s in cm.SOURCES_B for c in cm.SHIFTS for j in range(cm.N_B)]
    jobs.sort(key=lambda x: -cm.shift_cell(x[0], x[1])["n"] ** 3 * cm.shift_cell(x[0], x[1])["T"])
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(OUT, "B", f"{jb[0]}_{jb[1]}_{jb[2]:02d}.json"))]
    print(f"分布偏移：{len(jobs)} 个系统，待跑 {len(todo)}", flush=True)
    for _ in cm.pmap(_B_job, todo, workers):
        pass


def _big_job(args):
    n, j = args
    part = "C" if n == 256 else "D"
    cell = cm.big_cell(n)
    f = os.path.join(OUT, part, f"{cell['name']}_{j:02d}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    sys, y, yv, lam, A0, A_R, F_R, taus, head = _test_common(cell, "test1g", j, part, f"{cell['name']}_{j:02d}", kind="ss")
    out = {**head, "arms": _run_arms(ARMS_BC, sys, y, yv, lam, A0, taus, load_models("F128_500"), kind="ss")}
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    _print_job(f"{part} n={n} {j:02d}", out)
    return f


def part_big(n, workers):
    N = cm.N_C if n == 256 else cm.N_D
    for _ in cm.pmap(_big_job, [(n, j) for j in range(N)], workers):
        pass


ARMS_GE = ["learned_tied_L4", "scalar_tied", "EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]


def _ge_job(j):
    f = os.path.join(OUT, "ge", f"test_{j:02d}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    sys, y, yv, lam, A0, A_R, F_R, taus, head = _test_common(dict(GE), "test", j, "ge", f"test_{j:02d}")
    out = {**head, "arms": _run_arms(ARMS_GE, sys, y, yv, lam, A0, taus, load_models(GE["name"]), keep_trace=False)}
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    _print_job(f"GE {j:02d}", out)
    return f


def part_ge(workers):
    for _ in cm.pmap(_ge_job, list(range(GE["n_test"])), workers):
        pass


N_DIAG = 3 if not cm.SMOKE else 1
DIAG_EM_ITERS = 150 if not cm.SMOKE else 20


def _diag_job(args):
    import torch
    from unrolled_em import core as ls
    from unrolled_em.experiments import method as me
    name, j = args
    f = os.path.join(OUT, "diag", f"{name}_{j:02d}.npz")
    if os.path.exists(f):
        return f
    cell = cm.base_cell(name)
    sys, y, yv, info = cm.setup(cell, "test1c", j)
    z = np.load(p_ref("testI", f"{name}_{j:02d}"))
    A_R, A0, lam = z["A_R"], z["A0"], float(z["lam"])
    A = A0.copy()
    err = [A - A_R]
    for _ in range(DIAG_EM_ITERS):
        st = ls.estep_ss(A, sys.C, sys.q, sys.r, sys.P1, y)
        A = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
        err.append(A - A_R)
    err = np.array(err)
    stR = ls.estep_ss(A_R, sys.C, sys.q, sys.r, sys.P1, y)
    T = y.shape[0]
    muR = np.clip((T - 1) * np.diag(stR["Ps"]) / np.diag(stR["Phi"]), 0, 1)
    layers = load_models(name)["learned"][:cm.L_MAIN]
    A = A0.copy(); st = ls.estep_ss(A, sys.C, sys.q, sys.r, sys.P1, y); Rp = np.zeros_like(A)
    S_all, M_all, F_all = [], [], []
    for ly in layers:
        B = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
        F = me.features(A, B, Rp, st, sys, T)
        with torch.no_grad():
            s, m = ly.step_params(torch.tensor(F))
        S_all.append(s.numpy()); M_all.append(m.numpy()); F_all.append(F)
        o = me.layer_step(ly, A, st, Rp, sys, y, lam)
        A, st, Rp = o["A"], o["st"], o["Rp"]
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez_compressed(f, err=err.astype(np.float32), A_R=A_R, A_true=sys.A, muR=muR, S=np.array(S_all),
                        M=np.array(M_all), F=np.array(F_all).astype(np.float32))
    print(f"  诊断 {name} {j:02d}", flush=True)
    return f


def part_diag(workers):
    jobs = [(c, j) for c in cm.CELLS for j in range(N_DIAG)]
    for _ in cm.pmap(_diag_job, jobs, workers):
        pass


TIMING_ARMS = ["learned_tied_L4", "SQ_ss_K50", "DAAREM_tun"]
TIMING_TARGETS = [1.1, 1.01]
REPEATS = {32: 5, 64: 5, 128: 3, 256: 1}


class _StopRun(Exception):
    pass


def replay(arm, sys, y, lam, A0, models, k, K=50):
    from unrolled_em import core as ls
    if k == 0:
        return A0.copy(), 0.0, 0
    state = {"c": 0, "best": None}
    orig = ls.estep_ss

    def wrapped(A, *a, **kw):
        try:
            st = orig(A, *a, **kw)
        except np.linalg.LinAlgError:
            state["c"] += 1
            raise
        state["c"] += 1
        F = -st["ll"] + lam * float(np.sum(np.abs(A)))
        if np.isfinite(F) and (state["best"] is None or F <= state["best"][0]):
            state["best"] = (F, A.copy(), st)
        if state["c"] >= k and state["best"] is not None:
            raise _StopRun
        return st
    run = algo_for(arm, sys, y, lam, A0, models, [])
    ls.estep_ss = wrapped
    try:
        t0 = time.perf_counter()
        try:
            run()
        except _StopRun:
            pass
        _, Ab, sb = state["best"]
        x = ls.mstep(Ab, sb["Phi"], sb["Psi"], sys.q, lam, K=K)
        t = time.perf_counter() - t0
    finally:
        ls.estep_ss = orig
    return x, t, state["c"]


def part_timing():
    fs = [f for c in ("F32_500", "F64_500", "F128_500") for f in sorted(glob.glob(os.path.join(OUT, "testII", f"{c}_*.json")))]
    fs += sorted(glob.glob(os.path.join(OUT, "C", "*.json")))
    print(f"计时：{len(fs)} 个系统（单进程）", flush=True)
    bad = []
    for f in fs:
        rec = cm.load(f)
        cell = rec["cell"]
        out_f = os.path.join(OUT, "timing", os.path.basename(f))
        if os.path.exists(out_f):
            continue
        sys, y, yv, _ = cm.setup(cell, rec["split"], rec["j"])
        lam = rec["lam"]
        A0 = np.load(p_ref(os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-5]))["A0"]
        models = load_models(cell["source"])
        metric = cm.Recorder(sys, yv, 0.0, cell.get("metric", "exact"))
        out = {"cell": cell["name"], "n": cell["n"], "j": rec["j"], "arms": {}}
        for arm in TIMING_ARMS:
            out["arms"][arm] = {}
            for m in TIMING_TARGETS:
                i = cm.MULTS.index(m)
                k = rec["arms"][arm]["hits"][i]
                if k is None:
                    out["arms"][arm][str(m)] = {"h": None, "t": None}
                    continue
                ts = []
                for _ in range(REPEATS.get(cell["n"], 1)):
                    x, t, c = replay(arm, sys, y, lam, A0, models, k)
                    ts.append(t)
                assert c == k, (f, arm, m, c, k)
                ok = metric.metric(x) <= rec["taus"][i]
                if not ok:
                    bad.append((cell["name"], rec["j"], arm, m))
                out["arms"][arm][str(m)] = {"h": k, "t": float(np.median(ts)), "ts": ts, "reached": bool(ok)}
        cm.dump(out, out_f)
        print(f"  计时 {cell['name']} {rec['j']:02d}: " + "  ".join(
            f"{a} " + "/".join(("-" if v["t"] is None else f"{v['t']:.3f}s") for v in out["arms"][a].values())
            for a in TIMING_ARMS), flush=True)
    cm.dump({"bad": bad}, os.path.join(OUT, "timing_check.json"))
    print(f"回放点未达标 {len(bad)} 项 {bad[:5]}", flush=True)


def part_percall():
    import torch
    from unrolled_em import core as ls
    from unrolled_em.experiments import method as me
    lay = load_models("F128_500")["learned"][cm.L_MAIN - 1]
    out = {}

    def tm(fn, k):
        fn(); t = time.perf_counter()
        for _ in range(k):
            fn()
        return (time.perf_counter() - t) / k
    ns = (32, 64, 128, 256) if not cm.SMOKE else (6, 8)
    for n in ns:
        cell = dict(cm.big_cell(256) if n == 256 else cm.base_cell({32: "F32_500", 64: "F64_500", 128: "F128_500"}.get(n, "F32_500")))
        cell["n"] = n
        sys, y, _, _ = cm.setup(cell, "percall", 0)
        A = 0.5 * np.eye(n); lam = 0.01 * (y.shape[0] - 1) * sys.sigbar2 / sys.q
        k = {32: 20, 64: 20, 128: 5, 256: 2}.get(n, 5)
        t_ss = tm(lambda: ls.estep_ss(A, sys.C, sys.q, sys.r, sys.P1, y), k)
        t_ex = tm(lambda: ls.estep_exact(A, sys.C, sys.q, sys.r, sys.P1, y), max(1, k // 2))
        st = ls.estep_ss(A, sys.C, sys.q, sys.r, sys.P1, y)
        t_m = tm(lambda: ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50), k)
        B = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)

        def layer():
            F = me.features(A, B, np.zeros_like(A), st, sys, y.shape[0])
            me.surrogate(A, st["Phi"], st["Psi"], sys.q, lam); me.surrogate(B, st["Phi"], st["Psi"], sys.q, lam)
            with torch.no_grad():
                lay(torch.tensor(A), torch.tensor(B - A), torch.tensor(np.zeros_like(A)), torch.tensor(F))
            np.max(np.abs(np.linalg.eigvals(A)))
        t_l = tm(layer, k)
        out[n] = {"estep_ss": t_ss, "estep_exact": t_ex, "mstep": t_m, "layer_extra": t_l}
        print(n, {a: round(b * 1000, 2) for a, b in out[n].items()}, "ms", flush=True)
    cm.dump(out, os.path.join(OUT, "percall.json"))


def part_all(workers):
    steps = [("lambda", lambda: part_lambda(workers)), ("prep", lambda: part_prep(workers)),
             ("musel", lambda: part_musel(workers)), ("train", lambda: part_train(workers)),
             ("tune", lambda: part_tune(workers)), ("testI", lambda: part_testI(workers)),
             ("testII", lambda: part_testII(workers)),
             ("C", lambda: part_big(256, workers) if "F128_500" in cm.CELLS else None),
             ("ge", lambda: part_ge(workers) if GE["name"] in TRAIN_CELLS else None), ("B", lambda: part_B(workers)),
             ("diag", lambda: part_diag(workers))]
    for name, fn in steps:
        t = time.time()
        print(f"===== {name} =====", flush=True)
        fn()
        print(f"===== {name} 完成（{time.time() - t:.0f}s）=====", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["all", "lambda", "prep", "musel", "train", "tune", "testI", "testII", "B", "C", "D", "ge",
                             "diag", "timing", "percall"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=None, help="只跑这些格（逗号分隔，如 F32_500；试运行用），不含 GraphEM 协议")
    a = ap.parse_args()
    w = a.workers
    if a.cells:
        global TRAIN_CELLS
        cm.CELLS[:] = [c for c in cm.CELLS if c in a.cells.split(",")]
        cm.SOURCES_B[:] = [c for c in cm.SOURCES_B if c in cm.CELLS]
        TRAIN_CELLS = list(cm.CELLS)
    {"all": lambda: part_all(w), "lambda": lambda: part_lambda(w), "prep": lambda: part_prep(w),
     "musel": lambda: part_musel(w), "train": lambda: part_train(w), "tune": lambda: part_tune(w),
     "testI": lambda: part_testI(w), "testII": lambda: part_testII(w), "B": lambda: part_B(w),
     "C": lambda: part_big(256, w), "D": lambda: part_big(512, w), "ge": lambda: part_ge(w), "diag": lambda: part_diag(w),
     "timing": part_timing, "percall": part_percall}[a.part]()


if __name__ == "__main__":
    main()
