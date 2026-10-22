from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import glob
import time

import numpy as np

from unrolled_em.experiments import common as cm
from unrolled_em.experiments import run as R

OUT = os.path.join(cm.OUT, "g3")
P = "learned_tied_L4"
CLS = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
CLS_BIG = ["SQ_ss_K50", "DAAREM_tun"]

HEAT = {"HF64_500": (8, 8), "HF128_500": (8, 16)}
N_TRVA_H = {"HF64_500": (24, 6), "HF128_500": (24, 6)}
N_TEST_H = 30
N_TRVA_BIG = (24, 6)
TRANSFER_H = {"HF64_500": "F64_500", "HF128_500": "HF64_500"}
if cm.SMOKE:
    HEAT = {"HF64_500": (3, 4), "HF128_500": (4, 4)}
    N_TRVA_H = {"HF64_500": (3, 2), "HF128_500": (3, 2)}
    N_TEST_H = 2
    N_TRVA_BIG = (3, 2)


def heat_cell(name):
    nx, ny = HEAT[name]
    return {"name": name, "n": nx * ny, "nx": nx, "ny": ny, "T": 500 if not cm.SMOKE else 150, "obs": "full",
            "snr_db": 0, "source": name, "heat": True}


def big_train_cell():
    c = cm.big_cell(256)
    c.update(name="F256_500" if not cm.SMOKE else "F256_s", source="F256_500")
    return c


def heat_matrix(nx, ny, rng):
    n = nx * ny
    idx = lambda x, y: x * ny + y
    L = np.zeros((n, n))
    for x in range(nx):
        for y in range(ny):
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx < nx and y + dy < ny:
                    i, j, k = idx(x, y), idx(x + dx, y + dy), rng.uniform(0.5, 1.5)
                    L[i, j] -= k; L[j, i] -= k; L[i, i] += k; L[j, j] += k
    Mh = L + np.diag(rng.uniform(0.3, 0.7, size=n))
    dt = 1.0 / float(np.max(np.linalg.eigvalsh(Mh)))
    return np.eye(n) - dt * Mh, dt


def heat_setup(cell, split, j):
    from scipy.linalg import solve_discrete_lyapunov
    from unrolled_em import core as ls
    nx, ny = cell["nx"], cell["ny"]
    n = nx * ny
    A, dt = heat_matrix(nx, ny, np.random.default_rng(cm.seed("heat", cell["name"], split, j)))
    Sig = solve_discrete_lyapunov(A, cm.Q * np.eye(n))
    s2 = float(np.trace(Sig) / n)
    r = s2 * 10 ** (-cell["snr_db"] / 10.0)
    if cell["obs"] == "full":
        obs = list(range(n))
    else:
        obs = [x * ny + y for x in range(nx) for y in range(ny) if (x + y) % 2 == 0]
    C = np.eye(n)[obs]
    true = ls.Sys(A=A, C=C, q=float(cm.Q), r=float(r), P1=s2 * np.eye(n), sigbar2=s2)
    _, y = ls.simulate(true, cell["T"], np.random.default_rng(cm.seed("y", cell["name"], split, j)))
    _, yv = ls.simulate(true, cm.T_VAL if not cm.SMOKE else 100, np.random.default_rng(cm.seed("yv", cell["name"], split, j)))
    s2h = cm.scale_hat(y, true.C, true.r)
    sys = ls.Sys(A=A, C=C, q=true.q, r=true.r, P1=s2h * np.eye(n), sigbar2=s2h)
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    return sys, y, yv, {"sigbar2_true": s2, "s2_hat": s2h, "rho": rho, "dt": dt}


def setup_any(cell, split, j):
    return heat_setup(cell, split, j) if cell.get("heat") else cm.setup(cell, split, j)


def c_for(cell):
    if cell.get("heat"):
        return cm.load(os.path.join(OUT, "lambda.json"))[cell["name"]]["c"]
    return cm.c_of("F128_500")


def kind_for(cell):
    return "ss" if cell["n"] >= 256 or cell.get("metric") == "ss" else "exact"


def p_data(name, split, j):
    return os.path.join(OUT, "data", f"{name}_{split}_{j:02d}.npz")


def p_model(name):
    return os.path.join(OUT, "models", f"{name}_learned.pt")


def load_learned(name):
    from unrolled_em.experiments import method as me
    f = p_model(name)
    return me.load_layers(f) if os.path.exists(f) else R.load_models(name)["learned"]


def _lambda_job(args):
    from unrolled_em import core as ls
    name, j = args
    f = os.path.join(OUT, "lambda", f"{name}_{j}.json")
    if os.path.exists(f):
        return f
    cell = heat_cell(name)
    sys, y, yv, info = heat_setup(cell, "dev", j)
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
    old = cm.load(f) if os.path.exists(f) else {}
    if all(nm in old for nm in HEAT):
        return
    jobs = [(nm, j) for nm in HEAT if nm not in old for j in range(cm.N_DEV)]
    for _ in cm.pmap(_lambda_job, jobs, workers):
        pass
    sel = dict(old)
    for nm in [x for x in HEAT if x not in old]:
        rs = [cm.load(os.path.join(OUT, "lambda", f"{nm}_{j}.json")) for j in range(cm.N_DEV)]
        mean = {str(c): float(np.mean([r["M"][str(c)] for r in rs])) for c in cm.LAMBDA_C}
        sel[nm] = {"c": min(cm.LAMBDA_C, key=lambda c: mean[str(c)]), "dev_mean_M": mean}
    cm.dump(sel, f)
    print("λ（热传导网格）：" + ", ".join(f"{k}: c={v['c']}" for k, v in sel.items()), flush=True)


def train_cells():
    return [heat_cell(nm) for nm in HEAT] + [big_train_cell()]


def n_trva(cell):
    return N_TRVA_H[cell["name"]] if cell.get("heat") else N_TRVA_BIG


def _prep_job(args):
    cell, split, j = args
    f = p_data(cell["name"], split, j)
    if os.path.exists(f):
        return f
    sys, y, _, info = setup_any(cell, split, j)
    lam = cm.lam_of(c_for(cell), cell, sys)
    A0, init_name = cm.init(cell, sys, y, lam)
    t = time.time()
    A_R, nE, _ = cm.reference(sys, y, A0, lam, kind=kind_for(cell))
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, A_R=A_R, A0=A0, lam=lam, nE=nE, s2_hat=info["s2_hat"], sigbar2_true=info["sigbar2_true"], sec=time.time() - t)
    print(f"  参照解 {cell['name']} {split}{j:02d}: {nE} E 步 {time.time() - t:.0f}s", flush=True)
    return f


def part_prep(workers):
    jobs = [(c, s, j) for c in train_cells() for s, k in zip(("train", "val"), n_trva(c)) for j in range(k)]
    jobs.sort(key=lambda x: -x[0]["n"])
    todo = [jb for jb in jobs if not os.path.exists(p_data(jb[0]["name"], jb[1], jb[2]))]
    print(f"参照解：{len(jobs)} 个，待算 {len(todo)}", flush=True)
    for _ in cm.pmap(_prep_job, todo, workers):
        pass


def states(cell, split):
    from unrolled_em.experiments import method as me
    out = []
    for j in range(n_trva(cell)[0 if split == "train" else 1]):
        z = np.load(p_data(cell["name"], split, j))
        sys, y, _, _ = setup_any(cell, split, j)
        lam = float(z["lam"])
        cands = cm.init_candidates(cell, sys, y, lam)
        assert any(np.array_equal(z["A0"], A) for A in cands.values())
        out += me.make_states(sys, y, lam, z["A_R"], cands)
    return out


def _train_job(cell):
    from unrolled_em.experiments import method as me
    f = p_model(cell["name"])
    if os.path.exists(f):
        return f
    t = time.time()
    layers, log = me.train_chain(states(cell, "train"), states(cell, "val"), "learned", cm.L_MAIN, cm.TRAIN_STEPS,
                                 cm.TRAIN_LR, tag=cell["name"])
    me.save_layers(f, "learned", layers)
    cm.dump({"log": log, "sec": time.time() - t}, f.replace(".pt", "_log.json"))
    return f


def part_train(workers):
    jobs = [c for c in train_cells() if not os.path.exists(p_model(c["name"]))]
    jobs.sort(key=lambda c: -c["n"])
    print(f"训练：{[c['name'] for c in jobs]}", flush=True)
    for _ in cm.pmap(_train_job, jobs, min(workers, max(len(jobs), 1))):
        pass


def _tune_job(args):
    from unrolled_em import core as ls
    name, j = args
    f = os.path.join(OUT, "tune", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    cell = heat_cell(name)
    z = np.load(p_data(name, "val", j))
    sys, y, yv, _ = heat_setup(cell, "val", j)
    lam, A0 = float(z["lam"]), z["A0"]
    tau = cm.tau_of(ls.excess_nll(z["A_R"], sys, yv), 1.1)
    out = {"cell": name, "j": j, "tau": tau, "hits": {}}
    for o, k in R.GRID_DA:
        out["hits"][f"{o},{k}"] = R.run_arm("DAAREM", sys, y, lam, A0, cm.Recorder(sys, yv, tau), {"daarem": (o, k)})["hits_single"]
    out["SQ_ss_K50"] = R.run_arm("SQ_ss_K50", sys, y, lam, A0, cm.Recorder(sys, yv, tau), {})["hits_single"]
    cm.dump(out, f)
    print(f"  调参 {name} val{j:02d}: " + "  ".join(f"{g} {h}" for g, h in out["hits"].items()), flush=True)
    return f


def part_tune(workers):
    f = os.path.join(OUT, "tune_choice.json")
    old = cm.load(f) if os.path.exists(f) else {}
    if all(nm in old for nm in HEAT):
        return
    jobs = [(nm, j) for nm in HEAT if nm not in old for j in range(N_TRVA_H[nm][1])]
    for _ in cm.pmap(_tune_job, jobs, workers):
        pass
    choice = dict(old)
    for nm in [x for x in HEAT if x not in old]:
        rs = [cm.load(os.path.join(OUT, "tune", f"{nm}_{j:02d}.json")) for j in range(N_TRVA_H[nm][1])]
        gm = {g: R._gm([r["hits"][f"{g[0]},{g[1]}"] for r in rs]) for g in R.GRID_DA}
        best = min(gm.values())
        tied = [g for g in gm if gm[g] <= best * (1 + 1e-12)]
        cfg = R.DEFAULT_DA if R.DEFAULT_DA in tied else sorted(tied)[0]
        choice[nm] = {"config": list(cfg), "gm": {f"{g[0]},{g[1]}": x for g, x in gm.items()},
                      "gm_SQ_ss": R._gm([r["SQ_ss_K50"] for r in rs])}
        print(f"  DAAREM {nm}: {cfg}", flush=True)
    cm.dump(choice, f)


def _common(cell, split, j, tag):
    from unrolled_em import core as ls
    kind = kind_for(cell)
    sys, y, yv, info = setup_any(cell, split, j)
    lam = cm.lam_of(c_for(cell), cell, sys)
    A0, init_name = cm.init(cell, sys, y, lam)
    t = time.time()
    A_R, nE_R, rinfo = cm.reference(sys, y, A0, lam, kind=kind)
    sec_R = time.time() - t
    f = os.path.join(OUT, "refs", f"{tag}.npz")
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, A_R=A_R, A0=A0, lam=lam)
    if kind == "exact":
        M_R = ls.excess_nll(A_R, sys, yv)
    else:
        M_R = cm.Recorder(sys, yv, 0.0, "ss").metric(A_R)
    taus = [cm.tau_of(M_R, m) for m in cm.MULTS]
    head = {"cell": cell, "split": split, "j": j, "init": init_name, "lam": lam, "M_R": M_R, "nE_R": nE_R, "sec_R": sec_R,
            "taus": taus, "q_R": cm.quality(A_R, sys.A), **info}
    return sys, y, yv, lam, A0, taus, head, kind


def _run(label, arm, sys, y, yv, lam, A0, taus, models, kind):
    rec = cm.Recorder(sys, yv, taus[-1], kind, taus=taus)
    res = R.run_arm(arm, sys, y, lam, A0, rec, models)
    res.pop("hits_single", None)
    res.pop("trace", None)
    return label, res


def _heat_test_job(args):
    name, j = args
    f = os.path.join(OUT, "H", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    cell = heat_cell(name)
    t0 = time.time()
    sys, y, yv, lam, A0, taus, head, kind = _common(cell, "test", j, f"H_{name}_{j:02d}")
    da = tuple(cm.load(os.path.join(OUT, "tune_choice.json"))[name]["config"])
    native = {"learned": load_learned(name), "daarem": da}
    src = TRANSFER_H[name]
    transfer = {"learned_from": load_learned(src)}
    arms = {}
    for label, arm, mdl in [(P, P, native), (f"from_{src}", "tied_from64_L4", transfer),
                            ("EM_ss_K50", "EM_ss_K50", native), ("SQ_ss_K50", "SQ_ss_K50", native),
                            ("SQ_exact_K50", "SQ_exact_K50", native), ("DAAREM_tun", "DAAREM_tun", native)]:
        lb, res = _run(label, arm, sys, y, yv, lam, A0, taus, mdl, kind)
        arms[lb] = res
    out = {**head, "arms": arms, "sec": time.time() - t0}
    cm.dump(out, f)
    i11 = cm.MULTS.index(1.1)
    print(f"  H {name} {j:02d}: " + "  ".join(f"{k} {v['hits'][i11]}/{v['hits'][-1]}" for k, v in arms.items())
          + f" | {out['sec']:.0f}s", flush=True)
    return f


def part_heat(workers):
    jobs = sorted([(nm, j) for nm in HEAT for j in range(N_TEST_H)], key=lambda x: -heat_cell(x[0])["n"])
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(OUT, "H", f"{jb[0]}_{jb[1]:02d}.json"))]
    print(f"热传导网格测试：{len(jobs)} 个系统，待跑 {len(todo)}", flush=True)
    for _ in cm.pmap(_heat_test_job, todo, workers):
        pass


def _heat_a7_job(args):
    name, j = args
    f = os.path.join(OUT, "H_A7", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    from unrolled_em import core as ls
    cell = heat_cell(name)
    t0 = time.time()
    sys, y, yv, lam, A0, taus, head, kind = _common(cell, "test", j, f"H_{name}_{j:02d}")
    z = np.load(os.path.join(OUT, "refs", f"H_{name}_{j:02d}.npz"))
    A_R = z["A_R"]
    F_R = -ls.loglik(A_R, sys.C, sys.q, sys.r, sys.P1, y) + lam * float(np.sum(np.abs(A_R)))
    A_Rss, nE_Rss, _ = cm.reference(sys, y, A0, lam, kind="ss", max_E=R.REF_SS_MAX_E, rel_tol=R.REF_SS_TOL)
    F_Rss = -cm.loglik_ss(A_Rss, sys.C, sys.q, sys.r, y) + lam * float(np.sum(np.abs(A_Rss)))
    rich = {"y": y, "lam": lam, "refs": {"exact": (A_R, F_R), "ss": (A_Rss, F_Rss)}}
    da = tuple(cm.load(os.path.join(OUT, "tune_choice.json"))[name]["config"])
    native = {"learned": load_learned(name), "daarem": da}
    src = TRANSFER_H[name]
    transfer = {"learned_from": load_learned(src)}
    arms = {}
    for label, arm, mdl in [(P, P, native), (f"from_{src}", "tied_from64_L4", transfer),
                            ("EM_ss_K50", "EM_ss_K50", native), ("SQ_ss_K50", "SQ_ss_K50", native),
                            ("SQ_exact_K50", "SQ_exact_K50", native), ("DAAREM_tun", "DAAREM_tun", native)]:
        rr = {**rich, "own": "exact" if "exact" in arm else "ss"}
        rec = cm.Recorder(sys, yv, taus[-1], kind, taus=taus, rich=rr, stop_dist=R.STOP_DIST, stop_gap=R.STOP_GAP)
        res = R.run_arm(arm, sys, y, lam, A0, rec, mdl)
        res.pop("hits_single", None); res.pop("trace", None)
        arms[label] = res
    out = {**head, "nE_Rss": nE_Rss, "F_R": F_R, "F_Rss": F_Rss,
           "dist_Rss_R": float(np.linalg.norm(A_Rss - A_R) / np.linalg.norm(A_R)), "arms": arms, "sec": time.time() - t0}
    cm.dump(out, f)
    print(f"  H_A7 {name} {j:02d}: {time.time() - t0:.0f}s", flush=True)
    return f


def part_heat_a7(workers):
    jobs = sorted([(nm, j) for nm in HEAT for j in range(N_TEST_H)], key=lambda x: -heat_cell(x[0])["n"])
    todo = [jb for jb in jobs if not os.path.exists(os.path.join(OUT, "H_A7", f"{jb[0]}_{jb[1]:02d}.json"))]
    print(f"热传导网格（参数空间 / 目标值）：{len(jobs)} 个系统，待跑 {len(todo)}", flush=True)
    for _ in cm.pmap(_heat_a7_job, todo, workers):
        pass


def _xhits(r, arm, col, thr):
    ex = r["arms"][arm].get("extra") or []
    h = next((e[0] for e in ex if e[col] <= thr), None)
    return h if (h is not None and h <= cm.MAX_E) else None


def export_heat_a7():
    out = {}
    for nm in HEAT:
        recs = [cm.load(f) for f in sorted(glob.glob(os.path.join(OUT, "H_A7", f"{nm}_*.json")))]
        if not recs:
            continue
        arms = list(recs[0]["arms"])
        ent = {"n": len(recs), "nE_Rss_median": float(np.median([r["nE_Rss"] for r in recs])),
               "dist_Rss_R_median": float(np.median([r["dist_Rss_R"] for r in recs]))}
        for key, col, thr in (("dist5", 1, 0.05), ("dist1", 1, 0.01), ("gap1e-4", 3, 1e-4), ("gap1e-5", 3, 1e-5)):
            Hx = {a: [_xhits(r, a, col, thr) for r in recs] for a in arms}
            st = {a: cm.ratio_stats(Hx[a], Hx[P]) for a in CLS}
            o = min(CLS, key=lambda a: st[a]["G"])
            src = [a for a in arms if a.startswith("from_")][0]
            ent[key] = {"G": st[o]["G"], "ci": st[o]["ci"], "o_star": o, "median": {a: cm.med(Hx[a]) for a in arms},
                        "n_fail": {a: sum(x is None for x in Hx[a]) for a in arms},
                        "transfer_G": cm.ratio_stats(Hx[o], Hx[src])["G"]}
        lower = [r["j"] for r in recs if min(e[3] for e in r["arms"][P]["extra"]) < -1e-6]
        ent["lower_objective"] = lower
        out[nm] = ent
        print(nm, "  ".join(f"{k}: {ent[k]['G']:.2f} [{ent[k]['ci'][0]:.2f},{ent[k]['ci'][1]:.2f}] {ent[k]['o_star'][:6]} "
                            f"med {ent[k]['median'][P]:g}/{ent[k]['median'][ent[k]['o_star']]:g} 迁移 {ent[k]['transfer_G']:.2f}"
                            for k in ("dist5", "dist1", "gap1e-4", "gap1e-5")), flush=True)
    return out


def _cnative_job(j):
    f = os.path.join(OUT, "C_native", f"F256_{j:02d}.json")
    if os.path.exists(f):
        return f
    cell = cm.big_cell(256)
    rec_c = cm.load(os.path.join(cm.OUT, "C", f"{cell['name']}_{j:02d}.json"))
    z = np.load(os.path.join(cm.OUT, "refs", "C", f"{cell['name']}_{j:02d}.npz"))
    sys, y, yv, _ = cm.setup(cell, "test1g", j)
    lam, A0, taus = float(z["lam"]), z["A0"], rec_c["taus"]
    t0 = time.time()
    _, res = _run(P, P, sys, y, yv, lam, A0, taus, {"learned": load_learned(big_train_cell()["name"])}, "ss")
    cm.dump({"j": j, "arms": {"native_F256": res}, "sec": time.time() - t0}, f)
    print(f"  C 原生 {j:02d}: {res['hits'][cm.MULTS.index(1.1)]}/{res['hits'][-1]}（F128 迁移 "
          f"{rec_c['arms'][P]['hits'][cm.MULTS.index(1.1)]}/{rec_c['arms'][P]['hits'][-1]}）", flush=True)
    return f


def part_cnative(workers):
    js = [j for j in range(cm.N_C) if os.path.exists(os.path.join(cm.OUT, "C", f"{cm.big_cell(256)['name']}_{j:02d}.json"))]
    for _ in cm.pmap(_cnative_job, js, workers):
        pass


def _d_job(j):
    cell = cm.big_cell(512)
    f = os.path.join(OUT, "D", f"{cell['name']}_{j:02d}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    sys, y, yv, lam, A0, taus, head, kind = _common(cell, "test1g", j, f"D_{cell['name']}_{j:02d}")
    models = {"learned_from": load_learned(big_train_cell()["name"]), "daarem": R.load_models("F128_500")["daarem"]}
    arms = {}
    for label, arm in [("from_F256", "tied_from64_L4"), ("SQ_ss_K50", "SQ_ss_K50"), ("DAAREM_tun", "DAAREM_tun")]:
        lb, res = _run(label, arm, sys, y, yv, lam, A0, taus, models, kind)
        arms[lb] = res
    out = {**head, "arms": arms, "sec": time.time() - t0}
    cm.dump(out, f)
    i11 = cm.MULTS.index(1.1)
    print(f"  D n=512 {j:02d}: " + "  ".join(f"{k} {v['hits'][i11]}/{v['hits'][-1]}" for k, v in arms.items())
          + f" | {out['sec']:.0f}s", flush=True)
    return f


def part_d(workers):
    for _ in cm.pmap(_d_job, list(range(cm.N_D)), workers):
        pass


def _stats(recs, prim, cls, i):
    H = lambda a: [r["arms"][a]["hits"][i] for r in recs]
    st = {a: cm.ratio_stats(H(a), H(prim)) for a in cls}
    o = min(cls, key=lambda a: st[a]["G"])
    return {"G": st[o]["G"], "ci": st[o]["ci"], "o_star": o, "G_all": {a: st[a]["G"] for a in cls},
            "median": {a: cm.med(H(a)) for a in [prim] + cls}, "n_fail": {a: sum(x is None for x in H(a)) for a in [prim] + cls}}


def _qual(recs, arm, i):
    out = {}
    for key in ("delta", "f1"):
        xs = [r["arms"][arm]["q"][i][key] for r in recs if r["arms"][arm].get("q") and r["arms"][arm]["q"][i]]
        out[key] = float(np.median(xs)) if xs else None
    return out


def _rej(recs, arm):
    first, cont = {}, {}
    for r in recs:
        for k, x in enumerate(r["arms"][arm].get("log", [])):
            d = first if k < cm.L_MAIN else cont
            d[x] = d.get(x, 0) + 1
    rate = lambda d: 1 - d.get("ok", 0) / max(sum(d.values()), 1)
    return {"first": first, "cont": cont, "rate_first": rate(first), "rate_cont": rate(cont)}


def export():
    I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)
    V = {}
    for nm in HEAT:
        recs = [cm.load(f) for f in sorted(glob.glob(os.path.join(OUT, "H_A7", f"{nm}_*.json")))]
        if not recs:
            continue
        src = TRANSFER_H[nm]
        ent = {"n": len(recs), "rho": [float(np.min([r["rho"] for r in recs])), float(np.median([r["rho"] for r in recs])),
                                       float(np.max([r["rho"] for r in recs]))],
               "scale_rel_err_median": float(np.median([abs(r["s2_hat"] / r["sigbar2_true"] - 1) for r in recs])),
               "q_R": {k: float(np.median([r["q_R"][k] for r in recs])) for k in ("delta", "f1", "prec", "rec")},
               "nE_R_median": float(np.median([r["nE_R"] for r in recs])), "capped": sum(r["nE_R"] >= cm.REF_MAX_E for r in recs),
               "c": cm.load(os.path.join(OUT, "lambda.json"))[nm]["c"],
               "daarem": cm.load(os.path.join(OUT, "tune_choice.json"))[nm]["config"], "rejects": _rej(recs, P)}
        A = recs[0]["cell"]
        ent["nnz_per_row_max"] = 5
        for i, m in ((I11, "1.1"), (I101, "1.01")):
            s = _stats(recs, P, CLS, i)
            st_t = _stats(recs, f"from_{src}", CLS, i)
            s["quality"] = {a: _qual(recs, a, i) for a in [P] + CLS}
            s["transfer"] = {"src": src, "G": st_t["G"], "ci": st_t["ci"], "o_star": st_t["o_star"],
                             "median": st_t["median"][f"from_{src}"], "n_fail": st_t["n_fail"][f"from_{src}"],
                             "vs_native": cm.ratio_stats([r["arms"][f"from_{src}"]["hits"][i] for r in recs],
                                                         [r["arms"][P]["hits"][i] for r in recs])}
            ent[m] = s
        lf = p_model(nm).replace(".pt", "_log.json")
        if os.path.exists(lf):
            lg = cm.load(lf)
            ent["train_sec"] = lg["sec"]; ent["val_loss"] = [e["ratio"] for e in lg["log"]]
        V[nm] = ent
    cn = sorted(glob.glob(os.path.join(OUT, "C_native", "*.json")))
    if cn:
        recs = []
        for f in cn:
            a = cm.load(f)
            c = cm.load(os.path.join(cm.OUT, "C", f"{cm.big_cell(256)['name']}_{a['j']:02d}.json"))
            c["arms"]["native_F256"] = a["arms"]["native_F256"]
            recs.append(c)
        ent = {"n": len(recs)}
        for i, m in ((I11, "1.1"), (I101, "1.01")):
            s = _stats(recs, "native_F256", CLS_BIG, i)
            s["transfer_F128"] = _stats(recs, P, CLS_BIG, i)["G"]
            s["F128_vs_native"] = cm.ratio_stats([r["arms"][P]["hits"][i] for r in recs],
                                                 [r["arms"]["native_F256"]["hits"][i] for r in recs])
            s["median_F128"] = cm.med([r["arms"][P]["hits"][i] for r in recs])
            ent[m] = s
        ent["rejects_native"] = _rej(recs, "native_F256")
        lf = p_model(big_train_cell()["name"]).replace(".pt", "_log.json")
        if os.path.exists(lf):
            lg = cm.load(lf)
            ent["train_sec"] = lg["sec"]; ent["val_loss"] = [e["ratio"] for e in lg["log"]]
        ref_sec = [float(np.load(f)["sec"]) for f in glob.glob(os.path.join(OUT, "data", f"{big_train_cell()['name']}_*.npz"))]
        ent["ref_sec_total"] = float(np.sum(ref_sec)) if ref_sec else None
        V["F256_native"] = ent
    dd = [cm.load(f) for f in sorted(glob.glob(os.path.join(OUT, "D", "*.json")))]
    if dd:
        ent = {"n": len(dd), "nE_R_median": float(np.median([r["nE_R"] for r in dd])),
               "capped": sum(r["nE_R"] >= cm.REF_MAX_E for r in dd), "rejects": _rej(dd, "from_F256"),
               "sec_median": float(np.median([r["sec"] for r in dd]))}
        for i, m in ((I11, "1.1"), (I101, "1.01")):
            ent[m] = _stats(dd, "from_F256", CLS_BIG, i)
        V["D512"] = ent
    ha = export_heat_a7()
    if ha:
        V["heat_A7"] = ha
    cm.dump(V, os.path.join(OUT, "verdict_g3.json"))
    for k, e in V.items():
        if k == "heat_A7":
            continue
        msg = f"{k}: n={e['n']}"
        for m in ("1.1", "1.01"):
            if m in e:
                prim = [a for a in e[m]["median"] if a not in CLS][0]
                msg += f" | α={m} G {e[m]['G']:.2f} [{e[m]['ci'][0]:.2f}, {e[m]['ci'][1]:.2f}] ({e[m]['o_star']}) " \
                       f"med {e[m]['median'][prim]:g}/{e[m]['median'][e[m]['o_star']]:g} fail {e[m]['n_fail'][prim]}"
                if "transfer" in e[m]:
                    msg += f" 迁移 {e[m]['transfer']['G']:.2f}"
                if "transfer_F128" in e[m]:
                    msg += f" F128迁移 {e[m]['transfer_F128']:.2f}"
        print(msg, flush=True)


def part_all(w):
    for name, fn in [("lambda", part_lambda), ("prep", part_prep), ("train", part_train), ("tune", part_tune),
                     ("heatA7", part_heat_a7), ("cnative", part_cnative), ("D", part_d)]:
        t = time.time()
        print(f"===== {name} =====", flush=True)
        fn(w)
        print(f"===== {name} 完成（{time.time() - t:.0f}s）=====", flush=True)
    export()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["all", "lambda", "prep", "train", "tune", "heat", "heatA7", "cnative", "D", "export"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    w = a.workers
    {"all": lambda: part_all(w), "lambda": lambda: part_lambda(w), "prep": lambda: part_prep(w),
     "train": lambda: part_train(w), "tune": lambda: part_tune(w), "heat": lambda: part_heat(w),
     "heatA7": lambda: part_heat_a7(w),
     "cnative": lambda: part_cnative(w), "D": lambda: part_d(w), "export": export}[a.part]()


if __name__ == "__main__":
    main()
