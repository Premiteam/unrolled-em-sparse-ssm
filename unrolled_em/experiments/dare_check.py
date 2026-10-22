import sys, os, json, time, numpy as np
from unrolled_em import core as ls
from unrolled_em.experiments import common as cm
from unrolled_em.experiments import run as R
from scipy.linalg import solve_discrete_are

def pbh_undetectable(A, C, tol=1e-8):
    ev = np.linalg.eigvals(A); n = A.shape[0]; out = []
    for lam in ev:
        if abs(lam) >= 1 - 1e-9:
            M = np.vstack([A - lam * np.eye(n), C.astype(complex)])
            s = np.linalg.svd(M, compute_uv=False)[-1]
            out.append((float(abs(lam)), float(s)))
    return out

def job(args):
    name, split, j, arm = args
    cell = cm.base_cell(name)
    sysm, y, yv, info = R.any_setup(cell, split, j)
    lam = cm.lam_of(cm.c_of(cell["source"]), cell, sysm)
    A0, _ = cm.init(cell, sysm, y, lam)
    models = R.load_models(name)
    stats = {"calls": 0, "fail": 0, "rho_ge1": 0, "unstable_modes": 0, "min_pbh_sv_unstable": None,
             "nonstabilizing": 0, "max_rel_res": 0.0}
    orig = ls.estep_ss
    def wrapped(A, *a, **k):
        stats["calls"] += 1
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        if rho >= 1: stats["rho_ge1"] += 1
        for mod, sv in pbh_undetectable(A, sysm.C):
            stats["unstable_modes"] += 1
            m = stats["min_pbh_sv_unstable"]
            stats["min_pbh_sv_unstable"] = sv if m is None else min(m, sv)
        try:
            st = orig(A, *a, **k)
        except np.linalg.LinAlgError:
            stats["fail"] += 1
            raise
        n, m = A.shape[0], sysm.C.shape[0]
        P = ls.steady_pred_cov(A, sysm.C, sysm.q, sysm.r)
        S = sysm.C @ P @ sysm.C.T + sysm.r * np.eye(m)
        K = P @ sysm.C.T @ np.linalg.inv(S)
        res = A @ P @ A.T - A @ P @ sysm.C.T @ np.linalg.solve(S, sysm.C @ P @ A.T) + sysm.q * np.eye(n) - P
        stats["max_rel_res"] = max(stats["max_rel_res"], float(np.abs(res).max() / np.abs(P).max()))
        if np.max(np.abs(np.linalg.eigvals(A - A @ K @ sysm.C))) >= 1 - 1e-12:
            stats["nonstabilizing"] += 1
        return st
    ls.estep_ss = wrapped
    t = time.time(); err = None
    try:
        R.algo_for(arm, sysm, y, lam, A0, models, [])()
    except Exception as e:
        err = repr(e)[:200]
    finally:
        ls.estep_ss = orig
    stats.update(cell=name, split=split, j=j, arm=arm, sec=round(time.time() - t, 1), error=err)
    return stats

if __name__ == "__main__":
    import multiprocessing as mp
    arms = ["EM_ss_K50", "SQ_ss_K50", "DAAREM_tun", "learned_tied_L4"]
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = [(c, s, j, a) for s in ("test1g", "test1c") for c in ("P32_500", "P64_500") for j in range(30) for a in arms]
    if which == "one":
        jobs = [("P64_500", "test1g", 0, a) for a in arms]
    out = os.path.join(cm.HERE, 'results', 'v2', 'dare_check_' + which + '.jsonl')
    done = set()
    if os.path.exists(out):
        for l in open(out):
            z = json.loads(l); done.add((z["cell"], z["split"], z["j"], z["arm"]))
    jobs = [jb for jb in jobs if jb not in done]
    with mp.Pool(2) as pool, open(out, "a") as fo:
        for r in pool.imap_unordered(job, jobs):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            print(r["cell"], r["split"], r["j"], r["arm"], r["calls"], r["fail"], r["rho_ge1"], r["unstable_modes"], r["nonstabilizing"], r["sec"], r["error"], flush=True)
