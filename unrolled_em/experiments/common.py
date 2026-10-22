from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import hashlib
import json

import numpy as np

from unrolled_em import core as ls

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(PKG)
SMOKE = os.environ.get("V2_SMOKE", "") == "1"
OUT = os.path.join(HERE, "results", os.environ.get("V2_TAG") or ("v2_smoke" if SMOKE else "v2"))

PROTOCOL = json.load(open(os.path.join(PKG, "protocol.json"), encoding="utf-8"))
SEED_ROOT = PROTOCOL["seed_root"]
Q = PROTOCOL["q"]
T_VAL = PROTOCOL["T_val"]
LAMBDA_C = PROTOCOL["lambda_c"]
REF_MAX_E, REF_TOL = PROTOCOL["ref"]["max_E"], PROTOCOL["ref"]["rel_tol"]
LAMBDA_REF_MAX_E = PROTOCOL["lambda_ref_max_E"]
TAU_ABS = PROTOCOL["tau_abs"]

MULTS = [2.0, 1.5, 1.2, 1.1, 1.05, 1.02, 1.01]
MAX_E, FAIL = 100, 101
CELLS = ["F32_500", "F64_500", "F128_500", "P32_500", "P64_500"]
SHORT = {"F32_500": "F32", "F64_500": "F64", "F128_500": "F128", "P32_500": "P32", "P64_500": "P64"}

N_TRVA = {"F32_500": (48, 12), "F64_500": (24, 6), "F128_500": (24, 6), "P32_500": (48, 12), "P64_500": (24, 6)}
N_DEV = 2
N_TEST_I, N_TEST_II, N_B, N_C, N_D = 30, 30, 30, 12, 12
TRAIN_STEPS, TRAIN_LR = 2000, 1e-2
L_MAIN, L_MAX, L_ABL = 4, 6, 4
SHIFTS = {"T1000": {"T": 1000}, "T2000": {"T": 2000}, "SNRm5": {"snr_db": -5}, "SNRp5": {"snr_db": 5},
          "k1": {"per_row": 1}, "k4": {"per_row": 4}, "rho08": {"rho": 0.8}, "rho095": {"rho": 0.95}}
SOURCES_B = ["F64_500", "F128_500"]

_N = {"F32_500": 32, "F64_500": 64, "F128_500": 128, "P32_500": 32, "P64_500": 64}
_T = 500
if SMOKE:
    _N = {"F32_500": 6, "F64_500": 8, "F128_500": 10, "P32_500": 6, "P64_500": 8}
    _T = 150
    N_TRVA = {c: (3, 2) for c in CELLS}
    N_TEST_I, N_TEST_II, N_B, N_C, N_D = 2, 2, 1, 1, 1
    TRAIN_STEPS = 20
    SHIFTS = {"T1000": {"T": 300}, "SNRm5": {"snr_db": -5}}


def base_cell(name):
    obs = "partial" if name.startswith("P") else "full"
    return {"name": name, "n": _N[name], "T": _T, "obs": obs, "snr_db": 0, "rho": 0.9, "per_row": 2, "source": name}


def shift_cell(source, shift):
    c = base_cell(source)
    c.update(SHIFTS[shift])
    c["shift"] = shift
    return c


def big_cell(n):
    if SMOKE:
        n = {256: 14, 512: 16}[n]
    return {"name": f"F{n}_500" if not SMOKE else f"F{n}_s", "n": n, "T": _T, "obs": "full", "snr_db": 0, "rho": 0.9,
            "per_row": 2, "source": "F128_500", "metric": "ss"}


def load(p):
    return json.load(open(p, encoding="utf-8"))


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path + ".tmp", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=_json_default)
    os.replace(path + ".tmp", path)


def _json_default(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def seed(*parts):
    h = hashlib.sha256("|".join([str(SEED_ROOT)] + [str(p) for p in parts]).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63)


def pmap(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            yield from pool.imap_unordered(fn, jobs)
    else:
        for j in jobs:
            yield fn(j)


def scale_hat(y, C, r):
    v = float(np.mean(np.var(y, axis=0)))
    return max(v - r, 0.05 * v)


def setup(cell, split, j):
    true = ls.make_system(cell["n"], seed("sys", cell["name"], split, j), cell["snr_db"], obs=cell["obs"],
                          rho=cell["rho"], per_row=cell["per_row"], q=Q)
    _, y = ls.simulate(true, cell["T"], np.random.default_rng(seed("y", cell["name"], split, j)))
    _, yv = ls.simulate(true, T_VAL if not SMOKE else 100, np.random.default_rng(seed("yv", cell["name"], split, j)))
    s2 = scale_hat(y, true.C, true.r)
    sys = ls.Sys(A=true.A, C=true.C, q=true.q, r=true.r, P1=s2 * np.eye(cell["n"]), sigbar2=s2)
    return sys, y, yv, {"sigbar2_true": true.sigbar2, "s2_hat": s2}


def lam_of(c, cell, sys):
    return float(c) * (cell["T"] - 1) * sys.sigbar2 / sys.q


def c_of(source):
    return load(os.path.join(OUT, "lambda.json"))[source]["c"]


def project(A, rmax=0.95):
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    return A * min(1.0, rmax / rho) if rho > 0 else A


def init_candidates(cell, sys, y, lam):
    n = sys.A.shape[0]
    if cell["obs"] == "full":
        return {"EIV": project(ls.eiv(y, sys.r, sys.q, lam)), "LS": project(ls.eiv(y, 0.0, sys.q, lam)),
                "half": 0.5 * np.eye(n)}
    return {"half": 0.5 * np.eye(n), "zero": np.zeros((n, n))}


def init(cell, sys, y, lam):
    cand = init_candidates(cell, sys, y, lam)
    objs = {k: ls.objective(A, sys, y, lam) for k, A in cand.items()}
    best = min(objs, key=objs.get)
    return cand[best], best


def reference(sys, y, A0, lam, kind="exact", max_E=REF_MAX_E, rel_tol=REF_TOL):
    A = A0.copy()
    prev = None
    nE = 0
    obj = None
    snap = None
    for k in range(1, max_E + 1):
        fn = ls.estep_exact if kind == "exact" else ls.estep_ss
        st = fn(A, sys.C, sys.q, sys.r, sys.P1, y)
        obj = -st["ll"] + lam * float(np.sum(np.abs(A)))
        A = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=None)
        nE = k
        if k == max_E - 50:
            snap = A.copy()
        if prev is not None and abs(prev - obj) <= rel_tol * abs(prev):
            break
        prev = obj
    return A, nE, {"last_rel_change": None if prev is None else abs(prev - obj) / abs(prev),
                   "A_m50": snap if nE == max_E else None}


def loglik_ss(A, C, q, r, y):
    from scipy.linalg import cho_solve
    Pp, Pf, K, cf = ls._ss_gains(A, C, q, r)
    T, m = y.shape
    n = A.shape[0]
    E = np.empty((T, m)); xp = np.zeros(n)
    for t in range(T):
        E[t] = y[t] - C @ xp
        xp = A @ (xp + K @ E[t])
    ld = 2 * np.sum(np.log(np.diag(cf[0])))
    return float(-0.5 * (T * (ld + m * ls.LOG2PI) + np.sum(E * cho_solve(cf, E.T).T)))


def tau_of(M_R, mult):
    return mult * M_R + TAU_ABS


SUPP_TOL = 1e-10


def quality(Ahat, A):
    Ahat = np.asarray(Ahat)
    s_hat, s_true = np.abs(Ahat) > SUPP_TOL, np.abs(A) > 0
    tp = int(np.sum(s_hat & s_true)); fp = int(np.sum(s_hat & ~s_true)); fn = int(np.sum(~s_hat & s_true))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"delta": float(np.linalg.norm(Ahat - A) / np.linalg.norm(A)), "f1": f1, "prec": prec, "rec": rec,
            "nnz": int(np.sum(s_hat))}


class Recorder:

    def __init__(self, sys, yv, tau_stop, kind="exact", taus=None, rich=None, stop_dist=None, stop_gap=None):
        self.sys, self.yv, self.tau_stop, self.kind = sys, yv, tau_stop, kind
        self.stop_dist, self.stop_gap = stop_dist, stop_gap
        self.l_true = self._ll(sys.A, yv)
        self.Tm = yv.shape[0] * yv.shape[1]
        self.trace = []
        self.extra = []
        self.taus = taus
        self.q = [None] * len(taus) if taus else None
        self.rich = rich

    def _ll(self, A, y, kind=None):
        s = self.sys
        if (kind or self.kind) == "exact":
            return ls.loglik(A, s.C, s.q, s.r, s.P1, y)
        return loglik_ss(A, s.C, s.q, s.r, y)

    def metric(self, A):
        return float((self.l_true - self._ll(A, self.yv)) / self.Tm)

    def __call__(self, A, c):
        m = self.metric(A)
        self.trace.append((c, m))
        stop = m <= self.tau_stop
        if self.rich is not None:
            own = self.rich["own"]
            other = "ss" if own == "exact" else "exact"
            A_o, F_o = self.rich["refs"][own]
            A_x = self.rich["refs"][other][0]
            F = -self._ll(A, self.rich["y"], own) + self.rich["lam"] * float(np.sum(np.abs(A)))
            self.extra.append((c, float(np.linalg.norm(A - A_o) / np.linalg.norm(A_o)),
                               float(np.linalg.norm(A - self.sys.A) / np.linalg.norm(self.sys.A)),
                               float((F - F_o) / abs(F_o)), float(np.linalg.norm(A - A_x) / np.linalg.norm(A_x))))
            if self.stop_dist is not None:
                stop = stop and self.extra[-1][1] <= self.stop_dist and self.extra[-1][3] <= self.stop_gap
        if self.taus:
            for i, tau in enumerate(self.taus):
                if self.q[i] is None and m <= tau and c <= MAX_E:
                    self.q[i] = {"c": c, **quality(A, self.sys.A)}
        return stop


def hits_from_trace(trace, taus, max_E=MAX_E):
    out = []
    for tau in taus:
        h = next((c for c, m in trace if m <= tau), None)
        out.append(h if (h is not None and h <= max_E) else None)
    return out


BOOT_B, BOOT_SEED = 10000, 20260927


def ratio_stats(hc, hl, B=BOOT_B, seed_=BOOT_SEED):
    v = lambda x: FAIL if x is None else x
    hc = np.array([v(x) for x in hc], float); hl = np.array([v(x) for x in hl], float)
    both0 = (hc == 0) & (hl == 0)
    r = np.where(both0, 1.0, hc / np.where(hl == 0, np.nan, hl))
    lg = np.log(r)
    rng = np.random.default_rng(seed_)
    idx = rng.integers(0, len(lg), size=(B, len(lg)))
    Gb = np.exp(lg[idx].mean(axis=1))
    return {"G": float(np.exp(lg.mean())), "ci": (float(np.percentile(Gb, 2.5)), float(np.percentile(Gb, 97.5)))}


def boot_gm(x, B=BOOT_B, seed_=BOOT_SEED):
    lg = np.log(np.asarray(x, float))
    rng = np.random.default_rng(seed_)
    Gb = np.exp(lg[rng.integers(0, len(lg), size=(B, len(lg)))].mean(axis=1))
    return float(np.exp(lg.mean())), float(np.percentile(Gb, 2.5)), float(np.percentile(Gb, 97.5))


def sign_test_p(d):
    from scipy.stats import binomtest
    d = np.asarray(d, float)
    pos, neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    return 1.0 if pos + neg == 0 else float(binomtest(pos, pos + neg, 0.5, alternative="greater").pvalue)


def med(xs):
    return float(np.median([FAIL if x is None else x for x in xs]))
