from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_discrete_are, solve_discrete_lyapunov

LOG2PI = float(np.log(2 * np.pi))


@dataclass(frozen=True)
class Sys:
    A: np.ndarray
    C: np.ndarray
    q: float
    r: float
    P1: np.ndarray
    sigbar2: float


def make_system(n, seed, snr_db, obs="full", rho=0.9, per_row=2, q=0.1):
    rng = np.random.default_rng(int(seed))
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)
        others = rng.choice([j for j in range(n) if j != i], size=per_row, replace=False)
        A[i, others] = rng.choice([-1, 1], size=per_row) * rng.uniform(0.3, 0.8, size=per_row)
    A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    Sig = solve_discrete_lyapunov(A, q * np.eye(n))
    s2 = float(np.trace(Sig) / n)
    r = s2 * 10 ** (-snr_db / 10.0)
    if obs == "full":
        C = np.eye(n)
    elif obs == "partial":
        C = np.eye(n)[: n // 2]
    else:
        raise ValueError(obs)
    return Sys(A=A, C=C, q=float(q), r=float(r), P1=s2 * np.eye(n), sigbar2=s2)


def simulate(sys, T, rng, burn=200):
    n, m = sys.A.shape[0], sys.C.shape[0]
    sq = np.sqrt(sys.q)
    x = np.zeros(n)
    for _ in range(burn):
        x = sys.A @ x + sq * rng.standard_normal(n)
    X = np.empty((T, n))
    for t in range(T):
        X[t] = x
        x = sys.A @ x + sq * rng.standard_normal(n)
    Y = X @ sys.C.T + np.sqrt(sys.r) * rng.standard_normal((T, m))
    return X, Y


def _forward(A, C, q, r, P1, y, keep=True):
    T, m = y.shape
    n = A.shape[0]
    Q = q * np.eye(n)
    xp = np.zeros(n); Pp = P1.copy()
    ll = 0.0
    if keep:
        XP = np.empty((T, n)); PP = np.empty((T, n, n)); XF = np.empty((T, n)); PF = np.empty((T, n, n))
    for t in range(T):
        PCt = Pp @ C.T
        S = C @ PCt + r * np.eye(m)
        cf = cho_factor(S, lower=True)
        e = y[t] - C @ xp
        Se = cho_solve(cf, e)
        ll += -0.5 * (2 * np.sum(np.log(np.diag(cf[0]))) + m * LOG2PI + e @ Se)
        K = cho_solve(cf, PCt.T).T
        xf = xp + K @ e
        Pf = Pp - K @ PCt.T
        Pf = 0.5 * (Pf + Pf.T)
        if keep:
            XP[t], PP[t], XF[t], PF[t] = xp, Pp, xf, Pf
        xp = A @ xf
        Pp = A @ Pf @ A.T + Q
        Pp = 0.5 * (Pp + Pp.T)
    if keep:
        return ll, XP, PP, XF, PF
    return ll


def loglik(A, C, q, r, P1, y):
    return float(_forward(A, C, q, r, P1, y, keep=False))


def filter_means_exact(A, C, q, r, P1, y):
    return _forward(A, C, q, r, P1, y)[3]


def estep_exact(A, C, q, r, P1, y):
    ll, XP, PP, XF, PF = _forward(A, C, q, r, P1, y)
    T, n = XF.shape
    xs = XF[-1].copy(); Ps = PF[-1].copy()
    Phi = np.zeros((n, n)); Psi = np.zeros((n, n))
    for t in range(T - 2, -1, -1):
        cf = cho_factor(PP[t + 1], lower=True)
        J = cho_solve(cf, A @ PF[t]).T
        xs_t = XF[t] + J @ (xs - XP[t + 1])
        Ps_t = PF[t] + J @ (Ps - PP[t + 1]) @ J.T
        Ps_t = 0.5 * (Ps_t + Ps_t.T)
        Psi += Ps @ J.T + np.outer(xs, xs_t)
        Phi += Ps_t + np.outer(xs_t, xs_t)
        xs, Ps = xs_t, Ps_t
    return {"Phi": Phi, "Psi": Psi, "ll": float(ll)}


def steady_pred_cov(A, C, q, r):
    n, m = A.shape[0], C.shape[0]
    P = solve_discrete_are(A.T, C.T, q * np.eye(n), r * np.eye(m))
    return 0.5 * (P + P.T)


def _ss_gains(A, C, q, r):
    m = C.shape[0]
    Pp = steady_pred_cov(A, C, q, r)
    S = C @ Pp @ C.T + r * np.eye(m)
    cf = cho_factor(S, lower=True)
    K = cho_solve(cf, C @ Pp).T
    Pf = Pp - K @ C @ Pp
    Pf = 0.5 * (Pf + Pf.T)
    return Pp, Pf, K, cf


def filter_means_ss(A, C, q, r, y):
    Pp, Pf, K, cf = _ss_gains(A, C, q, r)
    T = y.shape[0]
    n = A.shape[0]
    XF = np.empty((T, n)); xp = np.zeros(n)
    for t in range(T):
        XF[t] = xp + K @ (y[t] - C @ xp)
        xp = A @ XF[t]
    return XF


def estep_ss(A, C, q, r, P1, y):
    Pp, Pf, K, cf = _ss_gains(A, C, q, r)
    T, m = y.shape
    n = A.shape[0]
    XF = np.empty((T, n)); XP = np.empty((T, n)); xp = np.zeros(n)
    E = np.empty((T, m))
    for t in range(T):
        XP[t] = xp
        E[t] = y[t] - C @ xp
        XF[t] = xp + K @ E[t]
        xp = A @ XF[t]
    ld = 2 * np.sum(np.log(np.diag(cf[0])))
    ll = -0.5 * (T * (ld + m * LOG2PI) + np.sum(E * cho_solve(cf, E.T).T))
    J = np.linalg.solve(Pp.T, (Pf @ A.T).T).T
    Ps = solve_discrete_lyapunov(J, Pf - J @ Pp @ J.T)
    Ps = 0.5 * (Ps + Ps.T)
    XS = np.empty_like(XF); XS[-1] = XF[-1]
    for t in range(T - 2, -1, -1):
        XS[t] = XF[t] + J @ (XS[t + 1] - XP[t + 1])
    X0, X1 = XS[:-1], XS[1:]
    Phi = X0.T @ X0 + (T - 1) * Ps
    Psi = X1.T @ X0 + (T - 1) * (Ps @ J.T)
    return {"Phi": Phi, "Psi": Psi, "ll": float(ll), "Ps": Ps}


def _soft(X, t):
    return np.sign(X) * np.maximum(np.abs(X) - t, 0.0)


def mstep(A0, Phi, Psi, q, lam, K=None, tol=1e-10, maxit=5000):
    if lam == 0.0 and K is None:
        return np.linalg.solve(Phi.T, Psi.T).T
    L = float(np.linalg.eigvalsh(0.5 * (Phi + Phi.T))[-1]) / q
    F = lambda X: (np.sum((X @ Phi) * X) - 2 * np.sum(X * Psi)) / (2 * q) + lam * np.sum(np.abs(X))
    x = A0.copy(); yk = A0.copy(); t = 1.0
    Fx = F(x)
    nit = maxit if K is None else int(K)
    for _ in range(nit):
        g = (yk @ Phi - Psi) / q
        z = _soft(yk - g / L, lam / L)
        Fz = F(z)
        x_prev = x
        if Fz <= Fx:
            x, Fx = z, Fz
        if K is None and np.linalg.norm(z - yk) <= tol * max(np.linalg.norm(yk), 1e-12):
            x = z if Fz <= Fx else x
            break
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        yk = x + (t / t_new) * (z - x) + ((t - 1) / t_new) * (x - x_prev)
        t = t_new
    return x


def eiv(y, r, q, lam, eig_floor=1e-8):
    T, n = y.shape
    Y0, Y1 = y[:-1], y[1:]
    S0 = Y0.T @ Y0 / (T - 1) - r * np.eye(n)
    w, V = np.linalg.eigh(0.5 * (S0 + S0.T))
    S0 = (V * np.maximum(w, eig_floor * max(w.max(), 1.0))) @ V.T
    S1 = Y1.T @ Y0 / (T - 1)
    return mstep(np.zeros((n, n)), S0, S1, 1.0, lam * q / (T - 1), K=None)


def excess_nll(Ahat, sys, yv):
    T, m = yv.shape
    l_hat = loglik(Ahat, sys.C, sys.q, sys.r, sys.P1, yv)
    l_true = loglik(sys.A, sys.C, sys.q, sys.r, sys.P1, yv)
    return float((l_true - l_hat) / (T * m))


def objective(A, sys, y, lam):
    return -loglik(A, sys.C, sys.q, sys.r, sys.P1, y) + lam * float(np.sum(np.abs(A)))


def first_hit(values, nE, tau):
    for v, k in zip(values, nE):
        if v <= tau:
            return k
    return None


def em_run(sys, y, A0, lam, estep="exact", K=None, max_E=100, record_obj=False, stop_fn=None):
    fn = estep_exact if estep == "exact" else estep_ss
    A = A0.copy()
    tr = {"n_E": [0], "A": [A.copy()], "t_E": [], "t_M": []}
    if record_obj:
        tr["obj"] = [objective(A, sys, y, lam)]
    if stop_fn is not None and stop_fn(A, 0):
        return tr
    for k in range(1, max_E + 1):
        t0 = time.perf_counter()
        st = fn(A, sys.C, sys.q, sys.r, sys.P1, y)
        t1 = time.perf_counter()
        A = mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=K)
        t2 = time.perf_counter()
        tr["n_E"].append(k); tr["A"].append(A.copy())
        tr["t_E"].append(t1 - t0); tr["t_M"].append(t2 - t1)
        if record_obj:
            tr["obj"].append(objective(A, sys, y, lam))
        if stop_fn is not None and stop_fn(A, k):
            break
    return tr


def squarem_step(a0, F, obj):
    F1 = F(a0)
    F2 = F(F1)
    r = F1 - a0
    v = F2 - 2 * F1 + a0
    nr, nv = np.linalg.norm(r), np.linalg.norm(v)
    alpha = min(-nr / nv, -1.0) if nv > 0 else -1.0
    ap = a0 - 2 * alpha * r + alpha ** 2 * v
    Fa = F(ap)
    o_ap, o_F1 = obj(ap), obj(F1)
    if not (np.isfinite(o_ap) and o_ap <= o_F1) or not np.all(np.isfinite(Fa)):
        return F2, {"fallback": True, "alpha": alpha}
    return Fa, {"fallback": False, "alpha": alpha}


def squarem_run(sys, y, A0, lam, K=50, max_E=100, stop_fn=None, estep="exact"):
    cache = {}
    fn = estep_exact if estep == "exact" else estep_ss

    def F(A):
        try:
            st = fn(A, sys.C, sys.q, sys.r, sys.P1, y)
        except np.linalg.LinAlgError:
            return np.full_like(A, np.nan)
        cache[A.tobytes()] = st["ll"]
        return mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=K)

    def obj(A):
        ll = cache.get(A.tobytes())
        if ll is None:
            return np.inf
        return -ll + lam * float(np.sum(np.abs(A)))

    A = A0.copy()
    tr = {"n_E": [0], "A": [A.copy()], "fallback": [], "sec": []}
    if stop_fn is not None and stop_fn(A, 0):
        return tr
    nE = 0
    while nE + 3 <= max_E:
        t0 = time.perf_counter()
        A, info = squarem_step(A, F, obj)
        nE += 3
        tr["n_E"].append(nE); tr["A"].append(A.copy())
        tr["fallback"].append(info["fallback"]); tr["sec"].append(time.perf_counter() - t0)
        cache.clear()
        if stop_fn is not None and stop_fn(A, nE):
            break
    return tr
