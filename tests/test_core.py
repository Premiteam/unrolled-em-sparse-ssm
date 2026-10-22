import numpy as np
import pytest

from unrolled_em import core as ls


def _small(n=3, T=6, obs="full", seed=0):
    sys = ls.make_system(n, seed=seed, snr_db=0.0, obs=obs)
    _, y = ls.simulate(sys, T, np.random.default_rng(seed + 100))
    return sys, y


def _dense_posterior(A, C, q, r, P1, y):
    T, m = y.shape
    n = A.shape[0]
    L = np.zeros((T * n, T * n))
    for t in range(T):
        for s in range(t + 1):
            L[t * n:(t + 1) * n, s * n:(s + 1) * n] = np.linalg.matrix_power(A, t - s)
    D = np.zeros((T * n, T * n))
    D[:n, :n] = P1
    for s in range(1, T):
        D[s * n:(s + 1) * n, s * n:(s + 1) * n] = q * np.eye(n)
    Sx = L @ D @ L.T
    H = np.kron(np.eye(T), C)
    Sy = H @ Sx @ H.T + r * np.eye(T * m)
    yv = y.reshape(-1)
    K = Sx @ H.T @ np.linalg.inv(Sy)
    mu = K @ yv
    Sp = Sx - K @ H @ Sx
    ll = -0.5 * (np.linalg.slogdet(2 * np.pi * Sy)[1] + yv @ np.linalg.solve(Sy, yv))
    mu = mu.reshape(T, n)
    Phi = sum(Sp[t * n:(t + 1) * n, t * n:(t + 1) * n] + np.outer(mu[t], mu[t]) for t in range(T - 1))
    Psi = sum(Sp[(t + 1) * n:(t + 2) * n, t * n:(t + 1) * n] + np.outer(mu[t + 1], mu[t]) for t in range(T - 1))
    return Phi, Psi, ll


@pytest.mark.parametrize("obs", ["full", "partial"])
def test_s1_exact_estep_matches_dense(obs):
    sys, y = _small(n=4, T=6, obs=obs)
    A = sys.A * 0.8
    st = ls.estep_exact(A, sys.C, sys.q, sys.r, sys.P1, y)
    Phi, Psi, ll = _dense_posterior(A, sys.C, sys.q, sys.r, sys.P1, y)
    np.testing.assert_allclose(st["Phi"], Phi, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(st["Psi"], Psi, rtol=1e-9, atol=1e-10)
    assert abs(st["ll"] - ll) < 1e-8
    assert abs(ls.loglik(A, sys.C, sys.q, sys.r, sys.P1, y) - ll) < 1e-8


def test_s2_steady_state():
    sys, y = _small(n=6, T=400, obs="partial", seed=3)
    Pss = ls.steady_pred_cov(sys.A, sys.C, sys.q, sys.r)
    xf_e = ls.filter_means_exact(sys.A, sys.C, sys.q, sys.r, Pss, y)
    xf_s = ls.filter_means_ss(sys.A, sys.C, sys.q, sys.r, y)
    np.testing.assert_allclose(xf_s, xf_e, rtol=1e-8, atol=1e-10)
    se = ls.estep_exact(sys.A, sys.C, sys.q, sys.r, sys.P1, y)
    ss = ls.estep_ss(sys.A, sys.C, sys.q, sys.r, sys.P1, y)
    for k in ("Phi", "Psi"):
        assert np.linalg.norm(ss[k] - se[k]) / np.linalg.norm(se[k]) < 0.01


def test_s3_mstep_kkt():
    rng = np.random.default_rng(1)
    n = 8
    X = rng.standard_normal((200, n))
    Phi = X.T @ X
    Psi = rng.standard_normal((n, n)) * 20 + Phi @ np.diag(rng.uniform(-.5, .5, n))
    q = 0.1
    A0 = ls.mstep(np.zeros((n, n)), Phi, Psi, q, 0.0, K=None)
    np.testing.assert_allclose(A0, np.linalg.solve(Phi.T, Psi.T).T, rtol=1e-6, atol=1e-8)
    lam = 50.0
    A = ls.mstep(np.zeros((n, n)), Phi, Psi, q, lam, K=None)
    G = (A @ Phi - Psi) / q
    nz = np.abs(A) > 1e-10
    assert np.all(np.abs(G[nz] + lam * np.sign(A[nz])) < 1e-6 * lam)
    assert np.all(np.abs(G[~nz]) <= lam * (1 + 1e-6))
    assert nz.sum() < n * n


def test_s4_em_monotone():
    sys, y = _small(n=8, T=80, obs="partial", seed=4)
    lam = 2.0
    tr = ls.em_run(sys, y, 0.5 * np.eye(8), lam, estep="exact", K=None, max_E=15, record_obj=True)
    obj = np.array(tr["obj"])
    assert np.all(np.diff(obj) <= 1e-8 * np.abs(obj[:-1]) + 1e-10)


def test_s5_eiv_consistent():
    sys = ls.make_system(6, seed=5, snr_db=0.0, obs="full")
    _, y = ls.simulate(sys, 20000, np.random.default_rng(6))
    A = ls.eiv(y, sys.r, sys.q, 0.0)
    assert np.sum((A - sys.A) ** 2) / np.sum(sys.A ** 2) < 0.01


def test_s6_metric():
    sys = ls.make_system(8, seed=7, snr_db=0.0, obs="partial")
    _, yv = ls.simulate(sys, 300, np.random.default_rng(8))
    assert abs(ls.excess_nll(sys.A, sys, yv)) < 1e-12
    rng = np.random.default_rng(9)
    Ah = sys.A + 0.05 * rng.standard_normal(sys.A.shape)
    e = ls.excess_nll(Ah, sys, yv)
    assert e > 0
    m = sys.C.shape[0]
    O, _ = np.linalg.qr(rng.standard_normal((8 - m, 8 - m)))
    S = np.eye(8); S[m:, m:] = O
    assert abs(ls.excess_nll(S @ Ah @ S.T, sys, yv) - e) < 1e-9


def test_s7_generator():
    s1 = ls.make_system(32, seed=11, snr_db=0.0, obs="full")
    s2 = ls.make_system(32, seed=11, snr_db=0.0, obs="full")
    np.testing.assert_array_equal(s1.A, s2.A)
    assert abs(np.max(np.abs(np.linalg.eigvals(s1.A))) - 0.9) < 1e-9
    assert np.all((np.abs(s1.A) > 0).sum(1) == 3)
    p = ls.make_system(32, seed=11, snr_db=0.0, obs="partial")
    assert p.C.shape == (16, 32)
    hi = ls.make_system(32, seed=11, snr_db=10.0, obs="full")
    assert abs(hi.r / s1.r - 0.1) < 1e-12


def test_s8_counting_and_first_hit():
    assert ls.first_hit([0.5, 0.3, 0.2, 0.1], [1, 2, 3, 4], 0.2) == 3
    assert ls.first_hit([0.5, 0.3], [1, 2], 0.1) is None
    assert ls.first_hit([0.05], [0], 0.1) == 0
    sys, y = _small(n=5, T=30, obs="full", seed=12)
    calls = {"n": 0}
    orig = ls.estep_exact

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    ls.estep_exact = spy
    try:
        tr = ls.em_run(sys, y, 0.5 * np.eye(5), 0.1, estep="exact", K=5, max_E=4)
    finally:
        ls.estep_exact = orig
    assert calls["n"] == 4 and tr["n_E"] == [0, 1, 2, 3, 4]


def test_s9_squarem_fallback():
    target = np.array([1.0, 1.0])
    F = lambda a: a + 0.5 * (target - a)
    obj = lambda a: float(np.sum(a ** 2))
    a0 = np.array([0.0, 0.0])
    a1, info = ls.squarem_step(a0, F, obj)
    np.testing.assert_allclose(a1, F(F(a0)))
    assert info["fallback"]
    obj2 = lambda a: float(np.sum((a - target) ** 2))
    a2, info2 = ls.squarem_step(a0, F, obj2)
    assert not info2["fallback"] and obj2(a2) <= obj2(F(F(a0)))
