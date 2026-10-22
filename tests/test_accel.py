import numpy as np
import pytest

from unrolled_em import accel as ac
from unrolled_em import core as ls


def _quad(n=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((80, n))
    Phi = X.T @ X
    Psi = rng.standard_normal((n, n)) * 5 + Phi @ np.diag(rng.uniform(-.6, .6, n))
    q = 0.1
    fg = lambda A: ((np.sum((A @ Phi) * A) - 2 * np.sum(A * Psi)) / (2 * q), (A @ Phi - Psi) / q)
    return Phi, Psi, q, fg


@pytest.mark.parametrize("obs", ["full", "partial"])
def test_q1_fisher_gradient(obs):
    sys = ls.make_system(4, seed=1, snr_db=0.0, obs=obs)
    _, y = ls.simulate(sys, 10, np.random.default_rng(2))
    A = sys.A * 0.9
    st = ls.estep_exact(A, sys.C, sys.q, sys.r, sys.P1, y)
    g = ac.fisher_grad(A, st, sys.q)
    f = lambda X: -ls.loglik(X, sys.C, sys.q, sys.r, sys.P1, y)
    h = 1e-6
    for (i, j) in [(0, 0), (1, 2), (3, 1), (2, 3)]:
        E = np.zeros_like(A); E[i, j] = h
        fd = (f(A + E) - f(A - E)) / (2 * h)
        assert abs(fd - g[i, j]) <= 1e-6 * max(1.0, abs(fd))


@pytest.mark.parametrize("solver", ["owlqn", "sparsa"])
def test_q2_q3_l1_quadratic(solver):
    Phi, Psi, q, fg = _quad()
    lam = 30.0
    want = ls.mstep(np.zeros_like(Psi), Phi, Psi, q, lam, K=None)
    run = ac.owlqn if solver == "owlqn" else ac.sparsa
    tr = run(fg, np.zeros_like(Psi), lam, max_evals=3000, tol=1e-12)
    got = tr["A"][-1]
    assert np.linalg.norm(got - want) <= 1e-6 * max(np.linalg.norm(want), 1.0)
    assert np.sum(np.abs(want) > 0) < want.size


def test_q4_anderson():
    rng = np.random.default_rng(3)
    d = 4
    M = rng.standard_normal((d, d)); M = 0.8 * M / np.max(np.abs(np.linalg.eigvals(M)))
    b = rng.standard_normal(d)
    xs = np.linalg.solve(np.eye(d) - M, b)
    Fobj = lambda x: (M @ x + b, float(np.sum((x - xs) ** 2)))
    tr = ac.anderson(Fobj, np.zeros(d), max_evals=40, m=d, stop_fn=lambda x, k: np.linalg.norm(x - xs) < 1e-9)
    assert np.linalg.norm(tr["A"][-1] - xs) < 1e-9 and tr["n_E"][-1] <= d + 2
    cnt = {"n": 0}

    def Fbad(x):
        cnt["n"] += 1
        v = float(np.sum((x - xs) ** 2))
        return M @ x + b, (v if cnt["n"] <= 3 else -v)
    tr2 = ac.anderson(Fbad, np.zeros(d), max_evals=12, m=d)
    assert tr2["rejects"] >= 1 and tr2["memory_resets"] >= 1


def test_q5_counting():
    Phi, Psi, q, fg = _quad(seed=4)
    calls = {"n": 0}

    def fg_spy(A):
        calls["n"] += 1
        return fg(A)
    for run in (ac.owlqn, ac.sparsa):
        calls["n"] = 0
        tr = run(fg_spy, np.zeros_like(Psi), 10.0, max_evals=25)
        assert tr["n_E"][-1] == calls["n"] and np.all(np.diff(tr["n_E"]) >= 0)
    calls["n"] = 0

    def F_spy(x):
        calls["n"] += 1
        return 0.5 * x + 1.0, float(np.sum((x - 2.0) ** 2))
    tr = ac.anderson(F_spy, np.zeros(3), max_evals=10, m=2)
    assert tr["n_E"][-1] == calls["n"] and np.all(np.diff(tr["n_E"]) >= 0)


def test_q6_squarem_ss_uses_steady_state():
    sys = ls.make_system(5, seed=6, snr_db=0.0, obs="full")
    _, y = ls.simulate(sys, 60, np.random.default_rng(7))
    calls = {"ss": 0, "ex": 0}
    o_ss, o_ex = ls.estep_ss, ls.estep_exact

    def s1(*a, **k):
        calls["ss"] += 1
        return o_ss(*a, **k)

    def s2(*a, **k):
        calls["ex"] += 1
        return o_ex(*a, **k)
    ls.estep_ss, ls.estep_exact = s1, s2
    try:
        tr = ls.squarem_run(sys, y, 0.5 * np.eye(5), 1.0, K=10, max_E=6, estep="ss")
    finally:
        ls.estep_ss, ls.estep_exact = o_ss, o_ex
    assert calls["ss"] == 6 and calls["ex"] == 0 and tr["n_E"][-1] == 6


def test_q5b_counting_with_rejections():
    cnt = {"n": 0}

    def Fbad(x):
        cnt["n"] += 1
        v = float(np.sum((x - 2.0) ** 2))
        return 0.5 * x + 1.0, (v if cnt["n"] <= 3 else -v)
    tr = ac.anderson(Fbad, np.zeros(3), max_evals=10, m=2)
    assert tr["rejects"] >= 1 and tr["n_E"][-1] == cnt["n"]
