import os

import numpy as np
import pytest
import torch

from unrolled_em import core as ls
from unrolled_em.preliminary import classical as r0
from unrolled_em import unroll as u2


def _sysdata(n=8, obs="full", seed=0, T=80):
    sys = ls.make_system(n, seed=seed, snr_db=0.0, obs=obs)
    _, y = ls.simulate(sys, T, np.random.default_rng(seed + 1))
    return sys, y


class Fixed:
    def __init__(self, cand):
        self.cand = cand

    def __call__(self, A, R, Rp, F):
        return torch.as_tensor(self.cand)


class Spy:
    def __enter__(self):
        self.n = 0
        self.orig = ls.estep_ss

        def spy(*a, **k):
            self.n += 1
            return self.orig(*a, **k)
        ls.estep_ss = spy
        return self

    def __exit__(self, *a):
        ls.estep_ss = self.orig


@pytest.mark.parametrize("mode", ["learned", "scalar"])
@pytest.mark.parametrize("theta", ["sq", "softplus", "proj"])
def test_u1_identity_and_bounds(mode, theta):
    sys, y = _sysdata(obs="partial")
    lam = 2.0
    A = 0.5 * np.eye(8)
    st = u2.estep(A, sys, y)
    B = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
    layer = u2.Layer(mode, theta=theta)
    layer.set_identity()
    F = u2.features(A, B, np.zeros_like(A), st, sys, y.shape[0])
    with torch.no_grad():
        out = layer(torch.tensor(A), torch.tensor(B - A), torch.tensor(np.ones_like(A)), torch.tensor(F)).numpy()
    np.testing.assert_allclose(out, B, atol=1e-12, rtol=0)
    s, m = layer.step_params(torch.zeros(3, 3, u2.N_FEAT))
    np.testing.assert_allclose(s.detach().numpy(), 1.0, atol=1e-14)
    np.testing.assert_allclose(m.detach().numpy(), 0.0, atol=1e-14)
    assert float(layer.theta()) == 0.0
    if mode == "learned":
        with torch.no_grad():
            layer.mlp[-1].bias.copy_(torch.tensor([1e3, -1e3]))
        s, m = layer.step_params(torch.zeros(2, 2, u2.N_FEAT))
    else:
        with torch.no_grad():
            layer.a.fill_(1e3); layer.b.fill_(-1e3)
        s, m = layer.step_params(None)
    assert torch.all(s <= u2.S_MAX) and torch.all(m >= -u2.M_MAX)


def test_u2_guard():
    sys, y = _sysdata(n=8, obs="full", seed=3)
    lam = 1.0
    A = 0.5 * np.eye(8)
    st = u2.estep(A, sys, y)
    B = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
    Rp = np.zeros_like(A)
    with Spy() as sp:
        o = u2.layer_step(Fixed(2.0 * np.eye(8)), A, st, Rp, sys, y, lam)
    assert o["reject"] == "rho" and o["nE"] == 1 and sp.n == 1
    np.testing.assert_array_equal(o["A"], B)
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((8, 8)))
    with Spy() as sp:
        o = u2.layer_step(Fixed(0.9 * Q), A, st, Rp, sys, y, lam)
    assert o["reject"] == "obj" and o["nE"] == 2 and sp.n == 2
    np.testing.assert_array_equal(o["A"], B)
    with Spy() as sp:
        o = u2.layer_step(Fixed(B), A, st, Rp, sys, y, lam)
    assert o["reject"] is None and o["nE"] == 1 and sp.n == 1
    assert u2.fobj(o["A"], o["st"], lam) <= u2.fobj(A, st, lam)


def test_u3_advance_matches_run():
    sys, y = _sysdata(n=8, obs="partial", seed=4)
    lam = 1.0
    torch.manual_seed(1)
    layers = [u2.Layer("learned") for _ in range(2)]
    with torch.no_grad():
        for ly in layers:
            for p in ly.parameters():
                p.add_(0.2 * torch.randn_like(p))
    A0 = 0.5 * np.eye(8)
    states = u2.make_states(sys, y, lam, A_R=np.zeros((8, 8)), inits={"half": A0})
    for ly in layers:
        states = u2.advance(ly, states)
    res = u2.run_unrolled(layers, sys, y, lam, A0, stop_fn=lambda A, k: False, max_E=len(layers) + 1)
    np.testing.assert_array_equal(res["A_layers"][-1], states[0]["A"])


def test_u4_counting_with_rejections():
    sys, y = _sysdata(n=6, obs="full", seed=6, T=60)
    lam = 1.0
    A0 = 0.5 * np.eye(6)
    Q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((6, 6)))
    bad = Fixed(0.9 * Q)
    seen = []

    def stop(A, k):
        seen.append(k)
        return k > 0
    with Spy() as sp:
        res = u2.run_unrolled([bad], sys, y, lam, A0, stop_fn=stop, max_E=20)
    assert res["hit_E"] == 3 and res["n_E"] == 3 == sp.n and res["rejects"] == ["obj"]
    assert seen == [0, 3]
    st = u2.estep(A0, sys, y)
    B = ls.mstep(A0, st["Phi"], st["Psi"], sys.q, lam, K=50)
    with Spy() as sp:
        res = u2.run_unrolled([Fixed(B)], sys, y, lam, A0, stop_fn=lambda A, k: k > 0, max_E=20)
    assert res["hit_E"] == 2 and sp.n == 2 and res["rejects"] == [None]
    with Spy() as sp:
        res = u2.run_unrolled([Fixed(B)], sys, y, lam, A0, stop_fn=lambda A, k: False, max_E=11)
    assert res["hit_E"] is None and res["n_E"] == sp.n and res["n_E"] <= 11
    assert res["n_E"] == 2 + 2 + 3 + 3


def test_u5_mu_and_equivariance():
    sys, y = _sysdata(n=8, obs="partial", seed=12)
    A = 0.5 * np.eye(8)
    st = u2.estep(A, sys, y)
    B = ls.mstep(A, st["Phi"], st["Psi"], sys.q, 1.0, K=50)
    F = u2.features(A, B, np.zeros_like(A), st, sys, y.shape[0])
    T = y.shape[0]
    mu = (T - 1) * np.diag(st["Ps"]) / np.diag(st["Phi"])
    assert np.all((mu >= 0) & (mu <= 1))
    np.testing.assert_allclose(F[2, 5, 8], mu[2]); np.testing.assert_allclose(F[2, 5, 9], mu[5])
    assert F[3, 3, 7] == 1.0 and F[3, 4, 7] == 0.0
    m = sys.C.shape[0]
    assert mu[m:].mean() > mu[:m].mean()
    sys, y = _sysdata(n=8, obs="full", seed=2)
    torch.manual_seed(0)
    layer = u2.Layer("learned")
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(0.3 * torch.randn_like(p))
    A = 0.3 * np.eye(8) + 0.05 * np.random.default_rng(3).standard_normal((8, 8))
    Rp = 0.01 * np.random.default_rng(4).standard_normal((8, 8))
    o = u2.layer_step(layer, A, u2.estep(A, sys, y), Rp, sys, y, 1.0)
    P = np.eye(8)[np.random.default_rng(5).permutation(8)]
    from dataclasses import replace
    sysP = replace(sys, A=P @ sys.A @ P.T)
    AP = P @ A @ P.T
    oP = u2.layer_step(layer, AP, u2.estep(AP, sysP, y @ P.T), P @ Rp @ P.T, sysP, y @ P.T, 1.0)
    np.testing.assert_allclose(oP["cand"], P @ o["cand"] @ P.T, atol=1e-7)


def test_u6_fisher_loss():
    rng = np.random.default_rng(7)
    n = 5
    X = rng.standard_normal((40, n)); PhiR = X.T @ X
    AR = rng.standard_normal((n, n)); Al = AR + 0.1 * rng.standard_normal((n, n))
    D = Al - AR
    want = np.trace(D @ PhiR @ D.T) / np.trace(AR @ PhiR @ AR.T)
    got = u2.fisher_rel(torch.tensor(Al), torch.tensor(AR), torch.tensor(PhiR)).item()
    assert abs(got - want) < 1e-12
    got2 = u2.fisher_rel(torch.tensor(Al), torch.tensor(AR), torch.tensor(3.0 * np.eye(n))).item()
    assert abs(got2 - np.sum(D ** 2) / np.sum(AR ** 2)) < 1e-12


@pytest.mark.parametrize("cname", ["F32_500", "P32_500"])
def test_u7_init_augmentation(cname):
    pp = r0._load(os.path.join(r0.HERE, "protocol.json"))
    cell = next(c for c in pp["cells"] if c["name"] == cname)
    sys, y, _ = r0._setup(pp, cell, "train", 0)
    lam = r0._lam(cell, sys, 0.01)
    cands = u2.init_candidates(cell, sys, y, lam)
    assert set(cands) == ({"EIV", "LS", "half"} if cell["obs"] == "full" else {"half", "zero"})
    A0, name, _ = r0._init(cell, sys, y, lam, return_all=True)
    np.testing.assert_array_equal(cands[name], A0)
    AR = 0.5 * np.eye(32)
    states = u2.make_states(sys, y, lam, AR, cands)
    assert len(states) == len(cands) and all(s["A_R"] is AR for s in states)
    assert sorted(s["init"] for s in states) == sorted(cands)


def test_u8_seeds():
    pp = r0._load(os.path.join(r0.HERE, "protocol.json"))
    names = ["F32_500", "F64_500", "F128_500", "P32_500", "P64_500"]
    old = {r0._seed(pp["seed_root"], "sys", c, s, j) for c in names for s in ("train", "val", "test", "dev") for j in range(60)}
    new = {r0._seed(pp["seed_root"], "sys", c, "test1b", j) for c in names for j in range(20)}
    assert len(new) == 100 and not (old & new)
    cell = next(c for c in pp["cells"] if c["name"] == "F32_500")
    a, b = r0._setup(pp, cell, "test1b", 4), r0._setup(pp, cell, "test1b", 4)
    np.testing.assert_array_equal(a[1], b[1])


def test_u9_projected_theta():
    rng = np.random.default_rng(8)
    n, S = 6, 10

    def batch(thr):
        out = []
        for _ in range(S):
            A = rng.standard_normal((n, n)) * 0.3
            R = rng.standard_normal((n, n)) * 0.1
            Z = A + R
            out.append({"A": A, "B": A + R, "Rp": np.zeros((n, n)), "F": np.zeros((n, n, u2.N_FEAT)),
                        "target": np.sign(Z) * np.maximum(np.abs(Z) - thr, 0), "PhiR": np.eye(n)})
        return out
    ly = u2.Layer("scalar", theta="proj")
    ly.set_identity()
    b = batch(0.05)
    u2.train_layer(ly, b, b, steps=300, lr=0.01)
    assert 0.03 < float(ly.theta()) < 0.07
    ly2 = u2.Layer("scalar", theta="proj")
    with torch.no_grad():
        ly2.t.fill_(0.02)
    b2 = batch(0.0)
    u2.train_layer(ly2, b2, b2, steps=300, lr=0.01)
    assert float(ly2.theta()) >= 0.0 and float(ly2.theta()) < 1e-3


def test_u4b_rejected_candidate_never_counts_as_hit():
    sys, y = _sysdata(n=6, obs="full", seed=6, T=60)
    Q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((6, 6)))
    cand = 0.9 * Q
    res = u2.run_unrolled([Fixed(cand)], sys, y, 1.0, 0.5 * np.eye(6),
                          stop_fn=lambda A, k: bool(np.array_equal(A, cand)), max_E=8)
    assert res["rejects"] == ["obj"] and res["hit_E"] is None
    res2 = u2.run_unrolled([Fixed(cand)], sys, y, 1.0, 0.5 * np.eye(6), stop_fn=lambda A, k: False, max_E=8,
                           cand_stop=lambda A, k: bool(np.array_equal(A, cand)))
    assert res2["hit_E_cand"] == 1


def test_u9b_projection_holds_at_zero():
    rng = np.random.default_rng(9)
    n = 6
    b = []
    for _ in range(10):
        A = rng.standard_normal((n, n)) * 0.3
        R = rng.standard_normal((n, n)) * 0.1
        Z = A + R
        b.append({"A": A, "B": A + R, "Rp": np.zeros((n, n)), "F": np.zeros((n, n, u2.N_FEAT)),
                  "target": Z + 0.05 * np.sign(Z), "PhiR": np.eye(n)})
    ly = u2.Layer("scalar", theta="proj")
    ly.set_identity()
    u2.train_layer(ly, b, b, steps=200, lr=0.01)
    assert float(ly.theta().detach()) == 0.0


@pytest.mark.parametrize("where", ["core", "unroll2"])
def test_u11_squarem_survives_failed_estep(where):
    sys, y = _sysdata(n=6, obs="full", seed=6, T=60)
    orig = ls.estep_ss
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == (3 if where == "core" else 4):
            raise np.linalg.LinAlgError("boom")
        return orig(*a, **k)
    ls.estep_ss = flaky
    try:
        if where == "core":
            tr = ls.squarem_run(sys, y, 0.5 * np.eye(6), 1.0, K=10, max_E=3, estep="ss")
            assert tr["fallback"] == [True] and tr["n_E"][-1] == 3 and np.all(np.isfinite(tr["A"][-1]))
        else:
            st = u2.estep(0.5 * np.eye(6), sys, y); calls["n"] = 1
            B = ls.mstep(0.5 * np.eye(6), st["Phi"], st["Psi"], sys.q, 1.0, K=50)
            calls["n"] = 0
            res = u2.run_unrolled([Fixed(B)], sys, y, 1.0, 0.5 * np.eye(6), stop_fn=lambda A, k: False, max_E=4)
            assert res["n_E"] == 4 == calls["n"] and np.all(np.isfinite(res["A"]))
    finally:
        ls.estep_ss = orig


def test_u12_loss_normalized_by_em():
    rng = np.random.default_rng(10)
    n = 5
    batch = []
    for scale in (1.0, 1e3):
        A = rng.standard_normal((n, n)) * scale
        X = rng.standard_normal((30, n)); PhiR = X.T @ X
        batch.append({"A": A, "B": A + 0.1 * rng.standard_normal((n, n)), "Rp": np.zeros((n, n)),
                      "F": np.zeros((n, n, u2.N_FEAT)), "target": rng.standard_normal((n, n)), "PhiR": PhiR})
    ly = u2.Layer("scalar"); ly.set_identity()
    with torch.no_grad():
        assert abs(u2.batch_loss(ly, batch).item() - 1.0) < 1e-12
