from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from unrolled_em import core as ls
from unrolled_em.preliminary import classical as r0

torch.set_default_dtype(torch.float64)
N_FEAT = 10
S_MAX, M_MAX = 10.0, 3.0
B0 = math.log(1.0 / (S_MAX - 1.0))


class Layer(nn.Module):
    def __init__(self, mode="learned", theta="proj", hidden=16):
        super().__init__()
        self.mode, self.theta_mode = mode, theta
        if mode == "learned":
            self.mlp = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 2))
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        elif mode == "scalar":
            self.a = nn.Parameter(torch.zeros(()))
            self.b = nn.Parameter(torch.zeros(()))
        else:
            raise ValueError(mode)
        if theta == "sq":
            self.c = nn.Parameter(torch.tensor(0.01))
        elif theta == "softplus":
            self.u = nn.Parameter(torch.tensor(math.log(math.expm1(1e-4))))
        elif theta == "proj":
            self.t = nn.Parameter(torch.tensor(1e-4))
        else:
            raise ValueError(theta)

    def theta(self):
        if self.theta_mode == "sq":
            return self.c ** 2
        if self.theta_mode == "softplus":
            return nn.functional.softplus(self.u)
        return self.t

    def project_(self):
        if self.theta_mode == "proj":
            with torch.no_grad():
                self.t.clamp_(min=0.0)

    def set_identity(self):
        with torch.no_grad():
            if self.mode == "learned":
                self.mlp[-1].weight.zero_(); self.mlp[-1].bias.zero_()
            else:
                self.a.zero_(); self.b.zero_()
            if self.theta_mode == "sq":
                self.c.zero_()
            elif self.theta_mode == "softplus":
                self.u.fill_(-math.inf)
            else:
                self.t.zero_()

    def step_params(self, F):
        if self.mode == "learned":
            h = self.mlp(F)
            h1, h2 = h[..., 0], h[..., 1]
        else:
            h1, h2 = self.a, self.b
        return S_MAX * torch.sigmoid(h1 + B0), M_MAX * torch.tanh(h2)

    def forward(self, A, R, Rp, F):
        s, m = self.step_params(F)
        Z = A + s * R + m * Rp
        return torch.sign(Z) * torch.clamp(torch.abs(Z) - self.theta(), min=0.0)


def estep(A, sys, y):
    return ls.estep_ss(A, sys.C, sys.q, sys.r, sys.P1, y)


def fobj(A, st, lam):
    return -st["ll"] + lam * float(np.sum(np.abs(A)))


def features(A, B, Rp, st, sys, T):
    n = A.shape[0]
    R = B - A
    sc = np.sqrt(np.mean(R ** 2)) + 1e-12
    obs = np.zeros(n); obs[np.argmax(sys.C, axis=1)] = 1.0
    dPhi = np.maximum(np.diag(st["Phi"]), 1e-300)
    d = np.log(dPhi / ((T - 1) * sys.sigbar2))
    mu = np.clip((T - 1) * np.diag(st["Ps"]) / dPhi, 0.0, 1.0)
    F = np.empty((n, n, N_FEAT))
    F[..., 0] = R / sc
    F[..., 1] = Rp / sc
    F[..., 2] = A
    F[..., 3] = B
    F[..., 4] = obs[:, None]
    F[..., 5] = obs[None, :]
    F[..., 6] = d[None, :]
    F[..., 7] = np.eye(n)
    F[..., 8] = mu[:, None]
    F[..., 9] = mu[None, :]
    return F


def layer_step(layer, A, st, Rp, sys, y, lam):
    B = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
    R = B - A
    F = features(A, B, Rp, st, sys, y.shape[0])
    with torch.no_grad():
        cand = layer(torch.tensor(A), torch.tensor(R), torch.tensor(Rp), torch.tensor(F)).numpy().copy()
    out = {"Rp": R, "cand": cand}
    if not np.all(np.isfinite(cand)) or float(np.max(np.abs(np.linalg.eigvals(cand)))) >= 1.0:
        out.update(A=B, st=estep(B, sys, y), nE=1, reject="rho")
        return out
    try:
        st_c = estep(cand, sys, y)
        ok = np.isfinite(st_c["ll"]) and fobj(cand, st_c, lam) <= fobj(A, st, lam)
    except np.linalg.LinAlgError:
        ok = False
    if ok:
        out.update(A=cand, st=st_c, nE=1, reject=None)
    else:
        out.update(A=B, st=estep(B, sys, y), nE=2, reject="obj")
    return out


def run_unrolled(layers, sys, y, lam, A0, stop_fn, max_E=100, cont="sq", cand_stop=None):
    n = {"E": 0}
    res = {"hit_E": None, "hit_E_cand": None, "rejects": [], "A_layers": []}

    def done(A):
        res.update(n_E=n["E"], A=A)
        if res["hit_E_cand"] is None:
            res["hit_E_cand"] = res["hit_E"]
        return res
    A = A0.copy()
    if stop_fn(A, 0):
        res["hit_E"] = 0
        return done(A)
    st = estep(A, sys, y); n["E"] += 1
    Rp = np.zeros_like(A)
    for layer in layers:
        if n["E"] + 1 > max_E:
            return done(A)
        before = n["E"]
        o = layer_step(layer, A, st, Rp, sys, y, lam)
        n["E"] += o["nE"]
        if cand_stop is not None and res["hit_E_cand"] is None and cand_stop(o["cand"], before):
            res["hit_E_cand"] = before
        A, st, Rp = o["A"], o["st"], o["Rp"]
        res["rejects"].append(o["reject"]); res["A_layers"].append(A.copy())
        if stop_fn(A, n["E"]):
            res["hit_E"] = n["E"]
            return done(A)
    if cont == "tied" and layers:
        while n["E"] + 1 <= max_E:
            o = layer_step(layers[-1], A, st, Rp, sys, y, lam)
            n["E"] += o["nE"]
            A, st, Rp = o["A"], o["st"], o["Rp"]
            res["rejects"].append(o["reject"])
            if stop_fn(A, n["E"]):
                res["hit_E"] = n["E"]
                return done(A)
        return done(A)
    cache = {A.tobytes(): st}
    lls = {}

    def Fmap(X):
        s = cache.pop(X.tobytes(), None)
        if s is None:
            n["E"] += 1
            try:
                s = estep(X, sys, y)
            except np.linalg.LinAlgError:
                return np.full_like(X, np.nan)
        lls[X.tobytes()] = s["ll"]
        return ls.mstep(X, s["Phi"], s["Psi"], sys.q, lam, K=50)

    def obj(X):
        ll = lls.get(X.tobytes())
        return np.inf if ll is None else -ll + lam * float(np.sum(np.abs(X)))
    first = True
    while n["E"] + (2 if first else 3) <= max_E:
        A, _ = ls.squarem_step(A, Fmap, obj)
        first = False
        lls.clear(); cache.clear()
        if stop_fn(A, n["E"]):
            res["hit_E"] = n["E"]
            return done(A)
    return done(A)


def init_candidates(cell, sys, y, lam):
    n = sys.A.shape[0]
    if cell["obs"] == "full":
        return {"EIV": r0._project(ls.eiv(y, sys.r, sys.q, lam)), "LS": r0._project(ls.eiv(y, 0.0, sys.q, lam)),
                "half": 0.5 * np.eye(n)}
    return {"half": 0.5 * np.eye(n), "zero": np.zeros((n, n))}


def make_states(sys, y, lam, A_R, inits, PhiR=None):
    if PhiR is None:
        PhiR = estep(A_R, sys, y)["Phi"]
    return [{"sys": sys, "y": y, "lam": lam, "A": A0.copy(), "st": estep(A0, sys, y), "Rp": np.zeros_like(A0),
             "A_R": A_R, "PhiR": PhiR, "init": name} for name, A0 in inits.items()]


def layer_batch(states):
    batch = []
    for s in states:
        B = ls.mstep(s["A"], s["st"]["Phi"], s["st"]["Psi"], s["sys"].q, s["lam"], K=50)
        batch.append({"A": s["A"], "B": B, "Rp": s["Rp"],
                      "F": features(s["A"], B, s["Rp"], s["st"], s["sys"], s["y"].shape[0]),
                      "target": s["A_R"], "PhiR": s["PhiR"]})
    return batch


def advance(layer, states):
    out = []
    for s in states:
        o = layer_step(layer, s["A"], s["st"], s["Rp"], s["sys"], s["y"], s["lam"])
        out.append({**s, "A": o["A"], "st": o["st"], "Rp": o["Rp"], "reject": o["reject"]})
    return out


def fisher_rel(Al, AR, PhiR):
    D = Al - AR
    return torch.sum((D @ PhiR) * D) / torch.sum((AR @ PhiR) * AR)


def _tensors(b):
    if "_t" not in b:
        t = tuple(torch.as_tensor(b[k]) for k in ("A", "B", "Rp", "F", "target", "PhiR"))
        with torch.no_grad():
            em = torch.clamp(fisher_rel(t[1], t[4], t[5]), min=1e-12)
        b["_t"] = t + (em,)
    return b["_t"]


def batch_loss(layer, batch):
    tot = 0.0
    for b in batch:
        A, B, Rp, F, tgt, PhiR, em = _tensors(b)
        tot = tot + fisher_rel(layer(A, B - A, Rp, F), tgt, PhiR) / em
    return tot / len(batch)


def train_layer(layer, train, val, steps=2000, lr=1e-2, log=None):
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    best, best_state = float("inf"), None
    for s in range(1, steps + 1):
        loss = batch_loss(layer, train)
        opt.zero_grad(); loss.backward(); opt.step()
        layer.project_()
        if s % 50 == 0 or s == steps:
            with torch.no_grad():
                v = batch_loss(layer, val).item()
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in layer.state_dict().items()}
            if log is not None:
                log.append({"step": s, "train": loss.item(), "val": v})
    if best_state is not None:
        layer.load_state_dict(best_state)
    return best
