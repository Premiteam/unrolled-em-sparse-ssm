from __future__ import annotations

import math

import torch
from torch import nn

from unrolled_em import unroll as u2

S_MIN, S_MAX, M_MAX = 0.1, u2.S_MAX, u2.M_MAX
MU_VARIANTS = ("j", "max", "or")


def rho_one(R):
    return torch.ones_like(R)


def rho_ait(R, Rp):
    ok = Rp != 0
    x = torch.where(ok, R / torch.where(ok, Rp, torch.ones_like(Rp)), torch.zeros_like(R))
    ok = ok & (x < 1) & torch.isfinite(x)
    s = torch.where(ok, 1.0 / (1.0 - torch.where(ok, x, torch.zeros_like(x))), torch.ones_like(R))
    return torch.clamp(s, S_MIN, S_MAX)


def rho_sec(A, Ap, R, Rp):
    dA, dR = A - Ap, R - Rp
    eps = 1e-12 * (torch.sqrt(torch.mean(R ** 2)) + 1e-300)
    ok = (dA != 0) & (torch.abs(dR) > eps)
    s = -dA / torch.where(ok, dR, torch.ones_like(dR))
    ok = ok & torch.isfinite(s) & (s > 0)
    s = torch.where(ok, s, torch.ones_like(s))
    return torch.clamp(s, S_MIN, S_MAX)


def mubar(F, variant):
    mi, mj = F[..., 8], F[..., 9]
    if variant == "j":
        return mj
    if variant == "max":
        return torch.maximum(mi, mj)
    if variant == "or":
        return 1.0 - (1.0 - mi) * (1.0 - mj)
    raise ValueError(variant)


def rho_mu(F, variant):
    mb = torch.clamp(mubar(F, variant), 0.0, 1.0 - 1.0 / S_MAX)
    return torch.clamp(1.0 / (1.0 - mb), 1.0, S_MAX)


def _rule_value(rule, variant, A, R, Rp, F, Ap=None):
    if rule == "one":
        return rho_one(R)
    if rule == "ait":
        return rho_ait(R, Rp)
    if rule == "mu":
        return rho_mu(F, variant)
    raise ValueError(rule)


class ZeroLayer(nn.Module):

    def __init__(self, rule, variant=None, cap=None):
        super().__init__()
        if rule not in ("sec", "ait", "mu", "one"):
            raise ValueError(rule)
        if rule == "mu" and variant not in MU_VARIANTS:
            raise ValueError(variant)
        self.rule, self.variant = rule, variant
        self.cap = cap
        self.Ap = None

    def reset(self):
        self.Ap = None

    def theta(self):
        return torch.tensor(0.0)

    def project_(self):
        pass

    def forward(self, A, R, Rp, F):
        if self.rule == "sec":
            s = torch.ones_like(R) if self.Ap is None else rho_sec(A, self.Ap, R, Rp)
        else:
            s = _rule_value(self.rule, self.variant, A, R, Rp, F)
        if self.cap is not None:
            s = torch.clamp(s, max=self.cap)
        self.Ap = A.detach().clone()
        return A + s * R


_U1 = math.log(math.expm1(1.0))


class TunedLayer(nn.Module):
    def __init__(self, rule, variant=None):
        super().__init__()
        if rule not in ("ait", "mu", "one"):
            raise ValueError(rule)
        if rule == "mu" and variant not in MU_VARIANTS:
            raise ValueError(variant)
        self.rule, self.variant = rule, variant
        self.u = nn.Parameter(torch.tensor(_U1))
        self.g = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.t = nn.Parameter(torch.tensor(1e-4))

    def c(self):
        return nn.functional.softplus(self.u)

    def gamma(self):
        return 2.0 * torch.sigmoid(self.g)

    def theta(self):
        return self.t

    def project_(self):
        with torch.no_grad():
            self.t.clamp_(min=0.0)

    def set_identity(self):
        with torch.no_grad():
            self.u.fill_(_U1); self.g.zero_(); self.b.zero_(); self.t.zero_()

    def step_s(self, rho):
        return torch.clamp(self.c() * rho ** self.gamma(), max=S_MAX)

    def forward(self, A, R, Rp, F):
        rho = _rule_value(self.rule, self.variant, A, R, Rp, F)
        s = self.step_s(rho)
        m = M_MAX * torch.tanh(self.b)
        Z = A + s * R + m * Rp
        return torch.sign(Z) * torch.clamp(torch.abs(Z) - self.theta(), min=0.0)
