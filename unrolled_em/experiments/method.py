from __future__ import annotations

import numpy as np
import torch

from unrolled_em import core as ls
from unrolled_em import handcrafted as hc
from unrolled_em import unroll as u2

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

N_FEAT = u2.N_FEAT
ETA = 1e-3
THETA = "proj"


def estep(A, sys, y):
    return ls.estep_ss(A, sys.C, sys.q, sys.r, sys.P1, y)


def fobj(A, st, lam):
    return -st["ll"] + lam * float(np.sum(np.abs(A)))


def surrogate(X, Phi, Psi, q, lam):
    return float((np.sum((X @ Phi) * X) - 2.0 * np.sum(X * Psi)) / (2.0 * q) + lam * np.sum(np.abs(X)))


def features(A, B, Rp, st, sys, T):
    n = A.shape[0]
    R = B - A
    sc = np.sqrt(np.mean(R ** 2)) + 1e-12
    obs = np.zeros(n); obs[np.argmax(sys.C, axis=1)] = 1.0
    dPhi = np.maximum(np.diag(st["Phi"]), 1e-300)
    d = np.log(dPhi / np.mean(dPhi))
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


def layer_step(layer, A, st, Rp, sys, y, lam, eta=None):
    eta = ETA if eta is None else eta
    B = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=50)
    delta = max(surrogate(A, st["Phi"], st["Psi"], sys.q, lam) - surrogate(B, st["Phi"], st["Psi"], sys.q, lam), 0.0)
    R = B - A
    F = features(A, B, Rp, st, sys, y.shape[0])
    with torch.no_grad():
        cand = layer(torch.tensor(A), torch.tensor(R), torch.tensor(Rp), torch.tensor(F)).numpy().copy()
    F_old = fobj(A, st, lam)
    out = {"Rp": R, "cand": cand, "F_old": F_old, "delta": delta}
    if not np.all(np.isfinite(cand)) or float(np.max(np.abs(np.linalg.eigvals(cand)))) >= 1.0:
        stB = estep(B, sys, y)
        out.update(A=B, st=stB, nE=1, reject="rho", F_new=fobj(B, stB, lam))
        return out
    reason = "obj"
    try:
        st_c = estep(cand, sys, y)
        F_c = fobj(cand, st_c, lam)
        ok = bool(np.isfinite(st_c["ll"]) and F_c <= F_old - eta * delta)
        if not ok and np.isfinite(F_c) and F_c <= F_old:
            reason = "eta"
    except np.linalg.LinAlgError:
        ok = False
    if ok:
        out.update(A=cand, st=st_c, nE=1, reject=None, F_new=F_c)
    else:
        stB = estep(B, sys, y)
        out.update(A=B, st=stB, nE=2, reject=reason, F_new=fobj(B, stB, lam))
    return out


def run_unrolled(layers, sys, y, lam, A0, max_E=100, cont="tied", log=None, eta=None):
    log = [] if log is None else log
    n = {"E": 0}
    A = A0.copy()
    st = estep(A, sys, y); n["E"] += 1
    Rp = np.zeros_like(A)
    seq = list(layers)
    k = 0
    while n["E"] + 1 <= max_E:
        if k < len(seq):
            layer = seq[k]
        elif cont == "tied" and seq:
            layer = seq[-1]
        else:
            break
        o = layer_step(layer, A, st, Rp, sys, y, lam, eta)
        n["E"] += o["nE"]
        A, st, Rp = o["A"], o["st"], o["Rp"]
        log.append((o["reject"] or "ok", o["F_old"], o["F_new"], o["delta"]))
        k += 1
    if cont == "sq" and n["E"] + 2 <= max_E:
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
    return {"A": A, "n_E": n["E"], "log": log}


def new_layer(arm, mu_variant=None):
    if arm == "learned":
        return u2.Layer("learned", THETA)
    if arm == "scalar":
        return u2.Layer("scalar", THETA)
    if arm == "ait_t":
        return hc.TunedLayer("ait")
    if arm == "mu_t":
        return hc.TunedLayer("mu", mu_variant)
    raise ValueError(arm)


def zero_layers(rule, n_layers, mu_variant=None):
    one = hc.ZeroLayer(rule, mu_variant if rule == "mu" else None)
    return [one] * n_layers


def save_layers(path, arm, layers, mu_variant=None, log=None):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"arm": arm, "mu_variant": mu_variant, "state": [ly.state_dict() for ly in layers]}, path)


def load_layers(path):
    d = torch.load(path, weights_only=False)
    out = []
    for s in d["state"]:
        ly = new_layer(d["arm"], d["mu_variant"]); ly.load_state_dict(s); ly.eval(); out.append(ly)
    return out


def make_states(sys, y, lam, A_R, inits):
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


def advance(layer, states, eta=None):
    out = []
    for s in states:
        o = layer_step(layer, s["A"], s["st"], s["Rp"], s["sys"], s["y"], s["lam"], eta)
        out.append({**s, "A": o["A"], "st": o["st"], "Rp": o["Rp"], "reject": o["reject"]})
    return out


def rej_count(states):
    r = [s.get("reject") for s in states]
    return {"rho": r.count("rho"), "obj": r.count("obj"), "eta": r.count("eta"), "n": len(r)}


def train_chain(tr, va, arm, n_layers, steps, lr=1e-2, mu_variant=None, eta=None, tag=""):
    import time
    torch.manual_seed(0)
    layers, log = [], []
    for l in range(n_layers):
        t = time.time()
        btr, bva = layer_batch(tr), layer_batch(va)
        em = u2.Layer("scalar", THETA); em.set_identity()
        with torch.no_grad():
            l_em = u2.batch_loss(em, bva).item()
        layer = new_layer(arm, mu_variant)
        curve = []
        best = u2.train_layer(layer, btr, bva, steps=steps, lr=lr, log=curve)
        layer.eval()
        del btr, bva
        layers.append(layer)
        tr, va = advance(layer, tr, eta), advance(layer, va, eta)
        entry = {"layer": l + 1, "val_em": l_em, "val_best": best, "ratio": best / l_em, "sec": time.time() - t,
                 "theta": float(layer.theta().detach()), "reject_train": rej_count(tr), "reject_val": rej_count(va),
                 "curve": curve}
        if arm in ("ait_t", "mu_t"):
            entry.update(c=float(layer.c().detach()), gamma=float(layer.gamma().detach()),
                         m=float(hc.M_MAX * torch.tanh(layer.b).detach()))
        log.append(entry)
        print(f"  训练 {tag} {arm} 层{l + 1}: 验证 EM一步 {l_em:.4f} → {best:.4f}（比 {best / l_em:.3f}，"
              f"{time.time() - t:.0f}s）拒绝 {entry['reject_val']}", flush=True)
    return layers, log
