from __future__ import annotations

import numpy as np

from unrolled_em import core as ls


def fisher_grad(A, st, q):
    return -(st["Psi"] - A @ st["Phi"]) / q


def _l1(x):
    return float(np.sum(np.abs(x)))


def _pseudo_grad(x, g, lam):
    p = g + lam * np.sign(x)
    z = x == 0
    gp, gm = g[z] + lam, g[z] - lam
    p[z] = np.where(gp < 0, gp, np.where(gm > 0, gm, 0.0))
    return p


def _two_loop(p, S, Y):
    q = p.ravel().copy()
    al = []
    for s, y in reversed(list(zip(S, Y))):
        rho = 1.0 / float(y @ s)
        a = rho * float(s @ q)
        al.append((a, rho, s, y))
        q -= a * y
    if S:
        q *= float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
    for a, rho, s, y in reversed(al):
        b = rho * float(y @ q)
        q += (a - b) * s
    return q.reshape(p.shape)


def owlqn(fg, x0, lam, max_evals=100, m=10, tol=None, stop_fn=None, c1=1e-4):
    tr = {"n_E": [0], "A": [x0.copy()]}
    if stop_fn is not None and stop_fn(x0, 0):
        return tr
    x = x0.copy()
    f, g = fg(x); nE = 1
    F = f + lam * _l1(x)
    S, Y = [], []
    while nE < max_evals:
        p = _pseudo_grad(x, g, lam)
        if not np.any(p):
            break
        d = -_two_loop(p, S, Y)
        d[d * p >= 0] = 0.0
        if not np.any(d):
            d = -p
        xi = np.sign(x); xi[x == 0] = np.sign(-p[x == 0])
        alpha = 1.0 if S else min(1.0, 1.0 / max(np.linalg.norm(p), 1e-300))
        accepted = False
        while nE < max_evals:
            xn = x + alpha * d
            xn[np.sign(xn) != xi] = 0.0
            fn, gn = fg(xn); nE += 1
            Fn = fn + lam * _l1(xn)
            if np.isfinite(Fn) and Fn <= F + c1 * float(np.sum(p * (xn - x))):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        s, yv = (xn - x).ravel(), (gn - g).ravel()
        if float(s @ yv) > 1e-12 * float(s @ s):
            S.append(s); Y.append(yv)
            if len(S) > m:
                S.pop(0); Y.pop(0)
        step = np.linalg.norm(xn - x)
        x, f, g, F = xn, fn, gn, Fn
        tr["n_E"].append(nE); tr["A"].append(x.copy())
        if stop_fn is not None and stop_fn(x, nE):
            return tr
        if tol is not None and step <= tol * max(np.linalg.norm(x), 1.0):
            break
    if tr["n_E"][-1] != nE:
        tr["n_E"].append(nE); tr["A"].append(x.copy())
    return tr


def sparsa(fg, x0, lam, max_evals=100, M=5, sigma=1e-4, eta=2.0, tol=None, stop_fn=None,
           a_min=1e-30, a_max=1e30):
    tr = {"n_E": [0], "A": [x0.copy()]}
    if stop_fn is not None and stop_fn(x0, 0):
        return tr
    x = x0.copy()
    f, g = fg(x); nE = 1
    hist = [f + lam * _l1(x)]
    alpha = 1.0
    while nE < max_evals:
        accepted = False
        while nE < max_evals:
            xn = ls._soft(x - g / alpha, lam / alpha)
            fn, gn = fg(xn); nE += 1
            Fn = fn + lam * _l1(xn)
            if np.isfinite(Fn) and Fn <= max(hist[-M:]) - 0.5 * sigma * alpha * float(np.sum((xn - x) ** 2)):
                accepted = True
                break
            alpha = min(alpha * eta, a_max)
        if not accepted:
            break
        s, yv = (xn - x).ravel(), (gn - g).ravel()
        ss = float(s @ s)
        step = np.sqrt(ss)
        if ss > 0:
            alpha = float(np.clip(float(s @ yv) / ss, a_min, a_max))
        x, f, g = xn, fn, gn
        hist.append(Fn)
        tr["n_E"].append(nE); tr["A"].append(x.copy())
        if stop_fn is not None and stop_fn(x, nE):
            return tr
        if tol is not None and step <= tol * max(np.linalg.norm(x), 1.0):
            break
    if tr["n_E"][-1] != nE:
        tr["n_E"].append(nE); tr["A"].append(x.copy())
    return tr


def anderson(Fobj, x0, max_evals=100, m=5, stop_fn=None, reg=1e-10):
    tr = {"n_E": [0], "A": [x0.copy()], "rejects": 0, "memory_resets": 0}
    if stop_fn is not None and stop_fn(x0, 0):
        return tr
    shape = np.shape(x0)
    x = np.array(x0, float).ravel()
    Fx, ox = Fobj(x.reshape(shape)); Fx = np.asarray(Fx, float).ravel(); nE = 1
    X, G = [x], [Fx - x]
    cand = Fx
    while nE < max_evals:
        Fc, oc = Fobj(cand.reshape(shape)); Fc = np.asarray(Fc, float).ravel(); nE += 1
        if np.isfinite(oc) and oc <= ox:
            x, Fx, ox = cand, Fc, oc
            X.append(x); G.append(Fx - x)
            X, G = X[-(m + 1):], G[-(m + 1):]
            tr["n_E"].append(nE); tr["A"].append(x.reshape(shape).copy())
            if stop_fn is not None and stop_fn(x.reshape(shape), nE):
                return tr
        else:
            tr["rejects"] += 1
            if len(X) > 1:
                tr["memory_resets"] += 1
            X, G = [x], [Fx - x]
            if np.array_equal(cand, Fx):
                break
            cand = Fx
            continue
        if len(G) >= 2:
            dG = np.stack([G[i + 1] - G[i] for i in range(len(G) - 1)], 1)
            dX = np.stack([X[i + 1] - X[i] for i in range(len(X) - 1)], 1)
            gk = G[-1]
            k = dG.shape[1]
            sc = float(np.sum(dG * dG)) / k
            if sc <= 0 or not np.isfinite(sc):
                cand = Fx
            else:
                Aug = np.vstack([dG, np.sqrt(reg * sc) * np.eye(k)])
                rhs = np.concatenate([gk, np.zeros(k)])
                gam = np.linalg.lstsq(Aug, rhs, rcond=None)[0]
                cand = x + gk - (dX + dG) @ gam
        else:
            cand = Fx
    if tr["n_E"][-1] != nE:
        tr["n_E"].append(nE); tr["A"].append(x.reshape(shape).copy())
    return tr


def make_fg(sys, y):
    def fg(A):
        try:
            st = ls.estep_exact(A, sys.C, sys.q, sys.r, sys.P1, y)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(A)
        if not np.isfinite(st["ll"]):
            return np.inf, np.zeros_like(A)
        return -st["ll"], fisher_grad(A, st, sys.q)
    return fg


def make_em_map(sys, y, lam, K=50):
    def Fobj(A):
        try:
            st = ls.estep_exact(A, sys.C, sys.q, sys.r, sys.P1, y)
        except np.linalg.LinAlgError:
            return A, np.inf
        FA = ls.mstep(A, st["Phi"], st["Psi"], sys.q, lam, K=K)
        return FA, -st["ll"] + lam * _l1(A)
    return Fobj
