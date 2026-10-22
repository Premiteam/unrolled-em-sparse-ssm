# DAAREM: damped Anderson acceleration with restarts and epsilon-monotonicity
# (N. C. Henderson and R. Varadhan, Journal of Computational and Graphical Statistics, 28(4):834-846, 2019).
#
# The functions damping_find and daarem_core are a line-by-line Python translation of the files
# R/DampingFind.R and R/daarem_base_objfn.R of the R package daarem, version 0.7,
# by Nicholas Henderson and Ravi Varadhan, distributed under the GNU General Public License version 2.
#
# Modified by Ruiqi Li in September 2026: translated from R to Python; added an optional check that
# rejects a candidate before its objective is evaluated (argument precheck of daarem_core); added the
# wrapper daarem_run for the sparse state space model, which stops the iteration when no new E-step is made.
#
# This file is distributed under the GNU General Public License version 2. See LICENSE and NOTICE.

from __future__ import annotations

import math

import numpy as np

from unrolled_em import core as ls


STALL = 50


class Fail(Exception):
    pass


def _propose_hook(x):
    return x


def damping_find(uy_sq, dvec, aa, kappa, sk, Ftf, lambda_start=None, r_start=None, maxit=10):
    if lambda_start is None or (isinstance(lambda_start, float) and math.isnan(lambda_start)):
        lambda_start = 100.0
    if r_start is None or (isinstance(r_start, float) and math.isnan(r_start)):
        r_start = 0.0
    uy_sq = np.asarray(uy_sq, float).ravel()
    dvec = np.asarray(dvec, float).ravel()
    if np.sum(dvec == 0.0) > 0:
        ind = dvec > 0
        dvec = dvec[ind]
        uy_sq = uy_sq[ind]
    pow_ = kappa - sk
    target = math.exp(-0.5 * math.log1p(aa ** pow_))
    d_sq = dvec * dvec
    betahat_ls = uy_sq / d_sq
    betahat_ls_norm = math.sqrt(float(np.sum(betahat_ls)))
    vk = target * betahat_ls_norm
    if vk == 0:
        return lambda_start, r_start
    lam = lambda_start - r_start / vk
    LL = (betahat_ls_norm * (betahat_ls_norm - vk)) / float(np.sum(uy_sq / (d_sq * d_sq)))
    UU = Ftf / vk
    l_stop = math.exp(-0.5 * math.log1p(aa ** (pow_ + 0.5)))
    u_stop = math.exp(-0.5 * math.log1p(aa ** (pow_ - 0.5)))
    s_norm, phi_ratio = None, None
    for _ in range(maxit):
        if lam <= LL or lam >= UU:
            lam = max(1e-4 * UU, math.sqrt(LL * UU))
        d_lambda = (dvec / (d_sq + lam)) ** 2
        d_prime = d_lambda / (d_sq + lam)
        s_norm = math.sqrt(float(np.sum(uy_sq * d_lambda)))
        phi_val = s_norm - vk
        phi_der = -1.0 * float(np.sum(uy_sq * d_prime)) / s_norm
        phi_ratio = phi_val / phi_der
        if u_stop * betahat_ls_norm >= s_norm >= l_stop * betahat_ls_norm:
            break
        UU = UU if phi_val >= 0 else lam
        LL = max(LL, lam - phi_ratio)
        lam = lam - (s_norm * phi_ratio) / vk
    if s_norm is None:
        return lam, r_start
    return lam, s_norm * phi_ratio


def daarem_core(fixptfn, objfn, par, maxiter, order=5, kappa=25, alpha=1.2, mon_tol=0.01, cycl_mon_tol=0.0,
                precheck=None, budget=None, trace=None):
    x0 = np.asarray(par, float).ravel()
    num_params = x0.size
    nlag = min(order, int(math.ceil(num_params / 2)))
    Fdiff = np.zeros((num_params, nlag))
    Xdiff = np.zeros((num_params, nlag))
    obj_vals = []

    def safe_obj(x):
        try:
            return float(objfn(x))
        except Fail:
            return float("nan")

    xold = x0
    try:
        xnew = np.asarray(fixptfn(xold), float).ravel()
    except Fail:
        return xold
    obj_vals.append(safe_obj(xold))
    obj_vals.append(safe_obj(xnew))
    fold = xnew - xold
    k = 1
    count = 0
    shrink_count = 0
    lambda_ridge = 100000.0
    r_penalty = 0.0
    ell_star = obj_vals[1]
    while k < maxiter:
        if budget is not None and budget():
            break
        count += 1
        try:
            ftmp = np.asarray(fixptfn(xnew), float).ravel()
        except Fail:
            break
        fnew = ftmp - xnew
        Fdiff[:, count - 1] = fnew - fold
        Xdiff[:, count - 1] = xnew - xold
        np_ = count
        Ftmp = Fdiff[:, :np_]
        Xtmp = Xdiff[:, :np_]
        U, dvec, Vt = np.linalg.svd(Ftmp, full_matrices=False)
        dvec_sq = dvec * dvec
        uy = U.T @ fnew
        uy_sq = uy * uy
        Ftf = math.sqrt(float(np.sum(uy_sq * dvec_sq)))
        lambda_ridge, r_penalty = damping_find(uy_sq, dvec, alpha, kappa, shrink_count, Ftf,
                                               lambda_start=lambda_ridge, r_start=r_penalty)
        dd = (dvec * uy) / (dvec_sq + lambda_ridge)
        gamma_vec = Vt.T @ dd
        xbar = xnew - Xtmp @ gamma_vec
        fbar = fnew - Ftmp @ gamma_vec
        x_propose = _propose_hook(xbar + fbar)
        xnew_before = xnew
        new_val, failed = None, False
        if precheck is not None and not precheck(x_propose):
            failed = True
        else:
            try:
                new_val = float(objfn(x_propose))
            except Fail:
                failed = True
        if not failed and not math.isnan(obj_vals[k]) and not math.isnan(new_val):
            if new_val >= obj_vals[k] - mon_tol:
                obj_vals.append(new_val)
                fold = fnew
                xold = xnew
                xnew = x_propose
                shrink_count += 1
                branch = "accept"
            else:
                fold = fnew
                xold = xnew
                xnew = fold + xold
                obj_vals.append(safe_obj(xnew))
                branch = "reject"
        else:
            fold = fnew
            xold = xnew
            xnew = fold + xold
            obj_vals.append(safe_obj(xnew))
            count = 0
            branch = "fail"
        if count == nlag:
            count = 0
            if obj_vals[k + 1] < ell_star - cycl_mon_tol:
                shrink_count = max(shrink_count - nlag, -2 * kappa)
            ell_star = obj_vals[k + 1]
        if trace is not None:
            trace.append({"branch": branch, "s": shrink_count, "count_after": count, "lambda": lambda_ridge,
                          "xnew_before": xnew_before.copy(), "xnew_after": xnew.copy()})
        k += 1
    return xnew


def daarem_run(sys, y, A0, lam, K=50, max_E=100, estep="ss", order=5, kappa=25, alpha=1.2, mon_tol=0.01,
               cycl_mon_tol=0.0):
    shape = A0.shape
    cache, order_keys = {}, []
    n = {"E": 0}

    def E(x):
        key = x.tobytes()
        if key in cache:
            return cache[key]
        n["E"] += 1
        fn = ls.estep_ss if estep == "ss" else ls.estep_exact
        try:
            st = fn(x.reshape(shape), sys.C, sys.q, sys.r, sys.P1, y)
        except np.linalg.LinAlgError:
            st = None
        cache[key] = st
        order_keys.append(key)
        while len(order_keys) > 8:
            cache.pop(order_keys.pop(0), None)
        return st

    def fixpt(x):
        st = E(x)
        if st is None:
            raise Fail
        return ls.mstep(x.reshape(shape), st["Phi"], st["Psi"], sys.q, lam, K=K).ravel()

    def obj(x):
        st = E(x)
        if st is None:
            raise Fail
        return st["ll"] - lam * float(np.sum(np.abs(x)))

    def precheck(x):
        if not np.all(np.isfinite(x)):
            return False
        return float(np.max(np.abs(np.linalg.eigvals(x.reshape(shape))))) < 1.0

    stall = {"last": -1, "k": 0}

    def budget():
        if n["E"] >= max_E:
            return True
        stall["k"] = stall["k"] + 1 if n["E"] == stall["last"] else 0
        stall["last"] = n["E"]
        return stall["k"] >= STALL

    tr = []
    x = daarem_core(fixpt, obj, A0.ravel(), maxiter=10 ** 6, order=order, kappa=kappa, alpha=alpha,
                    mon_tol=mon_tol, cycl_mon_tol=cycl_mon_tol, precheck=precheck,
                    budget=budget, trace=tr)
    return {"A": x.reshape(shape), "n_E": n["E"], "branches": [t["branch"] for t in tr]}
