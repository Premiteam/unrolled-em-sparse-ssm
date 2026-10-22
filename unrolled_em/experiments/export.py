from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from unrolled_em.experiments import common as cm

OUT = cm.OUT
PD = os.path.join(OUT, "paper_data")
M = cm.MULTS
I11, I101 = M.index(1.1), M.index(1.01)
P = "learned_tied_L4"
CLS_A = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
CLS_SS = ["EM_ss_K50", "SQ_ss_K50", "DAAREM_tun"]
CLS_BC = ["SQ_ss_K50", "DAAREM_tun"]
CLS_I = ["SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
FORM = ["sec0", "ait0", "mu0", "ait_t", "mu_t"]
FIG_ARMS = [(P, "prop"), ("EM_ss_K50", "em"), ("SQ_ss_K50", "sq"), ("DAAREM_tun", "da")]
v = lambda x: cm.FAIL if x is None else x


def _load_dir(sub):
    return [cm.load(f) for f in sorted(glob.glob(os.path.join(OUT, sub, "*.json")))]


def _by_cell(recs):
    out = {}
    for r in recs:
        out.setdefault(r["cell"]["name"], []).append(r)
    return out


def H(recs, arm, i):
    return [r["arms"][arm]["hits"][i] for r in recs]


def o_star(recs, i, cls, prim=P):
    st = {a: cm.ratio_stats(H(recs, a, i), H(recs, prim, i)) for a in cls}
    o = min(cls, key=lambda a: st[a]["G"])
    return o, st[o]


def qmed(recs, arm, i, key):
    xs = [r["arms"][arm]["q"][i][key] for r in recs if r["arms"][arm].get("q") and r["arms"][arm]["q"][i]]
    return float(np.median(xs)) if xs else None


def write(name, header, rows):
    os.makedirs(PD, exist_ok=True)
    with open(os.path.join(PD, name), "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(str(x) for x in r) + "\n")


def rej_stats(recs, arm, first=cm.L_MAIN):
    tot = {"ok": 0, "rho": 0, "obj": 0, "eta": 0}
    fst = {"ok": 0, "rho": 0, "obj": 0, "eta": 0}
    for r in recs:
        for k, x in enumerate(r["arms"][arm].get("log") or []):
            tot[x] += 1
            if k < first:
                fst[x] += 1
    rate = lambda d: (sum(d.values()) - d["ok"]) / max(1, sum(d.values()))
    return {"all": tot, "first": fst, "rate_all": rate(tot), "rate_first": rate(fst),
            "rate_cont": rate({k: tot[k] - fst[k] for k in tot}),
            "rho_share_first": fst["rho"] / max(1, fst["rho"] + fst["obj"] + fst["eta"])}


def mono_stats(recs, arm):
    n = inc = 0
    worst = 0.0
    for r in recs:
        for F_old, F_new, _ in r["arms"][arm].get("F") or []:
            n += 1
            if F_new > F_old + 1e-12 * abs(F_old):
                inc += 1
                worst = max(worst, (F_new - F_old) / abs(F_old))
    return {"steps": n, "increases": inc, "max_rel_increase": worst}


def extra_hits(r, arm, col, thr):
    ex = r["arms"][arm].get("extra") or []
    h = next((c for c, *rest in ex if (rest[col - 1] <= thr)), None)
    return h if (h is not None and h <= cm.MAX_E) else None


def part_II():
    recs = _by_cell(_load_dir("testII"))
    out = {"cells": {}}
    for c in cm.CELLS:
        R = recs.get(c, [])
        if not R:
            continue
        ent = {"n": len(R), "mults": {}}
        for i, m in enumerate(M):
            o, st = o_star(R, i, CLS_A)
            o_ex, st_ex = o_star(R, i, [a for a in CLS_A if a != "EM_ss_K50"])
            e = {"G": st["G"], "ci": st["ci"], "o_star": o, "G_noEM": st_ex["G"],
                 "median": {a: cm.med(H(R, a, i)) for a in R[0]["arms"]},
                 "n_fail": {a: sum(x is None for x in H(R, a, i)) for a in R[0]["arms"]}}
            if m in (1.1, 1.01):
                e["delta"] = {a: qmed(R, a, i, "delta") for a in R[0]["arms"]}
                e["f1"] = {a: qmed(R, a, i, "f1") for a in R[0]["arms"]}
                e["L4_over_L3"] = cm.ratio_stats(H(R, P, i), H(R, "learned_tied_L3", i))
            ent["mults"][str(m)] = e
        ent["q_R"] = {k: float(np.median([r["q_R"][k] for r in R])) for k in ("delta", "f1", "prec", "rec")}
        ent["q_Rss"] = {k: float(np.median([r["q_Rss"][k] for r in R])) for k in ("delta", "f1")}
        ent["dist_Rss_R"] = {"median": float(np.median([r["dist_Rss_R"] for r in R])),
                             "max": float(np.max([r["dist_Rss_R"] for r in R]))}
        ent["rejects"] = rej_stats(R, P)
        ent["mono"] = mono_stats(R, P)
        capped = [r for r in R if r["nE_R"] >= cm.REF_MAX_E]
        ent["ref"] = {"nE_median": float(np.median([r["nE_R"] for r in R])), "capped": len(capped),
                      "capped_max_last_rel": max([r["ref_last_rel_change"] for r in capped], default=None),
                      "capped_M_change_last50": [abs(r["M_R"] - r["M_R_m50"]) / abs(r["M_R"]) for r in capped
                                                 if r.get("M_R_m50") is not None]}
        s = [r["s2_hat"] / r["sigbar2_true"] - 1 for r in R]
        ent["scale_rel_err"] = {"median_abs": float(np.median(np.abs(s))), "max_abs": float(np.max(np.abs(s)))}
        ent["init"] = {k: sum(r["init"] == k for r in R) for k in sorted({r["init"] for r in R})}
        a7 = {}
        for name, col, thr in (("dist5", 1, 0.05), ("dist1", 1, 0.01), ("gap1e-4", 3, 1e-4), ("gap1e-5", 3, 1e-5)):
            Hx = {a: [extra_hits(r, a, col, thr) for r in R] for a in R[0]["arms"]}
            st = {a: cm.ratio_stats(Hx[a], Hx[P]) for a in CLS_SS}
            o2 = min(CLS_SS, key=lambda a: st[a]["G"])
            a7[name] = {"G": st[o2]["G"], "ci": st[o2]["ci"], "o_star": o2, "median": {a: cm.med(x) for a, x in Hx.items()},
                        "n_fail": {a: sum(x is None for x in Hx[a]) for a in Hx}}
        ent["A7"] = a7
        if c == "F128_500":
            ent["sq_cont_vs_tied_101"] = {**cm.ratio_stats(H(R, "learned_L4_sq", I101), H(R, P, I101)),
                                          "median_sq": cm.med(H(R, "learned_L4_sq", I101)), "median_tied": cm.med(H(R, P, I101)),
                                          "fail_sq": sum(x is None for x in H(R, "learned_L4_sq", I101)),
                                          "fail_tied": sum(x is None for x in H(R, P, I101))}
        out["cells"][c] = ent
    tr = {}
    for key, c, arm in (("32to64", "F64_500", "tied_from32_L4"), ("64to128", "F128_500", "tied_from64_L4")):
        R = recs.get(c, [])
        if R:
            tr[key] = {}
            for m in (1.1, 1.01):
                i = M.index(m)
                o, st = o_star(R, i, CLS_A, prim=arm)
                tr[key][str(m)] = {"G": st["G"], "ci": st["ci"], "o_star": o,
                                   "loss_vs_native": cm.ratio_stats(H(R, arm, i), H(R, P, i))["G"]}
    out["transfer"] = tr
    rows = []
    for k, c in enumerate(cm.CELLS):
        R = recs.get(c, [])
        if R:
            rows.append([k, cm.SHORT[c]] + [cm.med(H(R, a, I11)) for a, _ in FIG_ARMS] + [cm.med(H(R, a, I101)) for a, _ in FIG_ARMS])
    write("main_median.csv", ["x", "setting"] + [f"{t}110" for _, t in FIG_ARMS] + [f"{t}101" for _, t in FIG_ARMS], rows)
    for c in ("F64_500", "F128_500"):
        R = recs.get(c, [])
        if R:
            write(f"tol_{cm.SHORT[c]}.csv", ["i", "alpha"] + [t for _, t in FIG_ARMS],
                  [[i, m] + [cm.med(H(R, a, i)) for a, _ in FIG_ARMS] for i, m in enumerate(M)])
    R = recs.get("F128_500", [])
    if R:
        arms = [P, "SQ_ss_K50", "DAAREM_tun"]
        md = {a: (cm.med(H(R, a, I11)), cm.med(H(R, a, I101))) for a in arms}
        dist = sorted((sum(abs(v(r["arms"][a]["hits"][I11]) - md[a][0]) / md[a][0]
                           + abs(v(r["arms"][a]["hits"][I101]) - md[a][1]) / md[a][1] for a in arms), r["j"]) for r in R)
        jt = dist[0][1]
        r = next(x for x in R if x["j"] == jt)
        out["trace_system"] = {"j": jt, "hits": {a: r["arms"][a]["hits"] for a in r["arms"]}}
        for a, t in FIG_ARMS:
            write(f"trace_{t}.csv", ["e", "r"], [[cc, mm / r["M_R"]] for cc, mm in r["arms"][a]["trace"] if cc <= 60])
            write(f"distR_{t}.csv", ["e", "d", "g"], [[e[0], e[1], max(e[3], 1e-12)] for e in r["arms"][a]["extra"] if e[0] <= 60])
    return out


def _excess(trace, M_R, k, floor=0.01):
    m = [mm for cc, mm in trace if cc <= k][-1]
    return max((m - M_R) / M_R, floor)


def part_I():
    recs = _by_cell(_load_dir("testI"))
    if not recs:
        return None
    cells_I = [c for c in cm.CELLS if c in recs]
    names = {"learned": P, "scalar": "scalar_tied", "DAAREM": "DAAREM_tun", **{h: f"{h}_tied" for h in FORM}}
    abl, layers = {"mults": {}}, {"mults": {}}
    for i, m in enumerate(M):
        if m not in (1.1, 1.01):
            continue
        cells, dS, dH = {}, [], []
        lay = {}
        for c in cells_I:
            R = recs[c]
            sq_arm = min(("SQ_ss_K50", "SQ_exact_K50"), key=lambda a: cm.ratio_stats(H(R, a, i), H(R, P, i))["G"])
            sq = H(R, sq_arm, i)
            vs = {k: cm.ratio_stats(sq, H(R, a, i))["G"] for k, a in names.items()}
            fst = {h: cm.ratio_stats(H(R, f"{h}_tied", i), H(R, P, i)) for h in FORM}
            hstar = min(sorted(FORM), key=lambda h: fst[h]["G"])
            sc = cm.ratio_stats(H(R, "scalar_tied", i), H(R, P, i))
            cells[c] = {"squarem_arm": sq_arm, "vs_squarem": vs, "h_star": hstar, "formula_over_learned": fst[hstar]["G"],
                        "formula_ci": fst[hstar]["ci"], "formula_all": {h: fst[h]["G"] for h in FORM},
                        "scalar_over_learned": sc["G"], "scalar_ci": sc["ci"],
                        "median": {k: cm.med(H(R, a, i)) for k, a in {**names, "SQUAREM": sq_arm}.items()},
                        "n_fail": {k: sum(x is None for x in H(R, a, i)) for k, a in {**names, "SQUAREM": sq_arm}.items()}}
            dS += [v(a) - v(b) for a, b in zip(H(R, "scalar_tied", i), H(R, P, i))]
            dH += [v(a) - v(b) for a, b in zip(H(R, f"{hstar}_tied", i), H(R, P, i))]
            le = {}
            for l in range(1, cm.L_MAX + 1):
                a = f"learned_tied_L{l}"
                o, st = o_star(R, i, CLS_I, prim=a)
                vs3 = cm.ratio_stats(H(R, a, i), H(R, "learned_tied_L3", i))
                le[l] = {"G": st["G"], "ci": st["ci"], "o_star": o, "median": cm.med(H(R, a, i)), "vs_L3": vs3["G"],
                         "vs_L3_ci": vs3["ci"], "n_fail": sum(x is None for x in H(R, a, i)),
                         "rejects": rej_stats(R, a, first=l)}
            lay[c] = le
        sg = lambda d: {"p": cm.sign_test_p(d), "wins": int(np.sum(np.array(d) > 0)), "losses": int(np.sum(np.array(d) < 0)),
                        "ties": int(np.sum(np.array(d) == 0))}
        abl["mults"][str(m)] = {"cells": cells, "sign_scalar": sg(dS), "sign_formula": sg(dH)}
        layers["mults"][str(m)] = lay
    exc = {}
    for c in cells_I:
        R = recs[c]
        h = abl["mults"]["1.1"]["cells"][c]["h_star"]
        L_ = [_excess(r["arms"][P]["trace"], r["M_R"], 3) for r in R]
        S_ = [_excess(r["arms"]["scalar_tied"]["trace"], r["M_R"], 3) for r in R]
        F_ = [_excess(r["arms"][f"{h}_tied"]["trace"], r["M_R"], 3) for r in R]
        gs = cm.boot_gm(np.array(S_) / np.array(L_)); gh = cm.boot_gm(np.array(F_) / np.array(L_))
        exc[c] = {"h_star": h, "scalar_over_learned": gs[0], "scalar_ci": gs[1:], "formula_over_learned": gh[0],
                  "formula_ci": gh[1:], "learned_at_floor": int(sum(x <= 0.01 for x in L_)),
                  "learned_hits_le3_1p1": int(sum(v(x) <= 3 for x in H(R, P, I11)))}
    abl["excess3"] = exc
    abl["zero_rejects"] = {c: {a: rej_stats(recs[c], a) for a in ("sec0_tied", "ait0_tied", "mu0_tied")} for c in cells_I}
    logs = {}
    for c in cells_I:
        logs[c] = {}
        for arm in ("learned", "scalar", "ait_t", "mu_t"):
            f = os.path.join(OUT, "models", f"{c}_{arm}_log.json")
            if os.path.exists(f):
                lg = cm.load(f)
                logs[c][arm] = [{k: e.get(k) for k in ("layer", "ratio", "val_em", "val_best", "sec", "theta", "gamma", "c",
                                                        "reject_val")} for e in lg]
    layers["val_loss"] = {c: [e["ratio"] for e in logs[c].get("learned", [])] for c in cells_I}
    abl["train_logs"] = logs
    return abl, layers


def part_B():
    recs = _load_dir("B")
    out = {}
    for s in cm.SOURCES_B:
        for cond in cm.SHIFTS:
            R = [r for r in recs if r["cell"]["source"] == s and r["cell"].get("shift") == cond]
            if not R:
                continue
            e = {"n": len(R), "scale_rel_err_median_abs": float(np.median([abs(r["s2_hat"] / r["sigbar2_true"] - 1) for r in R]))}
            for m in (1.1, 1.01):
                i = M.index(m)
                o, st = o_star(R, i, CLS_BC)
                cls_med = min(cm.med(H(R, a, i)) for a in CLS_BC)
                e[str(m)] = {"G": st["G"], "ci": st["ci"], "o_star": o, "cls_median": cls_med, "no_room": cls_med <= 5,
                             "prop_median": cm.med(H(R, P, i)), "n_fail_prop": sum(x is None for x in H(R, P, i)),
                             "n_fail_cls_best": min(sum(x is None for x in H(R, a, i)) for a in CLS_BC)}
            out[f"{s}/{cond}"] = e
    conds = list(cm.SHIFTS)
    rows = []
    for k, cond in enumerate(conds):
        row = [len(conds) - 1 - k, cond]
        for s in cm.SOURCES_B:
            e = out.get(f"{s}/{cond}")
            row += [round(e["1.1"]["G"], 4), round(e["1.01"]["G"], 4), int(e["1.1"]["no_room"])] if e else ["nan", "nan", 0]
        rows.append(row)
    write("shift.csv", ["y", "cond", "g64a", "g64b", "nh64", "g128a", "g128b", "nh128"], rows)
    return out


def part_big(sub):
    R = _load_dir(sub)
    if not R:
        return None
    e = {"n": len(R)}
    for m in (1.1, 1.01):
        i = M.index(m)
        o, st = o_star(R, i, CLS_BC)
        e[str(m)] = {"G": st["G"], "ci": st["ci"], "o_star": o, "prop_median": cm.med(H(R, P, i)),
                     "median": {a: cm.med(H(R, a, i)) for a in R[0]["arms"]}}
    return e


def part_GE():
    R = _load_dir("ge")
    if not R:
        return None
    out = {"n": len(R), "q_R": {k: float(np.median([r["q_R"][k] for r in R])) for k in ("delta", "f1")},
           "lambda_c": cm.load(os.path.join(OUT, "lambda.json"))["GE16_1000"]["c"],
           "daarem": cm.load(os.path.join(OUT, "tune_choice.json"))["GE16_1000"]["config"], "mults": {}}
    for m in (1.1, 1.01):
        i = M.index(m)
        o, st = o_star(R, i, CLS_A)
        out["mults"][str(m)] = {"G": st["G"], "ci": st["ci"], "o_star": o,
                                "median": {a: cm.med(H(R, a, i)) for a in R[0]["arms"]},
                                "delta": {a: qmed(R, a, i, "delta") for a in R[0]["arms"]},
                                "f1": {a: qmed(R, a, i, "f1") for a in R[0]["arms"]}}
    return out


def part_timing(II, C):
    T = _load_dir("timing")
    if not T:
        return None
    out = {}
    for m in ("1.1", "1.01"):
        for n in sorted({t["n"] for t in T}):
            tn = [t for t in T if t["n"] == n]
            best = None
            for o in CLS_BC:
                rT, rE, sav = [], [], []
                for t in tn:
                    Lp, Cc = t["arms"][P][m], t["arms"][o][m]
                    if Lp["h"] is None or Cc["h"] is None:
                        continue
                    rT.append(1.0 if (Lp["h"] == 0 or Cc["h"] == 0) else Cc["t"] / Lp["t"])
                    rE.append(1.0 if (Lp["h"] == 0 or Cc["h"] == 0) else Cc["h"] / Lp["h"])
                    sav.append(Cc["t"] - Lp["t"])
                if not rT:
                    continue
                g, lo, hi = cm.boot_gm(rT)
                gE = float(np.exp(np.mean(np.log(rE))))
                if best is None or g < best["G_T"]:
                    best = {"o_star": o, "G_T": g, "ci": (lo, hi), "G_E_same": gE, "eff": g / gE, "n": len(rT),
                            "mean_saving_s": float(np.mean(sav)), "mean_t_prop": float(np.mean([t["arms"][P][m]["t"] for t in tn
                                                                                                if t["arms"][P][m]["t"] is not None]))}
            out[f"{m}/{n}"] = best
    rows = []
    cells = {32: "F32_500", 64: "F64_500", 128: "F128_500"}
    for n in sorted({t["n"] for t in T}):
        e = II["cells"][cells[n]]["mults"]["1.1"] if n in cells else (C or {}).get("1.1")
        tm = out.get(f"1.1/{n}")
        if e and tm:
            rows.append([n, e["G"], e["G"] - e["ci"][0], e["ci"][1] - e["G"], tm["G_T"], tm["G_T"] - tm["ci"][0], tm["ci"][1] - tm["G_T"]])
    write("scaling.csv", ["n", "gE", "gElo", "gEhi", "gT", "gTlo", "gThi"], rows)
    pc = os.path.join(OUT, "percall.json")
    if os.path.exists(pc):
        d = cm.load(pc)
        write("percall.csv", ["n", "exact", "ss", "mstep", "layer"],
              [[n, d[n]["estep_exact"] * 1e3, d[n]["estep_ss"] * 1e3, d[n]["mstep"] * 1e3, d[n]["layer_extra"] * 1e3]
               for n in sorted(d, key=int)])
        out["percall"] = d
        out["layer_overhead_frac"] = {n: d[n]["layer_extra"] / (d[n]["estep_ss"] + d[n]["mstep"]) for n in d}
    return out


def part_diag():
    from scipy.stats import spearmanr
    fs = sorted(glob.glob(os.path.join(OUT, "diag", "*.npz")))
    if not fs:
        return None
    mv = cm.load(os.path.join(OUT, "mu_choice.json"))["choice"] if os.path.exists(os.path.join(OUT, "mu_choice.json")) else "or"
    out, pts_rate, pts_step = {}, {}, {}
    pts_ideal = {}
    for f in fs:
        c = re.match(r"(.+)_\d\d\.npz", os.path.basename(f)).group(1)
        z = np.load(f)
        err, A_R, mu = z["err"].astype(float), z["A_R"], z["muR"]
        dist = np.abs(err)
        n = A_R.shape[0]
        mub = 1 - np.outer(1 - mu, 1 - mu)
        lo, hi = 10, min(60, dist.shape[0] - 1)
        rates = np.full((n, n), np.nan)
        for i_ in range(n):
            for j_ in range(n):
                d = dist[lo:hi + 1, i_, j_]
                ok = d > 1e-10
                if ok.sum() >= 10:
                    r = d[1:][ok[1:] & ok[:-1]] / d[:-1][ok[1:] & ok[:-1]]
                    if len(r) >= 5:
                        rates[i_, j_] = float(np.median(r))
        sel = (np.abs(A_R) > 0) & np.isfinite(rates)
        pts_rate.setdefault(c, []).append(np.c_[mub[sel], rates[sel], np.eye(n)[sel]])
        for k in range(3):
            step = err[k + 1] - err[k]
            big = np.abs(step) > 1e-3 * np.sqrt(np.mean(step ** 2))
            s_star = np.where(big, -err[k] / np.where(big, step, 1.0), np.nan)
            okk = big & np.isfinite(s_star) & (np.abs(A_R) > 0)
            pts_ideal.setdefault((c, k), []).append(np.c_[mub[okk], s_star[okk], np.eye(n)[okk]])
        S, F = z["S"], z["F"].astype(float)
        for l in range(S.shape[0]):
            pts_step.setdefault((c, l), []).append(np.c_[mub.ravel(), S[l].ravel(), np.eye(n).ravel(), F[l][..., 0].ravel(),
                                                        F[l][..., 1].ravel()])
        step0 = err[1] - err[0]
        big0 = (np.abs(step0) > 1e-3 * np.sqrt(np.mean(step0 ** 2))) & (np.abs(A_R) > 0)
        s0 = -err[0][big0] / step0[big0]
        pts_ideal.setdefault((c, "learned1"), []).append(np.c_[S[0][big0], s0])
    for (c, k), L in pts_ideal.items():
        X = np.vstack(L)
        e = out.setdefault(c, {}).setdefault("ideal", {})
        if k == "learned1":
            e["spearman_learnedS1_vs_ideal0"] = spearmanr(X[:, 0], X[:, 1]).correlation if len(X) > 3 else None
            rng = np.random.default_rng(2)
            idx = rng.permutation(len(X))[:2000]
            write(f"ideal_vs_learned_{cm.SHORT.get(c, c)}.csv", ["S", "sstar"], [[f"{a:.5f}", f"{b:.5f}"] for a, b in X[idx]])
            continue
        e[f"iter{k}"] = {"spearman_sstar_mubar": spearmanr(X[:, 0], X[:, 1]).correlation if len(X) > 3 else None,
                         "median_sstar_diag": float(np.median(X[X[:, 2] == 1, 1])) if np.any(X[:, 2] == 1) else None,
                         "median_sstar_off": float(np.median(X[X[:, 2] == 0, 1])) if np.any(X[:, 2] == 0) else None,
                         "iqr_sstar": [float(np.percentile(X[:, 1], 25)), float(np.percentile(X[:, 1], 75))]}
    for c, L in pts_rate.items():
        X = np.vstack(L)
        rho = spearmanr(X[:, 0], X[:, 1]).correlation if len(X) > 3 else None
        out.setdefault(c, {})["rate_vs_mubar"] = {"spearman": rho, "n_entries": int(len(X)),
                                                  "rate_median_diag": float(np.median(X[X[:, 2] == 1, 1])) if np.any(X[:, 2] == 1) else None,
                                                  "rate_median_off": float(np.median(X[X[:, 2] == 0, 1])) if np.any(X[:, 2] == 0) else None}
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))[:2000]
        write(f"motiv_{cm.SHORT.get(c, c)}.csv", ["mubar", "rate", "diag"], [[f"{a:.5f}", f"{b:.5f}", int(d)] for a, b, d in X[idx]])
    for (c, l), Lst in pts_step.items():
        X = np.vstack(Lst)
        e = {"spearman_S_mubar": spearmanr(X[:, 0], X[:, 1]).correlation,
             "diag_over_off": float(np.mean(X[X[:, 2] == 1, 1]) / np.mean(X[X[:, 2] == 0, 1]))}
        ok = (X[:, 4] != 0) & np.isfinite(X[:, 3] / np.where(X[:, 4] == 0, 1, X[:, 4]))
        if ok.sum() > 10:
            ratio = X[ok, 3] / X[ok, 4]
            ait = np.where(ratio < 1, 1 / (1 - np.clip(ratio, -1e6, 0.999999)), 1.0)
            e["spearman_S_aitken"] = spearmanr(ait, X[ok, 1]).correlation
        out.setdefault(c, {})[f"layer{l + 1}"] = e
        if l in (0, cm.L_MAIN - 1):
            rng = np.random.default_rng(1)
            idx = rng.permutation(len(X))[:2000]
            write(f"steps_L{l + 1}_{cm.SHORT.get(c, c)}.csv", ["mubar", "S", "diag"],
                  [[f"{a:.5f}", f"{b:.5f}", int(d)] for a, b, d in X[idx, :3]])
    out["mu_variant"] = mv
    return out


def part_cost():
    out = {}
    for c in cm.CELLS + ["GE16_1000"]:
        e = {}
        f = os.path.join(OUT, "models", f"{c}_learned_log.json")
        if os.path.exists(f):
            lg = cm.load(f)
            e["train_sec_L4"] = float(sum(x["sec"] for x in lg[:cm.L_MAIN]))
            e["train_sec_all"] = float(sum(x["sec"] for x in lg))
        ref = [np.load(p) for p in glob.glob(os.path.join(OUT, "data", f"{c}_*.npz"))]
        if ref:
            e["ref_sec_total"] = float(sum(float(z["sec"]) for z in ref))
            e["ref_nE_median"] = float(np.median([int(z["nE"]) for z in ref]))
            e["n_trva"] = len(ref)
        out[c] = e
    out["lambda"] = cm.load(os.path.join(OUT, "lambda.json"))
    out["daarem"] = {c: v_["config"] for c, v_ in cm.load(os.path.join(OUT, "tune_choice.json")).items()}
    return out


def main():
    out = {"II": part_II()}
    r = part_I()
    if r:
        out["abl"], out["layers"] = r
        json.dump(r[0], open(os.path.join(PD, "ablation.json"), "w"), indent=1, ensure_ascii=False, default=float)
        json.dump(r[1], open(os.path.join(PD, "layers.json"), "w"), indent=1, ensure_ascii=False, default=float)
    out["B"] = part_B()
    out["C"] = part_big("C")
    out["D"] = part_big("D")
    tr = out["II"]["transfer"]
    if out["C"]:
        tr["128to256"] = {m: out["C"][m] for m in ("1.1", "1.01")}
    if all(k in tr for k in ("32to64", "64to128", "128to256")):
        write("transfer.csv", ["x", "G", "lo", "hi"],
              [[k, tr[key]["1.1"]["G"], tr[key]["1.1"]["G"] - tr[key]["1.1"]["ci"][0], tr[key]["1.1"]["ci"][1] - tr[key]["1.1"]["G"]]
               for k, key in enumerate(("32to64", "64to128", "128to256"))])
    out["GE"] = part_GE()
    out["timing"] = part_timing(out["II"], out["C"])
    out["diag"] = part_diag()
    out["cost"] = part_cost()
    cm.dump(out, os.path.join(OUT, "verdict.json"))
    II = out["II"]["cells"]
    for m in ("1.1", "1.01"):
        print(f"测试集 II α={m}: " + "  ".join(f"{cm.SHORT[c]} G {II[c]['mults'][m]['G']:.2f}"
                                               f"[{II[c]['mults'][m]['ci'][0]:.2f},{II[c]['mults'][m]['ci'][1]:.2f}] "
                                               f"med {II[c]['mults'][m]['median'][P]:g} o*{II[c]['mults'][m]['o_star'][:6]}"
                                               for c in II))
    print("A7 dist1: " + "  ".join(f"{cm.SHORT[c]} G {II[c]['A7']['dist1']['G']:.2f} med {II[c]['A7']['dist1']['median'][P]:g}"
                                   f" fail {II[c]['A7']['dist1']['n_fail'][P]}" for c in II))
    print("单调性：" + "  ".join(f"{cm.SHORT[c]} {II[c]['mono']['increases']}/{II[c]['mono']['steps']}" for c in II))
    print("η 生效（只因 ηΔ 被拒）：" + "  ".join(f"{cm.SHORT[c]} {II[c]['rejects']['all']['eta']}" for c in II))
    print("尺度估计相对误差中位：" + "  ".join(f"{cm.SHORT[c]} {II[c]['scale_rel_err']['median_abs']:.3f}" for c in II))
    print("写出：", sorted(os.listdir(PD)))


if __name__ == "__main__":
    main()
