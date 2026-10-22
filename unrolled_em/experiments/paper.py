from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from unrolled_em.experiments import common as cm

RES = os.path.join(cm.HERE, "results")
V2 = os.path.join(RES, "v2")
V2C = os.path.join(RES, "v2c")
SUPP = os.path.join(RES, "v2_supp")
RB = os.path.join(RES, "review_b")
G3 = os.path.join(V2, "g3")
OUT = os.path.join(cm.HERE, "paper")
P = "learned_tied_L4"
M = cm.MULTS
I11, I101 = M.index(1.1), M.index(1.01)
CELLS = cm.CELLS
SH = cm.SHORT
POOL_ALL = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun", "SQbt1_ss", "SQbt2_ss", "SQbt3_ss", "SQbt3_exact",
            "SQpkg_ss", "SQpkg16_ss", "QN1_ss", "QN3_ss"]
POOL_SS = ["EM_ss_K50", "SQ_ss_K50", "DAAREM_tun", "SQbt1_ss", "SQbt2_ss", "SQbt3_ss", "SQpkg_ss", "SQpkg16_ss",
           "QN1_ss", "QN3_ss"]
PARAM = (("dist5", 1, 0.05), ("dist1", 1, 0.01), ("gap1e-4", 3, 1e-4), ("gap1e-5", 3, 1e-5))
NUM = {}
LINES = []


def J(*p):
    return cm.load(os.path.join(*p))


def say(s=""):
    LINES.append(s)
    print(s, flush=True)


def put(key, val):
    NUM[key] = val


def f2(x):
    return f"{x:.2f}"


def ci(st):
    return f"{st['G']:.2f} [{st['ci'][0]:.2f}, {st['ci'][1]:.2f}]"


def rng(vals, fmt="{:.2f}"):
    vals = [v for v in vals if v is not None]
    return (fmt + " to " + fmt).format(min(vals), max(vals))


def med(xs):
    return cm.med(xs)


def xh(arm, col, thr):
    ex = arm.get("extra") or []
    h = next((e[0] for e in ex if e[col] <= thr), None)
    return h if (h is not None and h <= cm.MAX_E) else None


def strongest(H, me, pool):
    st = {a: cm.ratio_stats(H[a], H[me]) for a in pool}
    o = min(pool, key=lambda a: st[a]["G"])
    return o, st[o], st


_TII = {}


def test_ii(c):
    if c in _TII:
        return _TII[c]
    rows = []
    for j in range(cm.N_TEST_II):
        fn = f"{c}_{j:02d}.json"
        base = J(V2, "testII", fn)
        arms = dict(base["arms"])
        arms.update({k: v for k, v in J(V2, "sqbt", "testII", fn)["arms"].items() if k != "SQ_ss_K50"})
        arms.update(J(RB, "sq16", fn)["arms"])
        arms.update(J(RB, "qn", fn)["arms"])
        arms["v2_" + P] = arms[P]
        arms[P] = J(V2C, "testII", fn)["arms"][P]
        rows.append({"meta": base, "arms": arms, "j": j})
    _TII[c] = rows
    return rows


def hits(rows, arm, i):
    return [r["arms"][arm]["hits"][i] for r in rows]


def phits(rows, arm, col, thr):
    return [xh(r["arms"][arm], col, thr) for r in rows]


def part_main():
    say("== Table III: E-step ratio to the strongest classical method, test set II ==")
    T = {}
    miss = {a: 0 for a in (P, "SQpkg_ss", "DAAREM_tun")}
    for c in CELLS:
        R = test_ii(c)
        row = {}
        for m, i in ((1.1, I11), (1.01, I101)):
            H = {a: hits(R, a, i) for a in POOL_ALL + [P]}
            o, st, allst = strongest(H, P, POOL_ALL)
            row[str(m)] = {"G": st["G"], "ci": st["ci"], "strongest": o,
                           "median": {a: med(H[a]) for a in (P, "EM_ss_K50", "SQpkg_ss", "DAAREM_tun")},
                           "G_vs_SQpkg_ss": allst["SQpkg_ss"]["G"]}
            if m == 1.01:
                for a in miss:
                    miss[a] += sum(x is None for x in H[a])
        T[c] = row
        say(f"{SH[c]:5s} " + "  |  ".join(
            f"{ci(row[m])} ({row[m]['strongest']})  " + " ".join(f"{row[m]['median'][a]:g}" for a in (P, "EM_ss_K50", "SQpkg_ss", "DAAREM_tun"))
            for m in ("1.1", "1.01")) + "   [ratio, proposed, EM, SQUAREM, DAAREM]")
    put("table3", T)
    put("missed_alpha101", miss)
    say(f"missed alpha = 1.01 within 100 E-steps (150 systems): proposed {miss[P]}, SQUAREM {miss['SQpkg_ss']}, DAAREM {miss['DAAREM_tun']}")
    full = [T[c][m]["G"] for c in CELLS[:3] for m in ("1.1",)]
    full101 = [T[c]["1.01"]["G"] for c in CELLS[:3]]
    part = [T[c][m]["G"] for c in CELLS[3:] for m in ("1.1", "1.01")]
    say(f"full observation: {rng(full)} at alpha = 1.1, {rng(full101)} at alpha = 1.01; partial observation {rng(part)}")
    worst = max(T[c][m]["G_vs_SQpkg_ss"] / T[c][m]["G"] for c in CELLS for m in ("1.1", "1.01"))
    say(f"with SQUAREM (reference implementation, default bound) as the only competitor the ratios are at most {100 * (worst - 1):.0f}% larger")
    say()

    say("== Table VI: median E-steps of the classical variants at alpha = 1.1 / 1.01 ==")
    names = [("Fallback", "SQ_ss_K50"), ("Fallback (exact)", "SQ_exact_K50"), ("SqS1", "SQbt1_ss"), ("SqS2", "SQbt2_ss"),
             ("SqS3", "SQbt3_ss"), ("SqS3 (exact)", "SQbt3_exact"), ("Reference", "SQpkg_ss"), ("Reference gamma0=16", "SQpkg16_ss"),
             ("QN, 1 pair", "QN1_ss"), ("QN, 3 pairs", "QN3_ss"), ("Proposed", P)]
    tab = {}
    for lab, a in names:
        tab[lab] = {SH[c]: [med(hits(test_ii(c), a, I11)), med(hits(test_ii(c), a, I101))] for c in CELLS}
        say(f"{lab:22s} " + "  ".join(f"{v[0]:g} / {v[1]:g}" for v in tab[lab].values()))
    put("table6", tab)
    qn = [med(phits(test_ii(c), q, 1, 0.01)) for c in CELLS for q in ("QN1_ss", "QN3_ss")]
    rf = [med(phits(test_ii(c), "SQpkg_ss", 1, 0.01)) for c in CELLS]
    say(f"quasi-Newton EM, median E-steps to 1% distance: {rng(qn, '{:g}')}; reference SQUAREM {rng(rf, '{:g}')}")
    say()

    say("== Table VII: relative error delta and support F1 when the target is first met (proposed / strongest) ==")
    q7 = {}
    better_delta = lower_f1 = 0
    f1_gap = 0.0
    for c in CELLS:
        R = test_ii(c)
        e = {}
        for m, i in (("1.1", I11), ("1.01", I101)):
            o = T[c][m]["strongest"]
            qd = lambda a, k: float(np.median([r["arms"][a]["q"][i][k] for r in R if r["arms"][a]["q"][i] is not None]))
            e[m] = {"delta": (qd(P, "delta"), qd(o, "delta")), "f1": (qd(P, "f1"), qd(o, "f1"))}
            better_delta += e[m]["delta"][0] < e[m]["delta"][1]
            if e[m]["f1"][0] < e[m]["f1"][1]:
                lower_f1 += 1
                f1_gap = max(f1_gap, e[m]["f1"][1] - e[m]["f1"][0])
        q7[c] = e
        say(f"{SH[c]:5s} delta {e['1.1']['delta'][0]:.3f} / {e['1.1']['delta'][1]:.3f}  {e['1.01']['delta'][0]:.3f} / {e['1.01']['delta'][1]:.3f}"
            f"   F1 {e['1.1']['f1'][0]:.3f} / {e['1.1']['f1'][1]:.3f}  {e['1.01']['f1'][0]:.3f} / {e['1.01']['f1'][1]:.3f}")
    put("table7", q7)
    say(f"relative error smaller than the strongest competitor in {better_delta} of 10 cases; F1 lower in {lower_f1}, by at most {f1_gap:.3f}")
    qr = {c: (np.median([r["meta"]["q_R"]["delta"] for r in test_ii(c)]), np.median([r["meta"]["q_R"]["f1"] for r in test_ii(c)])) for c in CELLS}
    say("reference solutions: relative error " + ", ".join(f"{SH[c]} {v[0]:.2f}" for c, v in qr.items())
        + "; F1 " + ", ".join(f"{SH[c]} {v[1]:.2f}" for c, v in qr.items()))
    say()


def part_param():
    say("== Table IV: E-step ratios in parameter space, test set II ==")
    T = {}
    for c in CELLS:
        R = test_ii(c)
        row = {}
        for name, col, thr in PARAM:
            H = {a: phits(R, a, col, thr) for a in POOL_SS + [P, "v2_" + P]}
            o, st, allst = strongest(H, P, POOL_SS)
            o2, st2, _ = strongest(H, "v2_" + P, POOL_SS)
            row[name] = {"G": st["G"], "ci": st["ci"], "strongest": o, "fail": sum(x is None for x in H[P]),
                         "median": {a: med(H[a]) for a in (P, "EM_ss_K50", "SQpkg_ss")},
                         "G_without_handover": st2["G"],
                         "handover_factor": float(np.exp(np.mean(np.log([cm.FAIL if x is None else x for x in H["v2_" + P]])
                                                                 - np.log([cm.FAIL if x is None else x for x in H[P]]))))}
        T[c] = row
        say(f"{SH[c]:5s} " + "  ".join(f"{k} {ci(v)} ({v['strongest']})" for k, v in row.items()))
    put("table4", T)
    say("missed the 1% distance within 100 E-steps: " + ", ".join(f"{SH[c]} {T[c]['dist1']['fail']}" for c in CELLS))
    say("without the handover (layers of Section III, fourth layer reused): ratios at 1% distance "
        + ", ".join(f"{SH[c]} {T[c]['dist1']['G_without_handover']:.2f}" for c in CELLS))
    say("reduction of the E-steps to 1% distance by the handover (geometric mean): "
        + ", ".join(f"{SH[c]} {T[c]['dist1']['handover_factor']:.2f}" for c in CELLS))
    lower = []
    for c in CELLS:
        for r in test_ii(c):
            ex = r["arms"][P].get("extra") or []
            if ex and min(e[3] for e in ex) < -1e-6 and xh(r["arms"][P], 1, 0.01) is None:
                lower.append((c, r["j"], ex[-1][1]))
    say(f"systems on which the proposed method reached a lower objective than the reference and missed 1% distance: {len(lower)}"
        f" ({sum(c == 'F128_500' for c, _, _ in lower)} in F128), closest distance {rng([d for *_, d in lower], '{:.3f}')}")
    put("lower_objective", lower)
    say()


def part_setup():
    say("== Section IV-A and II-B: reference solutions, scale estimate, steady state fixed point ==")
    allr = [r["meta"] for c in CELLS for r in test_ii(c)]
    capped = [x for x in allr if x["nE_R"] >= cm.REF_MAX_E]
    stable = [x for x in capped if x.get("M_R_m50") is not None and abs(x["M_R"] - x["M_R_m50"]) / abs(x["M_R"]) < 0.01]
    say(f"reference solutions at the cap of {cm.REF_MAX_E} E-steps: {len(capped)} of {len(allr)}; predictive error changed by less than 1% over the last 50 E-steps on {len(stable)}")
    sc = {c: float(np.median([abs(r["meta"]["s2_hat"] / r["meta"]["sigbar2_true"] - 1) for r in test_ii(c)])) for c in CELLS}
    say("scale estimate, median relative error: " + ", ".join(f"{SH[c]} {100 * v:.1f}%" for c, v in sc.items()))
    dd = {c: float(np.median([r["meta"]["dist_Rss_R"] for r in test_ii(c)])) for c in CELLS}
    say("steady state versus exact fixed point, median relative distance: " + ", ".join(f"{SH[c]} {100 * v:.1f}%" for c, v in dd.items()))
    ex = []
    for c in CELLS:
        for r in test_ii(c):
            t = r["meta"]["taus"]
            ex.append((t[I11] - cm.TAU_ABS) / 1.1)
    lj = J(V2, "lambda.json")
    say("selected constant c: " + ", ".join(f"{SH.get(k, k)} {v['c'] if isinstance(v, dict) else v}" for k, v in lj.items()))
    say(f"excess error of the reference solution E(A_R): median {np.median(ex):.3f}, minimum {np.min(ex):.3f}; "
        f"the additive term raises alpha by at most {cm.TAU_ABS / np.min(ex):.3f}")
    put("setup", {"capped": len(capped), "capped_stable": len(stable), "scale": sc, "dist_ss": dd,
                  "excess_median": float(np.median(ex)), "excess_min": float(np.min(ex))})
    say()


def write_csv(name, header, rows):
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with open(os.path.join(OUT, "data", name), "w", encoding="utf-8") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join("nan" if v is None else str(v) for v in r) + "\n")


def part_fig_tol():
    R = test_ii("F128_500")
    arms = [(P, "prop"), ("EM_ss_K50", "em"), ("SQpkg_ss", "sq"), ("DAAREM_tun", "da")]
    write_csv("tol_F128.csv", ["i", "alpha"] + [t for _, t in arms],
              [[i, m] + [med(hits(R, a, i)) for a, _ in arms] for i, m in enumerate(M)])
    say("wrote data/tol_F128.csv (Fig. 3)")


def trace_system():
    R = test_ii("F128_500")
    arms = [P, "SQ_ss_K50", "DAAREM_tun"]
    v = lambda x: cm.FAIL if x is None else x
    md = {a: (med(hits(R, a, I11)), med(hits(R, a, I101))) for a in arms}
    d = sorted((sum(abs(v(r["arms"][a]["hits"][I11]) - md[a][0]) / md[a][0] + abs(v(r["arms"][a]["hits"][I101]) - md[a][1]) / md[a][1]
                    for a in arms), r["j"]) for r in R)
    return d[0][1]


class StampedLog(list):

    def __init__(self, rec):
        super().__init__()
        self.rec, self.stamps = rec, []

    def append(self, x):
        self.stamps.append(self.rec.trace[-1][0] if self.rec.trace else 0)
        super().append(x)


def part_fig_trace():
    from unrolled_em import anytime as at
    from unrolled_em.experiments import exp_hand as EH
    from unrolled_em.experiments import exp_review as X
    from unrolled_em.experiments import run as RN
    name, j = "F128_500", trace_system()
    old, sys, y, yv, lam, A0, taus, refs = X._test_ii(name, j)
    c = J(V2C, "choice.json")
    models = RN.load_models(name)
    models["learned"] = models["learned"][:cm.L_MAIN]
    out = {}
    for arm, tag in ((P, "prop"), ("EM_ss_K50", "em"), ("SQpkg_ss", "sq"), ("DAAREM_tun", "da")):
        rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus, rich=X._rich(y, lam, refs, "ss"), stop_dist=-1.0, stop_gap=-np.inf)
        if arm == P:
            EH.VARIANTS["_cur"] = {"r": int(c["r"]), "amax0": float(c["amax0"]), "safe": True}
            log = StampedLog(rec)
            fn_run = lambda: EH.run_hand(models["learned"], sys, y, lam, A0, "_cur", log=log)
        elif arm == "SQpkg_ss":
            fn_run = X._new_arm(arm, sys, y, lam, A0)
        else:
            fn_run = RN.algo_for(arm, sys, y, lam, A0, models, [])
        at.run(fn_run, sys, lam, 50, rec, A0)
        ex = [e for e in rec.extra if e[0] <= 60]
        write_csv(f"distR_{tag}.csv", ["e", "d", "g"], [[e[0], e[1], e[3] if e[3] > 0 else None] for e in ex])
        h = cm.hits_from_trace(rec.trace, taus)
        e01 = next((e[0] for e in rec.extra if e[1] <= 1e-3), None)
        six = next((e[1] for e in rec.extra if e[0] == 6), None)
        out[tag] = {"alpha": (h[I11], h[I101]), "dist_after_6": six, "E_to_0.1pct": e01}
        if arm == P:
            out[tag]["handover_at_E"] = next((e for x, e in zip(log, log.stamps) if x[0] == "handover"), None)
    put("fig2", {"system": j, **out})
    say(f"== Fig. 2: F128 test system {j} ==")
    for t, o in out.items():
        say(f"{t}: alpha 1.1 / 1.01 after {o['alpha'][0]} / {o['alpha'][1]} E-steps; distance after 6 E-steps {100 * o['dist_after_6']:.1f}%; "
            f"within 0.1% after {o['E_to_0.1pct']} E-steps" + (f"; handed over to SQUAREM after E-step {o.get('handover_at_E')}" if t == "prop" else ""))
    say("wrote data/distR_*.csv (Fig. 2)")
    say()

FORM = ["sec0", "ait0", "mu0", "ait_t", "mu_t"]
POOL_TR = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun", "SQbt3_ss", "SQbt1_ss", "SQbt2_ss", "SQpkg_ss", "SQbt3_exact"]
POOL_I = ["SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun", "SQbt1_ss", "SQbt2_ss", "SQbt3_ss", "SQbt3_exact", "SQpkg_ss"]
_TI = {}


def test_i(c):
    if c in _TI:
        return _TI[c]
    rows = []
    for j in range(cm.N_TEST_I):
        fn = f"{c}_{j:02d}.json"
        base = J(V2, "testI", fn)
        arms = dict(base["arms"])
        arms.update(J(V2C, "testI", fn)["arms"])
        rows.append({"meta": base, "arms": arms, "j": j})
    _TI[c] = rows
    return rows


def wins(a, b):
    v = lambda x: cm.FAIL if x is None else x
    d = [v(x) - v(y) for x, y in zip(a, b)]
    return d, int(sum(x > 0 for x in d))


def part_ablation():
    say("== Table V: ablation on test set I (E-step ratio of the variant to the learned extrapolation; xi after three E-steps) ==")
    ver = J(V2, "verdict.json")["abl"]["excess3"]
    T = {}
    dS, dF, dC, dL = {"1.1": [], "1.01": []}, {"1.1": [], "1.01": []}, {"1.1": [], "1.01": []}, {"1.1": [], "1.01": []}
    lo = []
    cls_g, lin_g, lo_cl = [], [], []
    for c in CELLS:
        R = test_i(c)
        row = {}
        for m, i in (("1.1", I11), ("1.01", I101)):
            H = lambda a: hits(R, a, i)
            fst = {h: cm.ratio_stats(H(f"{h}_tied"), H(P)) for h in FORM}
            hs = min(sorted(FORM), key=lambda h: fst[h]["G"])
            sc = cm.ratio_stats(H("scalar_tied"), H(P))
            cl = cm.ratio_stats(H("class"), H(P))
            ln = cm.ratio_stats(H("linear"), H(P))
            row[m] = {"best_formula": hs, "formula": fst[hs], "constants": sc, "class": cl, "linear": ln}
            dF[m] += wins(H(f"{hs}_tied"), H(P))[0]
            dS[m] += wins(H("scalar_tied"), H(P))[0]
            dC[m] += wins(H("class"), H(P))[0]
            dL[m] += wins(H("linear"), H(P))[0]
            lo += [fst[hs]["ci"][0], sc["ci"][0]]
            cls_g.append(cl["G"])
            lin_g.append(ln["G"])
            lo_cl += [cl["ci"][0], ln["ci"][0]]
        x = ver[c]
        row["xi"] = {"formula": x["formula_over_learned"], "formula_ci": x["formula_ci"], "formula_h": x["h_star"],
                     "constants": x["scalar_over_learned"], "constants_ci": x["scalar_ci"],
                     "met_alpha11_within_3": x["learned_hits_le3_1p1"]}
        T[c] = row
        say(f"{SH[c]:5s} best formula {row['1.1']['formula']['G']:.2f} {row['1.01']['formula']['G']:.2f} xi {x['formula_over_learned']:.2f}"
            f" | tuned constants {row['1.1']['constants']['G']:.2f} {row['1.01']['constants']['G']:.2f} xi {x['scalar_over_learned']:.2f}"
            f"   (best formula {row['1.1']['best_formula']} / {row['1.01']['best_formula']})")
    put("table5", T)
    cnt = lambda d: int(sum(x > 0 for x in d))
    say(f"learned faster than tuned constants on {cnt(dS['1.1'])} and {cnt(dS['1.01'])} of 150 systems, than the best formula on "
        f"{cnt(dF['1.1'])} and {cnt(dF['1.01'])} (sign test p = {max(cm.sign_test_p(d) for d in list(dS.values()) + list(dF.values())):.1e} at most)")
    say(f"lower 95% bounds of the E-step ratios in Table V: at least {min(lo):.2f}; of xi: "
        + ", ".join(f"{SH[c]} {T[c]['xi']['formula_ci'][0]:.2f}/{T[c]['xi']['constants_ci'][0]:.2f}" for c in CELLS))
    say(f"F64: target alpha = 1.1 met within three E-steps on {T['F64_500']['xi']['met_alpha11_within_3']} of 30 systems")
    say(f"one pair of constants per class: {rng(cls_g)} times as many E-steps, slower on {cnt(dC['1.1'])} and {cnt(dC['1.01'])} of 150 "
        f"(sign test p = {max(cm.sign_test_p(dC['1.1']), cm.sign_test_p(dC['1.01'])):.1e} at most)")
    say(f"linear map of the features: {rng(lin_g)} times as many, slower on {cnt(dL['1.1'])} and {cnt(dL['1.01'])}; "
        f"lower 95% bounds at least {min(lo_cl):.2f}")
    say()

    say("== Table X: E-step ratio to the strongest classical method with one to six trained layers, test set I ==")
    TX = {}
    alt_dev = 0.0
    for c in CELLS:
        R = test_i(c)
        TX[c] = {}
        for l in range(1, cm.L_MAX + 1):
            e = []
            for i in (I11, I101):
                H = {a: hits(R, a, i) for a in POOL_I + [f"learned_tied_L{l}", f"learned_alt_L{l}"]}
                _, st, _ = strongest(H, f"learned_tied_L{l}", POOL_I)
                _, st2, _ = strongest(H, f"learned_alt_L{l}", POOL_I)
                e.append(st["G"])
                alt_dev = max(alt_dev, abs(st2["G"] / st["G"] - 1))
            TX[c][l] = e
    for l in range(1, cm.L_MAX + 1):
        say(f"{l} layers  " + "  ".join(f"{TX[c][l][0]:.2f} / {TX[c][l][1]:.2f}" for c in CELLS))
    put("table10", TX)
    say(f"one layer: {rng([TX[c][1][0] for c in CELLS])} at alpha = 1.1; four layers: {rng([TX[c][4][0] for c in CELLS])}")
    say(f"starting the bound at gamma_0 = 1 instead of 16 changes the entries by at most {100 * alt_dev:.1f}%")
    h1 = [r["arms"]["learned_tied_L1"]["n_layer_steps"] for c in CELLS for r in test_i(c) if r["arms"]["learned_tied_L1"]["handover_at"] is not None]
    say(f"with a single layer the handover came after a median of {np.median(h1):g} layer applications")
    vl = J(V2, "verdict.json")["layers"]["val_loss"]
    say("validation loss of layer l: " + "; ".join(f"l = {l}: {rng([vl[c][l - 1] for c in CELLS])}" for l in (1, 4, 6)))
    say()


def part_robust():
    say("== Table XI: E-step ratios under distribution shift (F64 and F128, alpha = 1.1 / 1.01) ==")
    labels = [("T1000", "T x 2"), ("T2000", "T x 4"), ("SNRm5", "SNR -5 dB"), ("SNRp5", "SNR +5 dB"), ("k1", "2 nonzeros per row"),
              ("k4", "5 nonzeros per row"), ("rho08", "rho(A) = 0.8"), ("rho095", "rho(A) = 0.95")]
    pool = ["SQ_ss_K50", "DAAREM_tun", "SQpkg_ss"]
    T = {}
    big = []
    for sh, lab in labels:
        T[sh] = {}
        for c in ("F64_500", "F128_500"):
            rows = []
            for j in range(cm.N_B):
                fn = f"{c}_{sh}_{j:02d}.json"
                a = dict(J(V2, "B", fn)["arms"])
                a["SQpkg_ss"] = J(V2, "sqpkg_rest", "B", fn)
                a[P] = J(V2C, "B", fn)["arms"][P]
                rows.append({"arms": a, "meta": J(V2, "B", fn)})
            e = {}
            for m, i in (("1.1", I11), ("1.01", I101)):
                H = {a: hits(rows, a, i) for a in pool + [P]}
                o, st, _ = strongest(H, P, pool)
                e[m] = {"G": st["G"], "ci": st["ci"], "strongest": o, "strongest_median": med(H[o])}
            e["dagger"] = e["1.1"]["strongest_median"] <= 5
            e["rejected"] = (sum(r["arms"][P]["layer_reject"] for r in rows), sum(r["arms"][P]["n_layer_steps"] for r in rows))
            e["ref_nnz"] = float(np.median([r["meta"]["q_R"]["nnz"] for r in rows]))
            e["ref_recall"] = float(np.median([r["meta"]["q_R"]["rec"] for r in rows]))
            e["sq_median"] = (med(hits(rows, "SQpkg_ss", I11)), med(hits(rows, "SQpkg_ss", I101)))
            T[sh][c] = e
            if not e["dagger"]:
                big.append(e["1.1"]["G"])
        say(f"{lab:20s} " + "   ".join(f"{T[sh][c]['1.1']['G']:.2f}{'+' if T[sh][c]['dagger'] else ''} / {T[sh][c]['1.01']['G']:.2f}" for c in ("F64_500", "F128_500"))
            + "   (+ strongest classical method needs at most five E-steps to alpha = 1.1)")
    put("table11", T)
    al = [T[sh][c]["1.01"]["G"] for sh, _ in labels for c in ("F64_500", "F128_500")]
    say(f"alpha = 1.01: ratio at least 2.0 in {sum(x >= 2.0 for x in al)} of 16 conditions; with the strongest method above five E-steps "
        f"({len(big)} conditions) the ratio at alpha = 1.1 is at least {min(big):.2f}")
    e = T["SNRp5"]["F128_500"]
    say(f"F128, SNR +5 dB: SQUAREM median {e['sq_median'][0]:g} and {e['sq_median'][1]:g} E-steps to alpha = 1.1 and 1.01; {e['rejected'][0]} of {e['rejected'][1]} candidates of the trained layers rejected; "
        f"ratio at alpha = 1.01 {ci(e['1.01'])}")
    e1, e2 = T["SNRm5"]["F128_500"], T["SNRm5"]["F64_500"]
    say(f"SNR -5 dB: reference solutions F128 median {e1['ref_nnz']:g} nonzeros, recall {e1['ref_recall']:.2f}; F64 recall {e2['ref_recall']:.2f}, "
        f"ratio at alpha = 1.01 {e2['1.01']['G']:.2f}")
    say()

    say("== Size transfer, n = 256 and n = 512 ==")
    tr = {}
    for key, c, arm in (("32->64", "F64_500", "tied_from32_L4"), ("64->128", "F128_500", "tied_from64_L4")):
        R = test_ii(c)
        for r in R:
            r["arms"][arm] = J(V2C, "testII", f"{c}_{r['j']:02d}.json")["arms"][arm]
        tr[key] = {}
        for m, i in (("1.1", I11), ("1.01", I101)):
            H = {a: hits(R, a, i) for a in POOL_TR + [arm, P]}
            _, st, _ = strongest(H, arm, POOL_TR)
            _, stn, _ = strongest(H, P, POOL_TR)
            tr[key][m] = {"G": st["G"], "ci": st["ci"], "native": stn["G"]}
    rows = []
    for j in range(cm.N_C):
        fn = f"F256_500_{j:02d}.json"
        a = dict(J(V2, "C", fn)["arms"])
        a["SQpkg_ss"] = J(V2, "sqpkg_rest", "C", fn)
        a.update(J(V2C, "C", fn)["arms"])
        rows.append({"arms": a})
    pool_c = ["SQ_ss_K50", "DAAREM_tun", "SQpkg_ss"]
    tr["128->256"], nat = {}, {}
    for m, i in (("1.1", I11), ("1.01", I101)):
        H = {a: hits(rows, a, i) for a in pool_c + [P, "native_F256"]}
        _, st, _ = strongest(H, P, pool_c)
        _, stn, _ = strongest(H, "native_F256", pool_c)
        tr["128->256"][m] = {"G": st["G"], "ci": st["ci"], "native": stn["G"]}
        nat[m] = {"G": stn["G"], "ci": stn["ci"]}
    say("transferred models at alpha = 1.1: " + ", ".join(f"{k} {v['1.1']['G']:.2f} (native {v['1.1']['native']:.2f})" for k, v in tr.items()))
    say(f"largest loss against native models: {100 * max(1 - v[m]['G'] / v[m]['native'] for v in tr.values() for m in ('1.1',)):.0f}%")
    say(f"n = 256: native model {ci(nat['1.1'])} and {ci(nat['1.01'])}; transferred {ci(tr['128->256']['1.1'])} and {ci(tr['128->256']['1.01'])}")
    rows = []
    for j in range(cm.N_D):
        fn = f"F512_500_{j:02d}.json"
        a = dict(J(G3, "D", fn)["arms"])
        a.pop("from_F256", None)
        a["SQpkg_ss"] = J(RB, "d512", fn)["arms"]["SQpkg_ss"]
        a[P] = J(V2C, "D", fn)["arms"]["from_F256"]
        rows.append({"arms": a})
    pool_d = ["SQ_ss_K50", "DAAREM_tun", "SQpkg_ss"]
    d512 = {}
    for m, i in (("1.1", I11), ("1.01", I101)):
        H = {a: hits(rows, a, i) for a in pool_d + [P]}
        _, st, _ = strongest(H, P, pool_d)
        d512[m] = {"G": st["G"], "ci": st["ci"], "median": med(H[P]), "median_SQUAREM": med(H["SQpkg_ss"])}
    say(f"n = 512 ({cm.N_D} systems, n = 256 model): median {d512['1.1']['median']:g} and {d512['1.01']['median']:g} E-steps; SQUAREM "
        f"{d512['1.1']['median_SQUAREM']:g} and {d512['1.01']['median_SQUAREM']:g}; ratios {ci(d512['1.1'])} and {ci(d512['1.01'])}")
    put("transfer", {"transfer": tr, "n256_native": nat, "n512": d512})
    say()

    say("== Penalty selected for support recovery (F32, F64) ==")
    lam = J(SUPP, "lambda.json")
    say("selected c: " + ", ".join(f"{SH.get(k, k)} {v['c'] if isinstance(v, dict) else v}" for k, v in lam.items()))
    sup = {}
    for c in ("F32_500", "F64_500"):
        rows = []
        for j in range(cm.N_TEST_II):
            fn = f"{c}_{j:02d}.json"
            base = J(SUPP, "testII", fn)
            a = dict(base["arms"])
            a.update({k: v for k, v in J(SUPP, "sqbt", "testII", fn)["arms"].items() if k != "SQ_ss_K50"})
            a[P] = J(V2C, "supp", fn)["arms"][P]
            rows.append({"arms": a, "meta": base})
        pool = [x for x in POOL_ALL if x in rows[0]["arms"]]
        pool_ss = [x for x in POOL_SS if x in rows[0]["arms"]]
        e = {}
        for m, i in (("1.1", I11), ("1.01", I101)):
            H = {a: hits(rows, a, i) for a in pool + [P]}
            o, st, _ = strongest(H, P, pool)
            qd = lambda a: float(np.median([r["arms"][a]["q"][i]["f1"] for r in rows if r["arms"][a]["q"][i] is not None]))
            e[m] = {"G": st["G"], "ci": st["ci"], "strongest": o, "EM_median": med(H["EM_ss_K50"]), "f1": (qd(P), qd(o))}
        H = {a: phits(rows, a, 1, 0.01) for a in pool_ss + [P]}
        o, st, _ = strongest(H, P, pool_ss)
        e["dist1"] = {"G": st["G"], "ci": st["ci"], "fail": sum(x is None for x in H[P]), "fail_SQUAREM": sum(x is None for x in H["SQpkg_ss"])}
        sup[c] = e
        say(f"{SH[c]}: {ci(e['1.1'])} and {ci(e['1.01'])}; EM median to alpha = 1.01 {e['1.01']['EM_median']:g}; F1 when met "
            f"{e['1.1']['f1'][0]:.3f}/{e['1.1']['f1'][1]:.3f}, {e['1.01']['f1'][0]:.3f}/{e['1.01']['f1'][1]:.3f}; 1% distance {ci(e['dist1'])}, "
            f"missed on {e['dist1']['fail']} (SQUAREM {e['dist1']['fail_SQUAREM']})")
    put("support_penalty", sup)
    say()


def part_heat():
    say("== Heat conduction networks HF64 and HF128 ==")
    out = {}
    for c in ("HF64_500", "HF128_500"):
        rows = []
        for j in range(30):
            fn = f"{c}_{j:02d}.json"
            base = J(G3, "H_A7", fn)
            a = {k: x for k, x in base["arms"].items() if k in ("EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun")}
            a["SQpkg_ss"] = J(V2, "sqpkg_rest", "heat", fn)
            a.update({k: x for k, x in J(V2C, "heat", fn)["arms"].items() if isinstance(x, dict)})
            rows.append({"arms": a, "meta": base})
        pool = ["EM_ss_K50", "SQ_ss_K50", "DAAREM_tun", "SQpkg_ss"]
        e = {"alpha_median": {a: (med(hits(rows, a, I11)), med(hits(rows, a, I101))) for a in pool + [P]}}
        for name, col, thr in PARAM:
            other = [k for k in rows[0]["arms"] if k.startswith("from_")]
            H = {a: phits(rows, a, col, thr) for a in pool + [P] + other}
            o, st, _ = strongest(H, P, pool)
            e[name] = {"G": st["G"], "ci": st["ci"], "strongest": o, "median": {a: med(H[a]) for a in pool + [P]}}
            for a in other:
                e[name][a] = strongest(H, a, pool)[1]["G"]
        e["rejected"] = (sum(r["arms"][P]["layer_reject"] for r in rows), sum(r["arms"][P]["n_layer_steps"] for r in rows))
        e["handover"] = sum(r["arms"][P]["handover_at"] is not None for r in rows)
        e["ref"] = {"recall": float(np.median([r["meta"]["q_R"]["rec"] for r in rows])), "delta": float(np.median([r["meta"]["q_R"]["delta"] for r in rows])),
                    "rho": (min(r["meta"]["rho"] for r in rows), max(r["meta"]["rho"] for r in rows))}
        out[c] = e
        say(f"{c[:-4]}: alpha medians (proposed) {e['alpha_median'][P]}, all methods within {max(max(v) for v in e['alpha_median'].values()):g}; "
            f"1% distance: EM {e['dist1']['median']['EM_ss_K50']:g}, proposed {e['dist1']['median'][P]:g}, strongest {e['dist1']['strongest']} {e['dist1']['median'][e['dist1']['strongest']]:g}")
        say(f"   ratios " + ", ".join(f"{k} {e[k]['G']:.2f}" for k, *_ in PARAM)
            + "".join(f"; layers of {a[5:-4]}: " + ", ".join(f"{e[k][a]:.2f}" for k, *_ in PARAM) for a in e["dist1"] if a.startswith("from_")))
        say(f"   rejected {e['rejected'][0]} of {e['rejected'][1]} candidates, handover on {e['handover']} systems; reference recall {e['ref']['recall']:.2f}, "
            f"relative error {e['ref']['delta']:.2f}, spectral radius {e['ref']['rho'][0]:.3f} to {e['ref']['rho'][1]:.3f}")
    put("heat", out)
    say()


def part_graphem():
    say("== GraphEM benchmark (n = 16) and GraphEM outer iteration ==")
    rows = []
    for j in range(50):
        fn = f"test_{j:02d}.json"
        a = dict(J(V2, "ge", fn)["arms"])
        a["SQpkg_ss"] = J(V2, "sqpkg_rest", "ge", fn)
        a[P] = J(V2C, "ge", fn)["arms"][P]
        rows.append({"arms": a})
    pool = ["EM_ss_K50", "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun", "SQpkg_ss"]
    ge = {}
    for m, i in (("1.1", I11), ("1.01", I101)):
        H = {a: hits(rows, a, i) for a in pool + [P]}
        o, st, _ = strongest(H, P, pool)
        ge[m] = {"G": st["G"], "strongest": o, "medians": {a: med(H[a]) for a in pool + [P]}}
    say(f"n = 16: every method reached alpha = 1.1 within a median of {max(ge['1.1']['medians'].values()):g} E-steps; ratio to the strongest classical method "
        f"{ge['1.1']['G']:.2f} and {ge['1.01']['G']:.2f}")
    adv = J(V2, "advisor", "summary.json")
    A = {}
    for j in range(50):
        d = J(V2, "advisor", "A", f"test_{j:02d}.json")
        A[j] = d["arms"]["EM_exact_full"]["hits"]
    gA = {m: cm.ratio_stats([A[j][i] for j in range(50)], [rows[j]["arms"][P]["hits"][i] for j in range(50)]) for m, i in (("1.1", I11), ("1.01", I101))}
    say(f"EM with exact E-step and fully solved M-step (GraphEM outer iteration) / proposed, n = 16: {ci(gA['1.1'])} and {ci(gA['1.01'])}")
    gC = {}
    for c in ("F64_500", "F128_500"):
        hc, hp, dc, dp = {m: [] for m in ("1.1", "1.01")}, {m: [] for m in ("1.1", "1.01")}, [], []
        for r in test_ii(c):
            d = J(V2, "advisor", "C", f"{c}_{r['j']:02d}.json")
            g = d["arms"]["EM_exact_full"]
            for m, i in (("1.1", I11), ("1.01", I101)):
                hc[m].append(g["hits"][i])
                hp[m].append(r["arms"][P]["hits"][i])
            dc.append(xh(g, 1, 0.01))
            dp.append(xh(r["arms"][P], 1, 0.01))
        gC[c] = {m: cm.ratio_stats(hc[m], hp[m]) for m in ("1.1", "1.01")}
        gC[c]["dist1"] = cm.ratio_stats(dc, dp)
        say(f"{SH[c]}: GraphEM outer iteration / proposed {ci(gC[c]['1.1'])} at alpha = 1.1, {ci(gC[c]['1.01'])} at 1.01, {ci(gC[c]['dist1'])} to 1% distance (own reference)")
    gl = [x for c in ("F64_500", "F128_500") for f in sorted(glob.glob(os.path.join(V2, "advisor", "C", f"{c}_*.json"))) for x in J(f)["mstep_gap"]]
    say(f"along the first 20 EM iterations, monotone FISTA with 50 iterations stops short of the M-step solution by a median of {100 * np.median(gl):.2f}% "
        f"and at most {100 * np.max(gl):.1f}% of the length of the EM update")
    put("graphem", {"n16": ge, "outer_n16": gA, "outer_testII": gC, "mstep_gap": {c: (adv["C"][c]["mstep_gap_median"], adv["C"][c]["mstep_gap_max"]) for c in ("F64_500", "F128_500")}})
    say()


def gm(x):
    return float(np.exp(np.mean(np.log(x))))


def part_timing():
    say("== Computing time (single process replays) ==")
    pc = {int(k): v for k, v in J(V2, "percall.json").items()}
    say("exact / steady state E-step time: " + ", ".join(f"n = {n}: {v['estep_exact'] / v['estep_ss']:.1f}" for n, v in sorted(pc.items())))
    say("steady state E-step / M-step time: " + ", ".join(f"n = {n}: {v['estep_ss'] / v['mstep']:.1f}" for n, v in sorted(pc.items())))
    say("learned layer / iteration (steady state E-step + M-step): " + ", ".join(
        f"n = {n}: {100 * v['layer_extra'] / (v['estep_ss'] + v['mstep']):.1f}%" for n, v in sorted(pc.items())))
    T = J(RB, "time16", "summary.json")["timing"]
    for n in (32, 64, 128, 256):
        say(f"n = {n}: " + "  ".join(
            f"alpha {m}: {T[f'{n}/{m}'][T[f'{n}/{m}']['fastest_by_GT']]['G_T']:.2f} against the faster version ({T[f'{n}/{m}']['fastest_by_GT']}; "
            f"default {T[f'{n}/{m}']['SQpkg_ss']['G_T']:.2f}, gamma_0 = 16 {T[f'{n}/{m}']['SQpkg16_ss']['G_T']:.2f})" for m in ("1.1", "1.01")))
    eff = [T[f"{n}/{m}"][o]["G_T"] / T[f"{n}/{m}"][o]["G_E"] for n in (32, 64, 128, 256) for m in ("1.1", "1.01") for o in ("SQpkg_ss", "SQpkg16_ss")]
    say(f"time ratio / E-step ratio: {rng(eff)}")
    t2 = {}
    for n, c in ((32, "F32_500"), (64, "F64_500"), (128, "F128_500"), (256, "F256_500")):
        rows = [J(f)["arms"] for f in sorted(glob.glob(os.path.join(V2C, "timing2", f"{c}_*.json")))]
        for m in ("1.1", "1.01"):
            for o in ("DAAREM_tun", "SQ_ss_K50", "SQpkg_ss"):
                ok = [x for x in rows if x[P][m]["t"] and x[o][m]["t"]]
                t2[f"{n}/{m}/{o}"] = gm([x[o][m]["t"] / x[P][m]["t"] for x in ok])
    say("separate run (DAAREM and SQUAREM with immediate fallback): " + "; ".join(
        f"n = {n}: " + ", ".join(f"{o} {t2[f'{n}/{m}/{o}']:.2f}" for m in ("1.1", "1.01") for o in ("DAAREM_tun", "SQ_ss_K50", "SQpkg_ss")) for n in (32, 64, 128, 256)))
    t512 = [J(f) for f in sorted(glob.glob(os.path.join(RB, "t512", "*.json")))]
    t512b = [J(f) for f in sorted(glob.glob(os.path.join(RB, "time16", "t512", "*.json")))]
    d = {}
    for m in ("1.1", "1.01"):
        ok = [t for t in t512 if t["arms"][P][m].get("t")]
        d[m] = {"median_t": float(np.median([t["arms"][P][m]["t"] for t in ok]))}
        for o in ("DAAREM_tun", "SQpkg_ss"):
            okk = [t for t in ok if t["arms"][o][m].get("t")]
            d[m][o] = gm([t["arms"][o][m]["t"] / t["arms"][P][m]["t"] for t in okk])
        okb = [t for t in t512b if t["arms"][P][m].get("t") and t["arms"]["SQpkg16_ss"][m].get("t")]
        if okb:
            d[m]["SQpkg16_ss"] = gm([t["arms"]["SQpkg16_ss"][m]["t"] / t["arms"][P][m]["t"] for t in okb])
    say(f"n = 512: median time {d['1.1']['median_t']:.0f} s and {d['1.01']['median_t']:.0f} s; ratios to DAAREM {d['1.1']['DAAREM_tun']:.2f} and {d['1.01']['DAAREM_tun']:.2f}, "
        f"to default SQUAREM {d['1.1']['SQpkg_ss']:.2f} and {d['1.01']['SQpkg_ss']:.2f}, to SQUAREM with gamma_0 = 16 "
        f"{d['1.1'].get('SQpkg16_ss', float('nan')):.2f} and {d['1.01'].get('SQpkg16_ss', float('nan')):.2f}")
    cost = J(V2, "verdict.json")["cost"]
    for c in CELLS:
        cost[c].setdefault("ref_sec_total", float("nan"))
    say("training of four layers, minutes: " + ", ".join(f"{SH[c]} {cost[c]['train_sec_L4'] / 60:.0f}" for c in CELLS)
        + f"; reference solutions up to {max(cost[c]['ref_sec_total'] for c in CELLS) / 60:.0f} minutes; "
        f"F128 in total {(cost['F128_500']['train_sec_L4'] + cost['F128_500']['ref_sec_total']) / 3600:.1f} hours")
    put("timing", {"percall": pc, "ratios": T, "separate_run": t2, "n512": d})
    say()


def part_switch():
    say("== Steady state E-step: switch to exact E-steps (Algorithm 1) ==")
    S = {}
    tot = within_gap = below = within1 = 0
    after_alpha = never = 0
    at = []
    dists = []
    f128_sw = set()
    for c in CELLS:
        k = 0
        for f in sorted(glob.glob(os.path.join(RB, "switch", f"{c}_*.json"))):
            d = J(f)
            a = d["arms"]["switch"]
            if a["switched_at"] is None:
                continue
            k += 1
            at.append(a["switched_at"])
            h = a["hits"]
            after_alpha += h[I101] is not None and h[I101] <= a["switched_at"]
            never += h[I101] is None
            p = test_ii(c)[d["j"]]["arms"][P]["hits"]
            ex = a["extra"]
            within_gap += any(e[3] <= 1e-5 for e in ex)
            below += min(e[3] for e in ex) < 0
            within1 += any(e[1] <= 0.01 for e in ex)
            dists.append(ex[-1][1])
            if c == "F128_500":
                f128_sw.add(d["j"])
        S[c] = k
        tot += k
    same = all(J(f)["arms"]["switch"]["hits"][i] == test_ii(os.path.basename(f)[:-8])[J(f)["j"]]["arms"][P]["hits"][i]
               for f in glob.glob(os.path.join(RB, "switch", "*.json")) for i in (I11, I101))
    miss128 = {r["j"] for r in test_ii("F128_500") if xh(r["arms"][P], 1, 0.01) is None}
    say("switched on " + ", ".join(f"{SH[c]} {k}" for c, k in S.items()) + f" systems ({tot} of 150), after {min(at)} to {max(at)} E-steps; "
        f"after alpha = 1.01 had been met on {after_alpha} (alpha = 1.01 not met within 100 E-steps on {never}); E-step counts at both targets as in Table III: {same}")
    say(f"after the switch: within a relative objective gap of 1e-5 on {within_gap}, below the reference value on {below}, within 1% distance on {within1}; "
        f"final distance {100 * min(dists):.1f}% to {100 * max(dists):.0f}% (median {100 * np.median(dists):.1f}%)")
    say(f"F128: switched systems equal the systems that missed 1% distance in Table IV: {f128_sw == miss128}")
    chk = [json.loads(l) for l in open(os.path.join(V2C, "switch_check.jsonl"), encoding="utf-8") if l.strip()]
    fails = [x for x in chk if x["fails"]]
    first = [min(f[1] for f in x["fails"]) for x in fails]
    worst = max(f[2] for x in fails for f in x["fails"])
    sqshare = sum(f[0].startswith("sq") for x in fails for f in x["fails"]) / sum(len(x["fails"]) for x in fails)
    met101, close = 0, 0
    for x, e1 in zip(fails, first):
        r = test_ii(x["name"])[x["j"]]["arms"][P]
        met101 += r["hits"][I101] is not None and r["hits"][I101] <= e1
        ex = [e for e in r["extra"] if e[0] <= e1]
        close += bool(ex) and ex[-1][1] <= 0.01
    tiny = sorted((e1, f"{x['name']} {x['j']:02d}", min(f[2] for f in x["fails"] if f[1] == e1)) for x, e1 in zip(fails, first))[-1]
    say(f"decrease test failed on {len(fails)} of {len(chk)} systems ({100 * sqshare:.1f}% of failures in SQUAREM cycles), by a relative increase of at most {worst:.1e}; "
        f"first failure at E-step {min(first)} to {max(first)} (latest: {tiny[1]}, relative increase {tiny[2]:.1e}), after alpha = 1.01 on {met101}; within 1% of the reference by then on {close}")
    put("switch", {"per_cell": S, "after": (min(at), max(at)), "within_gap": within_gap, "below": below, "within1": within1,
                   "dist": (min(dists), max(dists), float(np.median(dists))), "check_fails": len(fails), "check_worst": worst,
                   "check_first": (min(first), max(first)), "check_close": close})
    reach = {}
    for a in ("SQ_ss_K50", "SQ_exact_K50", "SQbt3_ss", "SQbt3_exact"):
        reach[a] = sum(r["arms"][a]["hits"][I101] is not None for c in CELLS for r in test_ii(c))
    say("systems on which SQUAREM reached alpha = 1.01 within 100 E-steps: " + ", ".join(f"{a} {v}" for a, v in reach.items()))
    dc = [json.loads(l) for l in open(os.path.join(V2, "dare_check_all.jsonl"), encoding="utf-8") if l.strip()]
    agg = {}
    for x in dc:
        g = agg.setdefault(x["arm"], {"rho_ge1": 0, "undetectable": 0, "calls": 0, "errors": 0})
        g["rho_ge1"] += x["rho_ge1"]
        g["calls"] += x["calls"]
        g["errors"] += x["error"] is not None
        g["undetectable"] += x["min_pbh_sv_unstable"] is not None and x["min_pbh_sv_unstable"] < 1e-8
    say(f"partial observation, {len({(x['cell'], x['split'], x['j']) for x in dc})} systems: " + "; ".join(
        f"{a}: rho >= 1 at {g['rho_ge1']} evaluated matrices, undetectable {g['undetectable']}, errors {g['errors']}" for a, g in agg.items()))
    put("dare", agg)
    say()


def part_stoprule():
    say("== Table VIII: median E-steps at the stop / systems with alpha = 1.01 met ==")
    arms = [(P, "Proposed"), ("EM_ss_K50", "EM"), ("SQpkg_ss", "SQUAREM"), ("DAAREM_tun", "DAAREM")]
    T = {}
    for eps in (1e-4, 1e-5, 1e-6):
        T[eps] = {}
        for a, lab in arms:
            T[eps][lab] = {}
            for c in CELLS:
                cs, ok, ds = [], 0, []
                for f in sorted(glob.glob(os.path.join(RB, "stoprule", f"{c}_*.json"))):
                    d = J(f)
                    ex = d["arms"][a]["extra"]
                    tr = dict((int(k), v) for k, v in d["arms"][a]["trace"])
                    g = {e[0]: e for e in ex}
                    stop = next((g[k] for k in sorted(g) if k >= 4 and (k - 3) in g and abs(g[k][3] - g[k - 3][3]) <= eps), ex[-1])
                    cs.append(stop[0])
                    ok += tr[stop[0]] <= d["taus"][I101]
                    ds.append(stop[1])
                T[eps][lab][SH[c]] = (med(cs), ok, float(np.median(ds)))
            if eps != 1e-6:
                say(f"eps {eps:g} {lab:9s} " + "  ".join(f"{v[0]:g} / {v[1]}" for v in T[eps][lab].values()))
    p6 = [v[0] for v in T[1e-6]["Proposed"].values()]
    m6 = min(v[1] for lab in T[1e-6] for v in T[1e-6][lab].values())
    say(f"eps 1e-6: proposed stopped after {rng(p6, '{:g}')} E-steps; at these stops all methods met alpha = 1.01 on at least {m6} systems")
    say("eps 1e-5, median distance at the stop: " + "; ".join(f"{lab} {rng([v[2] * 100 for v in T[1e-5][lab].values()], '{:.1f}')}%" for _, lab in arms))
    put("table8", {str(k): v for k, v in T.items()})
    say()


def part_spectrum():
    say("== Table IX: eigenvalues of the Jacobian of steady state EM at its fixed point ==")
    T = {}
    for c in ("P32_500", "F32_500", "F64_500", "F128_500", "P64_500"):
        fs = sorted(glob.glob(os.path.join(V2, "sqdiag", "jac", f"{c}_*.json")))
        D = [J(f) for f in fs]
        if not D:
            continue
        T[c] = {k: (min(d[k] for d in D), max(d[k] for d in D)) for k in ("support", "lambda1", "ratio2", "n_within_0.95", "n_within_0.9")}
        T[c]["n"] = len(D)
        e = T[c]
        say(f"{SH[c]:5s} ({e['n']} systems) support {e['support'][0]}-{e['support'][1]}  |b1| {e['lambda1'][0]:.3f}-{e['lambda1'][1]:.3f}  "
            f"|b2|/|b1| {e['ratio2'][0]:.3f}-{e['ratio2'][1]:.3f}  >=0.95 {e['n_within_0.95'][0]}-{e['n_within_0.95'][1]}  >=0.9 {e['n_within_0.9'][0]}-{e['n_within_0.9'][1]}")
    put("table9", T)
    fb = []
    for c in ("F32_500", "F64_500", "F128_500"):
        logs = [J(f)["arms"]["SQ_ss_K50"]["sq_log"] for f in sorted(glob.glob(os.path.join(V2, "sqdiag", "arms", f"{c}_*.json")))]
        cyc = [x for L in logs for x in L]
        fb.append((c, float(np.mean([x[1] for x in cyc])), float(np.median([x[0] for x in cyc]))))
    say("SQUAREM with immediate fallback (first ten systems): " + ", ".join(f"{SH[c]} rejected {100 * r:.0f}%, median gamma {g:.1f}" for c, r, g in fb))
    acc, gam, ext = [], [], []
    for c in CELLS:
        a1, n1, g, e = 0, 0, [], []
        for r in test_ii(c):
            for x in J(V2, "sqbt", "testII", f"{c}_{r['j']:02d}.json")["arms"]["SQbt3_ss"]["log"]:
                n1 += 1
                a1 += x["nb"] == 0
            for x in J(V2, "sqbt", "testII", f"{c}_{r['j']:02d}.json")["arms"]["SQpkg_ss"]["log"]:
                g.append(x["alpha"])
                e.append(x["extrap"])
        acc.append(a1 / n1)
        gam.append(float(np.median(g)))
        ext.append(float(np.mean(e)))
    say(f"backtracking SQUAREM (SqS3): first trial accepted in {rng([100 * x for x in acc], '{:.0f}')}% of cycles; reference implementation: median -gamma "
        f"{rng(gam, '{:.1f}')}, extrapolations accepted {rng([100 * x for x in ext], '{:.0f}')}%")
    say()


def part_misc():
    say("== Implementation details ==")
    ch = J(V2C, "choice.json")
    say(f"handover: nu = {ch['r']}, gamma_0 = {ch['amax0']:g}; criterion from {min(ch['gm_stop'].values()):.1f} to {max(ch['gm_stop'].values()):.1f}")
    changed = sum(r["arms"][P]["hits"][I101] != r["arms"]["v2_" + P]["hits"][I101] for c in CELLS for r in test_ii(c))
    say(f"the handover changed the E-steps to alpha = 1.01 on {changed} of 150 systems")
    nf = J(V2, "sqdiag", "nf", "summary.json")
    g = lambda arm, k: [x for c in nf for x in nf[c][arm][k]["G_vs_full"]]
    say(f"N_F = 10: up to {100 * (max(g('learned', '10')) - 1):.0f}% more E-steps for the proposed method; from N_F = 30 on, EM within "
        f"{100 * max(abs(x - 1) for k in range(30, 101, 10) for x in g('EM', str(k))):.0f}%, proposed within "
        f"{100 * max(abs(x - 1) for k in range(30, 101, 10) for x in g('learned', str(k))):.0f}%, SQUAREM and DAAREM within "
        f"{100 * max(abs(x - 1) for a in ('SQUAREM', 'DAAREM') for k in range(30, 101, 10) for x in g(a, str(k))):.0f}% of the fully solved case; "
        f"one M-step with N_F = 50 took {rng([100 * nf[c][x]['50']['M_over_E'] for c in nf for x in nf[c]], '{:.0f}')}% of a steady state E-step")
    sm = {}
    for c in ("P32_500", "P64_500", "F64_500"):
        rows = [(J(V2C, "smax", f"{c}_{j:02d}.json")["arms"]["smax20"], test_ii(c)[j]["arms"][P]) for j in range(cm.N_TEST_II)]
        sm[c] = {m: cm.ratio_stats([b["hits"][i] for a, b in rows], [a["hits"][i] for a, b in rows]) for m, i in (("1.1", I11), ("1.01", I101))}
    say("layers with s_max = 10 / layers with s_max = 20 (E-step ratio): " + "; ".join(f"{SH[c]} {ci(v['1.1'])}, {ci(v['1.01'])}" for c, v in sm.items()))
    sat = {}
    for c in ("P32_500", "P64_500"):
        z = [np.load(f) for f in sorted(glob.glob(os.path.join(V2, "diag", f"{c}_*.npz")))]
        sup = []
        for x in z:
            err = x["err"].astype(float)
            st = err[1] - err[0]
            big = (np.abs(st) > 1e-3 * np.sqrt(np.mean(st ** 2))) & (np.abs(x["A_R"]) > 0)
            sup.append(x["S"][0][big])
        sup = np.concatenate(sup)
        old = J(V2, "smax20", "summary.json")[c]["old"]["sat"]
        sat[c] = (float(old), float(np.mean(sup >= 0.99 * 10)))
    say("first trained layer at s_max = 10: " + ", ".join(f"{SH[c]} {100 * a:.0f}% of all entries, {100 * b:.0f}% of the entries of Fig. 4" for c, (a, b) in sat.items()))
    put("misc", {"handover_changed": changed, "smax": sm, "smax_sat": sat})
    say()


def part_fig_motiv():
    by = {}
    for f in sorted(glob.glob(os.path.join(V2, "diag", "*.npz"))):
        c = os.path.basename(f)[:-7]
        z = np.load(f)
        err, A_R, S = z["err"].astype(float), z["A_R"], z["S"]
        n = A_R.shape[0]
        st = err[1] - err[0]
        big = (np.abs(st) > 1e-3 * np.sqrt(np.mean(st ** 2))) & (np.abs(A_R) > 0)
        s0 = -err[0][big] / st[big]
        by.setdefault(c, []).append((s0, S[0][big], np.eye(n, dtype=bool)[big]))
    rows, info = [], {}
    for k, c in enumerate(CELLS):
        s = np.concatenate([x[0] for x in by[c]])
        l = np.concatenate([x[1] for x in by[c]])
        d = np.concatenate([x[2] for x in by[c]])
        rows.append([k, SH[c]] + [f"{np.median(v):.2f}" for v in (s[d], l[d], s[~d], l[~d])])
        info[c] = {"negative": float(np.mean(s <= 0)), "quartiles_positive": [float(np.percentile(s[s > 0], q)) for q in (25, 75)]}
    write_csv("motiv_steps.csv", ["x", "setting", "idd", "lrd", "ido", "lro"], rows)
    say("== Fig. 4 and Section V ==")
    vals = {r[1]: [float(x) for x in r[2:]] for r in rows}
    say("class medians of the ideal first steps: " + rng([v for r in vals.values() for v in (r[0], r[2])]))
    say("diagonal / off-diagonal ideal medians: " + ", ".join(f"{k} {v[0] / v[2]:.2f}" for k, v in vals.items())
        + "; largest deviation of the first trained layer from the ideal class medians: "
        + f"{100 * max(max(abs(v[1] / v[0] - 1), abs(v[3] / v[2] - 1)) for v in vals.values()):.0f}%")
    put("fig4", info)
    say("negative ideal first steps: " + ", ".join(f"{SH[c]} {100 * v['negative']:.0f}%" for c, v in info.items())
        + "; quartiles of the positive ones: " + ", ".join(f"{SH[c]} {v['quartiles_positive'][0]:.2f}/{v['quartiles_positive'][1]:.2f}" for c, v in info.items()))
    dg = J(V2, "verdict.json")["diag"]
    say("rank correlation of missing information with the ideal first steps: " + ", ".join(f"{SH[c]} {dg[c]['ideal']['iter0']['spearman_sstar_mubar']:.3f}" for c in CELLS))
    say("rank correlation of the first learned step with the ideal one: " + ", ".join(f"{SH[c]} {dg[c]['ideal']['spearman_learnedS1_vs_ideal0']:.2f}" for c in CELLS))
    say("median contraction factor over EM iterations 10 to 60: " + ", ".join(f"{SH[c]} {dg[c]['rate_vs_mubar']['rate_median_diag']:.3f}/{dg[c]['rate_vs_mubar']['rate_median_off']:.3f}" for c in CELLS)
        + "; rank correlation with missing information: " + ", ".join(f"{SH[c]} {dg[c]['rate_vs_mubar']['spearman']:.3f}" for c in CELLS))
    say("rank correlation of the learned steps with the Aitken step, layers 2 to 4: " + ", ".join(
        f"{SH[c]} " + "/".join(f"{dg[c][f'layer{l}'].get('spearman_S_aitken', float('nan')):.2f}" for l in (2, 3, 4)) for c in CELLS))
    say("wrote data/motiv_steps.csv (Fig. 4)")
    say()


def part_preliminary():
    base = os.path.join(RES, "preliminary")
    if not os.path.isdir(base):
        say("preliminary tests: results/preliminary not found, run unrolled_em.preliminary first")
        return
    out = {}
    for c in CELLS:
        fs = sorted(glob.glob(os.path.join(base, c, "test_*_0b.json")))
        H = {a: [J(f)["arms"][a]["hit_E"] for f in fs] for a in ("OWLQN", "SpaRSA", "SQ_ss_K50")}
        out[c] = {a: med(v) for a, v in H.items()}
        out[c]["n"] = len(fs)
    say("== Preliminary tests: median evaluations / E-steps to alpha = 1.1 ==")
    for a in ("OWLQN", "SpaRSA", "SQ_ss_K50"):
        say(f"{a}: " + ", ".join(f"{SH[c]} {out[c][a]:g}" for c in CELLS) + f"  ({out[CELLS[0]]['n']} systems per setting)")
    put("preliminary", out)
    say()


PARTS = {"setup": part_setup, "main": part_main, "param": part_param, "switch": part_switch, "stoprule": part_stoprule,
         "timing": part_timing, "graphem": part_graphem, "ablation": part_ablation, "robust": part_robust, "heat": part_heat,
         "spectrum": part_spectrum, "misc": part_misc, "preliminary": part_preliminary, "fig_tol": part_fig_tol,
         "fig_motiv": part_fig_motiv, "fig_trace": part_fig_trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all"] + list(PARTS))
    a = ap.parse_args()
    for name, fn in PARTS.items():
        if a.part in ("all", name):
            fn()
    os.makedirs(OUT, exist_ok=True)
    if a.part == "all":
        cm.dump(NUM, os.path.join(OUT, "numbers.json"))
        with open(os.path.join(OUT, "numbers.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
