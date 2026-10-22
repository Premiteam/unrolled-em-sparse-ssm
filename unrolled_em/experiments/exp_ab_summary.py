import glob
import json
import os

import numpy as np

from unrolled_em.experiments import common as cm

OUT = os.path.join(cm.OUT, "advisor")
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)


def summary_A():
    rs = [cm.load(f) for f in sorted(glob.glob(os.path.join(OUT, "A", "test_*.json")))]
    arms = list(rs[0]["arms"])
    S = {"n_sys": len(rs), "arms": {}, "repro": {}}
    for a in arms:
        h = lambda i: [r["arms"][a]["hits"][i] for r in rs]
        q = [r["arms"][a]["q"][I11] for r in rs]
        S["arms"][a] = {"med_1.1": cm.med(h(I11)), "med_1.01": cm.med(h(I101)),
                        "fail_1.01": sum(x is None for x in h(I101)),
                        "delta_1.1": float(np.median([x["delta"] for x in q])),
                        "f1_1.1": float(np.median([x["f1"] for x in q]))}
    for a in ("EM_ss_K50", "learned_tied_L4"):
        S["repro"][a] = sum(r["arms"][a]["hits"] == r["old"][a] for r in rs)
    pairs = [("EM_exact_full", "learned_tied_L4"), ("EM_exact_full", "learned_ss_full"),
             ("EM_exact_full", "learned_exact_full"), ("EM_ss_K50", "EM_exact_full"), ("EM_ss_K50", "EM_ss_full")]
    S["ratios"] = {}
    for c, p in pairs:
        for i, tag in ((I11, "1.1"), (I101, "1.01")):
            st = cm.ratio_stats([r["arms"][c]["hits"][i] for r in rs], [r["arms"][p]["hits"][i] for r in rs])
            S["ratios"][f"{c}/{p}@{tag}"] = st
    S["same_hits_EM_ss_K50_vs_full"] = sum(r["arms"]["EM_ss_K50"]["hits"] == r["arms"]["EM_ss_full"]["hits"] for r in rs)
    S["same_hits_learned_K50_vs_full"] = sum(r["arms"]["learned_tied_L4"]["hits"] == r["arms"]["learned_ss_full"]["hits"]
                                             for r in rs)
    return S


def summary_B():
    out = {}
    for f in sorted(glob.glob(os.path.join(OUT, "B", "*.json"))):
        d = cm.load(f)
        row = {}
        for a, r in d["arms"].items():
            L = r["log"]
            n_inc_after = sum(1 for x in L if x[4] == "exact" and x[2] > x[1])
            row[a] = {"hits": r["hits"], "switched_at": r.get("switched_at"), "last": r["extra"][-1],
                      "n_E_total": r["extra"][-1][0], "inc_after_switch": n_inc_after, "sec": r["sec"]}
        row["old_hits"] = d["old"]["hits"]
        row["old_last"] = d["old"]["extra"][-1]
        out[os.path.basename(f)] = row
    return out


def _first(extra, idx, thr):
    return next((c for c, *v in extra if (v[idx - 1] <= thr) and c <= cm.MAX_E), None)


def summary_C():
    out = {}
    for name in ("F64_500", "F128_500"):
        rs = [cm.load(f) for f in sorted(glob.glob(os.path.join(OUT, "C", f"{name}_*.json")))]
        if not rs:
            continue
        arms = list(rs[0]["arms"])
        S = {"n_sys": len(rs), "repro": {a: sum(r["arms"][a]["hits"] == r["old"][a]["hits"] for r in rs)
                                         for a in ("EM_ss_K50", "learned_tied_L4")},
             "mstep_gap_median": float(np.median([np.median(r["mstep_gap"]) for r in rs])),
             "mstep_gap_first_median": float(np.median([r["mstep_gap"][0] for r in rs])),
             "mstep_gap_max": float(np.max([np.max(r["mstep_gap"]) for r in rs])), "arms": {}, "ratios": {}}
        meas = {}
        for a in arms:
            m = {"1.1": [r["arms"][a]["hits"][I11] for r in rs], "1.01": [r["arms"][a]["hits"][I101] for r in rs],
                 "dist5": [_first(r["arms"][a]["extra"], 1, 0.05) for r in rs],
                 "dist1": [_first(r["arms"][a]["extra"], 1, 0.01) for r in rs],
                 "gap1e-4": [_first(r["arms"][a]["extra"], 3, 1e-4) for r in rs],
                 "gap1e-5": [_first(r["arms"][a]["extra"], 3, 1e-5) for r in rs]}
            meas[a] = m
            S["arms"][a] = {k: {"med": cm.med(v), "fail": sum(x is None for x in v)} for k, v in m.items()}
        S["same_hits_EM"] = sum(r["arms"]["EM_ss_K50"]["hits"] == r["arms"]["EM_ss_full"]["hits"] for r in rs)
        S["same_hits_learned"] = sum(r["arms"]["learned_tied_L4"]["hits"] == r["arms"]["learned_ss_full"]["hits"] for r in rs)
        for c, p in (("EM_ss_K50", "EM_ss_full"), ("EM_exact_full", "learned_tied_L4"), ("EM_exact_full", "learned_ss_full"),
                     ("learned_tied_L4", "learned_ss_full"), ("EM_ss_full", "learned_ss_full")):
            for k in ("1.1", "1.01", "dist5", "dist1", "gap1e-4", "gap1e-5"):
                S["ratios"][f"{c}/{p}@{k}"] = cm.ratio_stats(meas[c][k], meas[p][k])
        out[name] = S
    return out


if __name__ == "__main__":
    S = {"A": summary_A() if glob.glob(os.path.join(OUT, "A", "*.json")) else None,
         "B": summary_B() if glob.glob(os.path.join(OUT, "B", "*.json")) else None,
         "C": summary_C() if glob.glob(os.path.join(OUT, "C", "*.json")) else None}
    cm.dump(S, os.path.join(OUT, "summary.json"))
    print(json.dumps(S, indent=1, ensure_ascii=False, default=str))
