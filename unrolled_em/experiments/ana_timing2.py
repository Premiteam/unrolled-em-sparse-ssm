import json, glob, os, sys
import numpy as np
from unrolled_em.experiments import common as cm
IDX = (cm.MULTS.index(1.1), cm.MULTS.index(1.01))
R = os.path.join(cm.HERE, "results", "v2"); C = os.path.join(cm.HERE, "results", "v2c")
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C, "timing2")
P = "learned_tied_L4"; CL = {"SQpkg": "SQpkg_ss", "DA": "DAAREM_tun", "SQold": "SQ_ss_K50"}
J = lambda f: json.load(open(f))
gm = lambda x: float(np.exp(np.mean(np.log(x))))
T = {}
for n_, c in ((32, "F32_500"), (64, "F64_500"), (128, "F128_500"), (256, "F256_500")):
    fs = sorted(glob.glob(f"{D}/{c}_*.json"))
    if not fs: continue
    for m in ("1.1", "1.01"):
        res = {}
        for a, an in CL.items():
            rows = [J(f)["arms"] for f in fs]
            ok = [x for x in rows if x[P][m]["t"] is not None and x[an][m]["t"] is not None]
            res[a] = {"G_T": gm([x[an][m]["t"] / x[P][m]["t"] for x in ok]), "G_E": gm([x[an][m]["h"] / x[P][m]["h"] for x in ok]),
                      "save": float(np.mean([x[an][m]["t"] - x[P][m]["t"] for x in ok])), "n": len(ok), "nall": len(rows),
                      "tP": float(np.median([x[P][m]["t"] for x in ok])), "tO": float(np.median([x[an][m]["t"] for x in ok]))}
        o = min(res, key=lambda a: res[a]["G_T"])
        T[f"{n_}/{m}"] = {"o": o, **res[o], "eff": res[o]["G_T"] / res[o]["G_E"], "all": {a: round(r["G_T"], 3) for a, r in res.items()}}
out = {"timing": T}
v = J(f"{R}/verdict.json")
cost128 = v["cost"]["F128_500"]["train_sec_L4"] + v["cost"]["F128_500"]["ref_sec_total"]
cost64 = v["cost"]["F64_500"]["train_sec_L4"] + v["cost"]["F64_500"]["ref_sec_total"]
out["cost"] = {"128": cost128, "64": cost64}
if "128/1.1" in T:
    out["payback"] = {m: cost128 / T[f"128/{m}"]["save"] for m in ("1.1", "1.01")}
    I = {"1.1": None, "1.01": None}
    pt = {}
    for m in ("1.1", "1.01"):
        sv = []
        for f in sorted(glob.glob(f"{D}/F128_500_*.json")):
            b = os.path.basename(f); x = J(f)["arms"]; tII = J(f"{C}/testII/{b}")["arms"]
            M = tII[P].get("targets") or None
            i = {"1.1": IDX[0], "1.01": IDX[1]}[m]
            htr = tII["tied_from64_L4"]["hits"][i]; hp = tII[P]["hits"][i]; tp = x[P][m]["t"]
            to = x[CL[T[f"128/{m}"]["o"]]][m]["t"]
            if None in (htr, hp, tp, to): continue
            sv.append(to - tp * htr / hp)
        pt[m] = cost64 / float(np.mean(sv))
    out["payback_transfer"] = pt
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 3)))
