import sys, os, json, numpy as np
from unrolled_em import core as ls
from unrolled_em.experiments import common as cm
from unrolled_em.experiments import method as me
from unrolled_em.experiments import exp_hand as EH
H = os.path.join(cm.HERE, "results", "v2")
def check(name, j, r=2, amax0=16.0, max_E=cm.MAX_E):
    z = np.load(f"{H}/refs/testII/{name}_{j:02d}.npz")
    sys_, y, yv, _ = cm.setup(cm.base_cell(name), "test1g", j)
    lam, A0 = float(z["lam"]), z["A0"]
    layers = me.load_layers(f"{H}/models/{name}_learned.pt")[:cm.L_MAIN]
    fails = []; ncl = 0
    A = A0.copy(); st = me.estep(A, sys_, y); n = 1; Rp = np.zeros_like(A); L = len(layers); k = consec = 0
    hand = None
    while n + 1 <= max_E:
        if k >= L and consec >= r:
            hand = n; break
        o = me.layer_step(layers[min(k, L - 1)], A, st, Rp, sys_, y, lam)
        n += o["nE"]
        if o["reject"]:
            ncl += 1
            if not (o["F_new"] <= o["F_old"] - me.ETA * o["delta"]):
                fails.append(("layer", n, (o["F_new"] - o["F_old"]) / abs(o["F_old"])))
        A, st, Rp = o["A"], o["st"], o["Rp"]
        if k >= L: consec = consec + 1 if o["reject"] else 0
        k += 1
    if hand is not None:
        amax = amax0
        while n < max_E:
            F_A = me.fobj(A, st, lam)
            B1 = ls.mstep(A, st["Phi"], st["Psi"], sys_.q, lam, K=50)
            dA = max(me.surrogate(A, st["Phi"], st["Psi"], sys_.q, lam) - me.surrogate(B1, st["Phi"], st["Psi"], sys_.q, lam), 0.0)
            st1 = EH._estep(me, B1, sys_, y); n += 1
            if st1 is None or n >= max_E: break
            B2 = ls.mstep(B1, st1["Phi"], st1["Psi"], sys_.q, lam, K=50)
            r_, v = B1 - A, B2 - 2 * B1 + A
            rr, vv = float(np.sum(r_ * r_)), float(np.sum(v * v))
            a = float(np.sqrt(rr / vv)) if vv > 0 else 1.0
            a = max(1.0, min(amax, a)); acc, why = False, "em"
            if abs(a - 1.0) > 0.01:
                X = A + 2 * a * r_ + a * a * v; why = "rho"
                if EH._ok(X):
                    sX = EH._estep(me, X, sys_, y); n += 1; why = "obj"
                    if sX is not None and n < max_E:
                        Xt = ls.mstep(X, sX["Phi"], sX["Psi"], sys_.q, lam, K=50)
                        if EH._ok(Xt):
                            sXt = EH._estep(me, Xt, sys_, y); n += 1
                            if sXt is not None:
                                F_Xt = me.fobj(Xt, sXt, lam); acc = F_Xt <= F_A - me.ETA * dA
            if acc:
                A, st = Xt, sXt
            else:
                if n >= max_E: break
                s2 = EH._estep(me, B2, sys_, y); n += 1
                if s2 is None: break
                F1 = me.fobj(B1, st1, lam); F2 = me.fobj(B2, s2, lam)
                d1 = max(me.surrogate(B1, st1["Phi"], st1["Psi"], sys_.q, lam) - me.surrogate(B2, st1["Phi"], st1["Psi"], sys_.q, lam), 0.0)
                ncl += 2
                if not F1 <= F_A - me.ETA * dA: fails.append(("sq1", n - 1, (F1 - F_A) / abs(F_A)))
                if not F2 <= F1 - me.ETA * d1: fails.append(("sq2", n, (F2 - F1) / abs(F1)))
                A, st = B2, s2
                if why != "em":
                    if a == amax: amax = max(amax0, amax / 4.0)
                    a = 1.0
            if a == amax: amax *= 4.0
            if rr == 0.0: break
    return {"name": name, "j": j, "hand": hand, "ncl": ncl, "fails": fails}
if __name__ == "__main__":
    import multiprocessing as mp
    cells = sys.argv[1].split(",")
    jobs = [(c, j) for c in cells for j in range(30)]
    with mp.Pool(int(sys.argv[2])) as p:
        for res in p.starmap(check, jobs):
            print(json.dumps(res), flush=True)
