from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import time

import numpy as np
import torch
from torch import nn

from unrolled_em import anytime as at
from unrolled_em import core as ls
from unrolled_em import unroll as u2
from unrolled_em.experiments import common as cm
from unrolled_em.experiments import run as rn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

CELLS = list(cm.CELLS)
ARMS = ["class", "linear"]
CFG = {"tag": "", "steps": cm.TRAIN_STEPS}
I11, I101 = cm.MULTS.index(1.1), cm.MULTS.index(1.01)


class AltLayer(nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "class":
            self.a = nn.Parameter(torch.zeros(8))
            self.b = nn.Parameter(torch.zeros(8))
        elif mode == "linear":
            self.lin = nn.Linear(u2.N_FEAT, 2)
            nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
        else:
            raise ValueError(mode)
        self.t = nn.Parameter(torch.tensor(1e-4))

    def theta(self):
        return self.t

    def project_(self):
        with torch.no_grad():
            self.t.clamp_(min=0.0)

    def set_identity(self):
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()

    @staticmethod
    def class_index(F):
        return (F[..., 7] + 2.0 * F[..., 4] + 4.0 * F[..., 5]).round().long()

    def step_params(self, F):
        if self.mode == "class":
            k = self.class_index(F)
            h1, h2 = self.a[k], self.b[k]
        else:
            h = self.lin(F)
            h1, h2 = h[..., 0], h[..., 1]
        return u2.S_MAX * torch.sigmoid(h1 + u2.B0), u2.M_MAX * torch.tanh(h2)

    def forward(self, A, R, Rp, F):
        s, m = self.step_params(F)
        Z = A + s * R + m * Rp
        return torch.sign(Z) * torch.clamp(torch.abs(Z) - self.theta(), min=0.0)

    def describe(self):
        if self.mode == "class":
            s = (u2.S_MAX * torch.sigmoid(self.a + u2.B0)).detach().numpy()
            m = (u2.M_MAX * torch.tanh(self.b)).detach().numpy()
            names = {0: "off,u,u", 1: "diag,u,u", 2: "off,o,u", 3: "diag,o,u", 4: "off,u,o", 5: "diag,u,o",
                     6: "off,o,o", 7: "diag,o,o"}
            return {names[k]: (float(s[k]), float(m[k])) for k in range(8)}
        return {"W": self.lin.weight.detach().numpy().tolist(), "b": self.lin.bias.detach().numpy().tolist()}


def out_dir():
    return os.path.join(cm.OUT, "classabl" + (f"_{CFG['tag']}" if CFG["tag"] else ""))


def p_model(name, arm):
    return os.path.join(out_dir(), "models", f"{name}_{arm}.pt")


def train_chain(tr, va, mode, n_layers, steps, lr, tag):
    from unrolled_em.experiments import method as me
    torch.manual_seed(0)
    layers, log = [], []
    for l in range(n_layers):
        t = time.time()
        btr, bva = me.layer_batch(tr), me.layer_batch(va)
        em = u2.Layer("scalar", me.THETA); em.set_identity()
        with torch.no_grad():
            l_em = u2.batch_loss(em, bva).item()
        layer = AltLayer(mode)
        curve = []
        best = u2.train_layer(layer, btr, bva, steps=steps, lr=lr, log=curve)
        layer.eval()
        del btr, bva
        layers.append(layer)
        tr, va = me.advance(layer, tr), me.advance(layer, va)
        entry = {"layer": l + 1, "val_em": l_em, "val_best": best, "ratio": best / l_em, "sec": time.time() - t,
                 "theta": float(layer.theta().detach()), "reject_train": me.rej_count(tr), "reject_val": me.rej_count(va),
                 "params": layer.describe(), "curve": curve}
        log.append(entry)
        print(f"  训练 {tag} {mode} 层{l + 1}: 验证 EM一步 {l_em:.4f} → {best:.4f}（比 {best / l_em:.3f}，"
              f"{time.time() - t:.0f}s）拒绝 {entry['reject_val']}", flush=True)
    return layers, log


def states(name, split):
    from unrolled_em.experiments import method as me
    cell = cm.base_cell(name)
    out = []
    for j in range(cm.N_TRVA[name][0 if split == "train" else 1]):
        z = np.load(rn.p_data(name, split, j))
        sys, y, _, _ = cm.setup(cell, split, j)
        lam = float(z["lam"])
        cands = cm.init_candidates(cell, sys, y, lam)
        d = {k: float(np.max(np.abs(z["A0"] - A))) for k, A in cands.items()}
        k0 = min(d, key=d.get)
        assert d[k0] <= 1e-6, (name, split, j, d)
        cands[k0] = z["A0"]
        out += me.make_states(sys, y, lam, z["A_R"], cands)
    return out


def train_job(args):
    name, mode = args
    f = p_model(name, mode)
    if os.path.exists(f):
        return f
    t = time.time()
    layers, log = train_chain(states(name, "train"), states(name, "val"), mode, cm.L_MAIN, CFG["steps"],
                              cm.TRAIN_LR, tag=name)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    torch.save({"mode": mode, "state": [ly.state_dict() for ly in layers]}, f)
    cm.dump({"steps": CFG["steps"], "sec": time.time() - t, "log": log}, f.replace(".pt", "_log.json"))
    print(f"训练完成 {name} {mode}：{time.time() - t:.0f}s", flush=True)
    return f


def load_alt(f):
    d = torch.load(f, weights_only=False)
    out = []
    for s in d["state"]:
        ly = AltLayer(d["mode"]); ly.load_state_dict(s); ly.eval(); out.append(ly)
    return out


def run_layers(layers, sys, y, yv, lam, A0, taus):
    from unrolled_em.experiments import method as me
    rec = cm.Recorder(sys, yv, taus[-1], "exact", taus=taus)
    log = []
    t = time.time()
    at.run(lambda: me.run_unrolled(layers, sys, y, lam, A0, max_E=cm.MAX_E, cont="tied", log=log), sys, lam, 50, rec, A0)
    rej = [x[0] for x in log]
    return {"hits": cm.hits_from_trace(rec.trace, rec.taus), "trace": rec.trace, "q": rec.q, "sec": time.time() - t,
            "reject_rate": float(np.mean([r != "ok" for r in rej])) if rej else None,
            "reject_first4": rej[:cm.L_MAIN]}


def test_job(args):
    from unrolled_em.experiments import method as me
    name, j = args
    f = os.path.join(out_dir(), "testI", f"{name}_{j:02d}.json")
    if os.path.exists(f):
        return f
    t0 = time.time()
    old = cm.load(os.path.join(cm.OUT, "testI", f"{name}_{j:02d}.json"))
    z = np.load(rn.p_ref("testI", f"{name}_{j:02d}"))
    sys, y, yv, _ = cm.setup(cm.base_cell(name), "test1c", j)
    lam, A0, taus = float(z["lam"]), z["A0"], old["taus"]
    assert abs(lam - old["lam"]) <= 1e-9 * abs(lam)
    keep = ["learned_tied_L4", "scalar_tied", "ait_t_tied", "mu_t_tied", "sec0_tied", "ait0_tied", "mu0_tied",
            "SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun"]
    out = {"cell": name, "j": j, "taus": taus, "M_R": old["M_R"],
           "stored": {k: {"hits": old["arms"][k]["hits"], "trace": old["arms"][k]["trace"]} for k in keep}}
    out["learned_rerun"] = run_layers(me.load_layers(rn.p_model(name, "learned"))[:cm.L_MAIN], sys, y, yv, lam, A0, taus)
    for mode in ARMS:
        out[mode] = run_layers(load_alt(p_model(name, mode)), sys, y, yv, lam, A0, taus)
    out["sec"] = time.time() - t0
    cm.dump(out, f)
    h = lambda v: f"{v[I11]}/{v[I101]}"
    print(f"{name} {j:02d}: learned {h(out['learned_rerun']['hits'])}（表 {h(out['stored']['learned_tied_L4']['hits'])}）"
          f"  class {h(out['class']['hits'])}  linear {h(out['linear']['hits'])}"
          f"  scalar {h(out['stored']['scalar_tied']['hits'])} | {out['sec']:.0f}s", flush=True)
    return f


FORM = ["ait_t", "mu_t", "sec0", "ait0", "mu0"]


def _excess(trace, M_R, k=3, floor=0.01):
    m = [mm for cc, mm in trace if cc <= k][-1]
    return max((m - M_R) / M_R, floor)


def summary():
    import glob
    res = {}
    v = lambda x: cm.FAIL if x is None else x
    allD = {}
    for c in CELLS:
        L = [cm.load(f) for f in sorted(glob.glob(os.path.join(out_dir(), "testI", f"{c}_*.json")))]
        if not L:
            continue
        H = lambda key, i: [(d[key]["hits"] if key in ARMS + ["learned_rerun"] else d["stored"][key]["hits"])[i] for d in L]
        r = {"n_sys": len(L),
             "repro_learned": sum(d["learned_rerun"]["hits"] == d["stored"]["learned_tied_L4"]["hits"] for d in L)}
        P = "learned_rerun"
        for i, a in ((I11, "1.1"), (I101, "1.01")):
            row = {}
            for key in ARMS + ["scalar_tied"]:
                st = cm.ratio_stats(H(key, i), H(P, i))
                row[key] = {"G": st["G"], "ci": st["ci"], "median": cm.med(H(key, i)),
                            "n_fail": sum(x is None for x in H(key, i)),
                            "wins_learned": sum(v(x) > v(y) for x, y in zip(H(key, i), H(P, i))),
                            "losses_learned": sum(v(x) < v(y) for x, y in zip(H(key, i), H(P, i)))}
            fst = {h: cm.ratio_stats(H(f"{h}_tied", i), H(P, i)) for h in FORM}
            hstar = min(sorted(FORM), key=lambda h: fst[h]["G"])
            row["best_formula"] = {"h": hstar, "G": fst[hstar]["G"], "ci": fst[hstar]["ci"]}
            cls = {k: cm.ratio_stats(H(k, i), H(P, i)) for k in ("SQ_ss_K50", "SQ_exact_K50", "DAAREM_tun")}
            ks = min(cls, key=lambda k: cls[k]["G"])
            for key in ARMS + [P]:
                g = cm.ratio_stats(H(ks, i), H(key, i))
                row.setdefault("vs_strongest", {})[key] = {"vs": ks, "G": g["G"], "ci": g["ci"]}
            row["median_learned"] = cm.med(H(P, i))
            row["class_over_linear"] = cm.ratio_stats(H("class", i), H("linear", i))
            r[a] = row
        xi = {}
        Lx = [_excess(d[P]["trace"], d["M_R"]) for d in L]
        for key in ARMS + ["scalar_tied"]:
            tr = [(d[key]["trace"] if key in ARMS else d["stored"][key]["trace"]) for d in L]
            Xx = [_excess(t, d["M_R"]) for t, d in zip(tr, L)]
            g = cm.boot_gm(np.array(Xx) / np.array(Lx))
            xi[key] = {"G": g[0], "ci": g[1:]}
        r["xi"] = xi
        r["reject_rate"] = {k: float(np.mean([d[k]["reject_rate"] for d in L if d[k]["reject_rate"] is not None]))
                            if any(d[k]["reject_rate"] is not None for d in L) else None for k in ARMS + [P]}
        r["no_layer_run"] = {k: sum(d[k]["reject_rate"] is None for d in L) for k in ARMS + [P]}
        res[c] = r
        allD[c] = L
        print(f"\n=== {c}（{len(L)} 个系统；learned 复现 {r['repro_learned']}/{len(L)}）===")
        for a in ("1.1", "1.01"):
            row = r[a]
            s = "  ".join(f"{k} {row[k]['G']:.2f} [{row[k]['ci'][0]:.2f}, {row[k]['ci'][1]:.2f}] (比 learned 快/慢的系统数 {row[k]['losses_learned']}/{row[k]['wins_learned']})"
                          for k in ARMS + ["scalar_tied"])
            bf = row["best_formula"]
            print(f"  α={a}：变体/learned 的 E 步比：{s}  最好公式({bf['h']}) {bf['G']:.2f}  | 中位 learned {row['median_learned']}, "
                  f"class {row['class']['median']}, linear {row['linear']['median']}")
            vs = row["vs_strongest"]
            print("         对最强经典的 G：" + "  ".join(f"{k} {vs[k]['G']:.2f}" for k in vs) + f"（{vs['class']['vs']}）")
        print("  ξ（三次 E 步后超额误差，变体/learned）：" + "  ".join(f"{k} {x['G']:.2f}" for k, x in xi.items()))
    for a, i in (("1.1", I11), ("1.01", I101)):
        d = []
        for c, L in allD.items():
            d += [v(x["class"]["hits"][i]) - v(x["learned_rerun"]["hits"][i]) for x in L]
        res.setdefault("pooled", {})[a] = {"sign_p_class_slower": cm.sign_test_p(d),
                                           "class_slower": int(np.sum(np.array(d) > 0)),
                                           "class_faster": int(np.sum(np.array(d) < 0)), "ties": int(np.sum(np.array(d) == 0))}
    print("\n合并：", res.get("pooled"))
    cm.dump(res, os.path.join(out_dir(), "summary.json"))
    print(f"已写 {os.path.join(out_dir(), 'summary.json')}")


def _init(cfg):
    CFG.update(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["train", "test", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--js", default=None)
    ap.add_argument("--steps", type=int, default=cm.TRAIN_STEPS)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    CFG.update(tag=a.tag, steps=a.steps)
    if a.cells:
        CELLS[:] = a.cells.split(",")
    if a.arms:
        ARMS[:] = a.arms.split(",")
    if a.part == "summary":
        summary()
        return
    if a.part == "train":
        jobs = [(c, m) for c in CELLS for m in ARMS if not os.path.exists(p_model(c, m))]
        jobs.sort(key=lambda x: -cm.base_cell(x[0])["n"])
        fn = train_job
    else:
        missing = [(c, m) for c in CELLS for m in ARMS if not os.path.exists(p_model(c, m))]
        if missing:
            raise SystemExit(f"先训练：{missing}")
        js = [int(x) for x in a.js.split(",")] if a.js else list(range(cm.N_TEST_I))
        jobs = [(c, j) for c in CELLS for j in js
                if not os.path.exists(os.path.join(out_dir(), "testI", f"{c}_{j:02d}.json"))]
        jobs.sort(key=lambda x: -cm.base_cell(x[0])["n"])
        fn = test_job
    print(f"{a.part}：待跑 {len(jobs)}，{a.workers} 个进程 → {out_dir()}", flush=True)
    if a.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers, initializer=_init, initargs=(dict(CFG),)) as pool:
            for _ in pool.imap_unordered(fn, jobs):
                pass
    else:
        for jb in jobs:
            fn(jb)


if __name__ == "__main__":
    main()
