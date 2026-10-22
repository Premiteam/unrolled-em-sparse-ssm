from __future__ import annotations

import numpy as np

from unrolled_em import core as ls


class Hit(Exception):
    pass


class Tracker:
    def __init__(self, sys, lam, K, stop, record=False):
        self.sys, self.lam, self.K, self.stop = sys, lam, K, stop
        self.c = 0
        self.best = None
        self.hit = None
        self.reports = [] if record else None

    def _report(self, x):
        if self.reports is not None:
            self.reports.append((self.c, x.copy()))
        if self.stop(x, self.c):
            self.hit = self.c
            raise Hit

    def start(self, A0):
        self._report(A0)

    def on_estep(self, A, st):
        self.c += 1
        F = -st["ll"] + self.lam * float(np.sum(np.abs(A)))
        if np.isfinite(F) and (self.best is None or F <= self.best[0]):
            self.best = (F, A.copy(), st)
        if self.best is None:
            return
        _, Ab, sb = self.best
        self._report(ls.mstep(Ab, sb["Phi"], sb["Psi"], self.sys.q, self.lam, K=self.K))

    def on_fail(self):
        self.c += 1


def run(algo, sys, lam, K, stop, A0, record=False):
    tr = Tracker(sys, lam, K, stop, record)
    o_ex, o_ss = ls.estep_exact, ls.estep_ss

    def wrap(fn):
        def w(A, *a, **k):
            try:
                st = fn(A, *a, **k)
            except np.linalg.LinAlgError:
                tr.on_fail()
                raise
            tr.on_estep(A, st)
            return st
        return w
    ls.estep_exact, ls.estep_ss = wrap(o_ex), wrap(o_ss)
    try:
        tr.start(A0)
        algo()
    except Hit:
        pass
    finally:
        ls.estep_exact, ls.estep_ss = o_ex, o_ss
    return tr.hit, tr
