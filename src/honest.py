"""Nested evaluation of blending.

Blend weights fitted on OOF predictions and then scored on those same OOF
predictions are optimistically biased: the weights saw the answers. This splits
the OOF rows into outer folds, fits weights on K-1 of them and scores the held
-out fold, which estimates what blending actually buys at test time.

Compares three strategies so we can pick one that generalizes rather than one
that flatters the local number.
"""
import argparse

import numpy as np
from scipy.optimize import minimize

from . import config as C
from . import cv


def _fit_weights(A, y):
    k = A.shape[1]
    if k == 1:
        return np.ones(1)
    obj = lambda w: np.sqrt(((A @ w - y) ** 2).mean())
    r = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0.0, 1.0)] * k,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def evaluate(runs, n_outer=5, seed=C.SEED):
    """Return {strategy: official score} under honest nested evaluation."""
    strategies = ["optimized", "uniform", "best_single"]
    per_type = {s: {} for s in strategies}
    naive = {}

    for tt in C.TARGET_TYPES:
        oofs = np.column_stack([np.load(C.RUNS / r / f"oof_{tt}.npy") for r in runs])
        y = np.load(C.RUNS / runs[0] / f"ytrue_{tt}.npy")

        # the number we normally report: weights fitted and scored on all rows
        naive[tt] = cv.score(y, oofs @ _fit_weights(oofs, y))["r2"]

        preds = {s: np.zeros(len(y)) for s in strategies}
        for tr_idx, va_idx in cv.make_folds(len(y), n_splits=n_outer, seed=seed):
            w = _fit_weights(oofs[tr_idx], y[tr_idx])
            preds["optimized"][va_idx] = oofs[va_idx] @ w
            preds["uniform"][va_idx] = oofs[va_idx].mean(axis=1)
            best = int(np.argmin([np.sqrt(((oofs[tr_idx, j] - y[tr_idx]) ** 2).mean())
                                  for j in range(oofs.shape[1])]))
            preds["best_single"][va_idx] = oofs[va_idx, best]

        for s in strategies:
            per_type[s][tt] = (y, preds[s])

    print(f"runs: {runs}")
    print(f"\n  naive (weights fitted AND scored on all OOF rows):")
    print(f"    official = {np.mean([naive[t] for t in C.TARGET_TYPES]):.4f}   <- what we have been reporting")
    print(f"\n  honest ({n_outer}-fold nested, weights never see the rows they score):")
    for s in strategies:
        sc = cv.official_score(per_type[s])
        detail = {t: cv.score(*per_type[s][t])["r2"] for t in C.TARGET_TYPES}
        print(f"    {s:12s} official = {sc:.4f}  (tg={detail['tg']:.4f} egc={detail['egc']:.4f})")
    return per_type


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--outer", type=int, default=5)
    a = ap.parse_args()
    evaluate(a.runs, n_outer=a.outer)
