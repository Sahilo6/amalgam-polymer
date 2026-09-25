"""Does cutting features help? 3,857 of them against 2,028 egc rows is a lot.

Feature selection must happen INSIDE the evaluation split or the number is
meaningless: picking features on all the data and then scoring on a subset of
it is the same selection bias that inflates our OOF. So each repeat holds out
20%, ranks features using only the other 80%, trains on that 80% with the top-N,
and scores the untouched holdout.
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import KFold

from . import config as C
from . import cv, features, models
from .data import load_raw, subset

BLOCKS = ["desc_raw", "desc_cap", "morgan2", "maccs", "avalon", "polymer", "decomp_bb"]


def rank_features(X, y, seed):
    """Importance ranking from a quick model fit on the dev split only."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          colsample_bytree=0.5, verbose=-1, n_jobs=-1, random_state=seed)
    m.fit(X, y)
    return np.argsort(-m.feature_importances_)


def fit_predict(X, y, X_hold, cols, n_folds, seed):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = []
    for tr_i, va_i in kf.split(X):
        m = models.make("lgbm", seed=seed)()
        m.fit_fold(X[tr_i][:, cols], y[tr_i], X[va_i][:, cols], y[va_i])
        preds.append(m.predict(X_hold[:, cols]))
    return np.mean(preds, axis=0)


def study(top_ns=(200, 500, 1000, 2000), n_repeats=3, n_folds=5, hold_frac=0.2):
    tr, _ = load_raw()
    X_all = features.select(features.build(), BLOCKS)
    results = {n: {t: [] for t in C.TARGET_TYPES} for n in (0,) + tuple(top_ns)}

    for rep in range(n_repeats):
        rng = np.random.default_rng(2000 + rep)
        print(f"\n--- repeat {rep+1}/{n_repeats} ---")
        for tt in C.TARGET_TYPES:
            tr_t = subset(tr, tt)
            X = features.for_rows(tr_t, X_all).values.astype(np.float32)
            y = tr_t.target.values
            idx = rng.permutation(len(y))
            n_hold = int(len(y) * hold_frac)
            hold, dev = idx[:n_hold], idx[n_hold:]

            order = rank_features(np.nan_to_num(X[dev]), y[dev], C.SEED)
            allc = np.arange(X.shape[1])
            for n in (0,) + tuple(top_ns):
                cols = allc if n == 0 else order[:n]
                p = fit_predict(X[dev], y[dev], X[hold], cols, n_folds, C.SEED)
                r2 = cv.score(y[hold], p)["r2"]
                results[n][tt].append(r2)
                label = "all" if n == 0 else f"top{n}"
                print(f"  {tt:4s} {label:7s} ({len(cols):4d} cols) holdout r2 = {r2:.4f}")

    print(f"\n=== mean holdout R2 over {n_repeats} splits ===")
    base = None
    for n in (0,) + tuple(top_ns):
        per = {t: float(np.mean(results[n][t])) for t in C.TARGET_TYPES}
        off = float(np.mean(list(per.values())))
        if n == 0:
            base = off
        label = "all features" if n == 0 else f"top {n}"
        d = "" if n == 0 else f"  delta={off-base:+.4f}  {'BETTER' if off-base>0.001 else ''}"
        print(f"  {label:14s} official={off:.4f}  (tg={per['tg']:.4f} egc={per['egc']:.4f}){d}")
    (C.RUNS / "prune.json").write_text(json.dumps(
        {str(k): {t: results[k][t] for t in C.TARGET_TYPES} for k in results}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    study(n_repeats=a.repeats)
