"""Does 10-fold or seed-averaging actually beat 5-fold?

OOF scores at different fold counts are NOT comparable: 10-fold OOF is higher
than 5-fold OOF simply because each model trains on 90% of the data instead of
80%. Comparing them directly manufactures a gain that does not exist at test
time.

So this measures the only thing that matters: accuracy on data no fold ever
saw. Hold out 20%, fit each strategy on the remaining 80% exactly as the
notebook would, predict the holdout by averaging the fold models, and score.
Repeated over several holdout splits so the answer is not one lucky partition.
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import KFold

from . import config as C
from . import cv, features, models
from .data import load_raw, subset

BLOCKS = ["desc_raw", "desc_cap", "morgan2", "maccs", "avalon", "polymer"]


def fit_predict(model_name, X, y, X_hold, n_folds, seeds):
    """Train with the given fold count over one or more seeds; average the
    holdout predictions exactly as the notebook averages test predictions."""
    preds = []
    for seed in seeds:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_i, va_i in kf.split(X):
            m = models.make(model_name, seed=seed)()
            m.fit_fold(X[tr_i], y[tr_i], X[va_i], y[va_i])
            preds.append(m.predict(X_hold))
    return np.mean(preds, axis=0)


def study(model_name="lgbm", n_repeats=3, hold_frac=0.2):
    tr, _ = load_raw()
    X_all = features.select(features.build(), BLOCKS)

    strategies = {
        "5fold_1seed": (5, [C.SEED]),
        "10fold_1seed": (10, [C.SEED]),
        "5fold_3seed": (5, [C.SEED, C.SEED + 1, C.SEED + 2]),
    }
    results = {k: {t: [] for t in C.TARGET_TYPES} for k in strategies}

    for rep in range(n_repeats):
        rng = np.random.default_rng(1000 + rep)
        print(f"\n--- holdout repeat {rep + 1}/{n_repeats} ---")
        for tt in C.TARGET_TYPES:
            tr_t = subset(tr, tt)
            X = features.for_rows(tr_t, X_all).values.astype(np.float32)
            y = tr_t.target.values
            idx = rng.permutation(len(y))
            n_hold = int(len(y) * hold_frac)
            hold, dev = idx[:n_hold], idx[n_hold:]

            for name, (folds, seeds) in strategies.items():
                p = fit_predict(model_name, X[dev], y[dev], X[hold], folds, seeds)
                r2 = cv.score(y[hold], p)["r2"]
                results[name][tt].append(r2)
                print(f"  {tt:4s} {name:14s} holdout r2 = {r2:.4f}")

    print(f"\n=== {model_name}: mean holdout R2 over {n_repeats} splits ===")
    summary = {}
    for name in strategies:
        per = {t: float(np.mean(results[name][t])) for t in C.TARGET_TYPES}
        official = float(np.mean(list(per.values())))
        summary[name] = {"official": official, **per,
                         "std": {t: float(np.std(results[name][t])) for t in C.TARGET_TYPES}}
        print(f"  {name:14s} official={official:.4f}  "
              f"(tg={per['tg']:.4f} egc={per['egc']:.4f})")

    base = summary["5fold_1seed"]["official"]
    print(f"\n  vs 5fold_1seed baseline:")
    for name in strategies:
        if name == "5fold_1seed":
            continue
        d = summary[name]["official"] - base
        print(f"    {name:14s} {d:+.4f}  {'WORTH IT' if d > 0.001 else 'not worth the runtime'}")

    out = C.RUNS / f"foldstudy_{model_name}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  -> {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    study(a.model, n_repeats=a.repeats)
