"""Per-target random search.

tg (4143 rows) and egc (2028 rows) are different-sized problems scored with
equal weight, so they get independently tuned hyperparameters rather than one
shared config. Search uses 3 folds for speed; the winner is re-scored at the
full fold count before being trusted.
"""
import argparse
import json
import time

import numpy as np

from . import config as C
from . import cv, features, models
from .data import load_raw, subset

SPACES = {
    "lgbm": {
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [15, 31, 63, 127, 255],
        "min_child_samples": [5, 10, 20, 40],
        "colsample_bytree": [0.3, 0.4, 0.6, 0.8],
        "subsample": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0, 20.0],
        "reg_alpha": [0.0, 0.5, 2.0],
    },
    "xgb": {
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 10],
        "colsample_bytree": [0.3, 0.5, 0.7],
        "subsample": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1.0, 5.0, 20.0],
    },
}


def sample(space, rng):
    """One random config. Values stay plain Python scalars so they can be
    copied straight into the notebook and serialized to JSON."""
    return {k: v[int(rng.integers(len(v)))] for k, v in space.items()}


def search(model_name, target_type, n_trials=25, folds=3, seed=C.SEED):
    rng = np.random.default_rng(seed)
    tr, te = load_raw()
    X_all = features.build()
    tr_t, te_t = subset(tr, target_type), subset(te, target_type)
    X = features.for_rows(tr_t, X_all).values
    Xte = features.for_rows(te_t, X_all).values
    y = tr_t.target.values
    fold_idx = cv.make_folds(len(y), n_splits=folds, seed=seed)

    print(f"\n### tuning {model_name} / {target_type}  ({X.shape[0]} rows, {n_trials} trials, {folds} folds)")
    results = []
    base, _, base_s = cv.run_cv(models.make(model_name, seed=seed), X, y, Xte,
                                folds=fold_idx, verbose=False)
    print(f"  baseline r2 = {base_s['r2']:.4f}")
    results.append({"params": {}, "r2": base_s["r2"]})

    for i in range(n_trials):
        p = sample(SPACES[model_name], rng)
        t0 = time.time()
        try:
            _, _, s = cv.run_cv(models.make(model_name, seed=seed, **p), X, y, Xte,
                                folds=fold_idx, verbose=False)
        except Exception as e:
            print(f"  trial {i:2d}: FAILED ({type(e).__name__})")
            continue
        results.append({"params": p, "r2": s["r2"]})
        best = max(r["r2"] for r in results)
        flag = " <- best" if s["r2"] >= best else ""
        print(f"  trial {i:2d}: r2={s['r2']:.4f} ({time.time()-t0:.0f}s){flag}")

    results.sort(key=lambda r: -r["r2"])
    out = C.RUNS / f"tune_{model_name}_{target_type}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  BEST r2={results[0]['r2']:.4f}  params={results[0]['params']}")
    print(f"  -> {out}")
    return results[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm", choices=list(SPACES))
    ap.add_argument("--target", default=None, choices=C.TARGET_TYPES)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--folds", type=int, default=3)
    a = ap.parse_args()
    targets = [a.target] if a.target else C.TARGET_TYPES
    for t in targets:
        search(a.model, t, n_trials=a.trials, folds=a.folds)
