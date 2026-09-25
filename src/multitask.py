"""Does `egc` benefit from seeing `tg` data?

`egc` has 2,028 training rows but carries half the metric; `tg` has 4,143. The
two targets are physically different but share a structure->property mapping, so
a model trained on both might learn representations `egc` alone cannot support.

Design, chosen so the result stays comparable to the per-target models:
  - keep each target's OWN fold split (same seed, same K)
  - augment each training fold with ALL rows of the other target
  - standardise each target separately first, since tg (140 +/- 109) and egc
    (4.5 +/- 1.6) differ by ~70x in scale and a shared model must see one scale
  - add a task-indicator column so the model can still separate them
  - drop the 7 molecules carrying both labels from the augmenting set, so a
    validation molecule can never appear in training under its other label
"""
import argparse

import numpy as np

from . import config as C
from . import cv, features, models
from .data import load_raw, subset

BLOCKS = ["desc_raw", "desc_cap", "morgan2", "maccs", "avalon", "polymer"]


def run(target, model_name="lgbm", n_folds=10, seed=C.SEED):
    other = [t for t in C.TARGET_TYPES if t != target][0]
    tr, _ = load_raw()
    X_all = features.select(features.build(), BLOCKS)

    t_rows, o_rows = subset(tr, target), subset(tr, other)
    Xt = features.for_rows(t_rows, X_all).values.astype(np.float32)
    Xo = features.for_rows(o_rows, X_all).values.astype(np.float32)
    yt_raw, yo_raw = t_rows.target.values, o_rows.target.values

    mu_t, sd_t = yt_raw.mean(), yt_raw.std()
    mu_o, sd_o = yo_raw.mean(), yo_raw.std()
    yt, yo = (yt_raw - mu_t) / sd_t, (yo_raw - mu_o) / sd_o

    # task indicator: 0 = this target, 1 = the other
    Xt = np.hstack([Xt, np.zeros((len(Xt), 1), np.float32)])
    Xo = np.hstack([Xo, np.ones((len(Xo), 1), np.float32)])

    shared = set(t_rows.smiles) & set(o_rows.smiles)
    keep_o = ~o_rows.smiles.isin(shared).values
    Xo, yo = Xo[keep_o], yo[keep_o]
    print(f"  {target}: {len(Xt)} own rows + {len(Xo)} borrowed from {other} "
          f"({len(shared)} shared molecules excluded)")

    oof = np.zeros(len(yt))
    for i, (tr_i, va_i) in enumerate(cv.make_folds(len(yt), n_splits=n_folds, seed=seed)):
        X_fit = np.vstack([Xt[tr_i], Xo])
        y_fit = np.concatenate([yt[tr_i], yo])
        m = models.make(model_name, seed=seed)()
        m.fit_fold(X_fit, y_fit, Xt[va_i], yt[va_i])
        oof[va_i] = m.predict(Xt[va_i])

    pred = oof * sd_t + mu_t          # back to original units
    s = cv.score(yt_raw, pred)
    print(f"  {target}: multitask r2 = {s['r2']:.4f}")
    np.save(C.RUNS / f"_mt_{model_name}_oof_{target}.npy", pred)
    return s["r2"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    print(f"=== multi-task ({a.model}, {a.folds} folds) ===")
    res = {t: run(t, a.model, a.folds) for t in C.TARGET_TYPES}
    print(f"\n  official = {np.mean(list(res.values())):.4f}")
    print("\n  single-task baselines (same folds, same features):")
    import json
    for t in C.TARGET_TYPES:
        base = json.loads((C.RUNS / f"{a.model}_f10" / "scores.json").read_text())[t]["r2"]
        d = res[t] - base
        print(f"    {t:4s} single={base:.4f}  multi={res[t]:.4f}  delta={d:+.4f}"
              f"  {'HELPS' if d > 0.001 else 'no gain'}")
