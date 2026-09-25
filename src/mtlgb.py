"""Multi-task LightGBM over the electronic cluster.

The same idea as the multi-task GNN but for the tree path: stack every
electronic row into one table, standardise each target, add a one-hot task
indicator, and fit a single model. Cheap (minutes), and it reads the shared
physics through the feature space rather than a learned encoder, so its errors
should differ from the GNN's.

Folds are kept per target, matching the single-task runs, so the OOF vector
lines up for blending.
"""
import argparse

import numpy as np

from . import config as C
from . import cv, features, models
from .data import load_raw, subset

BLOCKS = ["desc_raw", "desc_cap", "morgan2", "maccs", "avalon", "polymer", "decomp_bb"]


def run(targets, model_name="lgbm", n_folds=5, seed=C.SEED, tag="mtlgb"):
    tr, _ = load_raw()
    X_all = features.select(features.build(), BLOCKS)

    blocks, meta = [], []
    for ti, t in enumerate(targets):
        d = subset(tr, t)
        X = features.for_rows(d, X_all).values.astype(np.float32)
        ind = np.zeros((len(d), len(targets)), np.float32); ind[:, ti] = 1.0
        blocks.append(np.hstack([X, ind]))
        mu, sd = d.target.mean(), d.target.std()
        fold = np.empty(len(d), int)
        for k, (_, va) in enumerate(cv.make_folds(len(d), n_splits=n_folds, seed=seed)):
            fold[va] = k
        for i, y in enumerate(d.target.values):
            meta.append((t, i, (y - mu) / sd, y, fold[i], mu, sd))
    X = np.vstack(blocks)
    yz = np.array([m[2] for m in meta]); fold = np.array([m[4] for m in meta])
    print(f"  {X.shape[0]} rows x {X.shape[1]} cols ({len(targets)} task indicators)")

    oof = np.zeros(len(meta))
    for k in range(n_folds):
        tr_i, va_i = np.where(fold != k)[0], np.where(fold == k)[0]
        m = models.make(model_name, seed=seed)()
        m.fit_fold(X[tr_i], yz[tr_i], X[va_i], yz[va_i])
        oof[va_i] = m.predict(X[va_i])

    res = {}
    for t in targets:
        idx = [i for i, m in enumerate(meta) if m[0] == t]
        order = np.array([meta[i][1] for i in idx])
        pred = np.empty(len(idx)); pred[order] = np.array([oof[i] * meta[i][6] + meta[i][5] for i in idx])
        ytrue = np.empty(len(idx)); ytrue[order] = np.array([meta[i][3] for i in idx])
        res[t] = cv.score(ytrue, pred)["r2"]
        np.save(C.RUNS / f"_{tag}_oof_{t}.npy", pred)
        print(f"  {t:4s} multi-task lgbm r2 = {res[t]:.4f}")
    print(f"\n  mean over cluster = {np.mean(list(res.values())):.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--tag", default="mtlgb")
    a = ap.parse_args()
    print(f"=== multi-task {a.model} over {C.ELECTRONIC} ===")
    run(C.ELECTRONIC, model_name=a.model, tag=a.tag)
