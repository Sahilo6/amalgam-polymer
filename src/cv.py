"""Fold builder, metrics, and the out-of-fold training loop.

OFFICIAL METRIC (confirmed from the competition Evaluation tab):

    Score = (R2_Tg + R2_Egc) / 2

R2 is computed separately per target and the two are averaged, so the two
properties carry EQUAL weight despite tg having ~70x the spread of egc.
Effort is worth splitting evenly between them; a gain of 0.01 R2 on egc is
worth exactly as much as 0.01 R2 on tg.
"""
import numpy as np
from sklearn.model_selection import KFold

from . import config as C


def make_folds(n_rows, n_splits=C.N_FOLDS, seed=C.SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_rows)))


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def official_score(per_type):
    """Competition metric: mean of per-target R2, from {type: (y_true, y_pred)}."""
    r2s = [score(yt, yp)["r2"] for yt, yp in per_type.values()]
    return float(np.mean(r2s))


def pooled_rmse(per_type):
    """Pooled RMSE over all rows. Diagnostic only - not the leaderboard metric."""
    sq, n = 0.0, 0
    for y_true, y_pred in per_type.values():
        e = np.asarray(y_pred, float) - np.asarray(y_true, float)
        sq += float((e ** 2).sum())
        n += len(e)
    return float(np.sqrt(sq / n))


def run_cv(model_fn, X, y, X_test, folds=None, verbose=True):
    """Train one model across folds. Returns (oof, test_pred, scores).

    model_fn() must return a fresh estimator exposing
    .fit_fold(X_tr, y_tr, X_va, y_va) and .predict(X).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    X_test = np.asarray(X_test, dtype=np.float32)
    if folds is None:
        folds = make_folds(len(y))

    oof = np.zeros(len(y))
    test_preds = np.zeros((len(folds), len(X_test)))

    for i, (tr_idx, va_idx) in enumerate(folds):
        m = model_fn()
        m.fit_fold(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx])
        oof[va_idx] = m.predict(X[va_idx])
        test_preds[i] = m.predict(X_test)
        if verbose:
            f = score(y[va_idx], oof[va_idx])
            print(f"  fold {i}: rmse={f['rmse']:.4f} mae={f['mae']:.4f} r2={f['r2']:.4f}")

    s = score(y, oof)
    if verbose:
        print(f"  OOF   : rmse={s['rmse']:.4f} mae={s['mae']:.4f} r2={s['r2']:.4f}")
    return oof, test_preds.mean(axis=0), s
