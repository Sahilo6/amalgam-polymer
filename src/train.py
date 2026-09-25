"""Train one model across both property types; save OOF + test predictions.

Each run writes runs/<name>/ containing per-type oof/test arrays, the row
ids they correspond to, and a scores.json. blend.py consumes these; nothing
is ever retrained just to blend.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from . import cv, features, models
from .data import load_raw, subset


def run(model_name, run_name=None, seed=C.SEED, n_folds=C.N_FOLDS,
        blocks=None, **model_kw):
    run_name = run_name or model_name
    out = C.RUNS / run_name
    out.mkdir(parents=True, exist_ok=True)

    tr, te = load_raw()
    X_all = features.build()
    if blocks is not None:
        X_all = features.select(X_all, blocks)
    scores, per_type = {}, {}
    t0 = time.time()

    for tt in C.TARGET_TYPES:
        tr_t, te_t = subset(tr, tt), subset(te, tt)
        Xtr = features.for_rows(tr_t, X_all).values
        Xte = features.for_rows(te_t, X_all).values
        y = tr_t.target.values

        print(f"\n[{run_name}] {tt}: train={Xtr.shape} test={Xte.shape}")
        folds = cv.make_folds(len(y), n_splits=n_folds, seed=seed)
        oof, test_pred, s = cv.run_cv(models.make(model_name, seed=seed, **model_kw),
                                      Xtr, y, Xte, folds=folds)

        np.save(out / f"oof_{tt}.npy", oof)
        np.save(out / f"test_{tt}.npy", test_pred)
        np.save(out / f"ytrue_{tt}.npy", y)
        np.save(out / f"testid_{tt}.npy", te_t.id.values)
        scores[tt] = s
        per_type[tt] = (y, oof)

    scores["official"] = cv.official_score(per_type)
    scores["pooled_rmse"] = cv.pooled_rmse(per_type)
    scores["_meta"] = dict(model=model_name, seed=seed, n_folds=n_folds,
                           n_features=X_all.shape[1], blocks=blocks,
                           secs=round(time.time() - t0, 1), params=model_kw)
    (out / "scores.json").write_text(json.dumps(scores, indent=2))

    print(f"\n=== {run_name} ===")
    for tt in C.TARGET_TYPES:
        s = scores[tt]
        print(f"  {tt:4s} rmse={s['rmse']:8.4f}  mae={s['mae']:8.4f}  r2={s['r2']:.4f}")
    print(f"  OFFICIAL SCORE (mean R2 over targets) = {scores['official']:.4f}")
    print(f"  ({scores['_meta']['secs']}s)")
    return scores


def leaderboard():
    """Every run's scores, best pooled RMSE first."""
    rows = []
    for d in sorted(C.RUNS.iterdir()):
        f = d / "scores.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        rows.append({"run": d.name, "official": s.get("official"),
                     **{f"{t}_rmse": s[t]["rmse"] for t in C.TARGET_TYPES if t in s},
                     **{f"{t}_r2": s[t]["r2"] for t in C.TARGET_TYPES if t in s}})
    return pd.DataFrame(rows).sort_values("official", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm", choices=list(models.REGISTRY))
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--blocks", nargs="+", default=None,
                    help="feature blocks to use (default: all)")
    ap.add_argument("--board", action="store_true", help="print run scoreboard and exit")
    a = ap.parse_args()
    if a.board:
        print(leaderboard().to_string(index=False))
    else:
        run(a.model, run_name=a.name, seed=a.seed, n_folds=a.folds, blocks=a.blocks)
