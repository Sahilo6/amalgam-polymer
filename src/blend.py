"""Optimize non-negative blend weights on OOF, then write a checked submission."""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as C
from . import cv
from .data import load_raw, subset


def load_runs(names):
    """{target_type: {run: (oof, test, ytrue, test_ids)}} for the given runs."""
    store = {tt: {} for tt in C.TARGET_TYPES}
    for n in names:
        d = C.RUNS / n
        for tt in C.TARGET_TYPES:
            store[tt][n] = (
                np.load(d / f"oof_{tt}.npy"),
                np.load(d / f"test_{tt}.npy"),
                np.load(d / f"ytrue_{tt}.npy"),
                np.load(d / f"testid_{tt}.npy"),
            )
    return store


def optimize_weights(oofs, y):
    """Non-negative weights summing to 1 that minimize RMSE on OOF."""
    A = np.column_stack(oofs)
    k = A.shape[1]
    if k == 1:
        return np.ones(1)

    def obj(w):
        return np.sqrt(((A @ w - y) ** 2).mean())

    res = minimize(obj, np.full(k, 1 / k), method="SLSQP",
                   bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def build(names, desc="blend", clip=True):
    store = load_runs(names)
    tr, te = load_raw()
    pieces, report = [], {}

    for tt in C.TARGET_TYPES:
        runs = store[tt]
        y = next(iter(runs.values()))[2]
        oofs = [runs[n][0] for n in names]
        w = optimize_weights(oofs, y)

        blend_oof = np.column_stack(oofs) @ w
        blend_test = np.column_stack([runs[n][1] for n in names]) @ w
        ids = next(iter(runs.values()))[3]

        if clip:
            lo, hi = tr[tr.target_type == tt].target.min(), tr[tr.target_type == tt].target.max()
            blend_test = np.clip(blend_test, lo, hi)

        s = cv.score(y, blend_oof)
        report[tt] = {"weights": dict(zip(names, np.round(w, 4).tolist())), **s}
        print(f"  {tt:4s} blend rmse={s['rmse']:8.4f} r2={s['r2']:.4f}  "
              f"weights={ {n: round(float(x),3) for n, x in zip(names, w)} }")
        pieces.append(pd.DataFrame({"id": ids, "target": blend_test}))
        report[tt]["_oof"] = (y, blend_oof)

    per_type = {tt: report[tt].pop("_oof") for tt in C.TARGET_TYPES}
    report["official"] = cv.official_score(per_type)
    print(f"  OFFICIAL SCORE (mean R2 over targets) = {report['official']:.4f}")

    sub = pd.concat(pieces).sort_values("id").reset_index(drop=True)
    validate(sub, te)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = C.SUBS / f"sub_{stamp}_{desc}.csv"
    sub.to_csv(path, index=False)
    (C.SUBS / f"sub_{stamp}_{desc}.json").write_text(
        json.dumps({"runs": names, **report}, indent=2, default=float))
    print(f"  wrote {path}  ({len(sub)} rows)")
    return path, report


def validate(sub, te):
    """Fail loudly rather than submitting a malformed file."""
    ss = pd.read_csv(C.SAMPLE_SUB)
    assert list(sub.columns) == list(ss.columns), f"columns {list(sub.columns)} != {list(ss.columns)}"
    assert len(sub) == len(te), f"{len(sub)} rows but test has {len(te)}"
    assert sub.id.is_unique, "duplicate ids"
    assert set(sub.id) == set(te.id), "id set does not match test.csv"
    assert np.isfinite(sub.target).all(), "non-finite predictions"
    print("  validation: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--desc", default="blend")
    ap.add_argument("--no-clip", action="store_true")
    a = ap.parse_args()
    build(a.runs, desc=a.desc, clip=not a.no_clip)
