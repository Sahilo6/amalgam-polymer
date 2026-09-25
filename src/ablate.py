"""Forward selection over feature blocks.

Starts from the v1 baseline block set and adds one candidate block at a time,
reporting each block's marginal effect on the official metric. Ranking only —
3 folds by default for speed. Anything that looks like a winner gets re-scored
at the full fold count before it is promoted to the notebook.
"""
import argparse
import json
import time

import numpy as np

from . import config as C
from . import cv, features, models
from .data import load_raw, subset


def score_blocks(blocks, model_name="lgbm", folds=3, seed=C.SEED, X_all=None):
    """Official score for one block set, plus the per-target breakdown."""
    tr, te = load_raw()
    if X_all is None:
        X_all = features.build()
    X_sel = features.select(X_all, blocks)

    per_type, detail = {}, {}
    for tt in C.TARGET_TYPES:
        tr_t, te_t = subset(tr, tt), subset(te, tt)
        Xtr = features.for_rows(tr_t, X_sel).values
        Xte = features.for_rows(te_t, X_sel).values
        y = tr_t.target.values
        fold_idx = cv.make_folds(len(y), n_splits=folds, seed=seed)
        oof, _, s = cv.run_cv(models.make(model_name, seed=seed), Xtr, y, Xte,
                              folds=fold_idx, verbose=False)
        per_type[tt] = (y, oof)
        detail[tt] = s["r2"]
    return cv.official_score(per_type), detail, X_sel.shape[1]


def forward(candidates, base=None, model_name="lgbm", folds=3, seed=C.SEED):
    base = list(base or features.BASELINE_BLOCKS)
    X_all = features.build()

    print(f"baseline blocks: {base}")
    t0 = time.time()
    base_score, base_detail, n_base = score_blocks(base, model_name, folds, seed, X_all)
    print(f"  BASELINE  official={base_score:.4f}  "
          f"(tg={base_detail['tg']:.4f} egc={base_detail['egc']:.4f})  "
          f"{n_base} cols  {time.time()-t0:.0f}s\n")

    results = []
    for blk in candidates:
        if blk in base:
            continue
        t0 = time.time()
        s, d, ncol = score_blocks(base + [blk], model_name, folds, seed, X_all)
        delta = s - base_score
        results.append({"block": blk, "official": s, "delta": delta,
                        "tg": d["tg"], "egc": d["egc"], "n_cols": ncol})
        flag = "  <-- HELPS" if delta > 0 else ""
        print(f"  +{blk:10s} official={s:.4f}  delta={delta:+.4f}  "
              f"(tg={d['tg']:.4f} egc={d['egc']:.4f})  "
              f"{ncol} cols  {time.time()-t0:.0f}s{flag}")

    results.sort(key=lambda r: -r["delta"])
    out = C.RUNS / f"ablate_{model_name}_f{folds}.json"
    out.write_text(json.dumps(
        {"base": base, "base_score": base_score, "base_detail": base_detail,
         "results": results}, indent=2))
    print(f"\n  ranking: " + ", ".join(f"{r['block']}({r['delta']:+.4f})" for r in results))
    print(f"  -> {out}")
    return results


def greedy(order, base=None, model_name="lgbm", folds=3, seed=C.SEED, tol=0.0005):
    """Add blocks cumulatively in the given order, keeping only those that still
    help once the earlier ones are present.

    Marginal gains measured in isolation do not add up: these fingerprints encode
    overlapping structure, so a block that helps alone can be redundant next to
    another. `tol` is the minimum gain worth the extra columns.
    """
    base = list(base or features.BASELINE_BLOCKS)
    X_all = features.build()

    cur = list(base)
    best, detail, ncol = score_blocks(cur, model_name, folds, seed, X_all)
    print(f"start {cur}\n  official={best:.4f} (tg={detail['tg']:.4f} egc={detail['egc']:.4f}) {ncol} cols\n")

    kept, log = [], []
    for blk in order:
        if blk in cur:
            continue
        t0 = time.time()
        s, d, ncol = score_blocks(cur + [blk], model_name, folds, seed, X_all)
        gain = s - best
        take = gain >= tol
        log.append({"block": blk, "official": s, "gain": gain, "kept": bool(take),
                    "tg": d["tg"], "egc": d["egc"], "n_cols": ncol})
        print(f"  +{blk:10s} official={s:.4f} gain={gain:+.4f} "
              f"(tg={d['tg']:.4f} egc={d['egc']:.4f}) {ncol} cols {time.time()-t0:.0f}s"
              f"  {'KEEP' if take else 'drop (redundant)'}")
        if take:
            cur.append(blk); kept.append(blk); best = s

    print(f"\n  FINAL blocks: {cur}")
    print(f"  official={best:.4f}  (started {score_blocks(base, model_name, folds, seed, X_all)[0]:.4f})")
    out = C.RUNS / f"greedy_{model_name}_f{folds}.json"
    out.write_text(json.dumps({"final_blocks": cur, "official": best, "log": log}, indent=2))
    print(f"  -> {out}")
    return cur, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--candidates", nargs="+",
                    default=["morgan3", "avalon", "rdkfp", "atompair", "torsion"])
    ap.add_argument("--greedy", action="store_true",
                    help="cumulative forward selection instead of isolated marginals")
    a = ap.parse_args()
    if a.greedy:
        greedy(a.candidates, model_name=a.model, folds=a.folds)
    else:
        forward(a.candidates, model_name=a.model, folds=a.folds)
