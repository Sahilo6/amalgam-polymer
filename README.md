# Amalgam 2026 — Polymer Property Prediction

Kaggle: https://www.kaggle.com/competitions/amalgam-2026
**Deadline: 15 September 2026.** 3 submissions/day, 2 final selections.

## Metric (confirmed from the Evaluation tab)

    Score = (R2_Tg + R2_Egc) / 2

R2 is computed per target and averaged, so **the two properties carry equal
weight** despite tg having ~70x the spread of egc. A 0.01 R2 gain on egc is
worth exactly as much as 0.01 on tg. This is why models are trained and blended
separately per target: there is never a reason to trade one against the other.

## Rules that constrain the approach

This is a **notebook-only** competition. The constraints are unusually strict
and several of them rule out standard Kaggle tactics:

- **6.2.2** Every stage — loading, preprocessing, train/val split, training,
  inference, submission generation — must run inside a single Kaggle notebook
  execution. Prediction-only CSV uploads are invalidated even if they score.
- **6.2.1 / 6.2.4** No external data. No pretrained weights, checkpoints or
  embeddings. No uploading processed features or cached artifacts. **This rules
  out ChemBERTa and any other pretrained molecular model.** Anything neural has
  to train from scratch inside the notebook.
- **7.1** The submission description must link the notebook; its default/pinned
  version must be exactly the one that produced the score; it must be shared
  (view access) with `Rohit Batra IITM`, `VIJITH P`, `shreyasri0301`.
- **7.2** Hosts re-run the pinned notebook after the deadline. It must complete
  within Kaggle's limits and reproduce the submitted score, or the submission is
  invalidated. All seeds must be set explicitly.
- Output file must be named `submission.csv`.

`kaggle_notebook/amalgam_2026_pipeline.ipynb` is the compliant artifact. The
`src/` package is for fast local experimentation only — its findings reach the
leaderboard as hyperparameters copied into the notebook, never as uploaded
files.

## The data

Long format: one row = one (polymer, property) pair.

| | rows | unique SMILES |
|---|---|---|
| train | 6,171 | 6,158 |
| test | 4,115 | 4,111 |

| type | meaning | train rows | mean | std | range |
|---|---|---|---|---|---|
| `tg` | glass transition temp (°C) | 4,143 | 140.1 | 109.4 | -118 → 490 |
| `egc` | chain band gap (eV) | 2,028 | 4.53 | 1.56 | 0.10 → 9.86 |

Test is split 1,543 public / 2,572 private. All 10,264 SMILES parse under RDKit;
every one is a repeat unit with exactly two `*` attachment points. Train/test
SMILES overlap is 5 molecules and one exact (smiles, target_type) pair — no
leakage worth exploiting.

## Features (2,606 after dropping constants)

Computed once per unique SMILES, then looked up per row:

- RDKit descriptors on the raw SMILES (217)
- RDKit descriptors on a `*`→carbon capped copy (217) — dummy atoms make
  Gasteiger-charge and BCUT descriptors return NaN, and capping recovers them
- Morgan count fingerprints, radius 2, 2048 bits
- MACCS keys (167)

## Local pipeline

    src/config.py    paths, target types, CV scheme
    src/data.py      loading + per-type subsetting
    src/features.py  SMILES -> features, cached to parquet
    src/cv.py        folds, metrics, official_score, OOF loop
    src/models.py    lgbm / xgb / cat / ridge behind one interface
    src/train.py     CLI: train one model on both targets, save OOF + test preds
    src/tune.py      per-target random search
    src/blend.py     OOF-fitted weights -> validated submission

    .venv/bin/python -m src.features            # build cache (~40s)
    .venv/bin/python -m src.train --model lgbm  # 5-fold OOF
    .venv/bin/python -m src.train --board       # scoreboard
    .venv/bin/python -m src.tune --model lgbm --trials 20
    .venv/bin/python -m src.blend --runs lgbm xgb cat --desc gbdt3

## OOF vs leaderboard calibration

Every submission's pair, tracked to catch CV/LB divergence early.

| date | config | naive OOF | honest OOF | public LB | gap |
|---|---|---|---|---|---|
| Sep 5 | lgbm only (INVALID: CLI upload) | 0.8962 | - | 0.888 | 0.008 |
| Sep 7 | lgbm+xgb+cat+ridge, v1 features | 0.9061 | 0.9055 | 0.890 | 0.0155 |
| Sep 8 | same model resubmitted (stale kernel, wasted) | 0.9061 | 0.9055 | 0.890 | - |
| Sep 8 | + avalon + polymer backbone | 0.9131 | 0.9126 | 0.898 | 0.0146 |
| Sep 9 | + 10-fold, ridge dropped | 0.9163 | 0.9160 | 0.902 | 0.0143 |
| Sep 10 | + backbone descriptors | 0.9168 | 0.9165 | 0.903 | 0.0135 |

**The gap is a stable ~0.015 offset, and deltas track 1:1**: OOF +0.0071
produced LB +0.008. So OOF improvements do reach the board undiscounted, and
OOF remains the thing to optimize (6,171 rows vs the public board's 1,543).

Public-LB noise, bootstrapped at 37% subset size, is std 0.0056 with a 0.019-wide
90% band. Never promote on a board move alone; a +0.005 change is one sigma of
nothing.

Blend-weight overfitting was measured with nested evaluation (`src/honest.py`)
and accounts for only -0.0007 of the gap, so the difference is a level shift
between our CV and the public subset, not a scoring flaw. Optimized blend
weights beat uniform averaging by ~0.015 honestly measured, so they stay.

## Scores (5-fold OOF, official metric)

| run | R2_tg | R2_egc | official |
|---|---|---|---|
| mean predictor | 0.000 | 0.000 | 0.000 |
| lgbm | 0.8901 | 0.9022 | 0.8962 |
| xgb | 0.8926 | 0.9064 | 0.8995 |
| cat | 0.9002 | 0.9043 | 0.9023 |
| ridge (all blocks) | 0.4635 | 0.6109 | 0.5372 |
| **Day 1 blend** | 0.9018 | 0.9105 | **0.9061** |
| lgbm + avalon | 0.8972 | 0.9085 | 0.9029 |
| xgb + avalon | 0.8991 | 0.9105 | 0.9048 |
| cat + avalon | 0.9026 | 0.9066 | 0.9046 |
| ridge, descriptors only | 0.8189 | 0.8228 | 0.8208 |
| **Day 2 blend** | 0.9039 | 0.9128 | **0.9084** |

OOF 0.8962 vs public LB 0.888 on the lgbm run — CV tracks the leaderboard, so
local iteration is trustworthy.

## Harness sanity check

A mean predictor pushed through the same CV scores R2 = 0.000 and RMSE equal to
the target standard deviation on both targets. If that ever stops being true,
the metric or the fold logic has broken.
