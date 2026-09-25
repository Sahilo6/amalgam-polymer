# Current state — updated Sept 9, 2026

Read this first on resuming. Full plan:
`~/.claude/plans/https-www-kaggle-com-competitions-amalga-cosmic-wolf.md`

## Position
- **1st-equal on public LB at 0.898** (tied Team Vizer, who win the tiebreak by
  submitting earlier). Deadline **Sept 15**. 3 submissions/day.
- Best local model: **0.9126 honest OOF** (nested), shipped Sept 8.
- Public LB noise is std 0.0056 (bootstrapped at 37% subset). The 0.008 spread
  across the top is 1.4 sigma. **Optimize OOF, never the board.**
- OOF->LB: stable ~0.015 offset, deltas track ~1:1.

## Shipped feature set (keep `features.BASELINE_BLOCKS` in sync)
`desc_raw, desc_cap, morgan2, maccs, avalon, polymer` — 3648 cols.
The 18 `polymer` backbone features were the biggest single win (+0.0056),
larger than 1024 Avalon bits.

## ON "continue" — DO THIS

**Sept 14, deadline Sept 15. 2nd at 0.911, leader 0.916.**

### 1. Check the 10-fold GNN (the only live experiment)

    grep -E "tg: GNN|egc: GNN|official" runs/gnn_f10.log | grep -v Warning
    pgrep -f "[s]rc.gnn" && echo RUNNING || echo done

Baseline to beat: 5-fold 3-seed plain = **0.9139 solo / 0.9232 blend**.
OOF lands at `runs/_gnn10f10s2_oof_<target>.npy` (tag f10s2, so 10 folds 2 seeds).

Measure the blend honestly (never naive OOF):

    .venv/bin/python -m src.honest --runs lgbm_d10 <gnn tag>
    # or the inline nested-blend snippet used all week

### 2. Decision rule (already agreed)

- **If blend > 0.9240**: worth shipping. Rebuild
  `kaggle_notebook/amalgam_2026_fast.ipynb` with GNN_FOLDS=10 and 2 seeds,
  parity-check GNN code vs `src/gnn.py`, then `./ship_fast.sh <naive_oof>`.
  Est ~6h on Kaggle. MUST start by ~18:00 Sept 14 to be safe.
- **If blend <= 0.9235**: DO NOT SHIP. The current 0.911 submission is already
  our best feasible config; a pointless 6h re-run only adds risk on the last day.

**If the ship watcher dies (wifi/sleep): use `./resume_fast.sh <oof>`.
It does NOT re-push. Running `ship_fast.sh` again would restart a 6h run.**

### 3. Rejected, do not revisit

- `lgbm + plain3 + polymer3` = 0.9240 honest but ~9h on Kaggle (limit is 9h).
  Not worth the risk of a killed run for +0.0008.
- Bigger GNN (h256/l4) = +0.0018 solo but **82h on Kaggle**. Infeasible.
- GNN hyperparameters: hidden192 +0.0000, bs32 +0.0001, lr2e-3 -0.0003,
  dropout0 -0.0005, lr5e-4 -0.0016. All flat. Local optimum.
- Blend weighting: simplex / NNLS / intercept / ridge stacking all = 0.9247.
  Maxed out.

### 4. Final checklist regardless of shipping

| slot | ref | notebook | config | OOF | LB |
|---|---|---|---|---|---|
| 1 | `56205719` | amalgam-2026-fast | lgbm + 3-seed GNN | 0.9233 | 0.911 |
| 2 | `56166656` | amalgam-2026-pipeline | 4-model + 1-seed GNN | 0.9221 | 0.910 |

Two finals must come from DIFFERENT notebooks (rule 7.1: one pinned version
each). Both kernels are already pinned to exactly these runs.

**User must do:** (a) rule 7.1 sharing with `Rohit Batra IITM`, `VIJITH P`,
`shreyasri0301` — STILL UNRESOLVED, biggest risk; (b) explicitly select the two
finals on Kaggle; (c) **Unstop upload** — mandatory on both platforms.

## Where we stand (Sept 13 end)

**2nd at 0.911.** "Snacks, Vibes, Perplexity" leads with 0.916. Deadline Sept 15.

We need ~0.928 OOF to reach 0.916. Every combination tops out at **0.9243**
(4-model + all three GNN variants), projecting to ~0.912. Seed returns have
collapsed: 1->3 seeds +0.0020, 3->6 seeds +0.0006. Realistic outcome is 2nd on
the public board.

The private set is a different 63% and our deficit is under one standard
deviation of board noise (0.0056), so the final standing is genuinely open.
Maximising honest OOF and finishing VALID is worth more than chasing 0.005.

| config | honest OOF | Kaggle wall time |
|---|---|---|
| lgbm + 1-seed GNN (scored 0.911) | 0.9214 | 2h 3min |
| 4-model + 1-seed GNN (scored 0.910) | 0.9218 | ~5h |
| lgbm + 3-seed GNN | 0.9231 | ~4.5h est |
| 4-model + 3-seed GNN | 0.9235 | ~6h est |

**To beat 0.914 we need OOF >= 0.9259** (recent OOF-LB gap 0.0119). Best current
config projects to ~0.912, so seed ensembling alone does not get us there.

**Multi-seed GNN is the win of Sept 12**: GNN solo 0.9066 -> 0.9151 with 3 seeds,
blend 0.9211 -> 0.9231. Note this is NOT the same as GBDT seed averaging, which
measured +0.0000 - neural nets have real initialisation variance.

Kaggle runs ~5.6x slower than this machine. Do not estimate runtime from local
timings; I got it wrong twice (predicted 3.5h got 5h, predicted 25min got 2h).

## Latest results (Sept 9)

10-fold sweep complete. Blend `lgbm_f10 + xgb_f10 + cat_f10` = **0.9163 naive,
0.9160 honest**. Notebook updated: `N_FOLDS = 10`, Ridge removed.

**Do not read 0.9131 -> 0.9163 as +0.0032.** 10-fold OOF is inflated vs 5-fold
because each model trains on 90% of rows instead of 80%. The honest gain,
measured on held-out data, is **+0.0019**.

**Ridge was dropped for a real reason**: at 10 folds one ill-conditioned tg fold
scored R2 = -53 and predicted -17080 (true range -118..490). Nine folds were
fine at ~0.83. The submission CSV is range-clipped so the output was safe, but
blend weights are fitted on OOF and would have been corrupted. Removing it
scores the same (0.9160 vs 0.9158 honest) and deletes the failure mode.

## Measured DEAD ends — do not retry
| idea | result |
|---|---|
| hyperparameter tuning (20 trials) | +0.005 tg, **+0.000 egc** |
| morgan3 / rdkfp / atompair / torsion | redundant once avalon present |
| dimer RDKit descriptors | -0.0010 |
| conjugation features (10) | +0.0002 despite -0.757 corr with Egc |
| multi-seed averaging (3 seeds) | **+0.0000** on honest holdout |
| ExtraTrees / SVR / KernelRidge in blend | **+0.0000** each |
| multi-task (share tg<->egc rows) | **-0.0079 tg, -0.0234 egc** |
| pseudo-labelling test rows | -0.0002 to -0.0023 |
| feature pruning (top 200-2000) | 0 to -0.0103 |
| bare backbone / largest sidechain / dimer backbone | -0.0005 to -0.0014 |
| polymer-aware GNN (backbone + edge features) | **+0.0000 in blend** |

Conclusion: **the blend has saturated; the bottleneck is features, not models.**
A strong correlation with the target does NOT mean incremental model value.

## Only lever confirmed alive
- **10-fold CV: +0.0019** measured on held-out data (`src/foldstudy.py`).

## The pattern: only polymer-aware decomposition works

I called the ceiling after multi-task failed. That was premature - the correct
conclusion was narrower. Eight hypotheses:

- generic ML (tuning, seed averaging, extra model families, multi-task): all ~0
- generic chemistry (extra fingerprints, dimer descriptors, conjugation): all ~0
- **polymer decomposition: +0.0056 (backbone scalars), +0.0025 (backbone
  descriptors)** - two for two

**Anything describing the chain AS A CHAIN pays. Anything treating these as
ordinary molecules does not.** Plan new ideas from that family first.

Untried in the winning family: backbone with side chains capped as H; largest
side chain described separately; backbone descriptors of the DIMER's backbone.

**Remaining effort goes to protecting the position, not chasing +0.002.**
At a 0.004 lead with +/-0.0056 board noise, disqualification is a far bigger
risk to winning than a missing fraction of a point.

1. **Reproducibility** (rule 7.2): hosts re-run the pinned notebook and void the
   submission if it does not reproduce. Two clean reproductions so far
   (diff 0.0002 and 0.0001). Do a dedicated fresh-kernel dry run.
2. **Host sharing** (rule 7.1): notebook must be shared with `Rohit Batra IITM`,
   `VIJITH P`, `shreyasri0301`, and pinned to the scoring version. UI-only, the
   user must do it. NOT YET CONFIRMED DONE.
3. **Final selection** (Sept 14): 2 finals chosen on OOF, not board rank.
4. **Unstop upload** (Sept 15): mandatory on BOTH platforms or the entry is void.

## Hard rules (competition, verified from the Rules page)
- Notebook-only. A CLI CSV upload scores but is INVALID.
- No external data, no pretrained weights (rules out ChemBERTa etc).
- Notebook must be linked in the submission description, pinned to the scoring
  version, shared with `Rohit Batra IITM`, `VIJITH P`, `shreyasri0301`.
- Hosts re-run the pinned notebook; if it does not reproduce, submission void.
- Sept 13 = reproducibility dry run. Sept 14 = pick 2 finals on OOF, not board
  rank (private is 2572 rows vs 1543 public). Sept 15 = submit early + Unstop.
