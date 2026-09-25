# Round 3 state — updated Sept 23

**Deadline: Sept 25 for BOTH the Kaggle submission and the report.**
CAT-2 starts Sept 27. Team name: NeonNull. Competition: `amalgam-final-round`.

## On resume, run this first

    cd ~/Desktop/PROJECTS/amalgam-polymer && ./check_r3.sh

- Kernel COMPLETE + pipeline "not running" -> `./resume_r3.sh 0.9086`
- Pipeline still RUNNING -> leave it, the kernel survives laptop sleep

## Position

**1st at 0.898** (P Grade 0.886, The Polymerians 0.865).

| stage | OOF | LB |
|---|---|---|
| day 1: per-target lgbm + GNN | 0.8888 | 0.868 |
| day 2: + multi-task GNN | 0.9050 | **0.898** |
| day 3: + 3-seed multi-task | 0.9086 | ready to submit, in `kernel_out_r3/` |
| day 4: + canonical features (invariance) | 0.9086 | SHIPPING, launched 13:37 Sept 23 |

## What worked / what didn't (for the report)

| idea | effect | fate |
|---|---|---|
| multi-task GNN over the electronic cluster | **+0.0157** | shipped |
| 3-seed multi-task | +0.0036 | shipped |
| canonical atom order (invariance fix) | -0.0001 score, invariance 3% -> **exactly 0** | shipped |
| PI1M masked-atom pretraining (100k x 20ep, 99% acc) | **-0.0038** | rejected |

Multi-task worked here and FAILED in Round 1 (-0.023) for a reason worth stating:
tg and egc share no physics; these six targets are all frontier-orbital
quantities (egc-egb +0.93, eps-nc +0.92, egc-nc -0.85).

## Invariance (a scored theme) — measured, not claimed

Before: randomised SMILES of the same molecule moved predictions by up to
**3.02% of target sd (tg)** and 1.22% (egc). Three causes, one root: atom order.
Tied shortest paths round rings, Gasteiger charges on dummy atoms, and Ipc
rounding are all order-dependent, and 65% of the dataset's SMILES are not
canonical. Fix: re-parse from canonical SMILES in `features._one`, and drop Ipc
(exceeds 1e13, not order-stable). After: feature deviation and prediction
deviation both **exactly 0.00**. See `src/invariance.py`.

## Report (due Sept 25)

Six figures done and eyeballed in `report/r3/`:
fig1 target correlation · fig2 row imbalance · fig3 multi-task gain ·
fig4 pred-vs-true all 7 · fig5 invariance before/after · fig6 architecture.

Build script pattern: `report/build_report.js` (Round 1 version, docx via npm
`docx`, then `soffice --headless --convert-to pdf`). Round 1 PDF at
`report/NeonNull_Amalgam_Report.pdf` for structure and tone.

**Still to write.** Template sections: strategy/novelty (multi-task cluster +
backbone features + invariance fix), EDA (figs 1-2), results (7-target table,
fig 4), challenges/pivots (220-row targets, pretraining rejected, invariance
bug found), roadmap incl. explainability for electronic AND optical separately
(eps/nc ARE the optical targets) and the invariance number.

## User actions outstanding

1. Submit day 3 (`kernel_out_r3/submission.csv`) if not already done
2. Sept 25: pick 2 finals, confirm notebook named "NeonNull Round 3" and shared
   with `Rohit Batra IITM`, `VIJITH P`, `shreyasri0301`
3. Unstop upload if required for Round 3
