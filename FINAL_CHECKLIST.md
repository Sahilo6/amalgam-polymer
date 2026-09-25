# NeonNull — Amalgam Round 3 final checklist

Everything below is on you (Kaggle UI, email, Unstop). The model and report are done.

## 1. Kaggle: select the two final submissions

Go to the competition → My Submissions → tick two.

| pick | submission ref | what it is | honest OOF | public |
|---|---|---|---|---|
| **1** | `56540098` | 6-seed multi-task GNN, canonical features | **0.9081** | 0.899 |
| **2** | `56493708` | 3-seed multi-task GNN | 0.9073 | 0.899 |

**Choose on OOF, not on the public score.** Measured: four of seven targets have
only ~54-82 rows in the public subset, giving the public score a standard
deviation of 0.0074. The 0.003 between us and first place is 0.40 sigma, and
there is a 69% chance of a swing that size from resampling alone. Public rank
here is close to a coin flip; OOF is computed on 7,409 labelled rows.

## 2. Kaggle: notebook requirements

- Title must be the team name: **NeonNull Round 3** (already set)
- Share with **Rohit Batra IITM**, **VIJITH P**, **shreyasri0301** as Viewers
  - Share button is on the EDITOR page: kaggle.com/code/sahilsadhwani25/neonnull-round-3/edit
- Only that one notebook should be shared (Round 1 guidance from Aryan)
- Default/pinned version must be the one that produced the submission

## 3. Report

`report/r3/NeonNull_R3_Report.pdf` — 5 pages, due today 23:59.
Covers all four template sections plus explainability for electronic AND
optical applications separately, and the measured invariance result.

## 4. Unstop

Upload the final CSV and/or report if Round 3 requires it, same as Round 1.
File: `kernel_out_r3/submission.csv` (4,940 rows).

## 5. Optional: GitHub (report appendix links to it)

The appendix cites github.com/Sahilo6/amalgam-polymer. Either create that repo
or tell me and I will change the link. To create it:

    cd ~/Desktop/PROJECTS/amalgam-polymer
    DEVELOPER_DIR=/Library/Developer/CommandLineTools git init
    DEVELOPER_DIR=/Library/Developer/CommandLineTools git add -A
    DEVELOPER_DIR=/Library/Developer/CommandLineTools git commit -m "Amalgam polymer property prediction: Rounds 1-3"
    gh repo create amalgam-polymer --public --source=. --push

(.gitignore already excludes .venv, data, runs, node_modules and page images;
about 87 files, ~2 MB.)
