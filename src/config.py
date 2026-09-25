"""Single source of truth for paths, targets and CV scheme.

ROUND=1 (default) is the original two-target competition; ROUND=3 is the finale
(seven targets, slug amalgam-final-round). Every path keys off it, so the two
rounds never share caches, runs or submissions.
"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUND = os.environ.get("ROUND", "1")
_SUB = "" if ROUND == "1" else f"r{ROUND}"

RAW = ROOT / "data" / _SUB / "raw" if _SUB else ROOT / "data" / "raw"
FEAT = ROOT / "data" / _SUB / "features" if _SUB else ROOT / "data" / "features"
RUNS = ROOT / "runs" / _SUB if _SUB else ROOT / "runs"
SUBS = ROOT / "submissions" / _SUB if _SUB else ROOT / "submissions"
for _d in (FEAT, RUNS, SUBS):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW / "train.csv"
TEST_CSV = RAW / "test.csv"
SAMPLE_SUB = RAW / "sample_submission.csv"
COMPETITION = "amalgam-final-round" if ROUND == "3" else "amalgam-2026"


def _discover_targets():
    """Long-format data: one row = one (polymer, property) pair. Targets are
    read from the file so the same code serves both rounds."""
    if not TRAIN_CSV.exists():
        return ["tg", "egc"]
    seen = []
    with open(TRAIN_CSV) as fh:
        for row in csv.DictReader(fh):
            if row["target_type"] not in seen:
                seen.append(row["target_type"])
    return sorted(seen)


TARGET_TYPES = _discover_targets()

# The six electronic/optical targets form one physical cluster (frontier-orbital
# quantities; egc-egb +0.93, eps-nc +0.92). tg is thermal and stands alone.
ELECTRONIC = [t for t in ["egc", "egb", "ei", "eea", "eps", "nc"] if t in TARGET_TYPES]

SEED = 42
N_FOLDS = 5 if ROUND == "3" else 10   # 10-fold on 220-row targets = 22-row folds
