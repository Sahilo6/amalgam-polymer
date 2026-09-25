#!/usr/bin/env bash
# Same five gates as ship.sh, pointed at the separate fast notebook.
# Two notebooks rather than two versions of one: rule 7.1 requires the pinned
# version to be the one that produced the submitted result, and only one version
# per notebook can be pinned.
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
KAGGLE=".venv/bin/kaggle"
KERNEL="sahilsadhwani25/amalgam-2026-fast"
COMP="amalgam-2026"
EXPECTED="${1:?usage: ./ship_fast.sh <expected_oof> [message]}"
MSG="${2:-lgbm+GNN fast config | notebook: https://www.kaggle.com/code/$KERNEL}"
die() { echo; echo "ABORT: $*" >&2; exit 1; }
logs_of() { "$KAGGLE" kernels logs "$1" 2>&1 | grep -o '"data":"[^"]*"' \
            | sed 's/^"data":"//; s/"$//' | sed 's/\\n/\n/g'; }
wait_for() {
  local s t=0
  while true; do
    s=$("$KAGGLE" kernels status "$1" 2>&1 | tail -1)
    case "$s" in
      *COMPLETE*) echo; return 0;;
      *ERROR*|*CANCEL*) echo; echo "--- last log lines ---"; logs_of "$1" | tail -30; return 1;;
      *RUNNING*|*QUEUED*) t=0; printf "\r  %s running" "$(date +%H:%M:%S)"; sleep 45;;
      *) t=$((t+1)); [ "$t" -ge 20 ] && { echo; echo "$s"; return 1; }
         printf "\r  %s transient(%d)" "$(date +%H:%M:%S)" "$t"; sleep 45;;
    esac
  done
}

echo "[1/4] pushing fast kernel (~30 min expected)"
PUSH=$("$KAGGLE" kernels push -p kernel_push_fast 2>&1) || die "push failed: $PUSH"
echo "$PUSH" | grep -q "successfully pushed" || die "push did not confirm: $PUSH"
wait_for "$KERNEL" || die "kernel run FAILED"

echo "[2/4] OOF gate (expected $EXPECTED)"
ACTUAL=$(logs_of "$KERNEL" | grep -oE "OFFICIAL OOF SCORE = [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
[ -n "$ACTUAL" ] || die "no OFFICIAL OOF SCORE in logs"
python3 -c "
import sys; a,e=float('$ACTUAL'),float('$EXPECTED')
print(f'      kernel OOF={a:.4f}  local OOF={e:.4f}  diff={a-e:+.4f}')
sys.exit(0 if abs(a-e)<=0.003 else 1)" || die "OOF mismatch >0.003"

echo "[3/4] runtime check"
logs_of "$KERNEL" | grep -oE "Wall time: [^\\\\]*" | tail -2

echo "[4/4] file gate"
rm -rf kernel_out_fast && mkdir -p kernel_out_fast
"$KAGGLE" kernels output "$KERNEL" -p kernel_out_fast >/dev/null 2>&1
F=kernel_out_fast/submission.csv
[ -f "$F" ] || die "no submission.csv"
python3 - "$F" data/raw/test.csv <<'PY' || exit 1
import pandas as pd, numpy as np, sys
s=pd.read_csv(sys.argv[1]); t=pd.read_csv(sys.argv[2])
assert list(s.columns)==["id","target"] and len(s)==len(t)==4115
assert s.id.is_unique and set(s.id)==set(t.id) and np.isfinite(s.target).all()
print(f"      {len(s)} rows, ids match, all finite")
PY
echo
echo "================= READY TO SUBMIT (fast) ================="
echo "  cd $(pwd) && $KAGGLE competitions submit -c $COMP \\"
echo "    -f kernel_out_fast/submission.csv -m \"$MSG\""
echo "=========================================================="
