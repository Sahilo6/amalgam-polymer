#!/usr/bin/env bash
# Resume the gates for an ALREADY-RUNNING fast kernel. Does not push, so it
# cannot restart the run. Safe to re-run as often as needed.
# Tolerates long network outages: a dropped wifi killed the original watcher
# after 20 retries while the Kaggle run continued unaffected.
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
KAGGLE=".venv/bin/kaggle"
KERNEL="sahilsadhwani25/neonnull-round-3"
COMP="amalgam-final-round"
EXPECTED="${1:?usage: ./resume_fast.sh <expected_oof> [message]}"
MSG="${2:-lgbm + 3-seed GNN ensemble | notebook: https://www.kaggle.com/code/$KERNEL}"
die() { echo; echo "ABORT: $*" >&2; exit 1; }
logs_of() { "$KAGGLE" kernels logs "$KERNEL" 2>&1 | grep -o '"data":"[^"]*"' \
            | sed 's/^"data":"//; s/"$//' | sed 's/\\n/\n/g'; }

echo "waiting on $KERNEL (tolerates network outages up to ~2h)"
t=0
while true; do
  s=$("$KAGGLE" kernels status "$KERNEL" 2>&1 | tail -1)
  case "$s" in
    *KernelWorkerStatus.COMPLETE*) echo; echo "COMPLETE"; break;;
    *KernelWorkerStatus.ERROR*|*KernelWorkerStatus.CANCEL*)
        echo; echo "kernel FAILED"; logs_of | tail -25; exit 1;;
    *KernelWorkerStatus.RUNNING*|*KernelWorkerStatus.QUEUED*)
        t=0; printf "\r  %s running" "$(date +%H:%M:%S)"; sleep 45;;
    *)  t=$((t+1))
        [ "$t" -ge 160 ] && die "160 consecutive API failures (~2h). Kernel may still be fine: re-run this script."
        printf "\r  %s offline/transient (%d)" "$(date +%H:%M:%S)" "$t"; sleep 45;;
  esac
done

echo "[OOF gate] expected $EXPECTED"
A=$(logs_of | grep -oE "OFFICIAL OOF SCORE = [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
[ -n "$A" ] || die "no OFFICIAL OOF SCORE in logs"
python3 -c "
import sys; a,e=float('$A'),float('$EXPECTED')
print(f'      kernel OOF={a:.4f}  local OOF={e:.4f}  diff={a-e:+.4f}')
sys.exit(0 if abs(a-e)<=0.006 else 1)" || die "OOF mismatch >0.003"

logs_of | grep -oE "Wall time: [^\\\\]*" | tail -2

rm -rf kernel_out_r3 && mkdir -p kernel_out_r3
"$KAGGLE" kernels output "$KERNEL" -p kernel_out_r3 >/dev/null 2>&1
F=kernel_out_r3/submission.csv
[ -f "$F" ] || die "no submission.csv in output"
python3 - "$F" data/r3/raw/test.csv <<'PY' || exit 1
import pandas as pd, numpy as np, sys
s=pd.read_csv(sys.argv[1]); t=pd.read_csv(sys.argv[2])
assert list(s.columns)==["id","target"] and len(s)==len(t)==4940
assert s.id.is_unique and set(s.id)==set(t.id) and np.isfinite(s.target).all()
print(f"      {len(s)} rows, ids match, all finite")
PY
echo
echo "================= READY TO SUBMIT ================="
echo "  cd $(pwd) && $KAGGLE competitions submit -c $COMP \\"
echo "    -f kernel_out_r3/submission.csv -m \"$MSG\""
echo "==================================================="
