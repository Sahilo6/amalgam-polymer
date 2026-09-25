#!/usr/bin/env bash
# Poll the Kaggle kernel until it finishes, then pull its output and submit.
# Transient API errors are retried -- only a real terminal status ends the loop.
# Safe to re-run at any time.
set -u
KERNEL="sahilsadhwani25/amalgam-2026-pipeline"
COMP="amalgam-2026"
KAGGLE="$(cd "$(dirname "$0")" && pwd)/.venv/bin/kaggle"
OUTDIR="$(cd "$(dirname "$0")" && pwd)/kernel_out"
MSG="${1:-lgbm+xgb+cat+ridge blend | notebook: https://www.kaggle.com/code/$KERNEL}"

echo "waiting on $KERNEL (Ctrl-C is safe; just re-run)"
transient=0
while true; do
  S=$("$KAGGLE" kernels status "$KERNEL" 2>&1 | tail -1)
  case "$S" in
    *COMPLETE*)
      echo; echo "kernel COMPLETE"; break;;
    *ERROR*|*CANCEL*|*KernelWorkerStatus.FAILED*)
      echo; echo "KERNEL FAILED: $S"; echo "--- last log lines ---"
      "$KAGGLE" kernels logs "$KERNEL" 2>&1 | grep -o '"data":"[^"]*"' \
        | sed 's/^"data":"//; s/"$//' | sed 's/\\n/\n/g' | tail -25
      exit 1;;
    *RUNNING*|*QUEUED*)
      transient=0
      printf "\r  %s  running" "$(date +%H:%M:%S)"; sleep 60;;
    *)
      # Network blip, rate limit, transient 5xx -- retry rather than give up.
      transient=$((transient+1))
      printf "\r  %s  transient (%d/20): %.60s" "$(date +%H:%M:%S)" "$transient" "$S"
      if [ "$transient" -ge 20 ]; then
        echo; echo "giving up after 20 consecutive API errors: $S"; exit 1
      fi
      sleep 60;;
  esac
done

rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
"$KAGGLE" kernels output "$KERNEL" -p "$OUTDIR" >/dev/null 2>&1
F="$OUTDIR/submission.csv"
[ -f "$F" ] || { echo "no submission.csv in kernel output:"; ls -la "$OUTDIR"; exit 1; }

ROWS=$(($(wc -l < "$F") - 1))
echo "submission.csv: $ROWS rows"
[ "$ROWS" -eq 4115 ] || { echo "WRONG ROW COUNT (expected 4115) - refusing to submit"; exit 1; }
head -3 "$F"

echo "submitting ..."
"$KAGGLE" competitions submit -c "$COMP" -f "$F" -m "$MSG"
sleep 25
"$KAGGLE" competitions submissions -c "$COMP" 2>&1 | head -4
