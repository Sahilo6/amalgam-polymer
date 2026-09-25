#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
K="sahilsadhwani25/amalgam-env-probe"
./.venv/bin/kaggle kernels push -p kernel_probe
echo "waiting for probe (should take ~1 min) ..."
while true; do
  S=$(./.venv/bin/kaggle kernels status "$K" 2>&1 | tail -1)
  case "$S" in
    *RUNNING*|*QUEUED*) printf "."; sleep 20;;
    *) echo; echo "$S"; break;;
  esac
done
echo "=== PROBE OUTPUT ==="
./.venv/bin/kaggle kernels logs "$K" 2>&1 \
  | grep -o '"data":"[^"]*"' | sed 's/^"data":"//; s/"$//' | sed 's/\\n$//' | sed 's/\\n/\n/g'
