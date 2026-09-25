#!/usr/bin/env bash
# One-command status. Read-only, safe to run any time.
cd "$(cd "$(dirname "$0")" && pwd)"
K=.venv/bin/kaggle
echo "=========== NEONNULL ROUND 3  $(date '+%H:%M %b %d') ==========="
echo "  DEADLINE TODAY: Kaggle submission + report"
echo

echo "--- last experiment (10-fold multi-task) ---"
if pgrep -f "[s]rc.mtgnn" >/dev/null; then
  N=$(grep -cE "seed[0-9] fold" runs/r3/mtgnn10f.log 2>/dev/null || echo 0)
  echo "  RUNNING  $N of 30 fold-runs  (elapsed $(ps -eo etime,command | grep '[s]rc.mtgnn' | head -1 | awk '{print $1}'))"
  echo "  beats shipped model only if honest OOF > 0.9085"
elif grep -q "mean over cluster" runs/r3/mtgnn10f.log 2>/dev/null; then
  grep -E "multi-task r2|mean over" runs/r3/mtgnn10f.log | sed 's/^/  /'
else
  echo "  not running"
fi

LOG=$(ls -t runs/r3/ship*.log runs/r3/resume*.log 2>/dev/null | head -1)
echo
echo "--- ship pipeline ---"
if pgrep -f "[s]hip_r3.sh|[r]esume_r3.sh" >/dev/null; then
  echo "  RUNNING for $(ps -eo etime,command | grep -E '[s]hip_r3.sh|[r]esume_r3.sh' | head -1 | awk '{print $1}')"
elif grep -q "READY TO SUBMIT" "$LOG" 2>/dev/null; then
  echo "  ready (already submitted — nothing pending)"
else
  echo "  idle"
fi

echo
echo "--- leaderboard ---"
$K competitions leaderboard -c amalgam-final-round --show 2>&1 | head -5 | sed 's/^/  /'
echo "  NOTE: public noise std = 0.0074. The 0.003 gap is 0.40 sigma — not a real difference."

echo
echo "--- model (frozen) ---"
echo "  honest OOF 0.9081  |  invariance measured exactly 0  |  19 experiments, 4 kept"

echo
echo "=============== YOUR REMAINING CHECKLIST ==============="
echo "  [ ] 1. Kaggle: select finals 56540098 + 56493708  (pick on OOF, not public)"
echo "  [ ] 2. Kaggle: share notebook with Rohit Batra IITM / VIJITH P / shreyasri0301"
echo "         -> kaggle.com/code/sahilsadhwani25/neonnull-round-3/edit  → Share"
echo "  [ ] 3. Submit report: report/r3/NeonNull_R3_Report.pdf  (due 23:59 today)"
echo "  [ ] 4. Unstop upload if Round 3 requires it"
echo "  [ ] 5. DECIDE: create github.com/Sahilo6/amalgam-polymer, or I remove"
echo "         that link from the report appendix (it is cited but does not exist)"
echo
echo "  full detail: cat FINAL_CHECKLIST.md"
echo "======================================================="
