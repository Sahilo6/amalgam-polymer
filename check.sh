#!/usr/bin/env bash
# One-command status + decision. Read-only: safe to run any number of times.
cd "$(cd "$(dirname "$0")" && pwd)"
V=.venv/bin/python
echo "=============== AMALGAM STATUS  $(date '+%H:%M %b %d') ==============="

echo
echo "--- local experiment (10-fold GNN) ---"
if pgrep -f "[s]rc.gnn" >/dev/null; then
  done_folds=$(grep -cE "seed[0-9] fold" runs/gnn_f10.log 2>/dev/null || echo 0)
  echo "  RUNNING  ($done_folds of 40 fold-runs done)"
else
  grep -E "tg: GNN|egc: GNN|official" runs/gnn_f10.log 2>/dev/null | grep -v Warning \
    || echo "  not started / no log"
fi

echo
echo "--- decision ---"
$V - <<'PY' 2>/dev/null || echo "  (waiting for the run to finish)"
import numpy as np, sys
from pathlib import Path
from scipy.optimize import minimize
from src import config as C, cv
tag="gnn10f10s2"
if not (C.RUNS/f"_{tag}_oof_tg.npy").exists(): sys.exit(1)
def load(n,tt):
    return np.load(C.RUNS/f"_{n}_oof_{tt}.npy") if n.startswith("gnn") \
           else np.load(C.RUNS/n/f"oof_{tt}.npy")
def fitw(A,y):
    k=A.shape[1]
    o=lambda w: np.sqrt(((A@w-y)**2).mean())
    r=minimize(o,np.full(k,1/k),method="SLSQP",bounds=[(0,1)]*k,
               constraints=[{"type":"eq","fun":lambda w:w.sum()-1}])
    w=np.clip(r.x,0,None); return w/w.sum()
def score(names,nested=True):
    per={}
    for tt in C.TARGET_TYPES:
        A=np.column_stack([load(n,tt) for n in names])
        y=np.load(C.RUNS/"lgbm_d10"/f"ytrue_{tt}.npy")
        if nested:
            p=np.zeros(len(y))
            for tri,vai in cv.make_folds(len(y),n_splits=5): p[vai]=A[vai]@fitw(A[tri],y[tri])
        else:
            p=A@fitw(A,y)
        per[tt]=(y,p)
    return cv.official_score(per)
h=score(["lgbm_d10",tag]); nv=score(["lgbm_d10",tag],nested=False)
print(f"  shipped config  honest=0.9232")
print(f"  10-fold GNN     honest={h:.4f}  (naive {nv:.4f} = the OOF-gate value)")
print()
if h>0.9240:
    print(f"  >>> SHIP IT. Rebuild the notebook, then:  ./ship_fast.sh {nv:.4f}")
elif h>0.9235:
    print(f"  >>> MARGINAL ({h:.4f}). Ship only if it is before 18:00.")
else:
    print(f"  >>> DO NOT SHIP. Keep the 0.911 submission; a 6h re-run adds risk, not score.")
PY

echo
echo "--- kaggle ---"
.venv/bin/kaggle competitions leaderboard -c amalgam-2026 --show 2>&1 | head -4 | sed 's/^/  /'
echo
for k in amalgam-2026-fast amalgam-2026-pipeline; do
  printf "  %-24s " "$k"
  .venv/bin/kaggle kernels status sahilsadhwani25/$k 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*' || echo "?"
done

echo
echo "--- YOUR CHECKLIST (not automatable) ---"
echo "  [ ] rule 7.1: share BOTH notebooks with Rohit Batra IITM / VIJITH P / shreyasri0301"
echo "  [ ] select finals 56205719 (fast) + 56166656 (pipeline) on the submissions page"
echo "  [ ] Unstop upload of kernel_out_fast/submission.csv"
echo "======================================================================"
