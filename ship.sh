#!/usr/bin/env bash
# Local notebook -> verified, submission-ready file on Kaggle.
#
# Every gate exists because something went wrong without it:
#   staleness  - Sept 7 and Sept 8 submitted byte-identical stale kernel output
#   pre-flight - two 2-hour runs died on a missing rdkit / wrong data path
#   OOF gate   - proves the kernel ran what we validated locally
#   file gate  - a malformed file would waste one of only 3 daily submissions
#
# Usage: ./ship.sh <expected_oof> ["submission message"]
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
KAGGLE=".venv/bin/kaggle"
KERNEL="sahilsadhwani25/amalgam-2026-pipeline"
PREFLIGHT="sahilsadhwani25/amalgam-preflight"
COMP="amalgam-2026"
NB="kaggle_notebook/amalgam_2026_pipeline.ipynb"
EXPECTED="${1:?usage: ./ship.sh <expected_oof> [message]}"
MSG="${2:-polymer+avalon blend | notebook: https://www.kaggle.com/code/$KERNEL}"

die() { echo; echo "ABORT: $*" >&2; exit 1; }

wait_for() {  # $1 = kernel slug
  local s transient=0
  while true; do
    s=$("$KAGGLE" kernels status "$1" 2>&1 | tail -1)
    case "$s" in
      *COMPLETE*) echo; return 0;;
      *ERROR*|*CANCEL*) echo; echo "--- last log lines ---"
        "$KAGGLE" kernels logs "$1" 2>&1 | grep -o '"data":"[^"]*"' \
          | sed 's/^"data":"//; s/"$//' | sed 's/\\n/\n/g' | tail -30
        return 1;;
      *RUNNING*|*QUEUED*) transient=0; printf "\r  %s running" "$(date +%H:%M:%S)"; sleep 45;;
      *) transient=$((transient+1)); [ "$transient" -ge 20 ] && { echo; echo "$s"; return 1; }
         printf "\r  %s transient(%d)" "$(date +%H:%M:%S)" "$transient"; sleep 45;;
    esac
  done
}

logs_of() { "$KAGGLE" kernels logs "$1" 2>&1 | grep -o '"data":"[^"]*"' \
            | sed 's/^"data":"//; s/"$//' | sed 's/\\n/\n/g'; }

# ---- gate 1: staleness -------------------------------------------------
echo "[1/5] staleness check"
rm -rf .shipchk && mkdir -p .shipchk
"$KAGGLE" kernels pull "$KERNEL" -p .shipchk >/dev/null 2>&1 || true
DEPLOYED=$(ls .shipchk/*.ipynb 2>/dev/null | head -1)
if [ -n "$DEPLOYED" ] && python3 - "$NB" "$DEPLOYED" <<'PY'
import json,sys,hashlib
def norm(p):
    nb=json.load(open(p))
    src="".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
    return hashlib.sha256(src.encode()).hexdigest()
sys.exit(0 if norm(sys.argv[1])==norm(sys.argv[2]) else 1)
PY
then
  die "deployed kernel is IDENTICAL to local notebook. Nothing new to ship.
       (This is exactly the failure that wasted Sept 7-8.)"
fi
echo "      local notebook differs from deployed - proceeding"

# ---- gate 2: pre-flight ------------------------------------------------
echo "[2/5] pre-flight (~3 min)"
python3 - <<'PY'
import json
nb=json.load(open('kaggle_notebook/amalgam_2026_pipeline.ipynb'))
code=[c for c in nb['cells'] if c['cell_type']=='code']
keep=[c for c in code if 'fit_blend_weights' not in ''.join(c['source'])
      and 'submission =' not in ''.join(c['source'])
      and 'Parallel(' not in ''.join(c['source'])]
extra={"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
  ("smi = pd.concat([train.smiles, test.smiles]).unique()[:25]\n"
   "out = [featurize_one(s) for s in smi]\n"
   "print('featurized', len(smi), 'failures:', sum(o is None for o in out),\n"
   "      'features:', len(out[0][0]))\n"
   "print('PRE-FLIGHT OK')\n").splitlines(keepends=True)}
json.dump({"cells":keep+[extra],"metadata":nb["metadata"],"nbformat":4,"nbformat_minor":5},
          open('kernel_preflight/preflight.ipynb','w'),indent=1)
PY
"$KAGGLE" kernels push -p kernel_preflight >/dev/null 2>&1 || die "pre-flight push failed"
wait_for "$PREFLIGHT" || die "pre-flight FAILED - fix before spending 2 hours"
logs_of "$PREFLIGHT" | grep -q "PRE-FLIGHT OK" || die "pre-flight did not report OK"
echo "      pre-flight OK"

# ---- gate 3: push the real kernel --------------------------------------
echo "[3/5] pushing pipeline kernel (run takes ~2h)"
cp "$NB" kernel_push/amalgam-2026-pipeline.ipynb
"$KAGGLE" kernels push -p kernel_push >/dev/null 2>&1 || die "kernel push failed"
wait_for "$KERNEL" || die "kernel run FAILED"

# ---- gate 4: OOF matches what we validated locally ----------------------
echo "[4/5] OOF gate (expected $EXPECTED)"
ACTUAL=$(logs_of "$KERNEL" | grep -oE "OFFICIAL OOF SCORE = [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
[ -n "$ACTUAL" ] || die "could not find OFFICIAL OOF SCORE in kernel logs"
python3 -c "
import sys
a,e=float('$ACTUAL'),float('$EXPECTED')
print(f'      kernel OOF={a:.4f}  local OOF={e:.4f}  diff={a-e:+.4f}')
sys.exit(0 if abs(a-e)<=0.002 else 1)" \
  || die "kernel OOF differs from local by >0.002 - the notebook is not running what we validated"

# ---- gate 5: the file itself -------------------------------------------
echo "[5/5] file gate"
rm -rf kernel_out && mkdir -p kernel_out
"$KAGGLE" kernels output "$KERNEL" -p kernel_out >/dev/null 2>&1
F=kernel_out/submission.csv
[ -f "$F" ] || die "no submission.csv in kernel output"
python3 - "$F" data/raw/test.csv <<'PY' || exit 1
import pandas as pd, numpy as np, sys
s=pd.read_csv(sys.argv[1]); t=pd.read_csv(sys.argv[2])
assert list(s.columns)==["id","target"], f"columns {list(s.columns)}"
assert len(s)==len(t)==4115, f"{len(s)} rows"
assert s.id.is_unique and set(s.id)==set(t.id), "id mismatch"
assert np.isfinite(s.target).all(), "non-finite predictions"
print(f"      {len(s)} rows, ids match, all finite")
PY

echo
echo "================= READY TO SUBMIT ================="
echo "kernel OOF $ACTUAL  |  file verified"
echo
echo "  cd $(pwd) && $KAGGLE competitions submit -c $COMP \\"
echo "    -f kernel_out/submission.csv -m \"$MSG\""
echo "==================================================="
