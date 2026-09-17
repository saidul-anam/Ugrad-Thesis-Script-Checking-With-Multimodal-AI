#!/usr/bin/env bash
# Run the whole pipeline for one exam set, start to finish.
#
#   scripts/run_set.sh SE_07_Q1
#
# Every stage is resumable, so re-running after an interruption picks up where it
# stopped rather than re-spending on work already done. Grading is done in three
# passes (k=1, then 2, then 3) instead of one k=3 pass so that COVERAGE COMPLETES
# FIRST: if the run is cut short, every answer has a grade rather than the first
# third of them having three.
#
# Output goes to logs/run_<set>.log as well as stdout.
set -u

SET_ID="${1:-SE_07_Q1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
LOG="$ROOT/logs/run_${SET_ID}.log"

cd "$ROOT" || exit 1
mkdir -p logs

stage() {
  local name="$1"; shift
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $name ===" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "--- $name exited $rc" | tee -a "$LOG"
  return $rc
}

echo "=== run_set.sh $SET_ID started $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"

stage "1. convert"        "$PY" scripts/01_convert.py    --set "$SET_ID"
stage "2. extract"        "$PY" scripts/02_extract.py    --set "$SET_ID"
stage "3. validate"       "$PY" scripts/03_validate.py   --set "$SET_ID"
stage "4. build_csv"      "$PY" scripts/04_build_csv.py  --set "$SET_ID"

for K in 1 2 3; do
  stage "5. evaluate k=$K" "$PY" scripts/eval/01_evaluate.py --set "$SET_ID" --k "$K"
  # Emit after every pass so a run cut short still leaves usable CSVs.
  stage "6. emit_csv (after k=$K)" "$PY" scripts/eval/02_emit_csv.py --set "$SET_ID"
  stage "7. emit_avg (after k=$K)" "$PY" scripts/eval/04_emit_avg.py --set "$SET_ID"
done

stage "8. eval_validate"  "$PY" scripts/eval/03_validate.py --set "$SET_ID"

echo "" | tee -a "$LOG"
echo "=== run_set.sh $SET_ID finished $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"
