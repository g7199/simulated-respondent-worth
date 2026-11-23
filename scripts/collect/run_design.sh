#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_design.sh <model> : fielding designs (greedy and random choice of the questions to field) for one model
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; m=$1
for d in "greedy half" "random half 1" "random half 2" "greedy 10" "random 10 1" "random 10 2"; do $PY $S/w_design.py collect $m $d >> $L/design_$m.log 2>&1; echo "DONE $d" >> $L/design_$m.log; done
echo "DESIGN_DONE" >> $L/design_$m.log
