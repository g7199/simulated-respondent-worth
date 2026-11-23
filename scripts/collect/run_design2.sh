#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_design2.sh <model> : more fielding designs at ten questions per report (three more random draws, one-per-battery heuristic, the simulator's own choice)
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; m=$1
for d in "random 10 3" "random 10 4" "random 10 5" "battery 10 1" "battery 10 2" "llm 10"; do $PY $S/w_design.py collect $m $d >> $L/design2_$m.log 2>&1; echo "DONE $d" >> $L/design2_$m.log; done
echo "DESIGN2_DONE" >> $L/design2_$m.log
