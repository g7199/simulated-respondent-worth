#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_design_gpt.sh <model> : fielding designs at ten questions per report through the Codex client (greedy, three random draws, the model's own choice)
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; m=$1
for d in "greedy 10" "random 10 1" "random 10 2" "random 10 3" "llm 10"; do $PY $S/w_design.py collect $m $d >> $L/design_$m.log 2>&1; echo "DONE $d" >> $L/design_$m.log; done
echo "DESIGN_DONE" >> $L/design_$m.log
