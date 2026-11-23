#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_more.sh <queue> <model> [...] : earlier wave in the prompt, and a second random split for the same-survey context
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; q=$1; shift 1
for m in "$@"; do $PY $S/w_wave.py collect $m >> $L/more_${q}.log 2>&1; W_SEED=12 $PY $S/w_groups.py collect $m ctx 100000 3 >> $L/more_${q}.log 2>&1; echo "DONE $m" >> $L/more_${q}.log; done
echo "MORE_DONE" >> $L/more_${q}.log
