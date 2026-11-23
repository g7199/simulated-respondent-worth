#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_rel.sh <queue> <workers> <model> [<model> ...] : other-pollsters context experiment, plain then ctx
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; q=$1; w=$2; shift 2
for m in "$@"; do for mode in plain ctx; do $PY $S/w_rel.py collect $m $mode $w >> $L/rl_${q}.log 2>&1; echo "DONE $m $mode" >> $L/rl_${q}.log; done; done
echo "RL_DONE" >> $L/rl_${q}.log
