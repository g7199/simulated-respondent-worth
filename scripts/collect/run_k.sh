#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_k.sh <queue> <workers> <model> [...] : same-survey context with only K = 3 or 10 real answers shown
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; q=$1; w=$2; shift 2
for m in "$@"; do for k in 3 10; do W_K=$k $PY $S/w_groups.py collect $m ctx 100000 $w >> $L/k_${q}.log 2>&1; echo "DONE $m $k" >> $L/k_${q}.log; done; done
echo "K_DONE" >> $L/k_${q}.log
