#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_ctl.sh <queue> <workers> <model> [...] : controls for the same-survey context (previous week's report; other pollsters)
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; q=$1; w=$2; shift 2
for m in "$@"; do for mode in ctxweek ctxoth; do $PY $S/w_groups.py collect $m $mode 100000 $w >> $L/ctl_${q}.log 2>&1; echo "DONE $m $mode" >> $L/ctl_${q}.log; done; done
echo "CTL_DONE" >> $L/ctl_${q}.log
