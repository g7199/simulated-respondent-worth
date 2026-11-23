#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_thin.sh <model> <m> [<m> ...] : thin-survey design for one model
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; m=$1; shift 1
for k in "$@"; do $PY $S/v_thin.py collect $m $k >> $L/thin_$m.log 2>&1; echo "DONE $k" >> $L/thin_$m.log; done
echo "THIN_DONE" >> $L/thin_$m.log
