#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_thin_o.sh <model> : thin survey, the target's own pilot result withheld from the prompt (it is pooled afterwards by the m + r rule)
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; m=$1
for k in 20 50; do $PY $S/v_thin.py collect $m $k others >> $L/thino_$m.log 2>&1; echo "DONE $k" >> $L/thino_$m.log; done
echo "THINO_DONE" >> $L/thino_$m.log
