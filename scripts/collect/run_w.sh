#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_w.sh <queue name> <mode list> <nq or all> <workers> <model> [<model> ...]
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs
q=$1; modes=$2; nq=$3; w=$4; shift 4
for m in "$@"; do for mode in $modes; do
  if [ "$nq" = "all" ]; then $PY $S/w_groups.py collect $m $mode 100000 $w >> $L/w_${q}.log 2>&1; else $PY $S/w_groups.py collect $m $mode $nq $w >> $L/w_${q}.log 2>&1; fi
  echo "DONE $m $mode $(date +%H:%M)" >> $L/w_${q}.log
done; done
echo "QUEUE_DONE $q $(date +%H:%M)" >> $L/w_${q}.log
