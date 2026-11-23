#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# usage: run_local.sh <tag> <port> <served>   : ladder (surveys, new polls, party) and the same-survey context experiment for a local model
PY="${PYTHON:-python}"; S=$REPO/src; L=$REPO/logs; tag=$1; port=$2; served=$3
cd $S
for what in survey polls party given; do $PY $S/x_ladder_local.py ${tag} $port $served $what >> $L/local_${served}.log 2>&1; echo "DONE $what $(date +%H:%M)" >> $L/local_${served}.log; done
for mode in plain ctx; do
  W_BC=10 $PY $S/w_groups.py collect local:$port:$served $mode 100000 6 >> $L/local_${served}.log 2>&1
  W_FILL=1 W_KEY=f W_BC=4 $PY $S/w_groups.py collect local:$port:$served $mode 100000 6 >> $L/local_${served}.log 2>&1
  W_FILL=1 W_KEY=g W_BC=1 LOCAL_RETRY_TEMP=1 $PY $S/w_groups.py collect local:$port:$served $mode 100000 6 >> $L/local_${served}.log 2>&1
  W_FILL=1 W_KEY=h W_BC=1 LOCAL_RETRY_TEMP=1 $PY $S/w_groups.py collect local:$port:$served $mode 100000 6 >> $L/local_${served}.log 2>&1
  echo "DONE $mode $(date +%H:%M)" >> $L/local_${served}.log
done
echo "LOCAL_DONE $(date +%H:%M)" >> $L/local_${served}.log
