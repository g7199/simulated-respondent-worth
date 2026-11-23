#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# a panel of sixteen simulated individuals for one model on the first fourteen batches: survey_<model>_panel<k>/
cd $REPO/src; M=$1; W=$2
for k in $(seq 0 15); do PANEL=$k SOL_MODEL=$M SOL_SUFFIX=_${M}_panel$k "${PYTHON:-python}" x_ladder.py 14 $W medium 2>&1 | grep -E "FAILED|14/14" | tail -n 2; done
echo PANEL_DONE
