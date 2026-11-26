#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# a panel of sixteen simulated individuals for one model on all fifty batches: survey_<model>_panel<k>/
cd $REPO/src; M=$1; W=$2
for k in $(seq 0 15); do PANEL=$k SOL_MODEL=$M SOL_SUFFIX=_${M}_panel$k "${PYTHON:-python}" x_ladder.py 50 $W medium 2>&1 | grep -E "FAILED|50/50" | tail -n 2; done
echo PANEL_DONE
