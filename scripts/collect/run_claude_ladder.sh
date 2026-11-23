#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$REPO/logs"
# stated shares from Claude models for the common survey items, the new polls (toplines and party groups), the GSS groups and the annotator prompts
cd $REPO/src; PY="${PYTHON:-python}"
for M in "$@"; do
  export SOL_MODEL=$M SOL_SUFFIX=_$M
  echo "== $M survey $(date +%H:%M)"; $PY x_ladder.py 50 4 medium 2>&1 | grep -v Warning | tail -8
  echo "== $M polls $(date +%H:%M)"; $PY y_polls.py collect $M 2>&1 | grep -v Warning | tail -6
  echo "== $M grp $(date +%H:%M)"; $PY x_sol_more.py grp 4 medium 2>&1 | grep -E "^grp [0-9]|FAILED" | tail -4
  echo "== $M ca $(date +%H:%M)"; $PY x_sol_more.py ca 4 medium 2>&1 | grep -E "^ca [0-9]|FAILED" | tail -4
  echo "== $M done $(date +%H:%M)"
done
echo ALL_DONE
