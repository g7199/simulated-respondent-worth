#!/bin/bash
# Recomputes every result, table and figure from the stored model answers. Makes no model calls and needs no GPU.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY="${PYTHON:-python}"; cd "$REPO/src"
[ -f "$REPO/data/polls_2026/items.json" ] || { echo "Data not found: run python scripts/prepare_data.py first (see README)."; exit 1; }
export PYTHONHASHSEED=0 MPLBACKEND=Agg
run(){ echo "[$(date +%H:%M:%S)] $*"; "$PY" "$@" > "$REPO/logs/$(basename "$1" .py).log" 2>&1; }
mkdir -p "$REPO/logs"
[ -f "$REPO/outputs/screen/t4c_llm.json" ] || { echo "Model answers not found: unpack the model answers archive first (see README)."; exit 1; }
run y_cov.py; run z_nest.py; run z_groups.py; run z_wave.py; run z_anchor.py; run z_panel.py; run z_single.py
run w_score.py; run w_old.py score; run w_rel.py score; run w_ctl.py; run w_extra.py; run w_wave.py score
run w_design.py score; run v_thin.py score; run v_alloc.py; run v_transfer.py; run w_perq.py; run w_free.py
run z_figs.py; run w_figs.py; run paper_figs.py
"$PY" "$REPO/verify.py"
