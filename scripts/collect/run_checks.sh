#!/bin/bash
# The protocol checks of Appendix B on all fifty batches: one item per call, a repeat run, a second wording, and GPT-6.1-sol at low and high reasoning.
# usage: run_checks.sh   (cached batches are skipped; run it again to retry failures)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd $REPO/src; PY="${PYTHON:-python}"
one() { env "$@" $PY x_ladder.py 50 "${W:-4}" "${EFFORT:-medium}" 2>&1 | grep -E "FAILED|/(50|981):" | tail -n 2; }
for M in claude-fable-5-2 claude-opus-5-5 gpt-6-astra gpt-6.1-sol; do one SOL_MODEL=$M SOL_SUFFIX=_${M}_single BATCH_MODE=single; done
for M in claude-fable-5-2 claude-opus-5-5 gpt-6-astra; do one SOL_MODEL=$M SOL_SUFFIX=_${M}_rep; done
one SOL_MODEL=gpt-6.1-sol SOL_SUFFIX=_rep
for M in claude-fable-5-2 claude-haiku-5-5 gpt-6.1-sol gpt-5.6-luna; do one SOL_MODEL=$M SOL_SUFFIX=_${M}_alt PROMPT_ALT=1; done
EFFORT=low one SOL_MODEL=gpt-6.1-sol SOL_SUFFIX=_low
EFFORT=high one SOL_MODEL=gpt-6.1-sol SOL_SUFFIX=_high
echo CHECKS_DONE

# Fairness pass (2026-11-25): every comparison on the same questions and with both developers.
# Group answers still incomplete are asked again one question per call. Claude: first with the usual answer schema (files f*.json), then the rest
# with the number of entries fixed by the schema (21 with everyone, 20 when shown; files s*.json). GPT: with the fixed number directly.
S21=$REPO/outputs/.cli_work/schema21.json; S20=$REPO/outputs/.cli_work/schema20.json
"${PYTHON:-python}" -c "import json; s=json.load(open('$REPO/outputs/.cli_work/schema.json')); a=s['properties']['items']; [json.dump({**s,'properties':{'items':{**a,'minItems':k,'maxItems':k}}},open(f,'w')) for k,f in ((21,'$S21'),(20,'$S20'))]"
for mode in joint shown; do W_FILL=1 W_KEY=f $PY w_groups.py collect claude-sonnet-5-5 $mode 246 4; done
W_FILL=1 W_KEY=f $PY w_groups.py collect claude-haiku-5-5 shown 246 4
W_SET=fresh W_FILL=1 W_KEY=f $PY w_groups.py collect claude-sonnet-5-5 joint 126 4
SOL_SCHEMA=$S21 W_FILL=1 W_KEY=s $PY w_groups.py collect claude-sonnet-5-5 joint 246 4
SOL_SCHEMA=$S21 W_SET=fresh W_FILL=1 W_KEY=s $PY w_groups.py collect claude-sonnet-5-5 joint 126 4
$PY ../scripts/collect/repair_ids.py ../outputs/sol/egjoint_claude-sonnet-5-5; W_SET=fresh $PY ../scripts/collect/repair_ids.py ../outputs/sol/egjoint_fresh_claude-sonnet-5-5
for M in gpt-5.6-luna gpt-6-luna gpt-5.6-sol; do SOL_SCHEMA=$S21 W_FILL=1 W_KEY=s $PY w_groups.py collect $M joint 246 4; SOL_SCHEMA=$S20 W_FILL=1 W_KEY=s $PY w_groups.py collect $M shown 246 4; done
# Thin surveys, answers from 100 respondents, the other developer's answers and the choice of questions, for three models of each developer
for M in gpt-6.1-sol gpt-6-astra gpt-5.6-luna; do
  for k in 20 50; do $PY v_thin.py collect $M $k; done
  W_NOISE=100 $PY w_groups.py collect $M ctx 246 4
  W_PEERS=claude-fable-5-2,claude-opus-5-5 $PY w_groups.py collect $M peers 246 4
done
W_NOISE=100 $PY w_groups.py collect claude-haiku-5-5 ctx 246 4
for M in gpt-6.1-sol gpt-6-astra gpt-5.6-luna claude-haiku-5-5; do for d in "greedy 10" "random 10 1" "random 10 2" "random 10 3" "random 10 4" "random 10 5"; do $PY w_design.py collect $M $d; done; done
echo FAIRNESS_DONE
