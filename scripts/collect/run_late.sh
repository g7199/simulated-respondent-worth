#!/bin/bash
# The later runs behind Figure 2a, the one-question check and the held-out replication.
# usage: run_late.sh one|draws|splits|fresh <model> [<model> ...]
#   one     the same-survey experiment with one target question per call
#   draws   1, 3 or 10 real answers in the prompt, several random draws on the first split
#   splits  four more random splits of every report, with 1, 3 or 10 answers or the whole other half
#   fresh   the fixed protocol on the two held-out weekly reports
# Open models that return a malformed batch: python src/fill_split.py <party|given|sep> local:<port>:<name>
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PY="${PYTHON:-python}"; cd "$REPO/src"
what=$1; shift
for m in "$@"; do
  case $what in
    one) for mode in plain ctx; do W_SINGLE=1 $PY w_groups.py collect $m $mode 100000 4; done;;
    draws) for s in 1 2 3 4 5; do W_K=1 W_KS=$s $PY w_groups.py collect $m ctx 100000 3; done
           for k in 3 10; do for s in 1 2 3 4; do W_K=$k W_KS=$s $PY w_groups.py collect $m ctx 100000 3; done; done;;
    splits) for s in 12 13 14 15; do W_SEED=$s $PY w_groups.py collect $m ctx 100000 3
              for k in 1 3 10; do W_K=$k W_SEED=$s $PY w_groups.py collect $m ctx 100000 3; done; done;;
    fresh) for mode in plain ctx joint shown; do W_SET=fresh $PY w_groups.py collect $m $mode 100000 3; done;;
    *) echo "usage: run_late.sh one|draws|splits|fresh <model> ..."; exit 1;;
  esac
done
