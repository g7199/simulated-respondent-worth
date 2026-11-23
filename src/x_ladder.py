"""One more simulator on the common survey items: the batches and the prompt of x_sol.py, through sol_common (Codex or Claude client).
  SOL_MODEL=<model> SOL_SUFFIX=_<model> python x_ladder.py [number of batches = 50] [workers = 4] [effort = medium]
Output: <repo>/outputs/sol/survey_<model>/bNNNN.json (read by y_cov.py)."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json
ROOT = (_REPO + ""); nb = int(sys.argv[1]) if len(sys.argv) > 1 else 50; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4; effort = sys.argv[3] if len(sys.argv) > 3 else "medium"
sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/outputs/screen"); argv = sys.argv; sys.argv = [sys.argv[0], "none"]
from r2pop_common import load
from sol_common import run
sys.argv = argv
I, E, qi, L0, L1, LQ = load(); K = I.K.values
HEAD = ("You are estimating the results of real opinion surveys. For each item below, a group of respondents is described and a question is given with its answer "
        "options. Estimate the percentage of that group who chose each option in the actual survey (percentages over the listed options, summing to 100). "
        "Answer from your own knowledge only: do not run commands, read files or search. Return JSON with one entry per item, in the order given: its id and a "
        "list \"p\" with one percentage per option, in the order of the options.")


if os.environ.get("PROMPT_ALT"):                                 # a second wording of the same request (how much of the error is the prompt's?)
    HEAD = ("Below are questions from real opinion surveys. Each comes with the population that was surveyed and the answer options that were offered. For every question, give your best "
            "estimate of how the answers were distributed in the actual survey, as percentages over the listed options (they should sum to 100). Rely only on what you know: do not use "
            "tools, files or search. Reply in JSON with one entry per question, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options.")


def item_text(i):
    r = I.iloc[i]
    if os.environ.get("PROMPT_ALT"):
        return f"### Question {i}\nSurvey question: {r.q}\nAnswer options:\n" + "\n".join(f"({j + 1}) {o}" for j, o in enumerate(r.opts)) + f"\nWho was asked: {r['pop']}"
    return f"### Item {i}\nRespondents: {r['pop']}\nQuestion: {r.q}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(r.opts))


def positive(got):
    assert all(sum(v) > 0 for v in got.values()), "all zero"


batches = json.load(open((_REPO + "/outputs/sol/survey/batches.json")))[:nb]
mode = os.environ.get("BATCH_MODE", "")
if mode == "shuffle":                                            # the same items regrouped at random into other batches of twenty (is the shared error a product of shared batches?)
    import numpy as np
    flat = [int(i) for b in batches for i in b]; perm = np.random.default_rng(int(os.environ.get("BATCH_SEED", "7"))).permutation(flat); batches = [[int(i) for i in perm[a:a + 20]] for a in range(0, len(perm), 20)]
elif mode == "single":                                           # one item per call
    batches = [[int(i)] for b in batches for i in b]
if os.environ.get("PANEL"):                                       # one simulated individual: the model answers as a respondent with a given profile and states how likely it is to choose each option
    import itertools
    PERS = [f"a {sex} aged {age}, {edu}, living in {place}" for sex, age, edu, place in itertools.product(("woman", "man"), ("under 40", "over 40"), ("with a university degree", "without a university degree"), ("a large city", "a small town or a rural area"))]
    who = PERS[int(os.environ["PANEL"])]
    HEAD = (f"You are answering survey questions as one respondent: {who}. Each item below names the survey you are taking part in and gives a question with its answer options. "
            "Answer as that person would. For each item, give the probability, in percent, that you would choose each option (summing to 100). "
            "Answer from your own knowledge only: do not run commands, read files or search. Return JSON with one entry per item, in the order given: its id and a "
            "list \"p\" with one percentage per option, in the order of the options.")
jobs = {f"b{k:04d}": (HEAD + "\n\n" + "\n\n".join(item_text(i) for i in b), {int(i): int(K[i]) for i in b}) for k, b in enumerate(batches)}
run("survey", jobs, workers, effort, check=positive)
