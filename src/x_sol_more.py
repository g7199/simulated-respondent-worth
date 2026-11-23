"""Stated distributions from a model behind a CLI client (see sol_common.py) for the annotator prompts and the GSS groups.
  python x_sol_more.py ca     choice among four responses (the 130 Community Alignment prompts of x_ca.py); 5 prompts per call
  python x_sol_more.py grp    answers of the population and of 20 groups to 36 GSS items; one item per call
Outputs: <repo>/outputs/sol/<use>/*.json (ids as documented in each branch).
"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json
import numpy as np
from sol_common import run, CLOSED

use = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6; effort = sys.argv[3] if len(sys.argv) > 3 else "medium"
ROOT = (_REPO + ""); SCREEN = ROOT + "/outputs/screen"
RET = " Return JSON with one entry per item, in the order given: its id and a list \"p\" of percentages"

if use == "ca":
    import pandas as pd
    R = json.load(open(f"{SCREEN}/t5_ca_raw.json"))
    if os.path.exists(f"{SCREEN}/t5_ca_raw_b.json"):
        for k, v in json.load(open(f"{SCREEN}/t5_ca_raw_b.json")).items():
            R.setdefault(k, v)
    need = ["pop"] + [f"perm{v}" for v in range(8)]
    items = [k for k, v in R.items() if all(a in v for a in need) and min(min(v[a]) for a in need) > -29]
    items = [k for k in items if sum(R[k]["counts"]) - 1 >= 25]                    # the 130 prompts of x_ca.py, same order
    d = pd.read_parquet((_REPO + "/data/ca/ca_long.parquet"), columns=["item", "prompt", "resp_a", "resp_b", "resp_c", "resp_d"]).drop_duplicates("item").set_index("item")
    json.dump(items, open((_REPO + "/outputs/sol/ca_items.json"), "w"))
    head = ("For each item below, a user asked an AI assistant a question and was shown four candidate responses. Many different people were asked which response they "
            "personally prefer. Estimate the percentage of people who chose each response. " + CLOSED + RET + " for responses A, B, C, D (summing to 100).")
    txt = lambda j: f"### Item {j}\nQuestion: {d.loc[items[j]].prompt}\n\n" + "\n\n".join(f"Response {'ABCD'[k]}: {d.loc[items[j]]['resp_' + 'abcd'[k]]}" for k in range(4))
    jobs = {f"b{a:03d}": (head + "\n\n" + "\n\n".join(txt(j) for j in range(a, min(a + 5, len(items)))), {j: 4 for j in range(a, min(a + 5, len(items)))}) for a in range(0, len(items), 5)}

elif use == "grp":
    sys.path.insert(0, SCREEN); sys.argv = [sys.argv[0]]
    from t4_run import ITEMS, PARTS
    names = list(ITEMS); groups = [(pn, g, lab) for pn, (desc, gs) in PARTS.items() for g, sent, lab in gs]
    json.dump({"items": names, "groups": [[pn, g] for pn, g, _ in groups]}, open((_REPO + "/outputs/sol/grp_keys.json"), "w"))
    jobs = {}
    for a, it in enumerate(names):
        q, opts, _ = ITEMS[it]
        p = ("The General Social Survey is a nationally representative survey of adults in the United States. Around the year 2020 its respondents were asked the question below. "
             "Estimate the percentage who chose each option, first among all respondents and then within each of the listed groups of respondents. " + CLOSED +
             RET + f" over the {len(opts)} options (summing to 100). Entry 0 is all respondents; entries 1 to {len(groups)} are the groups in the order listed.\n\n"
             f"Question: {q}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(opts)) + "\n\nGroups:\n" +
             "\n".join(f"{j + 1}. {PARTS[pn][0]}: {lab}" for j, (pn, g, lab) in enumerate(groups)))
        jobs[f"i{a:02d}"] = (p, {j: len(opts) for j in range(len(groups) + 1)})

else:
    sys.exit("usage: x_sol_more.py ca|grp [workers] [effort]")

print(use, len(jobs), "calls; first prompt:\n" + next(iter(jobs.values()))[0][:1200] + "\n...", flush=True)
run(use, jobs, workers, effort)
