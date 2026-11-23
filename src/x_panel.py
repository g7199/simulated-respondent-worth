"""A panel of simulated individuals as the read-out of an open model: sixteen personas (sex x age x education x place of living)
are appended to the population description, each is asked the question (next-token probabilities, two option orders), and the
panel's answer is the mean of the sixteen distributions.
  LLM_PORTS=8000 LLM_MODEL=qwen9b python x_panel.py qwen   ->  screen/r3_out/panel_qwen.json  {item index: [mean probabilities]}"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, itertools
import numpy as np
ROOT = (_REPO + ""); tag = sys.argv[1]
sys.path.insert(0, ROOT + "/outputs/screen"); argv = sys.argv; sys.argv = [sys.argv[0], "none"]
from r2pop_common import load, soft
from llm import score_options
sys.argv = argv
I = load()[0]; L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; batches = json.load(open((_REPO + "/outputs/sol/survey/batches.json")))[:50]; items = [int(i) for b in batches for i in b]
PERS = [f" You are a {sex} aged {age}, {edu}, living in {place}." for sex, age, edu, place in itertools.product(("woman", "man"), ("under 40", "over 40"), ("with a university degree", "without a university degree"), ("a large city", "a small town or a rural area"))]
prompt = lambda pop, q, opts: f"{pop}\n\n{q}\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + "\nAnswer with a single letter."
import os
OUTF = f"{ROOT}/screen/r3_out/panel_{tag}.json"; out = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
for k in sorted({int(I.K.values[i]) for i in items}):
    if all(str(i) in out for i in items if int(I.K.values[i]) == k):
        continue
    idx = [i for i in items if int(I.K.values[i]) == k]; P = []; key = []
    for i in idx:
        r = I.iloc[i]
        for p, o in itertools.product(range(len(PERS)), (0, 1)):
            P.append(prompt(r["pop"] + PERS[p], r.q, r.opts[::-1] if o else r.opts)); key.append((i, o))
    R = score_options(P, list(L[:k]), workers=48); acc = {i: np.zeros(k) for i in idx}
    for (i, o), x in zip(key, R):
        lp = np.array([x[c] for c in L[:k]]); acc[i] += soft(lp[::-1] if o else lp) / (2 * len(PERS))
    for i in idx:
        out[str(i)] = acc[i].tolist()
    json.dump(out, open(OUTF, "w")); print(k, "options:", len(idx), "items", flush=True)
print("panel answers for", len(out), "items")
