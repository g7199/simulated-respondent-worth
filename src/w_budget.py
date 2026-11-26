"""A fixed number of interviews. Two ways to spend 50 respondents per question on a questionnaire:
  ask every question of 50 people and pool with the cold simulator (Proposition 1, worth 50 + r in expectation);
  ask half of the questions of 100 people (pooled with the cold simulator) and have the simulator read them to answer the other half.   ->  results/w_budget.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, numpy as np
sys.path.insert(0, _REPO + "/src")
from w_groups import load
from w_util import read, MODELS as MODELS_
SOL = _REPO + "/outputs/sol"; T = load(); N = len(T); nz = lambda p: np.asarray(p, float) / np.sum(p); th = [nz(t["percent"]) for t in T]; V = np.array([float((x * (1 - x)).sum()) for x in th]); noise = V / np.array([t["n"] for t in T]); TH = json.load(open(_REPO + "/results/v_thin.json")); out = {"reading a thin survey over the pooling rule (paired, one draw of respondents)": {nm: {m: TH[nm][m]["reads / m + r rule"] for m in ("20", "50")} for nm in TH if nm in dict(MODELS_)}}
err = lambda P: np.array([float(((P[i] - th[i]) ** 2).sum()) for i in range(N)]) - noise
for nm, tag in MODELS_:
    if not _os.path.isdir(f"{SOL}/egctxn100_{tag}"):
        continue
    Gc = err(read(f"{SOL}/egplain_{tag}")); Gr = err(read(f"{SOL}/egctxn100_{tag}")); r = V.mean() / Gc.mean(); fielded = (100 * V + r * r * Gc) / (100 + r) ** 2
    out[nm] = {"cold": float(r), "read from 100 respondents on the other half": float(V.mean() / Gr.mean()), "fielded half, 100 respondents + cold simulator": float(V.mean() / fielded.mean()), "ask half, read the rest": float(V.mean() / (0.5 * (fielded + Gr)).mean()), "ask all of 50, pool": 50 + float(r)}
    print(nm, {k: round(v, 1) for k, v in out[nm].items()})
json.dump(out, open(_REPO + "/results/w_budget.json", "w"), indent=1)
