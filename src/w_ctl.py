"""Controls for the same-survey context (w_groups.py ctxweek, ctxoth), on the questions of the second to fifth report (those with a report of the week before).
  python w_ctl.py   -> results/w_ctl.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, numpy as np
sys.path.insert(0, (_REPO + "/src"))
from w_groups import load, norm
from w_util import read, MODELS
T = load(); N = len(T); SOL = (_REPO + "/outputs/sol"); rng = np.random.default_rng(0)
th = [np.asarray(t["percent"], float) / sum(t["percent"]) for t in T]; V = np.array([(x * (1 - x)).sum() for x in th]); noise = V / np.array([t["n"] for t in T]); waves = sorted({t["wave"] for t in T})
idx = np.array([i for i in range(N) if T[i]["wave"] != waves[0]]); bk = {}; BLOCK = np.array([bk.setdefault((t["wave"], norm(t["text"])), len(bk)) for t in T])
def draw():
    bl = np.unique(BLOCK[idx]); pick = rng.choice(bl, len(bl)); return np.concatenate([idx[BLOCK[idx] == b] for b in pick])
rate = lambda P, ii: float(V[ii].mean() / max(np.mean([(P[i] - th[i]) @ (P[i] - th[i]) for i in ii]) - noise[ii].mean(), 1e-9))
r2 = np.random.default_rng(11); half = {}; wave = np.array([t["wave"] for t in T])
for w in waves:
    ii = [i for i in range(N) if T[i]["wave"] == w]; perm = r2.permutation(ii)
    for k, i in enumerate(perm):
        half[int(i)] = k % 2
opt = lambda i: tuple(norm(o) for o in T[i]["options"]); shown = {i: [j for j in range(N) if wave[j] == wave[i] and half[j] != half[i]] for i in range(N)}
same = [[j for j in shown[i] if opt(j) == opt(i)] for i in range(N)]; sib = [[j for j in same[i] if norm(T[j]["text"]) == norm(T[i]["text"])] for i in range(N)]
SUBS = {"a sibling item shown in the same-survey condition": np.array([i for i in idx if sib[i]]), "no shared options in the same-survey condition": np.array([i for i in idx if not same[i]])}
res = {"targets": len(idx), "subsets": {k: len(v) for k, v in SUBS.items()}, "models": {}}; print(len(idx), "questions of the second to fifth report", res["subsets"])
D = [draw() for _ in range(1000)]
for nm, tag in MODELS:
    P = {k: read(f"{SOL}/eg{k}_{tag}") for k in ("plain", "ctx", "ctxweek", "ctxoth", "ctxs12")}
    if any(not all(i in P[k] for i in idx) for k in ("plain", "ctx", "ctxweek", "ctxoth")):
        continue
    o = {"cold": rate(P["plain"], idx)}
    for k, lab in (("ctxoth", "other pollsters, same weeks"), ("ctxweek", "same pollster, the week before"), ("ctx", "same survey")):
        o[lab] = [rate(P[k], idx), *np.percentile([rate(P[k], r) / rate(P["plain"], r) for r in D], [2.5, 97.5])]
    o["same survey over the week before"] = [o["same survey"][0] / o["same pollster, the week before"][0], *np.percentile([rate(P["ctx"], r) / rate(P["ctxweek"], r) for r in D], [2.5, 97.5])]
    o["by subset"] = {k: {"cold": rate(P["plain"], ii), "other pollsters, same weeks": rate(P["ctxoth"], ii), "same pollster, the week before": rate(P["ctxweek"], ii), "same survey": rate(P["ctx"], ii)} for k, ii in SUBS.items()}
    if all(i in P["ctxs12"] for i in range(N)):
        o["second random split, all questions"] = {"cold": rate(P["plain"], np.arange(N)), "same survey, first split": rate(P["ctx"], np.arange(N)), "same survey, second split": rate(P["ctxs12"], np.arange(N))}
    res["models"][nm] = o
    print(f"{nm:18s} cold {o['cold']:.0f} | other pollsters {o['other pollsters, same weeks'][0]:.0f} (x{o['other pollsters, same weeks'][0] / o['cold']:.2f} [{o['other pollsters, same weeks'][1]:.2f}, {o['other pollsters, same weeks'][2]:.2f}]) | week before {o['same pollster, the week before'][0]:.0f} (x{o['same pollster, the week before'][0] / o['cold']:.2f} [{o['same pollster, the week before'][1]:.2f}, {o['same pollster, the week before'][2]:.2f}]) | same survey {o['same survey'][0]:.0f} (x{o['same survey'][0] / o['cold']:.2f} [{o['same survey'][1]:.2f}, {o['same survey'][2]:.2f}]) | same survey / week before x{o['same survey over the week before'][0]:.2f} [{o['same survey over the week before'][1]:.2f}, {o['same survey over the week before'][2]:.2f}]"
          + (f" | second split: {o['second random split, all questions']['same survey, second split']:.0f} (first {o['second random split, all questions']['same survey, first split']:.0f})" if "second random split, all questions" in o else ""))
json.dump(res, open((_REPO + "/results/w_ctl.json"), "w"), indent=1)
