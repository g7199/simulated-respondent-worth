"""Models or real answers? From one simulator asked cold, add either other simulators (the best one at each step, with the best weights on these very questions, which
favours them) or real answers from the same survey in its prompt (1, 3 or 10 drawn at random, or the whole other half), over five random splits of each report.  ->  results/w_curve.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, numpy as np
sys.path.insert(0, _REPO + "/src")
from w_groups import load, norm
from w_util import read, MODELS
SPLITS = (12, 13, 14, 15); SOL = _REPO + "/outputs/sol"; T = load(); N = len(T); rng = np.random.default_rng(0); nz = lambda p: np.asarray(p, float) / np.sum(p); th = [nz(t["percent"]) for t in T]; V = np.array([float((x * (1 - x)).sum()) for x in th]); noise = V / np.array([t["n"] for t in T])
_b = {}; BLOCK = np.array([_b.setdefault((t["wave"], norm(t["text"])), len(_b)) for t in T]); bl = np.unique(BLOCK); by = [np.where(BLOCK == b)[0] for b in bl]; draws = [np.concatenate([by[j] for j in rng.integers(0, len(bl), len(bl))]) for _ in range(2000)]
COLD = {nm: read(f"{SOL}/egplain_{tag}") for nm, tag in MODELS}; COLD = {m: P for m, P in COLD.items() if len(P) == N}; names = list(COLD); E = {m: np.concatenate([COLD[m][i] - th[i] for i in range(N)]) for m in names}; cut = np.cumsum([0] + [len(x) for x in th])
Gi = np.stack([np.stack([[float(E[a][cut[i]:cut[i + 1]] @ E[b][cut[i]:cut[i + 1]]) for b in names] for a in names]) for i in range(N)])
rate = lambda P, x: float(V[x].mean() / max(np.mean([((P[int(i)] - th[int(i)]) ** 2).sum() for i in x]) - noise[x].mean(), 1e-9)); all_ = np.arange(N)


def set_rate(G, v):
    from nnweights import best_weights
    return float(best_weights(G, v).sum())


def models_curve(base, x=all_):
    G = Gi[x].mean(0) - noise[x].mean(); v = V[x].mean(); chosen = [names.index(base)]; out = [v / G[chosen[0], chosen[0]]]
    while len(chosen) < len(names):
        c = max((j for j in range(len(names)) if j not in chosen), key=lambda j: set_rate(G[np.ix_(chosen + [j], chosen + [j])], v)); chosen.append(c); out.append(set_rate(G[np.ix_(chosen, chosen)], v))
    return out


out = {"simulators": names, "curves": {}}
for nm, tag in (("Claude-Fable-5.2", "claude-fable-5-2"), ("GPT-6.1-sol", "gpt-6.1-sol")):
    A = COLD[nm]; Ks = {}; SP = [read(f"{SOL}/egctx_{tag}")] + [read(f"{SOL}/egctxs{s}_{tag}") for s in SPLITS]
    for k in (1, 3, 10):
        first = [f"{SOL}/egctx{k}x{s}_{tag}" for s in range(1, 6 if k == 1 else 5)] + ([] if k == 1 else [f"{SOL}/egctx{k}_{tag}"]); Ks[k] = [[read(d) for d in first]] + [[read(f"{SOL}/egctx{k}s{s}_{tag}")] for s in SPLITS]
        assert all(len(P) == N for d in Ks[k] for P in d), (nm, k)
    assert all(len(P) == N for P in SP), nm
    by_split = lambda k, x: [float(np.mean([rate(P, x) for P in d])) for d in Ks[k]]; mean = lambda k, x: float(np.mean(by_split(k, x))); half = lambda x: float(np.mean([rate(P, x) for P in SP]))
    o = {"cold": rate(A, all_), "more simulators": models_curve(nm), "real answers": {str(k): {"mean": mean(k, all_), "splits": by_split(k, all_)} for k in Ks}, "the other half": {"mean": half(all_), "splits": [rate(P, all_) for P in SP]}}
    d3 = [mean(3, x) - rate(A, x) for x in draws[:400]]; dh = [half(x) - rate(A, x) for x in draws[:400]]; m12 = [models_curve(nm, x)[-1] - rate(A, x) for x in draws[:200]]; o["gain from three real answers"] = [mean(3, all_) - o["cold"], float(np.percentile(d3, 2.5)), float(np.percentile(d3, 97.5))]; o["gain from the other half"] = [o["the other half"]["mean"] - o["cold"], float(np.percentile(dh, 2.5)), float(np.percentile(dh, 97.5))]; o["gain from every other simulator"] = [o["more simulators"][-1] - o["more simulators"][0], float(np.percentile(m12, 2.5)), float(np.percentile(m12, 97.5))]
    out["curves"][nm] = o; print(nm, "cold", round(o["cold"], 1), "| all other simulators", round(o["more simulators"][-1], 1), "| real answers", {k: (round(v["mean"], 1), round(min(v["splits"]), 1), round(max(v["splits"]), 1)) for k, v in o["real answers"].items()}, "| other half", round(o["the other half"]["mean"], 1), [round(x, 1) for x in o["the other half"]["splits"]])
    print("   gains: three answers", [round(x, 1) for x in o["gain from three real answers"]], "other half", [round(x, 1) for x in o["gain from the other half"]], "every other simulator", [round(x, 1) for x in o["gain from every other simulator"]], flush=True)
json.dump(out, open(_REPO + "/results/w_curve.json", "w"), indent=1)
