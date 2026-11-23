"""One item per call for the four strongest simulators (two developers), on the items of the first fourteen batches:
rates, error correlations, the pooled ratio G_ij / min and what the four are worth together, next to the same quantities for the batched main runs on the same items.
python z_single.py -> results/z_single.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, re, json, pickle
import numpy as np
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); Z = pickle.load(open(ROOT + "/results/y_cov_cache.pkl", "rb")); ids = Z["ids"]; h = Z["h"]; V = Z["V"]; C = Z["C"]; pos = {int(i): k for k, i in enumerate(ids)}
M = [("Claude-Fable-5.2", "claude-fable-5-2"), ("Claude-Opus-5.5", "claude-opus-5-5"), ("GPT-6-astra", "gpt-6-astra"), ("GPT-6.1-sol", "gpt-6.1-sol")]
def rd(d):
    P = {}
    for f in os.listdir(d):
        if re.match(r"b\d+\.json", f):
            for i, p in json.load(open(f"{d}/{f}"))["items"].items():
                p = np.asarray(p, float); P[int(i)] = p / p.sum()
    return P
S = {nm: rd(f"{SOL}/survey_{tag}_single") for nm, tag in M if os.path.isdir(f"{SOL}/survey_{tag}_single")}; nm_ = list(S); com = sorted(set.intersection(*[set(v) for v in S.values()]) & set(pos)); v = np.mean([V[pos[i]] for i in com])
def joint(G, vv):
    act = list(range(len(G)))
    while True:
        w = vv * np.linalg.solve(G[np.ix_(act, act)], np.ones(len(act)))
        if (w >= 0).all() or len(act) == 1:
            break
        act.pop(int(np.argmin(w)))
    return float(w.sum())
def analyse(get):
    E = [[get(nm, i) - h[pos[i]] for i in com] for nm in nm_]; G = np.array([[np.mean([a @ b for a, b in zip(E[x], E[y])]) for y in range(len(nm_))] for x in range(len(nm_))]); d = np.diag(G); iu = np.triu_indices(len(nm_), 1); cr = G / np.sqrt(np.outer(d, d))
    return {"rates": {nm: float(v / d[j]) for j, nm in enumerate(nm_)}, "error correlations": {f"{nm_[a]} / {nm_[b]}": float(cr[a, b]) for a, b in zip(*iu)}, "G_ij / min, pooled": float(G[iu].sum() / np.minimum.outer(d, d)[iu].sum()), "sum of rates": float((v / d).sum()), "rate of the set": joint(G, v), "best single": float((v / d).max())}
res = {"items": len(com), "models": nm_, "one item per call": analyse(lambda nm, i: S[nm][i]), "batches of twenty": analyse(lambda nm, i: C[nm][pos[i]])}
for k in ("one item per call", "batches of twenty"):
    r = res[k]; print(f"{k} ({len(com)} items): rates " + " ".join(f"{nm.replace('GPT-', '').replace('Claude-', '')} {x:.1f}" for nm, x in r["rates"].items()) + f" | correlations {min(r['error correlations'].values()):.2f}-{max(r['error correlations'].values()):.2f} | G_ij/min {r['G_ij / min, pooled']:.2f} | sum {r['sum of rates']:.1f}, set {r['rate of the set']:.1f}, best {r['best single']:.1f} (+{100 * (r['rate of the set'] / r['best single'] - 1):.0f}%)")
json.dump(res, open(ROOT + "/results/z_single.json", "w"), indent=1)
