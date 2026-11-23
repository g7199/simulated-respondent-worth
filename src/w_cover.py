"""The coverage-based effective sample size of Huang, Wu and Wang (2025) on the same answers: the largest k/C for which the interval
c +- z sqrt(c (1 - c) C / k) misses the published share on at most alpha/2 of the answer options (alpha = 0.05), with the simulator's stated share taken as its
limit for many draws. Compared with r = V / G.  ->  results/w_cover.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, pickle, numpy as np
from scipy.stats import norm as _n, spearmanr
Z = pickle.load(open(_REPO + "/results/y_cov_cache.pkl", "rb")); ALPHA = 0.05; zq = _n.ppf(1 - ALPHA / 2)


def kappa(C, H):
    c = np.concatenate([np.asarray(x, float) for x in C]); h = np.concatenate([np.asarray(x, float) for x in H]); d = (c - h) ** 2 / np.maximum(c * (1 - c), 1e-12); return float(zq ** 2 / np.quantile(d, 1 - ALPHA / 2))


def r(C, H):
    return float(np.mean([(x * (1 - x)).sum() for x in map(np.asarray, H)]) / np.mean([((np.asarray(c) - np.asarray(x)) ** 2).sum() for c, x in zip(C, H)]))


out = {}
for lab, names, C, H in (("older surveys", Z["names"], Z["C"], Z["h"]), ("new polls", Z["polls"]["models"], Z["polls"]["PM"], Z["polls"]["hp"])):
    k = {m: kappa(C[m], H) for m in names}; w = {m: r(C[m], H) for m in names}; a = np.array([k[m] for m in names]); b = np.array([w[m] for m in names])
    out[lab] = {"kappa": k, "r": w, "kappa range": [float(a.min()), float(a.max())], "rank correlation": float(spearmanr(a, b)[0]), "kappa over r": [float((a / b).min()), float(np.median(a / b)), float((a / b).max())]}
    print(lab, "kappa", round(a.min(), 1), "to", round(a.max(), 1), "| r", round(b.min(), 1), "to", round(b.max(), 1), "| rank corr", round(out[lab]["rank correlation"], 3), "| kappa/r min, median, max", [round(x, 2) for x in out[lab]["kappa over r"]])
    print("   ", {m.replace("Claude-", "").replace("GPT-", ""): round(k[m], 1) for m in names})
json.dump(out, open(_REPO + "/results/w_cover.json", "w"), indent=1)
