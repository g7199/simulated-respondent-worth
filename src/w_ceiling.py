"""The ceiling of Proposition 2. If every pair of sources shares at least rho times the smallest squared error, G_ij >= rho min_k G_kk, then no non-negative
combination of them, however many, is worth more than r_best / rho. Measured on the older surveys and the new polls.  ->  results/w_ceiling.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, pickle, numpy as np
Z = pickle.load(open(_REPO + "/results/y_cov_cache.pkl", "rb")); rng = np.random.default_rng(0); out = {}


def per_question(C, H, names):
    n = len(names); out_ = np.zeros((len(H), n, n))
    for q, h in enumerate(H):
        e = np.stack([np.asarray(C[m][q]) - np.asarray(h) for m in names]); out_[q] = e @ e.T
    return out_


def bound(G, v):
    d = np.diag(G); rho = G[np.triu_indices(len(G), 1)].min() / d.min(); return float(v / d.min() / rho), float(rho), float(v / d.min())


for lab, names, C, H in (("older surveys", Z["names"], Z["C"], Z["h"]), ("new polls", Z["polls"]["models"], Z["polls"]["PM"], Z["polls"]["hp"])):
    Gi = per_question(C, H, names); V = np.array([float((np.asarray(x) * (1 - np.asarray(x))).sum()) for x in H]); c, rho, best = bound(Gi.mean(0), V.mean()); B = [bound(Gi[x].mean(0), V[x].mean())[0] for x in (rng.integers(0, len(H), len(H)) for _ in range(2000))]
    out[lab] = {"best single": best, "rho": rho, "ceiling": c, "ceiling interval": [float(x) for x in np.percentile(B, [2.5, 97.5])]}; print(lab, {k: (round(v, 3) if not isinstance(v, list) else [round(x, 1) for x in v]) for k, v in out[lab].items()})
json.dump(out, open(_REPO + "/results/w_ceiling.json", "w"), indent=1)
