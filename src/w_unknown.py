"""What happens when a simulator's worth r is not known: its weight next to n real respondents is taken from other data.
  held-out programme: r from the other six; new polls: r from the older surveys.  -> results/w_unknown.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, pickle, numpy as np
Z = pickle.load(open(_REPO + "/results/y_cov_cache.pkl", "rb")); names = Z["names"]; h = Z["h"]; sid = np.asarray(Z["sid"]); V = np.array([float((x * (1 - x)).sum()) for x in h]); NS = (5, 20, 100); FR = (1.0, 0.5)
G = {m: np.array([float((np.asarray(c) - x) @ (np.asarray(c) - x)) for c, x in zip(Z["C"][m], h)]) for m in names}
hp = Z["polls"]["hp"]; Vp = np.array([float((x * (1 - x)).sum()) for x in hp]); Gp = {m: np.array([float((np.asarray(c) - x) @ (np.asarray(c) - x)) for c, x in zip(Z["polls"]["PM"][m], hp)]) for m in Z["polls"]["models"]}
worth = lambda v, g, n, w: v / ((n * v + w * w * g) / (n + w) ** 2)
rows = []
for m in names:
    for s in np.unique(sid):
        tr = sid != s; te = sid == s; rows.append((m, f"p{s}", V[te].mean(), G[m][te].mean(), V[tr].mean() / G[m][tr].mean()))
    rows.append((m, "new polls", Vp.mean(), Gp[m].mean(), V.mean() / G[m].mean()))
out = {}
for fr in FR:
    o = {}
    for n in NS:
        a = np.array([worth(v, g, n, fr * r) for _, _, v, g, r in rows]); best = np.array([n + v / g for _, _, v, g, _ in rows]); o[str(n)] = {"share of the cases better than the n respondents alone": float((a > n).mean()), "median share of the best possible gain": float(np.median((a - n) / (best - n))), "worst worth over n": float((a / n).min()), "median worth over best": float(np.median(a / best)), "lowest worth over best": float((a / best).min())}
    out[f"weight = {fr} x r from other data"] = o
out["ratio of transferred to true r"] = {k: float(f(np.array([r / (v / g) for _, _, v, g, r in rows]))) for k, f in (("median", np.median), ("min", np.min), ("max", np.max))}; out["cases"] = len(rows)
json.dump(out, open(_REPO + "/results/w_unknown.json", "w"), indent=1); print(json.dumps(out, indent=0)[:2400])
