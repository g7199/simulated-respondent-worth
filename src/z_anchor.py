"""Paper 3 (third revision): one real anchor. The same question asked of another population (same programme, same options) has a
known answer theta'. For a target population with unknown theta and a simulator with answers c (target) and c' (anchor):
  simulator          c
  anchor             theta'                       (another population's real answer)
  pooled             best fixed mixture of c and theta' (weight from the other programmes)
  contrast           theta' + (c - c')            (the simulator's contrast between the populations, added to the real anchor)
  free               theta' + b (c - c') + a (c - theta'), a and b from the other programmes
Each is priced in respondents (mean variance of one respondent / mean squared error). Pairs: every ordered pair of items of the common
set with the same programme, question and options; clusters for the bootstrap are question texts.
Usage: python z_anchor.py -> results/z_anchor.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, pickle
import numpy as np, pandas as pd
ROOT = (_REPO + ""); Z = pickle.load(open(ROOT + "/results/y_cov_cache.pkl", "rb")); rng = np.random.default_rng(0)
sys.path.insert(0, ROOT + "/outputs/screen"); sys.argv = [sys.argv[0], "none"]
from r2pop_common import load
I = load()[0]; names = Z["names"]; ids = Z["ids"]; h = Z["h"]; C = Z["C"]; V = Z["V"]; sid = Z["sid"]; S = Z["S"]; qid = Z["qid"]; N = len(ids)
opts = [tuple(I.opts.values[i]) for i in ids]; key = {}
for k in range(N):
    key.setdefault((int(sid[k]), int(qid[k]), opts[k]), []).append(k)
pairs = [(a, b) for g in key.values() if len(g) > 1 for a in g for b in g if a != b]; T = np.array([a for a, b in pairs]); A = np.array([b for a, b in pairs]); ps = sid[T]; pq = qid[T]
norm = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum(); sq = lambda a, b: float((a - b) @ (a - b))
print(f"{len(pairs)} ordered pairs (target, anchor) from {len(set(T))} items and {len(set(pq))} question texts; by programme: " + " ".join(f"{S[g][:6]} {int((ps == g).sum())}" for g in np.unique(ps)))
Vt = V[T]; uq = np.unique(pq); byq = [np.where(pq == q)[0] for q in uq]; draws = [np.concatenate([byq[j] for j in rng.integers(0, len(uq), len(uq))]) for _ in range(1000)]
res = {"pairs": len(pairs), "items": int(len(set(T))), "questions": int(len(uq)), "models": {}}
anc = np.array([sq(h[b], h[a]) for a, b in pairs])
for m in names:
    c = C[m]; arms = {"simulator": np.array([sq(c[a], h[a]) for a, b in pairs]), "anchor": anc, "contrast": np.array([sq(norm(h[b] + c[a] - c[b]), h[a]) for a, b in pairs])}
    pooled = np.zeros(len(pairs)); free = np.zeros(len(pairs)); coef = []
    for g in np.unique(ps):
        tr = np.where(ps != g)[0]; te = np.where(ps == g)[0]
        if len(tr) == 0:
            tr = te
        x1 = np.concatenate([c[T[x]] - h[A[x]] for x in tr]); x2 = np.concatenate([c[T[x]] - c[A[x]] for x in tr]); y = np.concatenate([h[T[x]] - h[A[x]] for x in tr])
        a1 = float(x1 @ y / (x1 @ x1)); bc = np.linalg.lstsq(np.column_stack([x2, x1]), y, rcond=None)[0]; coef.append([float(bc[0]), float(bc[1])])
        for x in te:
            pooled[x] = sq(norm(h[A[x]] + a1 * (c[T[x]] - h[A[x]])), h[T[x]]); free[x] = sq(norm(h[A[x]] + bc[0] * (c[T[x]] - c[A[x]]) + bc[1] * (c[T[x]] - h[A[x]])), h[T[x]])
    arms["pooled"] = pooled; arms["free"] = free; out = {}
    for k, v in arms.items():
        bs = [Vt[d].mean() / v[d].mean() for d in draws]; out[k] = [float(Vt.mean() / v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["coefficients of the free combination (contrast, simulator minus anchor), range over programmes"] = [[float(np.min([x[0] for x in coef])), float(np.max([x[0] for x in coef]))], [float(np.min([x[1] for x in coef])), float(np.max([x[1] for x in coef]))]]
    out["correlation of the simulator's errors on target and anchor"] = float(np.mean([(c[a] - h[a]) @ (c[b] - h[b]) for a, b in pairs]) / np.mean([sq(c[a], h[a]) for a, b in pairs]))
    out["by programme"] = {S[g]: {k: float(Vt[ps == g].mean() / v[ps == g].mean()) for k, v in arms.items()} for g in np.unique(ps)}
    res["models"][m] = out
    print(f"{m:18s} " + " | ".join(f"{k} {out[k][0]:.1f} [{out[k][1]:.1f}, {out[k][2]:.1f}]" for k in arms) + f" | error correlation target-anchor {out['correlation of the simulator' + chr(39) + 's errors on target and anchor']:.2f} | free coef {coef[0][0]:.2f}, {coef[0][1]:.2f}", flush=True)
json.dump(res, open(ROOT + "/results/z_anchor.json", "w"), indent=1)
