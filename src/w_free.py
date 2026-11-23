"""Groups: free coefficients on the overall result, the group and everyone, fitted on the other half of the questions.  -> results/w_free.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, numpy as np
sys.path.insert(0, _REPO + "/src")
from w_groups import load, GROUPS
from w_util import read, MODELS
T = load(); N = len(T); SOL = _REPO + "/outputs/sol"; nz_ = lambda p: np.asarray(p, float) / np.sum(p); cl = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum()
th = [nz_(t["percent"]) for t in T]; out = {}
for nm, tag in MODELS:
    J = read(f"{SOL}/egjoint_{tag}"); S = read(f"{SOL}/egshown_{tag}"); qs = [i for i in range(N) if all(100 * i + g in J for g in range(21)) and all(100 * i + g in S for g in range(1, 21))]
    if len(qs) < 200: continue
    cells = [(i, g) for i in qs for g in range(20)]; hg = [nz_(T[i]["groups"][GROUPS[g]]["p"]) for i, g in cells]; m = np.array([T[i]["groups"][GROUPS[g]]["n"] for i, g in cells], float)
    Vc = np.array([(x * (1 - x)).sum() for x in hg]); tc = (1 - 2 * m / np.array([T[i]["n"] for i, g in cells])) * Vc / m; fold = np.array([i % 2 for i, g in cells])
    X = [np.stack([th[i], J[100 * i + 1 + g], J[100 * i]], 1) for i, g in cells]            # per cell: K x 3
    est = [None] * len(cells); coef = []
    for f in (0, 1):
        tr = np.where(fold != f)[0]; A = np.concatenate([X[c] for c in tr]); y = np.concatenate([hg[c] for c in tr]); b = np.linalg.lstsq(A, y, rcond=None)[0]; coef.append(b)
        for c in np.where(fold == f)[0]: est[c] = cl(X[c] @ b)
    rate = lambda E: float(Vc.mean() / (np.mean([(E[c] - hg[c]) @ (E[c] - hg[c]) for c in range(len(cells))]) - tc.mean()))
    anchor = [cl(th[i] + J[100 * i + 1 + g] - J[100 * i]) for i, g in cells]; shown = [S[100 * i + 1 + g] for i, g in cells]
    out[nm] = {"anchor": rate(anchor), "free coefficients": rate(est), "shown": rate(shown), "coefficients on overall result, group, everyone": [[float(x) for x in b] for b in coef]}
    print(f"{nm:18s} anchor {rate(anchor):.0f} | free coefficients (fitted on the other half of the questions) {rate(est):.0f} [coef {np.round(coef[0], 2)}] | shown {rate(shown):.0f}")
json.dump(out, open(_REPO + "/results/w_free.json", "w"), indent=1)
