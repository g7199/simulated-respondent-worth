"""One overlap, two effects. For two errors of equal size that correlate rho, averaging multiplies what a source is worth by 2 / (1 + rho), and differencing against a known
answer (an anchor) by 1 / (2 (1 - rho)). Measured on pairs of simulators (older surveys) and on a simulator's errors for a group and for everyone (weekly polls, party splits).  ->  results/w_overlap.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, pickle, sys, numpy as np
sys.path.insert(0, _REPO + "/src")
from w_groups import load, GROUPS
from w_util import read, MODELS
SOL = _REPO + "/outputs/sol"; T = load(); N = len(T); NG = len(GROUPS); nz = lambda p: np.asarray(p, float) / np.sum(p); cl = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum(); sq = lambda a, b: float((a - b) @ (a - b)); out = {"weekly polls, twenty groups": {}}
th = [nz(t["percent"]) for t in T]; hg = [[nz(t["groups"][g]["p"]) for g in GROUPS] for t in T]
for nm, tag in MODELS:
    J = read(f"{SOL}/egjoint_{tag}"); qs = [i for i in range(N) if all(100 * i + g in J for g in range(21))]
    if len(qs) < 0.9 * N:
        continue
    cells = [(i, g) for i in qs for g in range(NG)]; Vc = np.array([float((hg[i][g] * (1 - hg[i][g])).sum()) for i, g in cells]); m = np.array([T[i]["groups"][GROUPS[g]]["n"] for i, g in cells], float); pi = m / np.array([T[i]["n"] for i, g in cells]); nzc = Vc / m; tc = (1 - 2 * pi) * nzc
    eg = [J[100 * i + 1 + g] - hg[i][g] for i, g in cells]; ea = [J[100 * i] - th[i] for i, g in cells]; Gg = np.mean([x @ x for x in eg]) - nzc.mean(); Ga = np.mean([x @ x for x in ea]); Gx = np.mean([x @ y for x, y in zip(eg, ea)]); rho = float(Gx / np.sqrt(Gg * Ga)); cold = float(Vc.mean() / Gg)
    dif = float(Vc.mean() / (np.mean([sq(cl(th[i] + J[100 * i + 1 + g] - J[100 * i]), hg[i][g]) for i, g in cells]) - tc.mean()))
    out["weekly polls, twenty groups"][nm] = {"rho": rho, "cold": cold, "difference estimate": dif, "observed gain": dif / cold, "gain for equal errors, 1 / (2 (1 - rho))": 1 / (2 * (1 - rho))}
Z = json.load(open(_REPO + "/results/z_groups.json"))["party"]["models"]; key = "correlation of the simulator's errors for the group and for all respondents"
out["party groups, new polls"] = {m: {"rho": v[key], "cold": v["persona"]["rate"], "difference estimate": v["contrast"]["rate"], "observed gain": v["contrast"]["rate"] / v["persona"]["rate"], "gain for equal errors, 1 / (2 (1 - rho))": 1 / (2 * (1 - v[key]))} for m, v in Z.items()}
C = pickle.load(open(_REPO + "/results/y_cov_cache.pkl", "rb")); names = C["names"]; h = C["h"]; E = {m: [np.asarray(C["C"][m][k]) - h[k] for k in range(len(h))] for m in names}; G = np.array([[np.mean([x @ y for x, y in zip(E[a], E[b])]) for b in names] for a in names]); V = float(np.mean([(np.asarray(x) * (1 - np.asarray(x))).sum() for x in h]))
pr = []
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        s, w = (a, b) if G[a, a] <= G[b, b] else (b, a); lam = max(0.0, min(1.0, (G[s, s] - G[s, w]) / (G[s, s] + G[w, w] - 2 * G[s, w]))); risk = (1 - lam) ** 2 * G[s, s] + lam ** 2 * G[w, w] + 2 * lam * (1 - lam) * G[s, w]
        pr.append((G[a, b] / np.sqrt(G[a, a] * G[b, b]), G[s, s] / risk))
pr = np.array(pr); out["pairs of simulators, older surveys"] = {"median rho": float(np.median(pr[:, 0])), "median gain of a pair": float(np.median(pr[:, 1])), "gain for equal errors at that rho, 2 / (1 + rho)": float(2 / (1 + np.median(pr[:, 0])))}
json.dump(out, open(_REPO + "/results/w_overlap.json", "w"), indent=1)
for k in ("weekly polls, twenty groups", "party groups, new polls"):
    d = out[k]; rr = np.array([v["observed gain"] / v["gain for equal errors, 1 / (2 (1 - rho))"] for v in d.values()]); print(k, len(d), "rho", round(min(v["rho"] for v in d.values()), 2), round(np.median([v["rho"] for v in d.values()]), 2), round(max(v["rho"] for v in d.values()), 2), "| observed gain", round(min(v["observed gain"] for v in d.values()), 2), round(np.median([v["observed gain"] for v in d.values()]), 2), round(max(v["observed gain"] for v in d.values()), 2), "| predicted", round(min(v["gain for equal errors, 1 / (2 (1 - rho))"] for v in d.values()), 2), round(np.median([v["gain for equal errors, 1 / (2 (1 - rho))"] for v in d.values()]), 2), round(max(v["gain for equal errors, 1 / (2 (1 - rho))"] for v in d.values()), 2), "| obs/pred", round(rr.min(), 2), round(np.median(rr), 2), round(rr.max(), 2))
    print("   ", {m.replace("Claude-", "").replace("GPT-", ""): (round(v["rho"], 2), round(v["observed gain"], 1), round(v["gain for equal errors, 1 / (2 (1 - rho))"], 1)) for m, v in d.items()})
print(out["pairs of simulators, older surveys"])
