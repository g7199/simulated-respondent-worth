"""Further checks for paper4 (no new calls except those already stored): reading other simulators' answers, shown answers from 100 respondents,
the gain without sibling items, single groups under design effects, and multiplicative / raking baselines for groups.  -> results/w_extra.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, numpy as np
sys.path.insert(0, (_REPO + "/src"))
from w_groups import load, norm, GROUPS, KIND
from w_util import read, MODELS
T = load(); N = len(T); SOL = (_REPO + "/outputs/sol"); rng = np.random.default_rng(0); nz_ = lambda p: np.asarray(p, float) / np.sum(p); cl = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum()
th = [nz_(t["percent"]) for t in T]; V = np.array([(x * (1 - x)).sum() for x in th]); noise = V / np.array([t["n"] for t in T]); wave = np.array([t["wave"] for t in T]); waves = sorted(set(wave))
bk = {}; BLOCK = np.array([bk.setdefault((t["wave"], norm(t["text"])), len(bk)) for t in T])
r2 = np.random.default_rng(11); half = {}
for w in waves:
    ii = [i for i in range(N) if wave[i] == w]; perm = r2.permutation(ii)
    for k, i in enumerate(perm):
        half[int(i)] = k % 2
opt = lambda i: tuple(norm(o) for o in T[i]["options"]); shown = {i: [j for j in range(N) if wave[j] == wave[i] and half[j] != half[i]] for i in range(N)}
sib = np.array([any(opt(j) == opt(i) and norm(T[j]["text"]) == norm(T[i]["text"]) for j in shown[i]) for i in range(N)]); ALL = np.arange(N); NOSIB = np.where(~sib)[0]
def rate(P, ii):
    ii = np.asarray(ii); return float(V[ii].mean() / max(np.mean([(P[i] - th[i]) @ (P[i] - th[i]) for i in ii]) - noise[ii].mean(), 1e-9))
def draws(ii, B=1000):
    ii = np.asarray(ii); bl = np.unique(BLOCK[ii]); return [np.concatenate([ii[BLOCK[ii] == b] for b in rng.choice(bl, len(bl))]) for _ in range(B)]
def setrate(Ps, ii):
    E = [[P[i] - th[i] for i in ii] for P in Ps]; G = np.array([[np.mean([x @ y for x, y in zip(a, b)]) - noise[ii].mean() for b in E] for a in E]); keep = list(range(len(Ps)))
    while True:
        w = V[ii].mean() * np.linalg.solve(G[np.ix_(keep, keep)], np.ones(len(keep)))
        if w.min() >= -1e-9 or len(keep) == 1:
            return float(w.sum())
        keep.pop(int(np.argmin(w)))
res = {"no sibling item shown": {"questions": int(len(NOSIB))}, "peers": {}, "shown answers from 100 respondents": {}, "groups": {}}
D = draws(NOSIB)
print(f"without sibling items: {len(NOSIB)} questions")
for nm, tag in MODELS:
    A = read(f"{SOL}/egplain_{tag}"); B = read(f"{SOL}/egctx_{tag}")
    if len(A) >= N and len(B) >= N:
        r = [rate(B, d) / rate(A, d) for d in D]; res["no sibling item shown"][nm] = {"cold": rate(A, NOSIB), "shown": rate(B, NOSIB), "ratio": [rate(B, NOSIB) / rate(A, NOSIB), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
        v = res["no sibling item shown"][nm]; print(f"   {nm:18s} {v['cold']:.0f} -> {v['shown']:.0f}  x{v['ratio'][0]:.2f} [{v['ratio'][1]:.2f}, {v['ratio'][2]:.2f}]")
PA = read(f"{SOL}/egplain_gpt-6-astra"); PB = read(f"{SOL}/egplain_gpt-6.1-sol"); DA = draws(ALL)
for nm, tag in MODELS:
    P = read(f"{SOL}/egpeers_{tag}"); A = read(f"{SOL}/egplain_{tag}"); C = read(f"{SOL}/egctx_{tag}"); Z = read(f"{SOL}/egctxn100_{tag}")
    if len(P) >= N:
        r = [rate(P, d) / rate(A, d) for d in DA]; res["peers"][nm] = {"cold": rate(A, ALL), "shown the cold answers of GPT-6-astra and GPT-6.1-sol": rate(P, ALL), "ratio": [rate(P, ALL) / rate(A, ALL), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                                                                         "the three cold simulators pooled with the best weights": setrate([A, PA, PB], ALL), "best of the three cold": max(rate(A, ALL), rate(PA, ALL), rate(PB, ALL))}
        v = res["peers"][nm]; print(f"peers {nm:18s} cold {v['cold']:.1f} | reads the two others' answers {v['shown the cold answers of GPT-6-astra and GPT-6.1-sol']:.1f} (x{v['ratio'][0]:.2f} [{v['ratio'][1]:.2f}, {v['ratio'][2]:.2f}]) | the three pooled {v['the three cold simulators pooled with the best weights']:.1f} | best of the three {v['best of the three cold']:.1f}")
    if len(Z) >= N:
        r = [rate(Z, d) / rate(A, d) for d in DA]; res["shown answers from 100 respondents"][nm] = {"cold": rate(A, ALL), "shown, 100 respondents": rate(Z, ALL), "shown, full sample": rate(C, ALL), "ratio to cold": [rate(Z, ALL) / rate(A, ALL), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
        v = res["shown answers from 100 respondents"][nm]; print(f"noise {nm:18s} cold {v['cold']:.1f} | shown answers of 100 respondents {v['shown, 100 respondents']:.1f} (x{v['ratio to cold'][0]:.2f} [{v['ratio to cold'][1]:.2f}, {v['ratio to cold'][2]:.2f}]) | of the full sample {v['shown, full sample']:.1f}")
# ---- groups
PART = {"sex": ["Male", "Female"], "age": ["18-29", "30-44", "45-64", "65+"], "education": ["No degree", "College grad"], "party": ["Dem", "Ind", "Rep"]}
for nm, tag in MODELS:
    J = read(f"{SOL}/egjoint_{tag}"); S = read(f"{SOL}/egshown_{tag}"); qs = [i for i in range(N) if all(100 * i + g in J for g in range(21)) and all(100 * i + g in S for g in range(1, 21))]
    if len(qs) < 220:
        continue
    o = {"by group": {}, "by kind, design effect": {}, "baselines": {}}
    def cellstats(cells, est):
        hg = [nz_(T[i]["groups"][GROUPS[g]]["p"]) for i, g in cells]; m = np.array([T[i]["groups"][GROUPS[g]]["n"] for i, g in cells], float); Vc = np.array([(x * (1 - x)).sum() for x in hg]); pi = m / np.array([T[i]["n"] for i, g in cells])
        e2 = np.array([(est[c] - hg[c]) @ (est[c] - hg[c]) for c in range(len(cells))]); return Vc, e2, Vc / m, pi
    for gname in ("Black", "Hispanic", "18-29", "White"):
        g = GROUPS.index(gname); cells = [(i, g) for i in qs]; out = {"respondents (median)": float(np.median([T[i]["groups"][gname]["n"] for i in qs]))}
        for lab, est, kind in (("cold", [J[100 * i + 1 + g] for i in qs], "p"), ("shown the overall result", [S[100 * i + 1 + g] for i in qs], "t"), ("overall result alone", [th[i] for i in qs], "t")):
            Vc, e2, nzc, pi = cellstats(cells, est); corr = nzc if kind == "p" else (1 - 2 * pi) * nzc; out[lab] = {str(d): float(Vc.mean() / max(e2.mean() - d * corr.mean(), 1e-9)) for d in (1.0, 1.5, 2.0)}
        o["by group"][gname] = out
    for kd in ("sex", "race", "age", "education", "2024 vote", "ideology", "MAGA", "party"):
        cells = [(i, g) for i in qs for g in range(20) if KIND[GROUPS[g]] == kd]; Vc, e2, nzc, pi = cellstats(cells, [S[100 * i + 1 + g] for i, g in cells]); o["by kind, design effect"][kd] = {str(d): float(Vc.mean() / max(e2.mean() - d * ((1 - 2 * pi) * nzc).mean(), 1e-9)) for d in (1.0, 1.5, 2.0)}
    # baselines on all cells: additive anchor, multiplicative anchor; on the four partitions that cover everyone: raking to the overall result with the published group sizes
    cells = [(i, g) for i in qs for g in range(20)]; add = [cl(th[i] + J[100 * i + 1 + g] - J[100 * i]) for i, g in cells]; mul = [cl(J[100 * i + 1 + g] * th[i] / np.clip(J[100 * i], 1e-3, None)) for i, g in cells]; sh = [S[100 * i + 1 + g] for i, g in cells]
    for lab, est in (("additive anchor", add), ("multiplicative anchor", mul), ("shown the overall result", sh)):
        Vc, e2, nzc, pi = cellstats(cells, est); o["baselines"][lab] = float(Vc.mean() / (e2.mean() - ((1 - 2 * pi) * nzc).mean()))
    pc = []; rk = []; ad = []; shp = []
    for i in qs:
        for kd, gl in PART.items():
            gi = [GROUPS.index(x) for x in gl]; wts = np.array([T[i]["groups"][x]["n"] for x in gl], float); wts /= wts.sum(); M = np.array([np.clip(J[100 * i + 1 + g], 1e-4, None) for g in gi])
            for _ in range(50):
                M = M * (th[i] / np.clip(wts @ M, 1e-9, None))[None, :]; M = M / M.sum(1, keepdims=True)
            for a, g in enumerate(gi):
                pc.append((i, g)); rk.append(M[a]); ad.append(cl(th[i] + J[100 * i + 1 + g] - J[100 * i])); shp.append(S[100 * i + 1 + g])
    for lab, est in (("raked to the overall result (sex, age, education, party)", rk), ("additive anchor on the same groups", ad), ("shown the overall result, same groups", shp)):
        Vc, e2, nzc, pi = cellstats(pc, est); o["baselines"][lab] = float(Vc.mean() / (e2.mean() - ((1 - 2 * pi) * nzc).mean()))
    res["groups"][nm] = o
    print(f"groups {nm:18s} Black: cold {o['by group']['Black']['cold']['1.0']:.0f}, overall result {o['by group']['Black']['overall result alone']['1.0']:.0f}, shown {o['by group']['Black']['shown the overall result']['1.0']:.0f} (deff 1.5: {o['by group']['Black']['shown the overall result']['1.5']:.0f}, 2: {o['by group']['Black']['shown the overall result']['2.0']:.0f}) | Hispanic shown {o['by group']['Hispanic']['shown the overall result']['1.0']:.0f} ({o['by group']['Hispanic']['shown the overall result']['1.5']:.0f}, {o['by group']['Hispanic']['shown the overall result']['2.0']:.0f}) | 18-29 shown {o['by group']['18-29']['shown the overall result']['1.0']:.0f} | baselines: " + ", ".join(f"{k} {v:.0f}" for k, v in o["baselines"].items()))
json.dump(res, open((_REPO + "/results/w_extra.json"), "w"), indent=1)
