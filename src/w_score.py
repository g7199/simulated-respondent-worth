"""Scores for w_groups.py: twenty groups on post-cutoff questions (persona, contrast on the real topline, topline shown) and
population answers with the other answers of the same survey shown.  ->  results/w_groups.json
  python w_score.py [nq]"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys, numpy as np
sys.path.insert(0, (_REPO + "/src"))
from w_groups import load, norm, GROUPS, KIND
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); T = load(); N = len(T); rng = np.random.default_rng(0)
nz_ = lambda p: np.asarray(p, float) / np.sum(p); cl = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum(); sq = lambda a, b: float((a - b) @ (a - b))


from w_util import read, MODELS


th = [nz_(t["percent"]) for t in T]; V = np.array([float((x * (1 - x)).sum()) for x in th]); noise = V / np.array([t["n"] for t in T]); wave = np.array([t["wave"] for t in T])
res = {"questions": N, "context": {}, "groups": {}}

# ------------------------------------------------------------ population answers with the rest of the survey shown
r2 = np.random.default_rng(11); half = {}
for w in sorted(set(wave)):
    ii = [i for i in range(N) if T[i]["wave"] == w]; perm = r2.permutation(ii)
    for k, i in enumerate(perm):
        half[int(i)] = k % 2
opt = lambda i: tuple(norm(o) for o in T[i]["options"])
shown = {i: [j for j in range(N) if wave[j] == wave[i] and half[j] != half[i]] for i in range(N)}
same = [[j for j in shown[i] if opt(j) == opt(i)] for i in range(N)]; sib = [[j for j in same[i] if norm(T[j]["text"]) == norm(T[i]["text"])] for i in range(N)]
has_sib = np.array([len(s) > 0 for s in sib]); has_same = np.array([len(s) > 0 for s in same]); b_same = {i: np.mean([th[j] for j in same[i]], 0) for i in range(N) if same[i]}
dr = rng.integers(0, N, (2000, N))
from sklearn.feature_extraction.text import TfidfVectorizer
_X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([t["title"] + " . " + t["question"] + " . " + " ; ".join(t["options"]) for t in T]); SIM = (_X @ _X.T).toarray()
_bk = {}; BLOCK = np.array([_bk.setdefault((t["wave"], norm(t["text"])), len(_bk)) for t in T]); NB = len(_bk)


def block_draw(idx):
    """resample batteries (questions of one report that share a stem), keeping the questions of idx"""
    idx = np.asarray(idx); bl = np.unique(BLOCK[idx]); pick = rng.choice(bl, len(bl)); return np.concatenate([idx[BLOCK[idx] == b] for b in pick])


def rate(P, idx):
    idx = np.asarray(idx); e = np.array([sq(P[i], th[i]) for i in idx]); return float(V[idx].mean() / max(e.mean() - noise[idx].mean(), 1e-9))


def pair(A, B, idx):
    idx = np.asarray(idx); ea = [A[i] - th[i] for i in idx]; eb = [B[i] - th[i] for i in idx]; nz = noise[idx].mean()
    a = np.mean([x @ x for x in ea]) - nz; b = np.mean([x @ x for x in eb]) - nz; c = np.mean([x @ y for x, y in zip(ea, eb)]) - nz
    lam = min(1.0, max(0.0, (a - c) / (a + b - 2 * c))); return float(V[idx].mean() / ((1 - lam) ** 2 * a + lam ** 2 * b + 2 * lam * (1 - lam) * c)), float(lam), float(c / np.sqrt(a * b))


SUB = {"all": np.arange(N), "a sibling item of the same battery shown": np.where(has_sib)[0], "same answer options shown, no sibling": np.where(has_same & ~has_sib)[0], "no question with the same options shown": np.where(~has_same)[0]}
print(f"{N} questions; " + "; ".join(f"{k}: {len(v)}" for k, v in SUB.items()))
CTX = {}
for nm, tag in MODELS:
    A = read(f"{SOL}/egplain_{tag}"); B = read(f"{SOL}/egctx_{tag}")
    if len(A) < N or len(B) < N:
        if A or B:
            print("   context:", nm, "incomplete", len(A), len(B))
        continue
    CTX[nm] = (A, B); o = {}
    for k, idx in SUB.items():
        bs = np.array([(rate(A, r), rate(B, r)) for r in (block_draw(idx) for _ in range(1000))])
        o[k] = {"plain": [rate(A, idx), *np.percentile(bs[:, 0], [2.5, 97.5])], "shown": [rate(B, idx), *np.percentile(bs[:, 1], [2.5, 97.5])], "ratio": [rate(B, idx) / rate(A, idx), *np.percentile(bs[:, 1] / bs[:, 0], [2.5, 97.5])]}
        if k in ("a sibling item of the same battery shown", "same answer options shown, no sibling"):
            o[k]["no model: mean of the shown answers on the same options"] = rate(b_same, idx); o[k]["plain pooled with that mean"] = pair(A, b_same, idx)[0]
    # the same real answers used next to the simulator instead of inside the prompt
    import re as _re
    nsm = [np.array([bool(_re.search(r"not sure|don.?t know|no opinion|refus|unsure|prefer not|undecided", x.lower())) for x in T[i]["options"]]) for i in range(N)]
    def cal(p, Tm, d, ns):
        q = np.log(np.clip(p, 1e-4, None)) / Tm - d * ns; q = np.exp(q - q.max()); return q / q.sum()
    TG = [.5, .6, .7, .8, .9, 1, 1.15, 1.3, 1.5, 1.75, 2, 2.5, 3]; DG = [-1, -.5, 0, .5, 1, 1.5, 2]; Ccal = {}; Csh = {}; Cboth = {}; Clog = {}; Csim = {}; Csimlog = {}
    lg = lambda p: np.log(np.clip(p, 1e-3, None)); sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    for i in range(N):
        ref = shown[i]; best = min(((sum(sq(cal(A[j], Tm, d, nsm[j]), th[j]) for j in ref), Tm, d) for Tm in TG for d in DG))
        Ccal[i] = cal(A[i], best[1], best[2], nsm[i])
        Csh[i] = cl(A[i] + np.mean([th[j] - A[j] for j in same[i]], 0)) if same[i] else A[i]
        Cboth[i] = cl(Ccal[i] + np.mean([th[j] - cal(A[j], best[1], best[2], nsm[j]) for j in same[i]], 0)) if same[i] else Ccal[i]
        if same[i]:
            wt = np.array([SIM[i, j] for j in same[i]]) ** 4; wt = wt / wt.sum() if wt.sum() > 0 else np.ones(len(same[i])) / len(same[i])
            Clog[i] = sm(lg(A[i]) + np.mean([lg(th[j]) - lg(A[j]) for j in same[i]], 0)); Csim[i] = cl(A[i] + sum(w_ * (th[j] - A[j]) for w_, j in zip(wt, same[i]))); Csimlog[i] = sm(lg(A[i]) + sum(w_ * (lg(th[j]) - lg(A[j])) for w_, j in zip(wt, same[i])))
        else:
            Clog[i] = Csim[i] = Csimlog[i] = A[i]
    o["post hoc"] = {k: {"calibrated on the shown questions": rate(Ccal, idx), "shifted by the error on shown questions with the same options": rate(Csh, idx), "both": rate(Cboth, idx), "shifted on the log scale": rate(Clog, idx),
                         "shifted, weighted by the similarity of the question texts": rate(Csim, idx), "shifted on the log scale, weighted by similarity": rate(Csimlog, idx)} for k, idx in SUB.items()}
    # is the prompt better than the best use next to the simulator? paired bootstrap over questions
    cand = {"calibrated on the shown questions": Ccal, "shifted by the error on shown questions with the same options": Csh, "both": Cboth, "shifted on the log scale": Clog, "shifted, weighted by the similarity of the question texts": Csim,
            "shifted on the log scale, weighted by similarity": Csimlog}; kb = max(cand, key=lambda k: rate(cand[k], np.arange(N)))
    bsr = [rate(B, r) / rate(cand[kb], r) for r in (block_draw(np.arange(N)) for _ in range(1000))]
    o["prompt over the best use next to the simulator"] = {"which": kb, "ratio": [rate(B, np.arange(N)) / rate(cand[kb], np.arange(N)), float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5))]}
    o["dose"] = {}
    for kk in (3, 10):
        Bk = read(f"{SOL}/egctx{kk}_{tag}")
        if len(Bk) >= N:
            o["dose"][str(kk)] = rate(Bk, np.arange(N))
    res["context"][nm] = o
    print(f"   {nm:18s} " + " | ".join(f"{k[:22]}: {v['plain'][0]:.0f} -> {v['shown'][0]:.0f} (x{v['ratio'][0]:.2f} [{v['ratio'][1]:.2f}, {v['ratio'][2]:.2f}])" + (f" [no model {v['no model: mean of the shown answers on the same options']:.0f}, pooled {v['plain pooled with that mean']:.0f}]" if 'plain pooled with that mean' in v else "") for k, v in o.items() if k in SUB))
for nm in CTX:
    v = res["context"][nm]; print(f"   {nm:18s} same real answers next to the simulator (all | sibling | no same options): calibrated {v['post hoc']['all']['calibrated on the shown questions']:.0f} | {v['post hoc']['a sibling item of the same battery shown']['calibrated on the shown questions']:.0f} | {v['post hoc']['no question with the same options shown']['calibrated on the shown questions']:.0f}; "
          f"shifted {v['post hoc']['all']['shifted by the error on shown questions with the same options']:.0f} | {v['post hoc']['a sibling item of the same battery shown']['shifted by the error on shown questions with the same options']:.0f}; both {v['post hoc']['all']['both']:.0f} | {v['post hoc']['a sibling item of the same battery shown']['both']:.0f}   vs in the prompt {v['all']['shown'][0]:.0f} | {v['a sibling item of the same battery shown']['shown'][0]:.0f} | {v['no question with the same options shown']['shown'][0]:.0f}")
if len(CTX) > 1:
    nm_ = list(CTX); res["context"]["pairs"] = {}
    for a in range(len(nm_)):
        for b in range(a + 1, len(nm_)):
            p0 = pair(CTX[nm_[a]][0], CTX[nm_[b]][0], np.arange(N)); p1 = pair(CTX[nm_[a]][1], CTX[nm_[b]][1], np.arange(N)); b0 = max(rate(CTX[nm_[a]][0], np.arange(N)), rate(CTX[nm_[b]][0], np.arange(N))); b1 = max(rate(CTX[nm_[a]][1], np.arange(N)), rate(CTX[nm_[b]][1], np.arange(N)))
            res["context"]["pairs"][f"{nm_[a]} + {nm_[b]}"] = {"plain": {"pair": p0[0], "best": b0, "gain": p0[0] / b0 - 1, "corr": p0[2]}, "shown": {"pair": p1[0], "best": b1, "gain": p1[0] / b1 - 1, "corr": p1[2]}}
    g0 = [v["plain"]["gain"] for v in res["context"]["pairs"].values()]; g1 = [v["shown"]["gain"] for v in res["context"]["pairs"].values()]
    print(f"   pairs of simulators ({len(g0)}): gain over the stronger, plain median {100 * np.median(g0):.1f}% (max {100 * max(g0):.1f}%); both shown the survey: median {100 * np.median(g1):.1f}% (max {100 * max(g1):.1f}%)")

def set_rate(G, Vm):
    """what a set of sources is worth (Proposition 1) with non-negative weights: the most negative weight is dropped and the system solved again"""
    keep = list(range(len(G)))
    while True:
        w = Vm * np.linalg.solve(G[np.ix_(keep, keep)], np.ones(len(keep)))
        if w.min() >= -1e-9 or len(keep) == 1:
            return float(w.sum()), len(keep)
        keep.pop(int(np.argmin(w)))


if len(CTX) > 2:
    nm_ = list(CTX); o = {}
    for k, col in (("asked cold", 0), ("shown the rest of the survey", 1)):
        E = [[CTX[m][col][i] - th[i] for i in range(N)] for m in nm_]; G = np.array([[np.mean([x @ y for x, y in zip(E[a], E[b])]) - noise.mean() for b in range(len(nm_))] for a in range(len(nm_))]); rs = V.mean() / np.diag(G); sr, kept = set_rate(G, V.mean())
        o[k] = {"simulators": len(nm_), "sum of rates": float(rs.sum()), "rate of the set": sr, "best single": float(rs.max()), "non-zero weights": kept}
    res["context"]["sets"] = o
    print("   sets: " + " | ".join(f"{k}: {v['simulators']} simulators sum {v['sum of rates']:.0f}, together {v['rate of the set']:.1f}, best {v['best single']:.1f}" for k, v in o.items()))

# ------------------------------------------------------------ twenty groups
GRP = {}
nq = int(sys.argv[1]) if len(sys.argv) > 1 else N
for nm, tag in MODELS:
    J = read(f"{SOL}/egjoint_{tag}"); S = read(f"{SOL}/egshown_{tag}")
    qs = [i for i in range(nq) if all(100 * i + g in J for g in range(21))]
    if len(qs) < 20:
        continue
    qs_s = [i for i in qs if all(100 * i + g in S for g in range(1, 21))]
    cells = [(i, g) for i in qs for g in range(20)]; C = len(cells); hg = [nz_(T[i]["groups"][GROUPS[g]]["p"]) for i, g in cells]; m = np.array([T[i]["groups"][GROUPS[g]]["n"] for i, g in cells], float)
    Vc = np.array([float((x * (1 - x)).sum()) for x in hg]); pi = m / np.array([T[i]["n"] for i, g in cells]); nzc = Vc / m; tc = (1 - 2 * pi) * nzc; qc = np.array([i for i, g in cells]); kind = np.array([KIND[GROUPS[g]] for i, g in cells]); gname = np.array([GROUPS[g] for i, g in cells])
    e = {"topline": (np.array([sq(th[i], hg[c]) for c, (i, g) in enumerate(cells)]), tc), "persona": (np.array([sq(J[100 * i + 1 + g], hg[c]) for c, (i, g) in enumerate(cells)]), nzc),
         "topline + contrast": (np.array([sq(cl(th[i] + J[100 * i + 1 + g] - J[100 * i]), hg[c]) for c, (i, g) in enumerate(cells)]), tc)}
    # free coefficients on the three numbers (overall result, the model's answer for the group and for everyone), fitted on the other half of the questions
    fold_ = np.array([i % 2 for i, g in cells]); X_ = [np.stack([th[i], J[100 * i + 1 + g], J[100 * i]], 1) for i, g in cells]; fr = np.zeros(C); cf = []
    for f_ in (0, 1):
        tr_ = np.where(fold_ != f_)[0]; b_ = np.linalg.lstsq(np.concatenate([X_[c] for c in tr_]), np.concatenate([hg[c] for c in tr_]), rcond=None)[0]; cf.append([float(x) for x in b_])
        for c in np.where(fold_ == f_)[0]:
            fr[c] = sq(cl(X_[c] @ b_), hg[c])
    e["free coefficients"] = (fr, tc)
    e["multiplicative anchor"] = (np.array([sq(cl(J[100 * i + 1 + g] * th[i] / np.clip(J[100 * i], 1e-3, None)), hg[c]) for c, (i, g) in enumerate(cells)]), tc)          # the group's answer times (real overall result / simulated overall result)
    # a difference without a model: the group's real gap to everyone on the fielded questions of the same survey that have the same answer options
    hasref = np.array([len(same[i]) > 0 for i, g in cells])
    e["topline + real difference on other questions"] = (np.array([sq(cl(th[i] + np.mean([nz_(T[j]["groups"][GROUPS[g]]["p"]) - th[j] for j in same[i]], 0)), hg[c]) if same[i] else np.nan for c, (i, g) in enumerate(cells)]), tc)
    ins = np.array([i in set(qs_s) for i, g in cells])
    if len(qs_s) >= 20:
        ok_s = set(qs_s); e["topline shown"] = (np.array([sq(S[100 * i + 1 + g], hg[c]) if i in ok_s else np.nan for c, (i, g) in enumerate(cells)]), tc)
    X = read(f"{SOL}/egshownx_{tag}"); qs_x = [i for i in qs if all(100 * i + g in X for g in range(1, 21))]
    if len(qs_x) >= 0.9 * N:
        ok_x = set(qs_x); e["topline and five other questions' full results shown"] = (np.array([sq(X[100 * i + 1 + g], hg[c]) if i in ok_x else np.nan for c, (i, g) in enumerate(cells)]), tc)
    bq = np.array([BLOCK[i] for i, g in cells]); ub = np.unique(bq); boots = [np.concatenate([np.where(bq == b)[0] for b in rng.choice(ub, len(ub))]) for _ in range(400)]

    def gr(k, idx):
        err, corr = e[k]; idx = idx[~np.isnan(err[idx])]; return float(Vc[idx].mean() / max(err[idx].mean() - corr[idx].mean(), 1e-9))
    o = {"questions": len(qs), "cells": C, "all": {k: [gr(k, np.arange(C)), *np.percentile([gr(k, r) for r in boots], [2.5, 97.5])] for k in e}, "by kind": {}, "by group": {}}
    for kd in ["sex", "race", "age", "education", "2024 vote", "ideology", "MAGA", "party"]:
        o["by kind"][kd] = {k: gr(k, np.where(kind == kd)[0]) for k in e}
    for g in GROUPS:
        o["by group"][g] = {k: gr(k, np.where(gname == g)[0]) for k in e}
    if "topline shown" in e:
        d = [gr("topline shown", r) / gr("topline + contrast", r) for r in boots]; o["shown over contrast"] = [o["all"]["topline shown"][0] / o["all"]["topline + contrast"][0], *np.percentile(d, [2.5, 97.5])]
        d = [gr("topline shown", r) / gr("persona", r) for r in boots]; o["shown over persona"] = [o["all"]["topline shown"][0] / o["all"]["persona"][0], *np.percentile(d, [2.5, 97.5])]
    if "topline and five other questions' full results shown" in e:
        kx = "topline and five other questions' full results shown"; d = [gr(kx, r) / gr("topline shown", r) for r in boots]; o["more shown over topline shown"] = [o["all"][kx][0] / o["all"]["topline shown"][0], *np.percentile(d, [2.5, 97.5])]
        print(f"      {nm}: with the full results of five other questions also shown {o['all'][kx][0]:.0f} [{o['all'][kx][1]:.0f}, {o['all'][kx][2]:.0f}], x{o['more shown over topline shown'][0]:.2f} [{o['more shown over topline shown'][1]:.2f}, {o['more shown over topline shown'][2]:.2f}] over the topline shown")
    o["cells with a same-options question fielded"] = {k: gr(k, np.where(hasref)[0]) for k in e}
    o["free coefficients on overall result, group, everyone"] = cf
    o["design effect"] = {str(d): {k: float(Vc.mean() / max(e[k][0][~np.isnan(e[k][0])].mean() - d * e[k][1][~np.isnan(e[k][0])].mean(), 1e-9)) for k in e} for d in (1.0, 1.5, 2.0)}
    res["groups"][nm] = o
    if "topline shown" in e and len(qs_s) >= 0.9 * N:
        GRP[nm] = {"q": set(qs_s), "S": S, "J": J}
    print(f"{nm:18s} {len(qs)} questions, {C} cells | " + " | ".join(f"{k} {v[0]:.0f} [{v[1]:.0f}, {v[2]:.0f}]" for k, v in o["all"].items()) + (f" | shown/persona x{o['shown over persona'][0]:.1f} [{o['shown over persona'][1]:.1f}, {o['shown over persona'][2]:.1f}]" if "shown over persona" in o else ""))
    print("      by kind: " + " | ".join(f"{kd}: " + "/".join(f"{o['by kind'][kd][k]:.0f}" for k in e) for kd in o["by kind"]))
if len(GRP) > 1:
    nm_ = list(GRP); cq = sorted(set.intersection(*[GRP[m]["q"] for m in nm_])); cells = [(i, g) for i in cq for g in range(20)]; hg = [nz_(T[i]["groups"][GROUPS[g]]["p"]) for i, g in cells]; m_ = np.array([T[i]["groups"][GROUPS[g]]["n"] for i, g in cells], float)
    Vc = np.array([float((x * (1 - x)).sum()) for x in hg]); tcm = ((1 - 2 * m_ / np.array([T[i]["n"] for i, g in cells])) * Vc / m_).mean(); Vm = Vc.mean(); o = {"questions": len(cq), "simulators": nm_}
    for k in ("topline shown", "topline + contrast"):
        E = [[(GRP[m]["S"][100 * i + 1 + g] if k == "topline shown" else cl(th[i] + GRP[m]["J"][100 * i + 1 + g] - GRP[m]["J"][100 * i])) - hg[c] for c, (i, g) in enumerate(cells)] for m in nm_]
        G = np.array([[np.mean([x @ y for x, y in zip(E[a], E[b])]) - tcm for b in range(len(nm_))] for a in range(len(nm_))]); rs = Vm / np.diag(G); sr, kept = set_rate(G, Vm); cr = G / np.sqrt(np.outer(np.diag(G), np.diag(G))); iu = np.triu_indices(len(nm_), 1)
        pg = []
        for a in range(len(nm_)):
            for b in range(a + 1, len(nm_)):
                s2, _ = set_rate(G[np.ix_([a, b], [a, b])], Vm); pg.append(s2 / max(rs[a], rs[b]) - 1)
        o[k] = {"sum of rates": float(rs.sum()), "rate of the set": sr, "best single": float(rs.max()), "non-zero weights": kept, "error correlations: min, median, max": [float(cr[iu].min()), float(np.median(cr[iu])), float(cr[iu].max())], "pair gain: median, max": [float(np.median(pg)), float(max(pg))]}
    res["groups"]["sets"] = o
    print(f"groups, {len(nm_)} simulators on {len(cq)} questions: " + " | ".join(f"{k}: sum {v['sum of rates']:.0f}, together {v['rate of the set']:.0f}, best {v['best single']:.0f}, pair gain median {100 * v['pair gain: median, max'][0]:.1f}% max {100 * v['pair gain: median, max'][1]:.1f}%, error corr {v['error correlations: min, median, max'][1]:.2f}" for k, v in o.items() if isinstance(v, dict)))
json.dump(res, open(ROOT + "/results/w_groups.json", "w"), indent=1)
