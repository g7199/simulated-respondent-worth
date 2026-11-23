"""Statistical alternatives to reading real answers in the prompt, on the 246 questions of the weekly polls.
All are fitted on the real answers of OTHER questions (five folds by question stem, tuning inside the training folds), which the simulator never sees.

A group's answer, given the question's real overall result:
  hierarchical   log(group / overall) with group-specific intercepts and slopes on the overall share and the text, shrunk by a Gaussian prior
  boosted        gradient-boosted trees on the same inputs and the group's real gap on other questions with the same options
  learned        the boosted model with the simulator's cold answers (group, everyone) as extra inputs
  small area     with m respondents per group, each estimate is pooled by the area-level (Fay-Herriot) weight m / (m + r), r from the other folds
A population's answer, given the real results of the other half of the same survey:
  learned        boosted trees on the cold answer, the text and the shown answers with the same options

  python w_strong.py  ->  results/w_strong.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, numpy as np, scipy.sparse as sp
sys.path.insert(0, _REPO + "/src")
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from w_groups import load, norm, GROUPS
from w_util import read, MODELS

from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)                       # the tree learner sums gradients in parallel, so its result depends slightly on the number of threads
SOL = _REPO + "/outputs/sol"; T = load(); N = len(T); NG = len(GROUPS); EPS = 0.005; CONFIGS = ((200, 0.06, 7), (300, 0.04, 15), (600, 0.02, 15)); M_SMALL = 25
nz = lambda p: np.asarray(p, float) / np.sum(p); cl = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum(); sq = lambda a, b: float((a - b) @ (a - b))
th = [nz(t["percent"]) for t in T]; hg = [[nz(t["groups"][g]["p"]) for g in GROUPS] for t in T]; K = [len(x) for x in th]; V = np.array([float((x * (1 - x)).sum()) for x in th]); noise = V / np.array([t["n"] for t in T])
wave = np.array([t["wave"] for t in T]); stems = np.unique([norm(t["text"]) for t in T], return_inverse=True)[1]; optkey = [tuple(norm(o) for o in t["options"]) for t in T]
folds = np.zeros(N, int)
for f, (_, te) in enumerate(GroupKFold(5).split(np.arange(N), groups=stems)):
    folds[te] = f
_b = {}; BLOCK = np.array([_b.setdefault((t["wave"], norm(t["text"])), len(_b)) for t in T])

# text of each answer option, in a few dimensions
texts = [f"{T[i]['title']} {T[i]['question']} answer {T[i]['options'][k]}" for i in range(N) for k in range(K[i])]
Z0 = TruncatedSVD(40, random_state=0).fit_transform(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit_transform(texts)); Z0 = PCA(24, random_state=0).fit_transform(Z0); Z0 = Z0 / Z0.std(0)
ZPOS = {x: j for j, x in enumerate((i, k) for i in range(N) for k in range(K[i]))}


def tune_and_predict(make_X, y, qrows, keep, categorical=None):
    """five outer folds by stem; the configuration is chosen on a held-out quarter of the training stems"""
    p = np.full(len(y), np.nan); fr = folds[qrows]
    for f in range(5):
        X = make_X(f); tr = np.where((fr != f) & keep)[0]; te = np.where((fr == f) & keep)[0]; a, b = next(GroupKFold(4).split(tr, groups=stems[qrows[tr]])); fit, val = tr[a], tr[b]; best = (np.inf, None)
        for it, lr, leaves in CONFIGS:
            m = HGB(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=20, l2_regularization=1.0, categorical_features=categorical, random_state=0, early_stopping=False).fit(X[fit], y[fit]); e = float(np.mean((m.predict(X[val]) - y[val]) ** 2))
            if e < best[0]:
                best = (e, (it, lr, leaves))
        it, lr, leaves = best[1]; p[te] = HGB(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=20, l2_regularization=1.0, categorical_features=categorical, random_state=0, early_stopping=False).fit(X[tr], y[tr]).predict(X[te])
    return p


# ------------------------------------------------------------------ a population's answer, with the other half of the same survey available
r2 = np.random.default_rng(11); half = {}
for w in sorted(set(wave)):
    ii = [i for i in range(N) if T[i]["wave"] == w]; perm = r2.permutation(ii)
    for k, i in enumerate(perm):
        half[int(i)] = k % 2
shown = {i: [j for j in range(N) if wave[j] == wave[i] and half[j] != half[i]] for i in range(N)}; same = [[j for j in shown[i] if optkey[j] == optkey[i]] for i in range(N)]
_X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([t["title"] + " . " + t["question"] + " . " + " ; ".join(t["options"]) for t in T]); SIM = (_X @ _X.T).toarray()
prow = np.array([(i, k) for i in range(N) for k in range(K[i])]); Zp = Z0[[ZPOS[(i, k)] for i, k in prow]]; lg = lambda x: np.log(np.asarray(x) + EPS)
rate = lambda P, idx: float(V[idx].mean() / max(np.mean([sq(P[i], th[i]) for i in idx]) - noise[idx].mean(), 1e-9))


def shown_features(A=None):
    """what the shown answers with the same options say about this option: their mean, and their error against the cold answer (plain and weighted by text similarity)"""
    out = np.full((len(prow), 3), np.nan)
    for r, (i, k) in enumerate(prow):
        if same[i]:
            w = np.array([SIM[i, j] ** 4 for j in same[i]]) + 1e-6; w = w / w.sum(); real = np.array([lg(th[j][k]) for j in same[i]]); out[r, 0] = real.mean(); base = np.array([lg(A[j][k]) for j in same[i]]) if A is not None else 0 * real
            out[r, 1] = (real - base).mean(); out[r, 2] = float(w @ (real - base))
    return out


def shares_pop(pred, base):
    out = [np.zeros(K[i]) for i in range(N)]
    for (i, k), v in zip(prow, pred):
        out[i][k] = v
    return {i: cl((base[i] + EPS) * np.exp(out[i])) for i in range(N)} if base is not None else {i: cl(np.exp(out[i])) for i in range(N)}


res = {"population": {}, "groups": {}, "folds": 5}; ypn = np.array([lg(th[i][k]) for i, k in prow]); Xn = np.column_stack([[K[i] for i, k in prow], prow[:, 1], shown_features()[:, 0], Zp])
Pn = shares_pop(tune_and_predict(lambda f: Xn, ypn, prow[:, 0], np.ones(len(prow), bool)), None); res["population without a simulator"] = rate(Pn, np.arange(N)); print("population, no simulator:", round(res["population without a simulator"], 1), flush=True)
for nm, tag in MODELS:
    A = read(f"{SOL}/egplain_{tag}"); B = read(f"{SOL}/egctx_{tag}")
    if len(A) < N or len(B) < N:
        continue
    la = np.array([lg(A[i][k]) for i, k in prow]); y = ypn - la; X = np.column_stack([la, [K[i] for i, k in prow], prow[:, 1], shown_features(A), Zp]); PL = shares_pop(tune_and_predict(lambda f: X, y, prow[:, 0], np.ones(len(prow), bool)), A)
    bl = np.unique(BLOCK); by = [np.where(BLOCK == b)[0] for b in bl]; rb = np.random.default_rng(0); dr = []
    for _ in range(2000):
        x = np.concatenate([by[j] for j in rb.integers(0, len(bl), len(bl))]); dr.append(rate(B, x) / rate(PL, x))
    all_ = np.arange(N); res["population"][nm] = {"cold": rate(A, all_), "learned": rate(PL, all_), "shown": rate(B, all_), "shown over learned": [rate(B, all_) / rate(PL, all_), float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]}
    v = res["population"][nm]; print(f"population {nm:18s} cold {v['cold']:.1f} learned {v['learned']:.1f} shown {v['shown']:.1f} ratio {v['shown over learned'][0]:.2f} [{v['shown over learned'][1]:.2f}, {v['shown over learned'][2]:.2f}]", flush=True)

# ------------------------------------------------------------------ a group's answer, with the question's real overall result
rows = np.array([(i, g, k) for i in range(N) for g in range(NG) for k in range(K[i])]); Ri, Rg, Rk = rows.T
span = {}; c_ = 0
for i in range(N):
    span[i] = (c_, c_ + NG * K[i]); c_ += NG * K[i]
y = np.array([lg(hg[i][g][k]) - lg(th[i][k]) for i, g, k in rows]); lt = np.array([np.log(th[i][k] + EPS) - np.log(1 - th[i][k] + EPS) for i, g, k in rows]); Z = Z0[[ZPOS[(i, k)] for i, g, k in rows]]
D = {i: y[span[i][0]:span[i][1]].reshape(NG, K[i]) for i in range(N)}; bucket = {}
for i, o in enumerate(optkey):
    bucket.setdefault(o, []).append(i)


def gap(f):
    """the group's mean real gap to everyone on other questions with the same options, from the training folds only (leaving the question itself out)"""
    out = np.full(len(rows), np.nan)
    for js in bucket.values():
        use = [j for j in js if folds[j] != f]
        if not use:
            continue
        tot = np.sum([D[j] for j in use], 0)
        for j in js:
            if folds[j] != f and len(use) > 1:
                out[span[j][0]:span[j][1]] = ((tot - D[j]) / (len(use) - 1)).ravel()
            elif folds[j] == f:
                out[span[j][0]:span[j][1]] = (tot / len(use)).ravel()
    return out


GAP = {f: gap(f) for f in range(5)}; Xb = np.column_stack([Rg, lt, [K[i] for i in Ri], Rk, Z]); ALL = np.ones(len(rows), bool)
gm = sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), Rg)), shape=(len(rows), NG)); Xh = sp.hstack([gm, gm.multiply(sp.csr_matrix(lt[:, None])), *[gm.multiply(sp.csr_matrix(Z[:, [j]])) for j in range(Z.shape[1])], sp.csr_matrix(np.column_stack([lt, Z]))]).tocsr()


def to_shares(pred):
    out = {(i, g): np.zeros(K[i]) for i in range(N) for g in range(NG)}
    for i, g, k, v in zip(Ri, Rg, Rk, pred):
        out[(i, g)][k] = v
    return {c: cl((th[c[0]] + EPS) * np.exp(v)) for c, v in out.items()}


def fit_hier():
    p = np.zeros(len(y)); fr = folds[Ri]
    for f in range(5):
        tr = np.where(fr != f)[0]; te = np.where(fr == f)[0]; a, b = next(GroupKFold(4).split(tr, groups=stems[Ri[tr]])); fit, val = tr[a], tr[b]
        al = min((3, 10, 30, 100, 300, 1000, 3000), key=lambda x: float(np.mean((Ridge(alpha=x).fit(Xh[fit], y[fit]).predict(Xh[val]) - y[val]) ** 2))); p[te] = Ridge(alpha=al).fit(Xh[tr], y[tr]).predict(Xh[te])
    return p


def cell_stats(qs):
    c = [(i, g) for i in qs for g in range(NG)]; Vc = np.array([float((hg[i][g] * (1 - hg[i][g])).sum()) for i, g in c]); m = np.array([T[i]["groups"][GROUPS[g]]["n"] for i, g in c], float); return c, Vc, (1 - 2 * m / np.array([T[i]["n"] for i, g in c])) * Vc / m


def score(P, qs):
    c, Vc, tc = cell_stats(qs); return float(Vc.mean() / max(np.mean([sq(P[x], hg[x[0]][x[1]]) for x in c]) - tc.mean(), 1e-9))


def small_area(P, qs, m=M_SMALL, seeds=3):
    """pooled with m respondents of the group by the area-level weight m / (m + r); r from the questions of the other folds"""
    c, Vc, tc = cell_stats(qs); rf = {f: score(P, [j for j in qs if folds[j] != f]) for f in range(5)}; out = []
    for s in range(seeds):
        rg = np.random.default_rng(100 + s); err = []
        for (i, g), t in zip(c, tc):
            r = max(rf[folds[i]], 0.1); d = rg.multinomial(m, hg[i][g]) / m; err.append(sq((m * d + r * P[(i, g)]) / (m + r), hg[i][g]) - (r / (m + r)) ** 2 * t)
        out.append(Vc.mean() / np.mean(err))
    return float(np.mean(out))


PH = to_shares(fit_hier()); PB = to_shares(tune_and_predict(lambda f: np.column_stack([Xb, GAP[f]]), y, Ri, ALL, categorical=[0])); print("groups: no-simulator models fitted", flush=True)
for nm, tag in MODELS:
    J = read(f"{SOL}/egjoint_{tag}"); S = read(f"{SOL}/egshown_{tag}"); qs = [i for i in range(N) if all(100 * i + g in J for g in range(21)) and all(100 * i + g in S for g in range(1, 21))]
    if len(qs) < 0.9 * N:
        continue
    has = np.isin(Ri, qs); a_ = np.zeros(len(y)); b_ = np.zeros(len(y)); s_ = np.zeros(len(y))
    for r in np.where(has)[0]:
        i, g, k = rows[r]; a_[r] = lg(J[100 * i + 1 + g][k]); b_[r] = lg(J[100 * i][k]); s_[r] = lg(S[100 * i + 1 + g][k])
    Xc = np.column_stack([Xb, a_, b_, a_ - b_]); PC = to_shares(np.nan_to_num(tune_and_predict(lambda f: np.column_stack([Xc, GAP[f]]), y, Ri, has, categorical=[0])))
    Xs = np.column_stack([Xc, s_, s_ - lg(np.array([th[i][k] for i, g, k in rows]))]); PX = to_shares(np.nan_to_num(tune_and_predict(lambda f: np.column_stack([Xs, GAP[f]]), y, Ri, has, categorical=[0])))
    PS = {(i, g): S[100 * i + 1 + g] for i in qs for g in range(NG)}; PT = {(i, g): th[i] for i in qs for g in range(NG)}
    c, Vc, tc = cell_stats(qs); st = stems[qs]; us = np.unique(st); by = [np.where(st == u)[0] for u in us]; qt = tc.reshape(len(qs), NG).mean(1); qv = Vc.reshape(len(qs), NG).mean(1)
    qe = {key: np.array([np.mean([sq(P[(i, g)], hg[i][g]) for g in range(NG)]) for i in qs]) for key, P in (("c", PC), ("s", PS))}; rb = np.random.default_rng(0); dr = []
    for _ in range(2000):
        x = np.concatenate([by[j] for j in rb.integers(0, len(us), len(us))]); dr.append((qe["c"][x].mean() - qt[x].mean()) / (qe["s"][x].mean() - qt[x].mean()))
    o = {"questions": len(qs), "overall result": score(PT, qs), "hierarchical": score(PH, qs), "boosted": score(PB, qs), "learned": score(PC, qs), "shown": score(PS, qs), "shown, then learned": score(PX, qs), "shown over learned": [score(PS, qs) / score(PC, qs), float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
         f"with {M_SMALL} respondents per group": {k: small_area(P, qs) for k, P in (("overall result", PT), ("hierarchical", PH), ("boosted", PB), ("learned", PC), ("shown", PS))}}
    res["groups"][nm] = o; print(f"groups {nm:18s} hier {o['hierarchical']:.0f} boosted {o['boosted']:.0f} learned {o['learned']:.0f} shown {o['shown']:.0f} shown+learned {o['shown, then learned']:.0f} ratio {o['shown over learned'][0]:.2f} [{o['shown over learned'][1]:.2f}, {o['shown over learned'][2]:.2f}] | m={M_SMALL}: " + " ".join(f"{k} {v:.0f}" for k, v in o[f"with {M_SMALL} respondents per group"].items()), flush=True)
json.dump(res, open(_REPO + "/results/w_strong.json", "w"), indent=1)
