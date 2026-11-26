"""Paper 3 (third revision): the structure of the errors that simulators share.

Reads results/y_cov_cache.pkl (written by y_cov.py) and tests, for the survey items, the new polls and the annotator prompts:
  nested   G_ij = min(G_ii, G_jj): the error of the weaker simulator is the error of the stronger one plus noise that is
           uncorrelated with it; equivalently the error correlation is sqrt(r_weak / r_strong), the value at which the weaker
           simulator receives zero weight next to the stronger one. Compared with 'one core plus independent noise'
           (G_ij = G_best for every pair) and with independence (G_ij = 0).
  ceiling  error correlation between two runs of the same model (GPT-6.1-sol, 280 items) and between reasoning settings
  stack    combiners fitted on the other programmes: convex pooling, unconstrained linear stacking with an intercept, and
           a log-linear pool with a free temperature (extremizing)
  price    does the disagreement between two simulators predict the error of the better one, and is a pseudo-count that
           depends on it worth more than a constant one
  anchor   rates of trivial predictors, uncalibrated rates, rates by programme, by country and tracker status of the new polls
  safety   interval for the largest overestimate of a rate, share of items on which pooling raises the expected error
  shape    the direction of the shared error: shrinkage towards the typical answer on the same scale and towards another population
Usage: python z_nest.py -> results/z_nest.json
"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, re, json, pickle, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); Z = pickle.load(open(ROOT + "/results/y_cov_cache.pkl", "rb")); rng = np.random.default_rng(0)
names = Z["names"]; J = len(names); h = Z["h"]; N = len(h); C = Z["C"]; V = Z["V"]; sid = Z["sid"]; S = Z["S"]; qid = Z["qid"]; ns = Z["ns"]; ids = Z["ids"]
soft = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
fl = lambda p, e: (np.asarray(p, float) + e) / (np.asarray(p, float) + e).sum()
res = {}


def gram(E):
    """per-item Gram matrices of a list of error lists"""
    return np.array([[[float(E[a][k] @ E[b][k]) for b in range(len(E))] for a in range(len(E))] for k in range(len(E[0]))])


def joint(Gs, Vv):
    from nnweights import best_weights
    full = best_weights(Gs, Vv)
    return float(full.sum()), full


def nested(Gi, Vv, labels, cl, tag, B=1000):
    """observed correlation of every pair against the nested value sqrt(r_weak / r_strong) and the one-core value sqrt(r_i r_j) / r_best"""
    G = Gi.mean(0); r = Vv.mean() / np.diag(G); n = len(labels); best = int(np.argmax(r)); rows = []
    for a, b in itertools.combinations(range(n), 2):
        rows.append({"pair": [labels[a], labels[b]], "observed": float(G[a, b] / np.sqrt(G[a, a] * G[b, b])), "nested": float(np.sqrt(min(r[a], r[b]) / max(r[a], r[b]))),
                     "one core": float(min(1.0, np.sqrt(r[a] * r[b]) / r[best])), "ratio": float(G[a, b] / min(G[a, a], G[b, b]))})
    o = np.array([x["observed"] for x in rows]); p = np.array([x["nested"] for x in rows]); c = np.array([x["one core"] for x in rows]); ratio = np.array([x["ratio"] for x in rows])
    # pooled ratio sum_ij G_ij / sum_ij min(G_ii, G_jj), with a bootstrap over clusters (question texts)
    iu = np.triu_indices(n, 1); cs = pd.DataFrame(Gi.reshape(len(Gi), -1)).groupby(cl).sum().values.reshape(-1, n, n); bs = []
    for _ in range(B):
        g = cs[rng.integers(0, len(cs), len(cs))].sum(0); d = np.diag(g); bs.append(g[iu].sum() / np.minimum.outer(d, d)[iu].sum())
    d = np.diag(G); pooled = float(G[iu].sum() / np.minimum.outer(d, d)[iu].sum()); rS, w = joint(G, Vv.mean())
    out = {"pairs": len(rows), "rates": {l: float(x) for l, x in zip(labels, r)}, "mean absolute gap, nested": float(np.abs(o - p).mean()), "largest gap, nested": float(np.abs(o - p).max()), "mean absolute gap, one core": float(np.abs(o - c).mean()),
           "mean absolute gap, independent": float(np.abs(o).mean()), "correlation of observed and nested": float(np.corrcoef(o, p)[0, 1]), "G_ij / min(G_ii, G_jj): pooled": [pooled, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "G_ij / min: min, median, max over pairs": [float(ratio.min()), float(np.median(ratio)), float(ratio.max())], "sum of rates": float(r.sum()), "rate of the set": rS, "best single": float(r.max()), "rows": rows}
    print(f"[nested] {tag}: {len(rows)} pairs | |observed - nested| mean {out['mean absolute gap, nested']:.3f} max {out['largest gap, nested']:.3f} | one core {out['mean absolute gap, one core']:.3f} | independent {out['mean absolute gap, independent']:.3f} | "
          f"corr(obs, nested) {out['correlation of observed and nested']:.2f} | G_ij/min pooled {pooled:.3f} [{out['G_ij / min(G_ii, G_jj): pooled'][1]:.3f}, {out['G_ij / min(G_ii, G_jj): pooled'][2]:.3f}], pairs {ratio.min():.2f}/{np.median(ratio):.2f}/{ratio.max():.2f} | "
          f"sum {r.sum():.1f} set {rS:.1f} best {r.max():.1f}", flush=True)
    return out


# ================================================================== nested structure
Err = [[C[m][k] - h[k] for k in range(N)] for m in names]; Gi = gram(Err); G = Gi.mean(0); rate = V.mean() / np.diag(G)
res["nested"] = {"surveys": nested(Gi, V, names, qid, "surveys")}
fam = lambda m: "GPT" if m.startswith("GPT") else "Claude" if m.startswith("Claude") else "open"
for a, b in (("GPT", "Claude"), ("open", "GPT"), ("open", "Claude"), ("GPT", "GPT"), ("Claude", "Claude"), ("open", "open")):
    rr = [x for x in res["nested"]["surveys"]["rows"] if sorted([fam(x["pair"][0]), fam(x["pair"][1])]) == sorted([a, b])]
    if rr:
        o = np.array([x["observed"] for x in rr]); p = np.array([x["nested"] for x in rr]); res["nested"]["surveys"][f"{a}-{b}"] = {"pairs": len(rr), "observed min/median/max": [float(o.min()), float(np.median(o)), float(o.max())], "mean absolute gap": float(np.abs(o - p).mean()), "mean signed gap": float((o - p).mean())}
        print(f"   {a}-{b}: {len(rr)} pairs, observed {o.min():.2f}/{np.median(o):.2f}/{o.max():.2f}, observed - nested mean {(o - p).mean():+.3f}, |.| {np.abs(o - p).mean():.3f}")
P = Z["polls"]; pn = P["models"]; hp = P["hp"]; NPn = len(hp); Vp = np.array([float((x * (1 - x)).sum()) for x in hp]); Ep = [[P["PM"][m][i] - hp[i] for i in range(NPn)] for m in pn]; Gp = gram(Ep)
res["nested"]["new polls"] = nested(Gp, Vp, pn, np.arange(NPn), "new polls")
A = Z["annot"]; cn = A["models"]; ph = A["ph"]; Ga_i = np.array([[((A["CA"][a] - ph) * (A["CA"][b] - ph)).sum(1) - A["noise"] for b in cn] for a in cn]).transpose(2, 0, 1)
res["nested"]["annotators"] = nested(Ga_i, A["Va"], cn, np.arange(len(ph)), "annotators")
# real sources against the simulators: far below the nested value, which is why they add
R = Z["real"]; res["nested"]["real sources"] = {}
for rn, lst in R.items():
    av = np.array([x is not None for x in lst]); kk = np.where(av)[0]; er = [lst[k] - h[k] for k in kk]; g = np.mean([e @ e for e in er]); rr = V[kk].mean() / g; row = {}
    for j, m in enumerate(names):
        gm = np.mean([Err[j][k] @ Err[j][k] for k in kk]); rm = V[kk].mean() / gm; ob = float(np.mean([e @ Err[j][k] for e, k in zip(er, kk)]) / np.sqrt(g * gm)); row[m] = {"observed": ob, "nested": float(np.sqrt(min(rr, rm) / max(rr, rm))), "rate of the simulator on these items": float(rm)}
    res["nested"]["real sources"][rn] = {"rate": float(rr), "items": int(av.sum()), "pairs": row}
    print(f"[nested] {rn} (rate {rr:.1f}) against each simulator, observed | nested: " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {v['observed']:.2f}|{v['nested']:.2f}" for m, v in row.items()))
json.dump(res, open(ROOT + "/results/z_nest.json", "w"), indent=1)

# ================================================================== ceiling: two runs of the same model
def stated(d):
    Pd = {}
    for f in sorted(os.listdir(d)):
        if re.match(r"b\d+\.json", f):
            for i, p in json.load(open(f"{d}/{f}"))["items"].items():
                p = np.asarray(p, float); Pd[int(i)] = np.log(fl(p / p.sum(), 1e-4))
    return Pd
pos = {int(i): k for k, i in enumerate(ids)}; kl = lambda hv, q: float((hv[hv > 0] * np.log(hv[hv > 0] / (0.999 * q + 0.001 / len(q))[hv > 0])).sum())
TG = [.6, .8, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20, 30]; DG = [0, .5, 1, 1.5, 2, 2.5, 3, 4, 5]
runs = {"main": stated(f"{SOL}/survey"), "repeat": stated(f"{SOL}/survey_rep"), "low": stated(f"{SOL}/survey_low"), "high": stated(f"{SOL}/survey_high")}
com = sorted(set.intersection(*[set(v) for v in runs.values()]) & set(pos)); kk = [pos[i] for i in com]
M = np.array([[[kl(h[pos[i]], soft(runs["main"][i] / T - d * ns[pos[i]])) for d in DG] for T in TG] for i in sorted(set(runs["main"]) & set(pos))]); a, b = np.unravel_index(np.argmin(M.mean(0)), (len(TG), len(DG)))
Er = {r: [soft(runs[r][i]) - h[pos[i]] for i in com] for r in runs}; Gr = gram(list(Er.values())).mean(0); Vr = V[kk].mean(); rr = Vr / np.diag(Gr); cr = Gr / np.sqrt(np.outer(np.diag(Gr), np.diag(Gr)))
avg = [(Er["main"][k] + Er["repeat"][k]) / 2 for k in range(len(com))]; j_astra = names.index("GPT-6-astra"); j_sol = names.index("GPT-6.1-sol")
c_other = float(np.mean([Er["main"][x] @ Err[j_astra][k] for x, k in enumerate(kk)]) / np.sqrt(Gr[0, 0] * np.mean([Err[j_astra][k] @ Err[j_astra][k] for k in kk])))
res["ceiling"] = {"items": len(com), "rates": {r: float(x) for r, x in zip(runs, rr)}, "error correlation, main and repeat": float(cr[0, 1]), "error correlation, low and high reasoning": float(cr[2, 3]), "error correlation, main and low / high": [float(cr[0, 2]), float(cr[0, 3])],
                  "rate of the mean of two runs": float(Vr / np.mean([e @ e for e in avg])), "rate of the pair of runs": joint(Gr[:2, :2], Vr)[0], "error correlation with GPT-6-astra on the same items": c_other}
print(f"[ceiling] GPT-6.1-sol, {len(com)} items: rates " + " ".join(f"{r} {x:.1f}" for r, x in zip(runs, rr)) + f" | error correlation main-repeat {cr[0, 1]:.3f}, low-high {cr[2, 3]:.3f}, main-low {cr[0, 2]:.3f}, main-high {cr[0, 3]:.3f} | "
      f"mean of two runs {res['ceiling']['rate of the mean of two runs']:.1f}, pair {res['ceiling']['rate of the pair of runs']:.1f} | with GPT-6-astra on these items {c_other:.3f}", flush=True)

# ================================================================== combiners fitted on the other programmes
def heldout_rate(pred):
    return float(V.mean() / np.mean([(pred[k] - h[k]) @ (pred[k] - h[k]) for k in range(N)]))
order = np.argsort(-rate); top = {"best two": list(order[:2]), "best four": list(order[:4]), "all": list(range(J))}; res["stack"] = {"best single (held-out calibration)": float(rate.max())}
Kk = np.array([len(x) for x in h])
for nm, idx in top.items():
    out = {}
    # (a) convex pooling: weights V G^-1 1 from the other programmes
    pred = [None] * N
    for g in np.unique(sid):
        tr = sid != g; _, w = joint(Gi[tr].mean(0)[np.ix_(idx, idx)], V[tr].mean())
        for k in np.where(~tr)[0]:
            pred[k] = sum(w[x] * C[names[j]][k] for x, j in enumerate(idx)) / w.sum()
    out["convex pooling"] = heldout_rate(pred)
    # (b) linear stacking with an intercept: least squares over the options of the training items, coefficients free
    X = [np.column_stack([C[names[j]][k] for j in idx] + [np.ones(Kk[k]) / Kk[k], ns[k].astype(float)]) for k in range(N)]; pred = [None] * N
    for g in np.unique(sid):
        tr = np.where(sid != g)[0]; beta = np.linalg.lstsq(np.vstack([X[k] for k in tr]), np.concatenate([h[k] for k in tr]), rcond=None)[0]
        for k in np.where(sid == g)[0]:
            p = np.clip(X[k] @ beta, 1e-6, None); pred[k] = p / p.sum()
    out["linear stacking"] = heldout_rate(pred)
    # (c) log-linear pool of the members with a free temperature and non-answer penalty (extremizing when the temperature is below one)
    L = [np.mean([np.log(fl(C[names[j]][k], 1e-4)) for j in idx], 0) for k in range(N)]; BG = [.6, .7, .8, .9, 1, 1.1, 1.2, 1.35, 1.5, 1.75, 2, 2.5]; DD = [-1, -.5, 0, .5, 1]
    Mx = np.array([[[float((soft(bb * L[k] - dd * ns[k]) - h[k]) @ (soft(bb * L[k] - dd * ns[k]) - h[k])) for dd in DD] for bb in BG] for k in range(N)]); pred_err = np.zeros(N); chosen = []
    for g in np.unique(sid):
        a_, b_ = np.unravel_index(np.argmin(Mx[sid != g].mean(0)), (len(BG), len(DD))); pred_err[sid == g] = Mx[sid == g, a_, b_]; chosen.append(BG[a_])
    out["log-linear pool, free temperature"] = float(V.mean() / pred_err.mean()); out["exponent chosen"] = [float(min(chosen)), float(max(chosen))]
    res["stack"][nm] = {"members": [names[j] for j in idx], **out}
    print(f"[stack] {nm:10s} ({', '.join(names[j].replace('GPT-', '') for j in idx[:4])}{'...' if len(idx) > 4 else ''}): convex {out['convex pooling']:.1f} | linear stacking {out['linear stacking']:.1f} | log-linear pool {out['log-linear pool, free temperature']:.1f} "
          f"(exponent {min(chosen)}-{max(chosen)}) | best single {rate.max():.1f}", flush=True)

# ================================================================== does disagreement price the error?
b1, b2 = int(order[0]), int(order[1]); e1 = Gi[:, b1, b1]; dis = np.array([float((C[names[b1]][k] - C[names[b2]][k]) @ (C[names[b1]][k] - C[names[b2]][k])) for k in range(N)])
rho = spearmanr(dis, e1).correlation; ter = np.digitize(dis, np.quantile(dis, [1 / 3, 2 / 3])); rt = [float(V[ter == t].mean() / e1[ter == t].mean()) for t in range(3)]
worth = {}
for n in (5, 20, 100):
    rc = np.zeros(N); rv = np.zeros(N)
    for g in np.unique(sid):
        tr = sid != g; te = np.where(~tr)[0]; q = np.quantile(dis[tr], [.2, .4, .6, .8]); bt = np.digitize(dis[tr], q); wc = V[tr].mean() / e1[tr].mean(); wb = np.array([V[tr][bt == t].mean() / e1[tr][bt == t].mean() for t in range(5)])
        for k in te:
            wv = wb[np.digitize(dis[k], q)]; rc[k] = (n * V[k] + wc * wc * e1[k]) / (n + wc) ** 2; rv[k] = (n * V[k] + wv * wv * e1[k]) / (n + wv) ** 2
    worth[n] = {"constant pseudo-count": float(V.mean() / rc.mean()), "pseudo-count by disagreement": float(V.mean() / rv.mean())}
res["price"] = {"simulators": [names[b1], names[b2]], "spearman of disagreement and squared error": float(rho), "rate by tertile of disagreement": rt, "worth held out": worth}
print(f"[price] disagreement of {names[b1]} and {names[b2]} against the squared error of {names[b1]}: Spearman {rho:.2f} | rate by tertile of disagreement (low to high) " + " / ".join(f"{x:.1f}" for x in rt) +
      " | worth held out, constant | by disagreement: " + " ".join(f"n={n}: {v['constant pseudo-count']:.1f} | {v['pseudo-count by disagreement']:.1f}" for n, v in worth.items()), flush=True)

# ================================================================== anchors: trivial predictors, uncalibrated rates, programmes, poll subsets
uni = [np.ones(len(x)) / len(x) for x in h]; r_uni = float(V.mean() / np.mean([(u - x) @ (u - x) for u, x in zip(uni, h)]))
raw_rate = {m: float(V.mean() / np.mean([(Z["raw"][m][k] - h[k]) @ (Z["raw"][m][k] - h[k]) for k in range(N)])) for m in names}
by_prog = {m: {S[g]: float(V[sid == g].mean() / Gi[sid == g][:, j, j].mean()) for g in np.unique(sid)} for j, m in enumerate(names)}; uni_prog = {S[g]: float(V[sid == g].mean() / np.mean([(uni[k] - h[k]) @ (uni[k] - h[k]) for k in np.where(sid == g)[0]])) for g in np.unique(sid)}
meta = P["meta"]; us = np.array([m["country"] == "United States" for m in meta]); trk = np.array([bool(m["tracker"]) for m in meta]); unip = np.array([float((np.ones(len(x)) / len(x) - x) @ (np.ones(len(x)) / len(x) - x)) for x in hp])
sub = {"all": np.ones(NPn, bool), "United States": us, "other countries": ~us, "not a tracker": ~trk, "United States, not a tracker": us & ~trk}
poll_sub = {nm: {"items": int(mk.sum()), "uniform": float(Vp[mk].mean() / unip[mk].mean()), **{m: float(Vp[mk].mean() / Gp[mk][:, j, j].mean()) for j, m in enumerate(pn)}} for nm, mk in sub.items()}
raw_poll = {m: float(Vp.mean() / np.mean([(np.asarray(P["raw"][m][i]) - hp[i]) @ (np.asarray(P["raw"][m][i]) - hp[i]) for i in range(NPn)])) for m in pn}
res["anchor"] = {"uniform": r_uni, "uniform by programme": uni_prog, "rates as read out, before calibration": raw_rate, "rates by programme": by_prog, "new polls by subset": poll_sub, "new polls as read out": raw_poll, "items by programme": {S[g]: int((sid == g).sum()) for g in np.unique(sid)}}
print(f"[anchor] uniform predictor: {r_uni:.1f} respondents (surveys), {poll_sub['all']['uniform']:.1f} (new polls) | as read out: " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {x:.1f}" for m, x in raw_rate.items()))
print("[anchor] by programme: uniform " + " ".join(f"{s[:6]} {x:.1f}" for s, x in uni_prog.items()))
for m in names:
    print(f"   {m:18s} " + " ".join(f"{s[:6]} {x:.1f}" for s, x in by_prog[m].items()))
for nm, v in poll_sub.items():
    print(f"[anchor] new polls, {nm} ({v['items']}): uniform {v['uniform']:.1f} | " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {v[m]:.1f}" for m in pn))

# ================================================================== safety: the largest overestimate with an interval; items on which pooling raises the expected error
ratios = {}; boots = []
cl = pd.factorize(qid)[0]
for j, m in enumerate(names):
    for g in np.unique(sid):
        te = sid == g; ratios[(m, S[g])] = float((V[~te].mean() / Gi[~te][:, j, j].mean()) / (V[te].mean() / Gi[te][:, j, j].mean()))
worst = max(ratios, key=ratios.get); jw = names.index(worst[0]); gw = S.index(worst[1])
uq = np.unique(cl); byq = {q: np.where(cl == q)[0] for q in uq}
for _ in range(1000):
    r = np.concatenate([byq[q] for q in rng.choice(uq, len(uq))]); te = sid[r] == gw; boots.append((V[r][~te].mean() / Gi[r][~te][:, jw, jw].mean()) / (V[r][te].mean() / Gi[r][te][:, jw, jw].mean()))
harm = {}
for m in (names[b1], "Qwen3.5-9B"):
    j = names.index(m); row = {}
    for n in (5, 20, 100):
        hm = np.zeros(N, bool)
        for g in np.unique(sid):
            te = sid == g; w = V[~te].mean() / Gi[~te][:, j, j].mean(); hm[te] = (n * V[te] + w * w * Gi[te][:, j, j]) / (n + w) ** 2 > V[te] / n
        row[n] = float(hm.mean())
    harm[m] = row
res["safety"] = {"largest ratio": [ratios[worst], float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "where": list(worst), "share of cells above 2": float(np.mean([v > 2 for v in ratios.values()])), "share of cells above 1.5": float(np.mean([v > 1.5 for v in ratios.values()])),
                 "median ratio": float(np.median(list(ratios.values()))), "share of items on which pooling raises the expected squared error": harm}
print(f"[safety] largest overestimate {ratios[worst]:.2f} [{res['safety']['largest ratio'][1]:.2f}, {res['safety']['largest ratio'][2]:.2f}] at {worst}; median ratio {res['safety']['median ratio']:.2f}; cells above 1.5: {100 * res['safety']['share of cells above 1.5']:.0f}% | "
      "items harmed in expectation: " + " ".join(f"{m.replace('GPT-', '')} " + "/".join(f"{100 * v:.0f}%" for v in row.values()) for m, row in harm.items()), flush=True)

# ================================================================== the direction of the shared error
res["shape"] = {}
for rn, lst in R.items():
    kk = [k for k in range(N) if lst[k] is not None]; f = [lst[k] - h[k] for k in kk]; F = np.mean([x @ x for x in f]); row = {}
    for j, m in enumerate(names):
        b = float(np.mean([Err[j][k] @ x for k, x in zip(kk, f)]) / F); g = np.mean([Err[j][k] @ Err[j][k] for k in kk]); row[m] = {"shrinkage": b, "share of the error along this direction": float(b * b * F / g)}
    res["shape"][rn] = row
    print(f"[shape] error regressed on ({rn} - truth): coefficient | share of squared error: " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {v['shrinkage']:.2f}|{100 * v['share of the error along this direction']:.0f}%" for m, v in row.items()))
# common error of the simulators (mean error) and the residual of each: how much of each simulator's error is the common one
com_e = [np.mean([Err[j][k] for j in range(J)], 0) for k in range(N)]; res["shape"]["share of each simulator's squared error that lies along the best simulator's error"] = {}
for j, m in enumerate(names):
    if j != b1:
        b = G[j, b1] / G[b1, b1]; res["shape"]["share of each simulator's squared error that lies along the best simulator's error"][m] = {"coefficient on the best simulator's error": float(b), "share": float(b * b * G[b1, b1] / G[j, j])}
print("[shape] coefficient of each simulator's error on the best simulator's error (nested value: 1): " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {v['coefficient on the best simulator' + chr(39) + 's error']:.2f}" for m, v in res["shape"]["share of each simulator's squared error that lies along the best simulator's error"].items()))
json.dump(res, open(ROOT + "/results/z_nest.json", "w"), indent=1)

# ================================================================== read-out control: the same open model asked to state percentages (the read-out of the proprietary models)
dq = f"{SOL}/survey_qwen9b-stated"
if os.path.isdir(dq) and len([f for f in os.listdir(dq) if re.match(r"b\d+\.json", f)]) >= 45:
    Ls = stated(dq); ok = np.array([int(i) in Ls for i in ids]); kk = np.where(ok)[0]
    M = np.array([[[kl(h[k], soft(Ls[int(ids[k])] / T - d * ns[k])) for d in DG] for T in TG] for k in kk]); cs = {}
    for g in np.unique(sid):
        a_, b_ = np.unravel_index(np.argmin(M[sid[kk] != g].mean(0)), (len(TG), len(DG)))
        for k in kk[sid[kk] == g]:
            cs[k] = soft(Ls[int(ids[k])] / TG[a_] - DG[b_] * ns[k])
    es = [cs[k] - h[k] for k in kk]; gs = np.mean([e @ e for e in es]); rs = V[kk].mean() / gs; raw_s = V[kk].mean() / np.mean([(soft(Ls[int(ids[k])]) - h[k]) @ (soft(Ls[int(ids[k])]) - h[k]) for k in kk]); row = {}
    for j, m in enumerate(names):
        gm = np.mean([Err[j][k] @ Err[j][k] for k in kk]); gx = np.mean([e @ Err[j][k] for e, k in zip(es, kk)]); row[m] = {"correlation": float(gx / np.sqrt(gs * gm)), "nested": float(np.sqrt(min(gs, gm) / max(gs, gm))), "ratio": float(gx / min(gs, gm))}
    jq = names.index("Qwen3.5-9B"); pair = joint(np.array([[gs, np.mean([e @ Err[jq][k] for e, k in zip(es, kk)])], [np.mean([e @ Err[jq][k] for e, k in zip(es, kk)]), np.mean([Err[jq][k] @ Err[jq][k] for k in kk])]]), V[kk].mean())[0]
    res["readout"] = {"items": int(len(kk)), "rate, stated percentages": float(rs), "rate, stated, as given": float(raw_s), "rate, next-token probabilities": float(V[kk].mean() / np.mean([Err[jq][k] @ Err[jq][k] for k in kk])), "the two read-outs together": pair, "against each simulator": row}
    print(f"[readout] Qwen3.5-9B asked to state percentages ({len(kk)} items): rate {rs:.1f} (as given {raw_s:.1f}); read through probabilities {res['readout']['rate, next-token probabilities']:.1f}; the two read-outs together {pair:.1f} | "
          "error correlation | nested value with each simulator: " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {v['correlation']:.2f}|{v['nested']:.2f}" for m, v in row.items()), flush=True)
    dp = f"{SOL}/polls_qwen9b-stated"
    if os.path.isdir(dp) and len([f for f in os.listdir(dp) if re.match(r"t\d+\.json", f)]) >= 13:
        Pq = {}
        for f in sorted(os.listdir(dp)):
            if re.match(r"t\d+\.json", f):
                for i, p in json.load(open(f"{dp}/{f}"))["items"].items():
                    Pq[int(i)] = np.log(fl(np.asarray(p, float) / np.sum(p), 1e-4))
        a_, b_ = np.unravel_index(np.argmin(M.mean(0)), (len(TG), len(DG))); nsp = P["nsp"]; eq = [soft(Pq[i] / TG[a_] - DG[b_] * nsp[i]) - hp[i] for i in range(NPn)]; gq = np.mean([e @ e for e in eq]); rowp = {}
        for j, m in enumerate(pn):
            gm = Gp[:, j, j].mean(); gx = np.mean([eq[i] @ Ep[j][i] for i in range(NPn)]); rowp[m] = {"correlation": float(gx / np.sqrt(gq * gm)), "nested": float(np.sqrt(min(gq, gm) / max(gq, gm)))}
        res["readout"]["new polls"] = {"rate, stated percentages": float(Vp.mean() / gq), "against each simulator": rowp}
        print(f"[readout] new polls: Qwen3.5-9B stated {Vp.mean() / gq:.1f} | correlation | nested: " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {v['correlation']:.2f}|{v['nested']:.2f}" for m, v in rowp.items()))
    json.dump(res, open(ROOT + "/results/z_nest.json", "w"), indent=1)

# ================================================================== robustness of the shared error, and pairs of equal strength from different developers
iu = np.triu_indices(J, 1); stat = lambda M_: [float(M_[iu].min()), float(np.median(M_[iu])), float(M_[iu].max())]
cosm = np.array([[np.mean([Err[a][k] @ Err[b][k] / (np.linalg.norm(Err[a][k]) * np.linalg.norm(Err[b][k]) + 1e-12) for k in range(N)]) for b in range(J)] for a in range(J)])
st_i = [j for j, m in enumerate(names) if fam(m) != "open"]; Eraw = [[Z["raw"][names[j]][k] - h[k] for k in range(N)] for j in st_i]; Graw = gram(Eraw); Gr_ = Graw.mean(0); cr_ = Gr_ / np.sqrt(np.outer(np.diag(Gr_), np.diag(Gr_))); ius = np.triu_indices(len(st_i), 1)
dr = np.diag(Gr_); rr_ = V.mean() / dr; rS_raw, _ = joint(Gr_, V.mean())
lst = R["same-scale answers"]; kk = [k for k in range(N) if lst[k] is not None]; f = [lst[k] - h[k] for k in kk]; F = np.mean([x @ x for x in f]); bj = [np.mean([Err[j][k] @ x for k, x in zip(kk, f)]) / F for j in range(J)]
Res = [[Err[j][k] - bj[j] * x for k, x in zip(kk, f)] for j in range(J)]; Gp_ = gram(Res).mean(0); cpart = Gp_ / np.sqrt(np.outer(np.diag(Gp_), np.diag(Gp_)))
corr_all = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
res["robust"] = {"error correlation, all pairs": stat(corr_all), "mean cosine of the error vectors of an item": stat(cosm), "stated shares as given (GPT and Claude)": {"correlations": [float(cr_[ius].min()), float(np.median(cr_[ius])), float(cr_[ius].max())],
                 "G_ij / min": float(Gr_[ius].sum() / np.minimum.outer(dr, dr)[ius].sum()), "sum of rates": float(rr_.sum()), "rate of the set": rS_raw, "best single": float(rr_.max())}, "partial correlation given the error of the same-scale answers": stat(cpart)}
print(f"[robust] error correlations over all {len(iu[0])} pairs: min {stat(corr_all)[0]:.2f} median {stat(corr_all)[1]:.2f} max {stat(corr_all)[2]:.2f} | cosine {stat(cosm)[0]:.2f}/{stat(cosm)[1]:.2f}/{stat(cosm)[2]:.2f} | partial given same-scale error {stat(cpart)[0]:.2f}/{stat(cpart)[1]:.2f}/{stat(cpart)[2]:.2f} | "
      f"stated shares as given ({len(st_i)} models): correlations {cr_[ius].min():.2f}/{np.median(cr_[ius]):.2f}/{cr_[ius].max():.2f}, G_ij/min {res['robust']['stated shares as given (GPT and Claude)']['G_ij / min']:.3f}, sum {rr_.sum():.1f} set {rS_raw:.1f} best {rr_.max():.1f}")
res["equal strength"] = {}
for a_, b_ in (("GPT-6-luna", "Claude-Haiku-5.5"), ("GPT-5.6-luna", "Claude-Haiku-5.5"), ("GPT-5.6-sol", "Claude-Sonnet-5.5"), ("GPT-6.1-terra", "Claude-Opus-5.5"), ("GPT-6.1-sol", "Claude-Opus-5.5"), ("GPT-6-astra", "Claude-Fable-5.2"), ("GPT-6.1-sol", "GPT-6-astra"), ("Qwen3.5-9B", "Gemma-4-12B"), ("Claude-Opus-5.5", "Claude-Fable-5.2")):
    if a_ in names and b_ in names:
        x, y = names.index(a_), names.index(b_); rS, _ = joint(G[np.ix_([x, y], [x, y])], V.mean()); res["equal strength"][f"{a_} + {b_}"] = {"rates": [float(rate[x]), float(rate[y])], "correlation": float(corr_all[x, y]), "rate of the pair": rS, "gain over the better": float(rS / max(rate[x], rate[y]) - 1)}
print("[pairs] " + " | ".join(f"{k}: {v['rates'][0]:.1f}, {v['rates'][1]:.1f} -> {v['rate of the pair']:.1f} (+{100 * v['gain over the better']:.0f}%, corr {v['correlation']:.2f})" for k, v in res["equal strength"].items()))
# bounded harm: pseudo-count kappa * r against respondents alone, worst case over n; closed form kappa^2 / (4 (kappa - 1)) for kappa >= 2
for kap in (1.5, 2.0, 2.66, 3.0):
    x = np.linspace(0.01, 500, 200000); worst = float(((x * x + kap * kap * x) / (x + kap) ** 2).max()); print(f"[harm] kappa {kap}: worst risk relative to respondents alone {worst:.4f}; closed form {max(1.0, kap * kap / (4 * (kap - 1))):.4f}")
json.dump(res, open(ROOT + "/results/z_nest.json", "w"), indent=1)

# ================================================================== ceiling for two more models (second developer)
res["ceiling more"] = {}
for nm, tag in (("Claude-Fable-5.2", "claude-fable-5-2"), ("Claude-Opus-5.5", "claude-opus-5-5"), ("GPT-6-astra", "gpt-6-astra")):
    d2 = f"{SOL}/survey_{tag}_rep"
    if nm in names and os.path.isdir(d2) and len([f for f in os.listdir(d2) if re.match(r"b\d+\.json", f)]) >= 13:
        r1 = stated(f"{SOL}/survey_{tag}"); r2 = stated(d2); com = sorted(set(r1) & set(r2) & set(pos)); kk = [pos[i] for i in com]
        M = np.array([[[kl(h[pos[i]], soft(r1[i] / T - d * ns[pos[i]])) for d in DG] for T in TG] for i in sorted(set(r1) & set(pos))]); a, b = np.unravel_index(np.argmin(M.mean(0)), (len(TG), len(DG)))
        E1 = [soft(r1[i]) - h[pos[i]] for i in com]; E2 = [soft(r2[i]) - h[pos[i]] for i in com]; g11 = np.mean([e @ e for e in E1]); g22 = np.mean([e @ e for e in E2]); g12 = np.mean([x @ y for x, y in zip(E1, E2)])
        avg = np.mean([((x + y) / 2) @ ((x + y) / 2) for x, y in zip(E1, E2)]); oth = {}
        for m2 in ("GPT-6-astra", "Claude-Opus-5.5", "Claude-Fable-5.2", "GPT-6.1-sol"):
            if m2 != nm:
                j2 = names.index(m2); oth[m2] = float(np.mean([x @ Err[j2][k] for x, k in zip(E1, kk)]) / np.sqrt(g11 * np.mean([Err[j2][k] @ Err[j2][k] for k in kk])))
        res["ceiling more"][nm] = {"items": len(com), "rates of the two runs": [float(V[kk].mean() / g11), float(V[kk].mean() / g22)], "error correlation of the two runs": float(g12 / np.sqrt(g11 * g22)), "rate of the mean of two runs": float(V[kk].mean() / avg), "error correlation with other simulators on these items": oth}
        print(f"[ceiling] {nm}, {len(com)} items: runs {V[kk].mean() / g11:.1f}, {V[kk].mean() / g22:.1f}; error correlation of the two runs {g12 / np.sqrt(g11 * g22):.3f}; mean of two runs {V[kk].mean() / avg:.1f} | with others: " + " ".join(f"{k_} {v_:.3f}" for k_, v_ in oth.items()))
json.dump(res, open(ROOT + "/results/z_nest.json", "w"), indent=1)

# ================================================================== round 4: stronger alternatives and formal tests
cl_ = pd.factorize(qid)[0]; ncl = cl_.max() + 1; CS = np.zeros((ncl, J, J)); np.add.at(CS, cl_, Gi); VS = np.bincount(cl_, V, minlength=ncl); iu = np.triu_indices(J, 1); rb = np.random.default_rng(1)
# (1) held-out fit of the off-diagonal second moments: nested (no free parameter), one factor with free loadings, one common error; clusters split in halves, 200 splits
def one_factor(Gm, it=500):
    lam = np.sqrt(np.clip(np.diag(Gm), 1e-9, None)) * 0.8
    for _ in range(it):
        for j in range(len(lam)):
            o = [x for x in range(len(lam)) if x != j]; lam[j] = float(Gm[j, o] @ lam[o] / (lam[o] @ lam[o]))
    return np.outer(lam, lam)
fit = {"nested": [], "one factor, free loadings": [], "one common error": [], "the other half itself": []}
for _ in range(200):
    half = rb.random(ncl) < 0.5; Ga = CS[half].sum(0) / np.bincount(cl_, minlength=ncl)[half].sum(); Gb = CS[~half].sum(0) / np.bincount(cl_, minlength=ncl)[~half].sum(); da = np.diag(Ga); sc = np.sqrt(np.outer(np.diag(Gb), np.diag(Gb)))[iu]
    rmse = lambda pred: float(np.sqrt(np.mean(((pred[iu] - Gb[iu]) / sc) ** 2)))
    fit["nested"].append(rmse(np.minimum.outer(da, da))); fit["one factor, free loadings"].append(rmse(one_factor(Ga))); fit["one common error"].append(rmse(np.full((J, J), da.min()))); fit["the other half itself"].append(rmse(Ga))
res["held-out fit"] = {k: float(np.mean(v)) for k, v in fit.items()}
print("[fit] held-out error of the predicted off-diagonal moments (in correlation units, 200 half splits): " + " | ".join(f"{k} {np.mean(v):.3f}" for k, v in fit.items()), flush=True)
# (2) encompassing: weight of the weaker simulator in the best combination of a pair, and what the pair is worth over the stronger one; bootstrap over question texts
B = 2000; draws = rb.integers(0, ncl, (B, ncl)); GB = np.array([CS[d].sum(0) for d in draws]); VB = np.array([VS[d].sum() for d in draws])
enc = []
for a, b in zip(*iu):
    s_, w_ = (a, b) if G[a, a] <= G[b, b] else (b, a)
    lam = lambda g: (g[..., s_, s_] - g[..., s_, w_]) / (g[..., s_, s_] + g[..., w_, w_] - 2 * g[..., s_, w_])
    def gain(g):
        l_ = np.clip(lam(g), 0, 1); risk = (1 - l_) ** 2 * g[..., s_, s_] + l_ ** 2 * g[..., w_, w_] + 2 * l_ * (1 - l_) * g[..., s_, w_]; return g[..., s_, s_] / risk - 1
    lb = lam(GB); gb = gain(GB)
    enc.append({"strong": names[s_], "weak": names[w_], "weight of the weaker": float(lam(G)), "interval": [float(np.percentile(lb, 2.5)), float(np.percentile(lb, 97.5))], "gain of the pair over the stronger": float(gain(G)), "gain interval": [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))],
                "ratio of rates": float(G[s_, s_] / G[w_, w_])})
lw = np.array([e["weight of the weaker"] for e in enc]); gn = np.array([e["gain of the pair over the stronger"] for e in enc]); rej = np.array([e["interval"][0] > 0 for e in enc]); rr_ = np.array([e["ratio of rates"] for e in enc])
res["encompassing"] = {"pairs": len(enc), "pairs in which the weaker has a weight above zero (95%)": int(rej.sum()), "weight of the weaker: median, max": [float(np.median(lw)), float(lw.max())], "gain: median, 90th percentile, max": [float(np.median(gn)), float(np.percentile(gn, 90)), float(gn.max())],
                       "pairs whose rates differ by more than a quarter": {"pairs": int((rr_ < 0.8).sum()), "weight above zero": int(rej[rr_ < 0.8].sum()), "median weight": float(np.median(lw[rr_ < 0.8])), "largest gain": float(gn[rr_ < 0.8].max())},
                       "pairs of similar rates": {"pairs": int((rr_ >= 0.8).sum()), "weight above zero": int(rej[rr_ >= 0.8].sum()), "median weight": float(np.median(lw[rr_ >= 0.8])), "largest gain": float(gn[rr_ >= 0.8].max())}, "rows": enc}
e_ = res["encompassing"]; print(f"[encompassing] {len(enc)} pairs: the weaker simulator has a weight above zero in {int(rej.sum())}; weight median {np.median(lw):.2f} max {lw.max():.2f}; gain of the pair over the stronger: median {100 * np.median(gn):.1f}%, 90th pct {100 * np.percentile(gn, 90):.1f}%, max {100 * gn.max():.1f}% | "
      f"rates differing by more than a quarter: {e_['pairs whose rates differ by more than a quarter']} | similar rates: {e_['pairs of similar rates']}", flush=True)
# (3) intervals for the rate of the set and for its excess over the best member
def setrate(g, v):
    return joint(g, v)[0]
sb = np.array([setrate(GB[x] / 1.0, VB[x]) for x in range(0, B, 2)]); bb = np.array([(VB[x] / np.diag(GB[x])).max() for x in range(0, B, 2)]); rS_all = setrate(G, V.mean())
res["set interval"] = {"rate of the set": [float(rS_all), float(np.percentile(sb, 2.5)), float(np.percentile(sb, 97.5))], "set over best member": [float(rS_all / rate.max() - 1), float(np.percentile(sb / bb - 1, 2.5)), float(np.percentile(sb / bb - 1, 97.5))], "sum of rates": float(rate.sum()), "best": float(rate.max())}
cp_ = pd.DataFrame(Gp.reshape(NPn, -1)).values.reshape(NPn, len(pn), len(pn)); dpl = rb.integers(0, NPn, (1000, NPn)); sp = []; bp = []
for d in dpl:
    g = cp_[d].mean(0); v = Vp[d].mean(); sp.append(setrate(g, v)); bp.append((v / np.diag(g)).max())
gpm = Gp.mean(0); rSp = setrate(gpm, Vp.mean()); rpb = (Vp.mean() / np.diag(gpm)).max()
res["set interval"]["new polls"] = {"rate of the set": [float(rSp), float(np.percentile(sp, 2.5)), float(np.percentile(sp, 97.5))], "set over best member": [float(rSp / rpb - 1), float(np.percentile(np.array(sp) / np.array(bp) - 1, 2.5)), float(np.percentile(np.array(sp) / np.array(bp) - 1, 97.5))], "sum of rates": float((Vp.mean() / np.diag(gpm)).sum()), "best": float(rpb)}
si = res["set interval"]; print(f"[set] surveys: sum {si['sum of rates']:.1f}, set {si['rate of the set'][0]:.1f} [{si['rate of the set'][1]:.1f}, {si['rate of the set'][2]:.1f}], best {si['best']:.1f}, set over best +{100 * si['set over best member'][0]:.1f}% [{100 * si['set over best member'][1]:.1f}, {100 * si['set over best member'][2]:.1f}] | "
      f"new polls: sum {si['new polls']['sum of rates']:.1f}, set {si['new polls']['rate of the set'][0]:.1f} [{si['new polls']['rate of the set'][1]:.1f}, {si['new polls']['rate of the set'][2]:.1f}], best {si['new polls']['best']:.1f}, +{100 * si['new polls']['set over best member'][0]:.1f}% [{100 * si['new polls']['set over best member'][1]:.1f}, {100 * si['new polls']['set over best member'][2]:.1f}]", flush=True)
# (4) common ignorance or common mistakes? With a no-model baseline b, signal s = c - b and t = theta - b: e = m - (1 - beta) t, where beta is the share of t the simulator recovers and m = s - beta t is what it asserts wrongly
res["ignorance and mistakes"] = {}
for bn, base in (("uniform answer", [np.ones(len(x)) / len(x) for x in h]), ("same-scale answers", R["same-scale answers"])):
    kk_ = [k for k in range(N) if base[k] is not None]; t_ = [h[k] - base[k] for k in kk_]; T_ = np.mean([x @ x for x in t_]); beta = []; Mis = []
    for j in range(J):
        s_ = [C[names[j]][k] - base[k] for k in kk_]; bj_ = float(np.mean([a @ b for a, b in zip(s_, t_)]) / T_); beta.append(bj_); Mis.append([a - bj_ * b for a, b in zip(s_, t_)])
    Gm_ = gram(Mis).mean(0); dm = np.diag(Gm_); cm_ = Gm_ / np.sqrt(np.outer(dm, dm)); Gk = np.array([[np.mean([Err[a][k] @ Err[b][k] for k in kk_]) for b in range(J)] for a in range(J)])
    ign = np.array([(1 - b) ** 2 * T_ for b in beta]); share_ign = ign / np.diag(Gk); cross_ign = np.array([[(1 - beta[a]) * (1 - beta[b]) * T_ / Gk[a, b] for b in range(J)] for a in range(J)])
    res["ignorance and mistakes"][bn] = {"items": len(kk_), "share of the truth's deviation from the baseline that each simulator recovers": {m: float(b) for m, b in zip(names, beta)}, "share of each simulator's squared error that is ignorance": {m: float(x) for m, x in zip(names, share_ign)},
                                         "share of the shared error (G_ij) that is ignorance: min, median, max": [float(cross_ign[iu].min()), float(np.median(cross_ign[iu])), float(cross_ign[iu].max())], "correlation of mistakes: min, median, max": [float(cm_[iu].min()), float(np.median(cm_[iu])), float(cm_[iu].max())],
                                         "nested ratio of mistakes, pooled": float(Gm_[iu].sum() / np.minimum.outer(dm, dm)[iu].sum())}
    v_ = res["ignorance and mistakes"][bn]; print(f"[ignorance] baseline {bn} ({len(kk_)} items): recovered share of the deviation " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {b:.2f}" for m, b in zip(names, beta)) +
          f" | ignorance as a share of own error {share_ign.min():.2f}-{share_ign.max():.2f}, of the shared error {v_['share of the shared error (G_ij) that is ignorance: min, median, max'][1]:.2f} (median) | mistakes: correlation {cm_[iu].min():.2f}/{np.median(cm_[iu]):.2f}/{cm_[iu].max():.2f}, nested ratio {v_['nested ratio of mistakes, pooled']:.2f}", flush=True)
# (5) the same analysis with every simulator calibrated on the other programmes (stated percentages included)
Ec = [[Z["Ccal"][m][k] - h[k] for k in range(N)] for m in names]; Gc_i = gram(Ec); res["calibrated variant"] = nested(Gc_i, V, names, qid, "every simulator calibrated"); res["calibrated variant"].pop("rows")
# (6) a second wording of the request, and a panel of simulated individuals
res["wording"] = {}
for nm, tag in (("Claude-Fable-5.2", "claude-fable-5-2"), ("Claude-Haiku-5.5", "claude-haiku-5-5"), ("GPT-6.1-sol", "gpt-6.1-sol"), ("GPT-5.6-luna", "gpt-5.6-luna")):
    d2 = f"{SOL}/survey_{tag}_alt"
    if nm in names and os.path.isdir(d2) and len([f for f in os.listdir(d2) if re.match(r"b\d+\.json", f)]) >= 13:
        r2 = stated(d2); com = sorted(set(r2) & set(pos)); kk_ = [pos[i] for i in com]; j_ = names.index(nm); E1 = [Err[j_][k] for k in kk_]; E2 = [soft(r2[i]) - h[pos[i]] for i in com]
        g11 = np.mean([e @ e for e in E1]); g22 = np.mean([e @ e for e in E2]); g12 = np.mean([x @ y for x, y in zip(E1, E2)]); pr_ = joint(np.array([[g11, g12], [g12, g22]]), V[kk_].mean())[0]
        res["wording"][nm] = {"items": len(com), "rates": [float(V[kk_].mean() / g11), float(V[kk_].mean() / g22)], "error correlation of the two wordings": float(g12 / np.sqrt(g11 * g22)), "the two wordings together": pr_}
        print(f"[wording] {nm}, {len(com)} items: rates {V[kk_].mean() / g11:.1f} (main wording), {V[kk_].mean() / g22:.1f} (second wording); errors correlate {g12 / np.sqrt(g11 * g22):.3f}; together {pr_:.1f}")
pf = ROOT + "/outputs/screen/r3_out/panel_qwen.json"
if os.path.exists(pf):
    Pn = json.load(open(pf)); Lp = {int(i): np.log(fl(np.asarray(p, float), 1e-4)) for i, p in Pn.items()}; kk_ = [k for k in range(N) if int(ids[k]) in Lp]
    M = np.array([[[kl(h[k], soft(Lp[int(ids[k])] / T - d * ns[k])) for d in DG] for T in TG] for k in kk_]); cpn = {}; sk = sid[kk_]
    for g in np.unique(sk):
        a_, b_ = np.unravel_index(np.argmin(M[sk != g].mean(0)), (len(TG), len(DG)))
        for k in np.array(kk_)[sk == g]:
            cpn[k] = soft(Lp[int(ids[k])] / TG[a_] - DG[b_] * ns[k])
    ep = [cpn[k] - h[k] for k in kk_]; gp_ = np.mean([e @ e for e in ep]); jq = names.index("Qwen3.5-9B"); row = {}
    for j, m in enumerate(names):
        gm = np.mean([Err[j][k] @ Err[j][k] for k in kk_]); gx = np.mean([e @ Err[j][k] for e, k in zip(ep, kk_)]); row[m] = {"correlation": float(gx / np.sqrt(gp_ * gm)), "nested": float(np.sqrt(min(gp_, gm) / max(gp_, gm))), "ratio": float(gx / min(gp_, gm))}
    gq = np.mean([Err[jq][k] @ Err[jq][k] for k in kk_]); gxq = np.mean([e @ Err[jq][k] for e, k in zip(ep, kk_)])
    res["panel"] = {"items": len(kk_), "rate of the panel": float(V[kk_].mean() / gp_), "rate of the single respondent": float(V[kk_].mean() / gq), "the two together": joint(np.array([[gp_, gxq], [gxq, gq]]), V[kk_].mean())[0], "against each simulator": row}
    print(f"[panel] Qwen3.5-9B as a panel of sixteen personas ({len(kk_)} items): rate {V[kk_].mean() / gp_:.1f}; single respondent {V[kk_].mean() / gq:.1f}; together {res['panel']['the two together']:.1f} | correlation | nested: " +
          " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {v['correlation']:.2f}|{v['nested']:.2f}" for m, v in row.items()))
json.dump(res, open(ROOT + "/results/z_nest.json", "w"), indent=1)

# ================================================================== round 5: by programme, batch control, calibration slope
res["by programme"] = {}
for g in np.unique(sid):
    mk = sid == g; Gg = Gi[mk].mean(0); Vg_ = V[mk].mean(); rg = Vg_ / np.diag(Gg); dg = np.diag(Gg); rS, _ = joint(Gg, Vg_); gains = []
    for a, b in zip(*np.triu_indices(J, 1)):
        s_, w_ = (a, b) if Gg[a, a] <= Gg[b, b] else (b, a); lam = min(1.0, max(0.0, (Gg[s_, s_] - Gg[s_, w_]) / (Gg[s_, s_] + Gg[w_, w_] - 2 * Gg[s_, w_]))); gains.append(Gg[s_, s_] / ((1 - lam) ** 2 * Gg[s_, s_] + lam ** 2 * Gg[w_, w_] + 2 * lam * (1 - lam) * Gg[s_, w_]) - 1)
    res["by programme"][S[g]] = {"items": int(mk.sum()), "sum of rates": float(rg.sum()), "rate of the set": rS, "best single": float(rg.max()), "best": names[int(np.argmax(rg))], "set over best": float(rS / rg.max() - 1), "G_ij / min, pooled": float(Gg[np.triu_indices(J, 1)].sum() / np.minimum.outer(dg, dg)[np.triu_indices(J, 1)].sum()),
                                 "pair gain: median, max": [float(np.median(gains)), float(np.max(gains))]}
    v = res["by programme"][S[g]]; print(f"[programme] {S[g]:16s} {v['items']:4d} items | sum {v['sum of rates']:6.1f} set {rS:6.1f} best {v['best single']:6.1f} ({v['best'].replace('GPT-', '').replace('Claude-', '')}) +{100 * v['set over best']:.0f}% | G_ij/min {v['G_ij / min, pooled']:.2f} | pair gain median {100 * v['pair gain: median, max'][0]:.1f}% max {100 * v['pair gain: median, max'][1]:.1f}%")
# programme-level bootstrap of the pooled quantities (seven clusters)
gs_ = np.unique(sid); bpr = []
for _ in range(1000):
    r = np.concatenate([np.where(sid == g)[0] for g in rb.choice(gs_, len(gs_))]); Gb_ = Gi[r].mean(0); vb = V[r].mean(); db = np.diag(Gb_); bpr.append((joint(Gb_, vb)[0] / (vb / db).max() - 1, Gb_[np.triu_indices(J, 1)].sum() / np.minimum.outer(db, db)[np.triu_indices(J, 1)].sum()))
bpr = np.array(bpr); res["programme bootstrap"] = {"set over best": [float(np.percentile(bpr[:, 0], 2.5)), float(np.percentile(bpr[:, 0], 97.5))], "G_ij / min": [float(np.percentile(bpr[:, 1], 2.5)), float(np.percentile(bpr[:, 1], 97.5))]}
print(f"[programme] bootstrap over programmes: set over best [{100 * res['programme bootstrap']['set over best'][0]:.1f}%, {100 * res['programme bootstrap']['set over best'][1]:.1f}%], G_ij/min [{res['programme bootstrap']['G_ij / min'][0]:.2f}, {res['programme bootstrap']['G_ij / min'][1]:.2f}]")
# new polls by country group
res["new polls by subset: sets"] = {}
for nm_, mk in (("United States", us), ("other countries", ~us)):
    Gg = Gp[mk].mean(0); vg = Vp[mk].mean(); dg = np.diag(Gg); rS, _ = joint(Gg, vg); res["new polls by subset: sets"][nm_] = {"sum of rates": float((vg / dg).sum()), "rate of the set": rS, "best single": float((vg / dg).max()), "G_ij / min, pooled": float(Gg[np.triu_indices(len(pn), 1)].sum() / np.minimum.outer(dg, dg)[np.triu_indices(len(pn), 1)].sum())}
    v = res["new polls by subset: sets"][nm_]; print(f"[programme] new polls, {nm_}: sum {v['sum of rates']:.0f} set {rS:.1f} best {v['best single']:.1f} | G_ij/min {v['G_ij / min, pooled']:.2f}")
# batch control: the same items in other batches of twenty, and one item per call
def asgiven(dname):
    return {i: soft(x) for i, x in stated(dname).items()} if os.path.isdir(dname) else {}
BT = {}
for nm, tag in (("Claude-Fable-5.2", "claude-fable-5-2"), ("GPT-6.1-sol", "gpt-6.1-sol"), ("Claude-Haiku-5.5", "claude-haiku-5-5"), ("GPT-6-astra", "gpt-6-astra")):
    base_dir = f"{SOL}/survey" + ("" if tag == "gpt-6.1-sol" else "_" + tag); BT[nm] = {"main": asgiven(base_dir), "shuffled": asgiven(f"{SOL}/survey_{tag}_shuf"), "single": asgiven(f"{SOL}/survey_{tag}_single"), "repeat": asgiven(f"{SOL}/survey" + ("_rep" if tag == "gpt-6.1-sol" else f"_{tag}_rep"))}
def ecorr(A, Bx):
    com = sorted(set(A) & set(Bx) & set(pos))
    if len(com) < 100:
        return None
    ea = [A[i] - h[pos[i]] for i in com]; eb = [Bx[i] - h[pos[i]] for i in com]; return float(np.mean([x @ y for x, y in zip(ea, eb)]) / np.sqrt(np.mean([x @ x for x in ea]) * np.mean([y @ y for y in eb]))), len(com)
def erate(A):
    com = sorted(set(A) & set(pos) & set(BT["GPT-6.1-sol"]["repeat"])); return float(np.mean([V[pos[i]] for i in com]) / np.mean([(A[i] - h[pos[i]]) @ (A[i] - h[pos[i]]) for i in com])) if len(com) >= 100 else None
res["batch control"] = {"within a model": {}, "between models": {}, "rates": {}}
first = sorted(set(BT["GPT-6.1-sol"]["repeat"]) & set(pos))
for nm in BT:
    mainf = {i: BT[nm]["main"][i] for i in first if i in BT[nm]["main"]}; row = {k: ecorr(mainf, BT[nm][k]) for k in ("repeat", "shuffled", "single") if BT[nm][k]}; res["batch control"]["within a model"][nm] = {k: v[0] for k, v in row.items() if v}
    res["batch control"]["rates"][nm] = {k: erate(BT[nm][k]) for k in ("main", "shuffled", "single") if BT[nm][k]}
    print(f"[batch] {nm}: error correlation of the main run with " + ", ".join(f"{k} {v[0]:.3f}" for k, v in row.items() if v) + " | rates " + " ".join(f"{k} {v:.1f}" for k, v in res["batch control"]["rates"][nm].items() if v))
for a_, b_ in (("Claude-Fable-5.2", "GPT-6.1-sol"), ("Claude-Fable-5.2", "GPT-6-astra"), ("GPT-6.1-sol", "GPT-6-astra"), ("Claude-Haiku-5.5", "GPT-6.1-sol"), ("Claude-Fable-5.2", "Claude-Haiku-5.5")):
    ma = {i: BT[a_]["main"][i] for i in first if i in BT[a_]["main"]}; mb = {i: BT[b_]["main"][i] for i in first if i in BT[b_]["main"]}; row = {"same batches": ecorr(ma, mb), "one in other batches": ecorr(ma, BT[b_]["shuffled"]) if BT[b_]["shuffled"] else None,
                                                                                                                   "one item per call (both)": ecorr(BT[a_]["single"], BT[b_]["single"]) if BT[a_]["single"] and BT[b_]["single"] else None, "one batched, one single": ecorr(ma, BT[b_]["single"]) if BT[b_]["single"] else None}
    res["batch control"]["between models"][f"{a_} / {b_}"] = {k: v[0] for k, v in row.items() if v}
    print(f"[batch] {a_} / {b_}: " + ", ".join(f"{k} {v[0]:.3f}" for k, v in row.items() if v))
# calibration slope of what the simulators assert (uniform baseline): regression of the truth's deviation on the simulator's deviation; one for a calibrated forecaster
uni = [np.ones(len(x)) / len(x) for x in h]; res["calibration slope"] = {}
for j, m in enumerate(names):
    s_ = [C[m][k] - uni[k] for k in range(N)]; t_ = [h[k] - uni[k] for k in range(N)]; res["calibration slope"][m] = float(np.mean([a @ b for a, b in zip(s_, t_)]) / np.mean([a @ a for a in s_]))
print("[slope] truth on assertion (1 = calibrated): " + " ".join(f"{m.replace('GPT-', '').replace('Claude-', '')} {v:.2f}" for m, v in res["calibration slope"].items()))
json.dump(res, open(ROOT + "/results/z_nest.json", "w"), indent=1)
