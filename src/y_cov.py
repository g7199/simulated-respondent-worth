"""Paper 3 (second revision): the redundancy of sources.

Squared-error view. For a source j, e_j = c_j - theta is its error on an item and V = sum_k theta_k (1 - theta_k) the variance of one
respondent. G_ij = E <e_i, e_j> over items. Then (exactly, for respondents drawn independently of the sources):
   rate of source j              r_j = V / G_jj
   pooled risk with weights w    (n V + w' G w) / (n + sum w)^2,  minimised by w* = V G^-1 1 (independent of n), worth n + r_S
   rate of a set S               r_S = V 1' G_S^-1 1   (= sum of rates when the errors are uncorrelated)
This script measures the correlation matrix of the errors of the ten simulators and the real sources, the rates of sets, and
checks on held-out programmes that the worth of the pooled estimate is n + r_S (transfer), under squared error and under KL.
Usage: python y_cov.py -> results/y_cov.json
"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, re, json, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, (_REPO + "/outputs/screen")); sys.argv = [sys.argv[0], "none"]
from r2pop_common import load, soft, fl

ROOT = (_REPO + ""); D = ROOT + "/outputs/screen/"; SOL = (_REPO + "/outputs/sol")
I, E, qi, L0, L1, LQ = load(); NALL = len(I); h = [np.asarray(x, float) / np.sum(x) for x in I.h.values]; K = I.K.values; src = I.src.values; S = list(pd.unique(src))
ns = [np.array([bool(re.search(r"refus|don.?t know|dont know|can.?t choose|cant choose|no answer|not sure|undecided|haven.?t heard", o.lower())) for o in os_]) for os_ in I.opts]
TG = [.6, .8, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20, 30]; DG = [0, .5, 1, 1.5, 2, 2.5, 3, 4, 5]; rng = np.random.default_rng(0)


def two_order(path):
    raw = json.load(open(path)); return {i: np.log(fl((soft(raw[f"{i}_0"]) + soft(raw[f"{i}_1"])) / 2, 1e-4)) for i in range(NALL)}


def stated(d):
    P = {}
    for f in sorted(os.listdir(d)):
        if re.match(r"b\d+\.json", f):
            for i, p in json.load(open(f"{d}/{f}"))["items"].items():
                p = np.asarray(p, float); P[int(i)] = np.log(fl(p / p.sum(), 1e-4))
    return P


MODELS = {"OLMo-3-7B": two_order(D + "r3_out/size_olmo.json"), "Qwen3.5-9B": {i: LQ[i] for i in range(NALL)}, "Gemma-4-12B": two_order(D + "r3_out/size_gemma12.json")}
STATED = [("GPT-6-luna", "gpt-6-luna"), ("GPT-5.6-luna", "gpt-5.6-luna"), ("GPT-5.6-sol", "gpt-5.6-sol"), ("GPT-6-sol", "gpt-6-sol"), ("GPT-6.1-terra", "gpt-6.1-terra"), ("GPT-6.1-sol", "gpt-6.1-sol"), ("GPT-6-astra", "gpt-6-astra"),
          ("Claude-Haiku-5.5", "claude-haiku-5-5"), ("Claude-Sonnet-5.5", "claude-sonnet-5-5"), ("Claude-Opus-5.5", "claude-opus-5-5"), ("Claude-Fable-5.2", "claude-fable-5-2"),
          ("Mistral-Small-24B", "mistral24b-stated"), ("Qwen3.8-27B", "qwen27b-stated")]          # two larger open models behind a local server, asked to state percentages like the proprietary ones
sdir = lambda use, tag: f"{SOL}/{use}" + ("" if tag == "gpt-6.1-sol" and use in ("survey", "ca", "grp") else "_" + tag)
nfiles = lambda d, pat: len([f for f in os.listdir(d) if re.match(pat, f)]) if os.path.isdir(d) else 0
for nm, tag in STATED:
    if nfiles(sdir("survey", tag), r"b\d+\.json") >= 49:
        MODELS[nm] = stated(sdir("survey", tag))
ids = np.array(sorted(set.intersection(*[set(v) for v in MODELS.values()]))); N = len(ids); sid = np.array([S.index(s) for s in src[ids]]); qid = qi[ids]
names = list(MODELS); J = len(names)
kl = lambda hv, q: float((hv[hv > 0] * np.log(hv[hv > 0] / (0.999 * q + 0.001 / len(q))[hv > 0])).sum())


def calibrated(Lg):
    """two-parameter rule fitted on the other programmes (squared-error criterion would give the same order; KL is kept for continuity)"""
    M = np.array([[[kl(h[i], soft(Lg[int(i)] / T - d * ns[i])) for d in DG] for T in TG] for i in ids]); cen = [None] * N
    for g in np.unique(sid):
        a, b = np.unravel_index(np.argmin(M[sid != g].mean(0)), (len(TG), len(DG)))
        for k in np.where(sid == g)[0]:
            cen[k] = soft(Lg[int(ids[k])] / TG[a] - DG[b] * ns[ids[k]])
    return cen


# primary read-out: stated percentages are used as given (nothing is fitted); the probabilities of the open models are calibrated on the other programmes.
# The calibrated version of every simulator is kept for the comparison in the appendix.
IS_STATED = {m: m in dict(STATED) for m in names}
Ccal = {m: calibrated(MODELS[m]) for m in names}
C = {m: ([soft(MODELS[m][int(i)]) for i in ids] if IS_STATED[m] else Ccal[m]) for m in names}
V = np.array([float((h[i] * (1 - h[i])).sum()) for i in ids])                           # variance of one respondent, per item
Err = {m: [C[m][k] - h[i] for k, i in enumerate(ids)] for m in names}
Gi = np.array([[[float(Err[a][k] @ Err[b][k]) for b in names] for a in names] for k in range(N)])   # per item Gram matrix
G = Gi.mean(0); Vm = V.mean(); rate = Vm / np.diag(G); corr = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
print(f"{N} items, {J} simulators; squared-error rates: " + " ".join(f"{m} {r:.1f}" for m, r in zip(names, rate)))
print("error correlations (rows/cols in the order above):"); print(np.round(corr, 2))
res = {"items": int(N), "models": names, "rate_mse": [float(x) for x in rate], "error_correlation": corr.tolist()}


def joint(Gs, Vv):
    """rate of a set and its weights, with weights constrained to be non-negative (exact solution, see nnweights.py)"""
    from nnweights import best_weights
    full = best_weights(Gs, Vv)
    return float(full.sum()), full


open_i = [j for j in range(J) if not names[j].startswith(("GPT", "Claude"))]; gpt_i = [j for j in range(J) if names[j].startswith("GPT")]; cl_i = [j for j in range(J) if names[j].startswith("Claude")]
sets = {"open models": open_i, "seven GPT models": gpt_i, "all simulators": list(range(J)), **({"Claude models": cl_i, "GPT and Claude models": gpt_i + cl_i} if cl_i else {}), "GPT-6.1-sol + GPT-6-astra": [names.index("GPT-6.1-sol"), names.index("GPT-6-astra")],
        "GPT-6.1-sol + Qwen3.5-9B": [names.index("GPT-6.1-sol"), names.index("Qwen3.5-9B")], "Qwen3.5-9B + Gemma-4-12B": [names.index("Qwen3.5-9B"), names.index("Gemma-4-12B")]}
res["sets"] = {}
for nm, idx in sets.items():
    # in-sample rate of the set, and the transfer test: weights from the other programmes, risk on the held-out programme
    rS, w = joint(G[np.ix_(idx, idx)], Vm); worth = {}
    for n in (5, 20, 100):
        num = 0.0; den = 0.0; pred = []
        risk = np.zeros(N)
        for g in np.unique(sid):
            tr = sid != g; te = sid == g; r_tr, w_tr = joint(Gi[tr].mean(0)[np.ix_(idx, idx)], V[tr].mean()); W = w_tr.sum(); pred.append((r_tr, te.sum()))
            for k in np.where(te)[0]:
                risk[k] = (n * V[k] + w_tr @ Gi[k][np.ix_(idx, idx)] @ w_tr) / (n + W) ** 2
        worth[n] = float(V.mean() / risk.mean())                                    # respondents whose mean squared error equals the pooled one
    pr = float(np.average([p for p, c in pred], weights=[c for p, c in pred]))
    res["sets"][nm] = {"sum of rates": float(rate[idx].sum()), "rate of the set": rS, "weights": {names[j]: float(x) for j, x in zip(idx, w)}, "rate of the set measured on other programmes": pr, "worth held out": worth}
    print(f"{nm:28s} sum of rates {rate[idx].sum():6.1f} | rate of the set {rS:5.1f} | weights " + " ".join(f"{names[j].replace('GPT-', '')}:{x:.1f}" for j, x in zip(idx, w) if x > 0.05) +
          f" | held out: predicted n + {pr:.1f}; worth at n=5/20/100: " + " / ".join(f"{worth[n]:.1f}" for n in (5, 20, 100)))
# single sources: transfer of the rate across programmes (rate measured on the other six against the rate on the held-out one)
res["by_programme"] = {}
for j, m in enumerate(names):
    row = {}
    for g in np.unique(sid):
        te = sid == g; row[S[g]] = {"held-out rate": float(V[te].mean() / Gi[te][:, j, j].mean()), "rate on the other programmes": float(V[~te].mean() / Gi[~te][:, j, j].mean())}
    res["by_programme"][m] = row
    print(f"{m:14s} rate by programme (held out | others): " + " ".join(f"{s[:5]} {v['held-out rate']:.1f}|{v['rate on the other programmes']:.1f}" for s, v in row.items()))
json.dump(res, open(ROOT + "/results/y_cov.json", "w"), indent=1)

# ------------------------------------------------------------------ real sources: answers to other questions on the same scale, another population
uqa = np.unique(qi); qfa = dict(zip(uqa, rng.permutation(len(uqa)) % 2)); gall = np.array([qfa[q] for q in qi]); sc = {}; dfa = pd.DataFrame({"scale": I.scale.values, "g": gall, "i": range(NALL)})
for (s_, g_), d_ in dfa.groupby(["scale", "g"]):
    other = dfa[(dfa.scale == s_) & (dfa.g != g_)].i.values
    if len(other):
        m = fl(np.mean([h[j] for j in other], 0), 1e-3)
        for i in d_.i.values:
            sc[int(i)] = m
rows_q = {}; opt_t = [tuple(o) for o in I.opts]
for i in range(NALL):
    rows_q.setdefault((src[i], int(qi[i])), []).append(i)
dn = {}
for i in range(NALL):
    c = [j for j in rows_q[(src[i], int(qi[i]))] if j != i and opt_t[j] == opt_t[i]]
    if c:
        dn[i] = soft(0.75 * np.log(fl(h[int(rng.choice(c))], 1e-3)))                     # another population's answers, tempered with a fixed power
REAL = {"same-scale answers": sc, "another population": dn}; allnames = names + list(REAL)
def err_of(nm, k):
    i = int(ids[k])
    if nm in MODELS:
        return Err[nm][k]
    return (REAL[nm][i] - h[i]) if i in REAL[nm] else None
res["real"] = {}
for rn in REAL:
    av = np.array([int(ids[k]) in REAL[rn] for k in range(N)]); e = [err_of(rn, k) for k in range(N)]; g = np.mean([e[k] @ e[k] for k in np.where(av)[0]]); r = V[av].mean() / g
    cors = {m: float(np.mean([e[k] @ Err[m][k] for k in np.where(av)[0]]) / np.sqrt(g * np.mean([Err[m][k] @ Err[m][k] for k in np.where(av)[0]]))) for m in names}
    res["real"][rn] = {"available": float(av.mean()), "rate_mse": float(r), "error correlation with each simulator": cors}
    print(f"{rn}: on {100 * av.mean():.0f}% of items, squared-error rate {r:.1f}; error correlation with the simulators: " + " ".join(f"{m.replace('GPT-', '')} {c:.2f}" for m, c in cors.items()))
both = np.array([int(ids[k]) in sc and int(ids[k]) in dn for k in range(N)]); kk = np.where(both)[0]
e1 = [err_of("same-scale answers", k) for k in kk]; e2 = [err_of("another population", k) for k in kk]
res["real"]["correlation between the two real sources"] = float(np.mean([a @ b for a, b in zip(e1, e2)]) / np.sqrt(np.mean([a @ a for a in e1]) * np.mean([b @ b for b in e2])))
print(f"the two real sources ({both.sum()} items with both): error correlation {res['real']['correlation between the two real sources']:.2f}")
# sets that mix a simulator with real sources, on the items where both real sources exist
res["mixed sets"] = {}
for m in names:
    S3 = [m, "same-scale answers", "another population"]; Gk = np.array([[[float(err_of(a, k) @ err_of(b, k)) for b in S3] for a in S3] for k in kk]); Gm = Gk.mean(0); Vk = V[kk].mean(); r1 = Vk / np.diag(Gm)
    out = {"rates": {a: float(x) for a, x in zip(S3, r1)}}
    for nm, idx in (("simulator + same-scale", [0, 1]), ("simulator + another population", [0, 2]), ("the two real sources", [1, 2]), ("all three", [0, 1, 2])):
        rS, w = joint(Gm[np.ix_(idx, idx)], Vk); out[nm] = {"sum of rates": float(r1[idx].sum()), "rate of the set": rS}
    res["mixed sets"][m] = out
    print(f"{m:14s} rates {' / '.join(f'{x:.1f}' for x in r1)} | " + " | ".join(f"{nm}: sum {v['sum of rates']:.1f}, set {v['rate of the set']:.1f}" for nm, v in out.items() if nm != "rates"))
# does pooling ever hurt? per programme, the simulator pooled with the rate measured on the other programmes, against respondents alone (squared error)
res["safety"] = {}
for m in ("Qwen3.5-9B", "GPT-6.1-sol", "GPT-6-astra"):
    j = names.index(m); row = {}
    for g in np.unique(sid):
        te = sid == g; w = V[~te].mean() / Gi[~te][:, j, j].mean(); true = V[te].mean() / Gi[te][:, j, j].mean()
        row[S[g]] = {"ratio of pseudo-count to true rate": float(w / true), "risk relative to respondents alone": {n: float(((n * V[te].mean() + w * w * Gi[te][:, j, j].mean()) / (n + w) ** 2) / (V[te].mean() / n)) for n in (5, 20, 100, 1000)}}
    res["safety"][m] = row
    print(f"{m:14s} pseudo-count / true rate by programme: " + " ".join(f"{s[:5]} {v['ratio of pseudo-count to true rate']:.2f}" for s, v in row.items()) + " | worst risk relative to respondents alone at n=1000: " + f"{max(v['risk relative to respondents alone'][1000] for v in row.values()):.3f}")
json.dump(res, open(ROOT + "/results/y_cov.json", "w"), indent=1)

# ================================================================== intervals for the squared-error rates (bootstrap over question texts)
def rate_ci(v, g, cl, B=1000):
    s = pd.DataFrame({"v": v, "g": g, "q": cl}).groupby("q")[["v", "g"]].sum().values; r = rng.integers(0, len(s), (B, len(s))); x = s[r, 0].sum(1) / s[r, 1].sum(1)
    return [float(v.sum() / g.sum()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
res["rate_mse_ci"] = {m: rate_ci(V, Gi[:, j, j], qid) for j, m in enumerate(names)}
# marginal value of a second simulator next to the best one, and of the best one next to each
best = int(np.argmax(rate)); res["pairs with the best"] = {}
for j, m in enumerate(names):
    if j != best:
        rS, w = joint(G[np.ix_([best, j], [best, j])], Vm); res["pairs with the best"][m] = {"rate of the pair": rS, "correlation": float(corr[best, j])}
print("best simulator:", names[best], f"{rate[best]:.1f}", "| rate of the pair with each other simulator: " + " ".join(f"{m.replace('GPT-', '')} {v['rate of the pair']:.1f}" for m, v in res["pairs with the best"].items()))

# ================================================================== polls fielded after training
PD = json.load(open((_REPO + "/data/polls_2026/items.json"))); NPn = len(PD); hp = [np.asarray(d["percent"], float) / np.sum(d["percent"]) for d in PD]
nsp = [np.array([bool(re.search(r"not sure|don.?t know|no opinion|refus|skip|unsure|prefer not|undecided|can.?t say|dont know", o.lower())) for o in d["options"]]) for d in PD]
def rule(Lg):                                                   # calibration fitted on the old survey items available for the model
    oi = sorted(Lg); M = np.array([[[kl(h[i], soft(Lg[i] / T - d * ns[i])) for d in DG] for T in TG] for i in oi]); a, b = np.unravel_index(np.argmin(M.mean(0)), (len(TG), len(DG))); return TG[a], DG[b]
def read(d, pat):
    P = {}
    for f in sorted(os.listdir(d)):
        if re.match(pat, f):
            for i, p in json.load(open(f"{d}/{f}"))["items"].items():
                p = np.asarray(p, float); P[int(i)] = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
    return P
def keep_stream(f, *a, **k):                                    # a model added later draws its bootstrap without moving the random stream of the others
    st = rng.bit_generator.state; out = f(*a, **k); rng.bit_generator.state = st; return out
PM = {}
PMcal = {}; PRAW = {}
for nm_, f_ in (("OLMo-3-7B", "polls_olmo.json"), ("Qwen3.5-9B", "polls_qwen.json"), ("Gemma-4-12B", "polls_gemma12.json")):       # the three smallest open models: next-token probabilities, rescaled by a rule fitted on the old surveys
    rawq = json.load(open(D + "r3_out/" + f_)); T_, d_ = rule(MODELS[nm_]); PRAW[nm_] = [(soft(rawq[f"{i}_0"]) + soft(rawq[f"{i}_1"])) / 2 for i in range(NPn)]
    PM[nm_] = [soft(np.log(fl(PRAW[nm_][i], 1e-4)) / T_ - d_ * nsp[i]) for i in range(NPn)]; PMcal[nm_] = PM[nm_]
for nm, tag in STATED:
    if nm not in MODELS or nfiles(f"{SOL}/polls_{tag}", r"t\d+\.json") < 13:
        continue
    P = read(f"{SOL}/polls_{tag}", r"t\d+\.json"); PRAW[nm] = [P[i] for i in range(NPn)]; T_, d_ = rule(MODELS[nm]); PMcal[nm] = [soft(np.log(fl(P[i], 1e-4)) / T_ - d_ * nsp[i]) for i in range(NPn)]; PM[nm] = [fl(P[i], 1e-4) for i in range(NPn)]
pn = list(PM); Vp = np.array([float((x * (1 - x)).sum()) for x in hp]); Ep = {m: [PM[m][i] - hp[i] for i in range(NPn)] for m in pn}
Gp = np.array([[[float(Ep[a][i] @ Ep[b][i]) for b in pn] for a in pn] for i in range(NPn)]); Gpm = Gp.mean(0); rp = Vp.mean() / np.diag(Gpm); cp = Gpm / np.sqrt(np.outer(np.diag(Gpm), np.diag(Gpm)))
pid = pd.factorize(np.array([f"{d_['source']}|{d_['fieldwork_start']}" for d_ in PD]))[0]              # bootstrap over poll releases: questions of one release share a sample
res["polls"] = {"items": NPn, "models": pn, "rate_mse_ci": {m: keep_stream(rate_ci, Vp, Gp[:, j, j], pid) if m in ("OLMo-3-7B", "Gemma-4-12B") else rate_ci(Vp, Gp[:, j, j], pid) for j, m in enumerate(pn)}, "error_correlation": cp.tolist()}
rS, w = joint(Gpm, Vp.mean()); old_idx = [names.index(m) for m in pn]; rO, wO = joint(G[np.ix_(old_idx, old_idx)], Vm)          # weights fitted on the old surveys, applied to the new polls
worth_tr = {n: float(Vp.mean() / np.mean([(n * Vp[i] + wO @ Gp[i] @ wO) / (n + wO.sum()) ** 2 for i in range(NPn)])) for n in (5, 20, 100)}
res["polls"]["all"] = {"sum of rates": float(rp.sum()), "rate of the set": rS, "best single": float(rp.max()), "weights from the old surveys": {m: float(x) for m, x in zip(pn, wO)}, "worth with those weights": worth_tr}
print("new polls, squared-error rates: " + " ".join(f"{m.replace('GPT-', '')} {res['polls']['rate_mse_ci'][m][0]:.1f}" for m in pn))
print(f"   error correlations between simulators on the new polls: min {cp[np.triu_indices(len(pn), 1)].min():.2f}, median {np.median(cp[np.triu_indices(len(pn), 1)]):.2f}, max {cp[np.triu_indices(len(pn), 1)].max():.2f}")
print(f"   all eight: sum of rates {rp.sum():.1f}, rate of the set {rS:.1f}, best single {rp.max():.1f}; with the weights from the old surveys, worth at n=5/20/100: " + " / ".join(f"{worth_tr[n]:.1f}" for n in (5, 20, 100)))
# the two real sources that need no model, on the new polls where they exist
optp = [tuple(x.strip().lower() for x in d["options"]) for d in PD]; qtp = [d["question"].strip().lower() for d in PD]; rngp = np.random.default_rng(0); scp = {}; dnp = {}
for i in range(NPn):
    o_ = [j for j in range(NPn) if optp[j] == optp[i] and qtp[j] != qtp[i]]
    if o_:
        scp[i] = fl(np.mean([hp[j] for j in o_], 0), 1e-3)
    c_ = [j for j in range(NPn) if j != i and optp[j] == optp[i] and qtp[j] == qtp[i] and PD[j]["population"] != PD[i]["population"]]
    if c_:
        dnp[i] = soft(0.75 * np.log(fl(hp[int(rngp.choice(c_))], 1e-3)))
res["polls"]["real"] = {k_: {"items": len(v_), "rate_mse": float(np.mean([Vp[i] for i in v_]) / np.mean([float((v_[i] - hp[i]) @ (v_[i] - hp[i])) for i in v_]))} for k_, v_ in (("same-scale answers", scp), ("another population", dnp))}
print("new polls, real sources:", res["polls"]["real"])
# party groups of the new polls: the poll's topline, the persona alone, and the persona's contrast added to the topline
PARTY = [i for i, d in enumerate(PD) if d.get("by_party") and {"Democrats", "Republicans"} <= set(d["by_party"])]; indk = lambda d: next(k for k in d["by_party"] if k.startswith("Independents"))
cells = [(i, g, key) for i in PARTY for g, key in enumerate(("Democrats", "Republicans", indk(PD[i])))]; hg = [np.asarray(PD[i]["by_party"][key], float) for i, g, key in cells]; hg = [x / x.sum() for x in hg]; Vg = np.array([float((x * (1 - x)).sum()) for x in hg])
et = [hp[i] - hg[c] for c, (i, g, key) in enumerate(cells)]; res["party"] = {"cells": len(cells), "topline rate": float(Vg.mean() / np.mean([e @ e for e in et])), "models": {}}
print(f"party groups ({len(cells)} cells): the topline is worth {res['party']['topline rate']:.1f} respondents of the group (squared error)")
PP = {}
OPEN_STATED = [("OLMo-3-7B", "olmo-stated"), ("Qwen3.5-9B", "qwen9b-stated"), ("Gemma-4-12B", "gemma12-stated")]
for nm, tag in OPEN_STATED + STATED:
    if nfiles(f"{SOL}/pollsparty_{tag}", r"p\d+\.json") < 23:
        continue
    P = read(f"{SOL}/pollsparty_{tag}", r"p\d+\.json"); PP[nm] = P; ep = [P[10 * i + 1 + g] - hg[c] for c, (i, g, key) in enumerate(cells)]
    ec = []
    for c, (i, g, key) in enumerate(cells):
        x = np.clip(hp[i] + (P[10 * i + 1 + g] - P[10 * i]), 1e-4, None); ec.append(x / x.sum() - hg[c])
    Gg = np.array([[np.mean([a @ b for a, b in zip(u, v)]) for v in (et, ep)] for u in (et, ep)]); rset, wset = joint(Gg, Vg.mean())
    res["party"]["models"][nm] = {"persona alone": float(Vg.mean() / Gg[1, 1]), "contrast added to the topline": float(Vg.mean() / np.mean([e @ e for e in ec])), "set of topline and persona": rset, "correlation of topline and persona errors": float(Gg[0, 1] / np.sqrt(Gg[0, 0] * Gg[1, 1]))}
    v = res["party"]["models"][nm]; print(f"   {nm:14s} persona alone {v['persona alone']:.1f} | contrast added to the topline {v['contrast added to the topline']:.1f} | optimal set of topline and persona {v['set of topline and persona']:.1f} | error correlation of topline and persona {v['correlation of topline and persona errors']:.2f}")

# ================================================================== annotators (truth = shares of about 31 annotators; its sampling variance is removed from every entry of G)
Rw = json.load(open(D + "t5_ca_raw.json"))
if os.path.exists(D + "t5_ca_raw_b.json"):
    for k_, v_ in json.load(open(D + "t5_ca_raw_b.json")).items():
        Rw.setdefault(k_, v_)
order = json.load(open(f"{SOL}/ca_items.json")); Cn = np.array([Rw[k_]["counts"] for k_ in order], float); Nn = Cn.sum(1); ph = Cn / Nn[:, None]; Va = (ph * (1 - ph)).sum(1) * Nn / (Nn - 1); noise = Va / Nn
CA = {"Qwen3.5-9B (probabilities)": np.array([soft(np.array(Rw[k_]["pop"]) * 0.36) for k_ in order])}
for nm, tag in STATED:
    if nfiles(sdir("ca", tag), r"b\d+\.json") < 26:
        continue
    P = read(sdir("ca", tag), r"b\d+\.json"); CA[nm] = np.array([P[j] for j in range(len(order))])
cn = list(CA); Ga = np.array([[np.mean(((CA[a] - ph) * (CA[b] - ph)).sum(1) - noise) for b in cn] for a in cn]); ra = Va.mean() / np.diag(Ga); ca_ = Ga / np.sqrt(np.outer(np.diag(Ga), np.diag(Ga))); rS, w = joint(Ga, Va.mean())
res["annotators"] = {"models": cn, "rate_mse": [float(x) for x in ra], "error_correlation": ca_.tolist(), "sum of rates": float(ra.sum()), "rate of the set": rS, "best single": float(ra.max())}
print("annotators, squared-error rates: " + " ".join(f"{m.replace('GPT-', '')} {x:.1f}" for m, x in zip(cn, ra)) + f" | correlations min {ca_[np.triu_indices(len(cn), 1)].min():.2f} median {np.median(ca_[np.triu_indices(len(cn), 1)]):.2f} | all: sum {ra.sum():.1f}, set {rS:.1f}, best {ra.max():.1f}")
json.dump(res, open(ROOT + "/results/y_cov.json", "w"), indent=1)
# everything the later analyses need (z_nest.py), so that they do not repeat the calibration
import pickle
pickle.dump({"names": names, "ids": ids, "sid": sid, "S": S, "qid": qid, "h": [h[i] for i in ids], "ns": [ns[i] for i in ids], "C": C, "Ccal": Ccal, "stated": IS_STATED, "raw": {m: [soft(MODELS[m][int(i)]) for i in ids] for m in names}, "V": V,
             "scale": I.scale.values[ids], "real": {rn: [REAL[rn].get(int(i)) for i in ids] for rn in REAL},
             "polls": {"models": pn, "PM": PM, "PMcal": PMcal, "raw": PRAW, "hp": hp, "nsp": nsp, "meta": [{k_: d.get(k_) for k_ in ("country", "tracker", "sample_size", "source", "topic", "fieldwork_start")} for d in PD]},
             "party": {"cells": cells, "hg": hg, "P": PP}, "annot": {"models": cn, "CA": CA, "ph": ph, "noise": noise, "Va": Va}}, open(ROOT + "/results/y_cov_cache.pkl", "wb"))
