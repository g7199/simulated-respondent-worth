"""Paper 3 (third revision): groups, under squared error, for every simulator.

A group's answer distribution is estimated from
  topline     the survey's own answer of all respondents (real data, no model)
  persona     the simulator asked for the group
  contrast    topline + (simulator for the group - simulator for all respondents), clipped at zero and renormalised
  pooled      the best fixed mixture of topline and persona (weights from the other half of the questions)
  free        topline + b (persona - simulator for all) + c (persona - topline), b and c from the other half of the questions
              (every combination of the three with weights summing to one; contrast is b = 1, c = 0)
  given       the simulator asked for the group after being shown the real topline (party groups of the new polls only)
and each is priced in respondents of the group: mean variance of one respondent / mean squared error.
Sampling noise of the group's observed answer is removed: the observed answer of a group with n respondents has variance V / n
around the true one; for estimates that contain the topline (of which the group is a share pi) the correction is (1 - 2 pi) V / n.
Data sets: party groups of the new polls (339 cells, n taken as a third of the poll's sample) and 20 groups of the General Social
Survey 2016-2022 (720 cells, effective n from the survey weights).
Usage: python z_groups.py -> results/z_groups.json   (needs results/y_cov_cache.pkl)
"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, re, sys, json, pickle
import numpy as np, pandas as pd

ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); SCREEN = ROOT + "/outputs/screen"; rng = np.random.default_rng(0)
Z = pickle.load(open(ROOT + "/results/y_cov_cache.pkl", "rb"))
STATED = [("GPT-6-luna", "gpt-6-luna"), ("GPT-5.6-luna", "gpt-5.6-luna"), ("GPT-5.6-sol", "gpt-5.6-sol"), ("GPT-6-sol", "gpt-6-sol"), ("GPT-6.1-terra", "gpt-6.1-terra"), ("GPT-6.1-sol", "gpt-6.1-sol"), ("GPT-6-astra", "gpt-6-astra"),
          ("Claude-Haiku-5.5", "claude-haiku-5-5"), ("Claude-Sonnet-5.5", "claude-sonnet-5-5"), ("Claude-Opus-5.5", "claude-opus-5-5"), ("Claude-Fable-5.2", "claude-fable-5-2")]
soft = lambda x: np.exp(np.asarray(x, float) - np.max(x)) / np.exp(np.asarray(x, float) - np.max(x)).sum()
norm = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum()
sq = lambda a, b: float((a - b) @ (a - b))
res = {}


def read(d, pat):
    P = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if re.match(pat, f):
                for i, p in json.load(open(f"{d}/{f}"))["items"].items():
                    p = np.asarray(p, float); P[int(i)] = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
    return P


def price(name, truth, Vc, noise, pi, top, per, mod, qcl, given=None, B=2000, bcl=None):
    """arms for one simulator; cells are rows; qcl = question of each cell (cross-fitting and bootstrap are by question)"""
    C = len(truth); fold = pd.factorize(qcl)[0] % 2; arms = {}
    arms["topline"] = np.array([sq(top[c], truth[c]) for c in range(C)]); arms["persona"] = np.array([sq(per[c], truth[c]) for c in range(C)])
    arms["contrast"] = np.array([sq(norm(top[c] + per[c] - mod[c]), truth[c]) for c in range(C)])
    pooled = np.zeros(C); free = np.zeros(C); coef = []
    for f in (0, 1):
        tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
        # pooled: topline + c (persona - topline)
        a = np.concatenate([per[c] - top[c] for c in tr]); y = np.concatenate([truth[c] - top[c] for c in tr]); cc = float(a @ y / (a @ a))
        X = np.column_stack([np.concatenate([per[c] - mod[c] for c in tr]), a]); bc = np.linalg.lstsq(X, y, rcond=None)[0]; coef.append([float(bc[0]), float(bc[1])])
        for c in te:
            pooled[c] = sq(norm(top[c] + cc * (per[c] - top[c])), truth[c]); free[c] = sq(norm(top[c] + bc[0] * (per[c] - mod[c]) + bc[1] * (per[c] - top[c])), truth[c])
    arms["pooled"] = pooled; arms["free"] = free
    if given is not None:
        arms["given"] = np.array([sq(given[c], truth[c]) for c in range(C)])
    corr = {"topline": (1 - 2 * pi) * noise, "persona": noise, "contrast": (1 - 2 * pi) * noise, "pooled": (1 - 2 * pi) * noise, "free": (1 - 2 * pi) * noise, "given": (1 - 2 * pi) * noise}
    bc_ = qcl if bcl is None else bcl; uq = np.unique(bc_); byq = [np.where(bc_ == q)[0] for q in uq]; draws = [np.concatenate([byq[j] for j in rng.integers(0, len(uq), len(uq))]) for _ in range(B)]
    out = {"coefficients of the free combination (on the contrast, on persona minus topline)": coef}
    if given is not None:
        dd_ = [arms["contrast"][x].mean() - arms["given"][x].mean() for x in draws]; out["squared error, contrast minus topline shown"] = [float(arms["contrast"].mean() - arms["given"].mean()), float(np.percentile(dd_, 2.5)), float(np.percentile(dd_, 97.5))]
    for k, v in arms.items():
        r = lambda idx, adj: float(Vc[idx].mean() / max(v[idx].mean() - (corr[k][idx].mean() if adj else 0.0), 1e-9))
        bs = [r(d, True) for d in draws]; out[k] = {"rate, as measured": r(np.arange(C), False), "rate": r(np.arange(C), True), "interval": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    ea = [per[c] - truth[c] for c in range(C)]; eb = [mod[c] - top[c] for c in range(C)]
    out["correlation of the simulator's errors for the group and for all respondents"] = float(np.mean([a @ b for a, b in zip(ea, eb)]) / np.sqrt(np.mean([a @ a for a in ea]) * np.mean([b @ b for b in eb])))
    d = [arms["persona"][x].mean() - arms["contrast"][x].mean() for x in draws]; out["squared error, persona minus contrast"] = [float(arms["persona"].mean() - arms["contrast"].mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    d = [arms["topline"][x].mean() - arms["contrast"][x].mean() for x in draws]; out["squared error, topline minus contrast"] = [float(arms["topline"].mean() - arms["contrast"].mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    print(f"   {name:20s} " + " | ".join(f"{k} {out[k]['rate']:.0f} [{out[k]['interval'][0]:.0f}, {out[k]['interval'][1]:.0f}] (raw {out[k]['rate, as measured']:.0f})" for k in arms) +
          f" | free coefficients {coef[0][0]:.2f}, {coef[0][1]:.2f} / {coef[1][0]:.2f}, {coef[1][1]:.2f} | errcorr {out['correlation of the simulator' + chr(39) + 's errors for the group and for all respondents']:.2f}", flush=True)
    return out, arms


# ================================================================== party groups of the new polls
D_opts = [d["options"] for d in json.load(open((_REPO + "/data/polls_2026/items.json")))]
P = Z["party"]; cells = P["cells"]; hg = P["hg"]; hp = Z["polls"]["hp"]; meta = Z["polls"]["meta"]; C = len(cells); Vg = np.array([float((x * (1 - x)).sum()) for x in hg])
ss = np.array([meta[i]["sample_size"] or np.nan for i, g, key in cells], float); ss = np.where(np.isnan(ss), np.nanmedian(ss), ss); ng = ss / 3; noise = Vg / ng; pi = np.full(C, 1 / 3); qcl = np.array([i for i, g, key in cells])
print(f"party groups of the new polls: {C} cells from {len(np.unique(qcl))} questions; poll sample sizes {int(np.min(ss))} to {int(np.max(ss))} (median {int(np.median(ss))}); noise of a group's observed answer is {100 * noise.mean() / Vg.mean() * 1:.2f}% of one respondent's variance")
res["party"] = {"cells": C, "questions": int(len(np.unique(qcl))), "median poll sample": float(np.median(ss)), "models": {}, "by party": {}}
top = [hp[i] for i, g, key in cells]; gi = np.array([g for i, g, key in cells])
rel = pd.factorize(np.array([f"{meta[i]['source']}|{meta[i]['fieldwork_start']}" for i, g, key in cells]))[0]              # poll release of each cell: the unit of the bootstrap
print(f"   {len(np.unique(rel))} poll releases")
qs = read(f"{SOL}/pollsparty_qwen9b-stated", r"p\d+\.json")                 # the open model asked to state percentages, as the proprietary models are
for nm, tag in [("Qwen3.5-9B", "qwen9b-stated")] + STATED:
    if nm not in P["P"] and not (tag == "qwen9b-stated" and len(qs) >= 4 * len(np.unique(qcl))):
        continue
    Pm = qs if tag == "qwen9b-stated" else P["P"][nm]; per = [Pm[10 * i + 1 + g] for i, g, key in cells]; mod = [Pm[10 * i] for i, g, key in cells]
    G = read(f"{SOL}/pollsgiven_{tag}", r"g\d+\.json"); has = np.array([10 * i + 1 + g in G for i, g, key in cells]); sel = np.arange(C)
    if has.all():
        given = [G[10 * i + 1 + g] for i, g, key in cells]
    elif has.mean() >= 0.9:                                      # a few questions not returned in the required format: every arm of this simulator is priced on the cells that have all of them
        sel = np.where(has)[0]; given = [G[10 * cells[c][0] + 1 + cells[c][1]] for c in sel]; print(f"   ({nm}: {int((~has).sum())} cells without the topline-shown answer are left out)")
    else:
        given = None
    sub = lambda L: [L[c] for c in sel]
    res["party"]["models"][nm], arms = price(nm, sub(hg), Vg[sel], noise[sel], pi[sel], sub(top), sub(per), sub(mod), qcl[sel], given, bcl=rel[sel])
    res["party"]["models"][nm]["cells"] = int(len(sel)); gi_ = gi[sel]
    res["party"]["by party"][nm] = {lab: {k: float(Vg[sel][gi_ == g].mean() / (v[gi_ == g].mean() - ((1 if k == "persona" else 1 / 3) * noise[sel][gi_ == g]).mean())) for k, v in arms.items() if k in ("topline", "persona", "contrast", "given")} for g, lab in enumerate(("Democrats", "Republicans", "Independents"))}

def across(tag, EV, Vv, qc, corr, B=1000):
    """errors of several simulators' estimates of the same cells: correlations between simulators, and what the set is worth against its best member.
    The noise of the observed truth is common to every simulator's measured error, so the same correction is subtracted from every entry."""
    nm_ = list(EV); n_ = len(nm_); Gk = np.array([[[float(EV[a][c] @ EV[b][c]) for b in nm_] for a in nm_] for c in range(len(Vv))]) - corr[:, None, None]; Gm = Gk.mean(0); d_ = np.diag(Gm); cr = Gm / np.sqrt(np.outer(d_, d_)); iu_ = np.triu_indices(n_, 1)
    def setr(g, v):
        act = list(range(n_))
        while True:
            w = v * np.linalg.solve(g[np.ix_(act, act)], np.ones(len(act)))
            if (w >= 0).all() or len(act) == 1:
                break
            act.pop(int(np.argmin(w)))
        return float(w.sum())
    uq_ = np.unique(qc); byq_ = [np.where(qc == q)[0] for q in uq_]; ex = []
    for _ in range(B):
        r = np.concatenate([byq_[j] for j in rng.integers(0, len(uq_), len(uq_))]); g = Gk[r].mean(0); v = Vv[r].mean(); ex.append(setr(g, v) / (v / np.diag(g)).max() - 1)
    rS = setr(Gm, Vv.mean()); best = float((Vv.mean() / d_).max())
    out = {"simulators": nm_, "error correlation: min, median, max": [float(cr[iu_].min()), float(np.median(cr[iu_])), float(cr[iu_].max())], "G_ij / min, pooled": float(Gm[iu_].sum() / np.minimum.outer(d_, d_)[iu_].sum()), "sum of rates": float((Vv.mean() / d_).sum()), "rate of the set": rS, "best single": best,
           "set over best": [rS / best - 1, float(np.percentile(ex, 2.5)), float(np.percentile(ex, 97.5))]}
    print(f"   [across simulators] {tag}: error correlations {cr[iu_].min():.2f}/{np.median(cr[iu_]):.2f}/{cr[iu_].max():.2f}, G_ij/min {out['G_ij / min, pooled']:.2f}; sum of rates {out['sum of rates']:.0f}, set {rS:.0f}, best {best:.0f} (+{100 * (rS / best - 1):.0f}% [{100 * out['set over best'][1]:.0f}, {100 * out['set over best'][2]:.0f}])", flush=True)
    return out


full = [nm for nm, tag in STATED if nm in P["P"] and res["party"]["models"].get(nm, {}).get("cells") == C]
EP = {nm: [P["P"][nm][10 * i + 1 + g] - hg[c] for c, (i, g, key) in enumerate(cells)] for nm in full}
EC = {nm: [norm(hp[i] + P["P"][nm][10 * i + 1 + g] - P["P"][nm][10 * i]) - hg[c] for c, (i, g, key) in enumerate(cells)] for nm in full}
EM = {nm: [P["P"][nm][10 * i] - hp[i] for c, (i, g, key) in enumerate(cells)] for nm in full}
res["party"]["across simulators"] = {"persona": across("persona", EP, Vg, rel, noise), "topline + contrast": across("topline + contrast", EC, Vg, rel, noise / 3)}
GV = {}
for nm, tag in STATED:
    G_ = read(f"{SOL}/pollsgiven_{tag}", r"g\d+\.json")
    if nm in full and all(10 * i + 1 + g in G_ for i, g, key in cells):
        GV[nm] = [G_[10 * i + 1 + g] - hg[c] for c, (i, g, key) in enumerate(cells)]
res["party"]["across simulators"]["topline shown"] = across("topline shown", GV, Vg, rel, noise / 3)
# a contrast without a model: the mean real contrast of the same party on the other questions with the same answer options
ot = {}
for c, (i, g, key) in enumerate(cells):
    ot.setdefault((tuple(D_opts[i]), g), []).append(c)
nb = np.full(C, np.nan); have = np.zeros(C, bool); est = [None] * C
for (o_, g), cs_ in ot.items():
    for c in cs_:
        oth = [x for x in cs_ if cells[x][0] != cells[c][0]]
        if len(oth) >= 2:
            est[c] = norm(top[c] + np.mean([hg[x] - top[x] for x in oth], 0)); have[c] = True
hv = np.where(have)[0]; e_same = np.array([sq(est[c], hg[c]) for c in hv]); e_top = np.array([sq(top[c], hg[c]) for c in hv])
res["party"]["contrast without a model"] = {"cells": int(len(hv)), "topline + mean real contrast on the same answer scale": float(Vg[hv].mean() / (e_same.mean() - (noise[hv] / 3).mean())), "topline alone on these cells": float(Vg[hv].mean() / (e_top.mean() - (noise[hv] / 3).mean())),
                                            "models on these cells": {nm: {"persona": float(Vg[hv].mean() / (np.mean([EP[nm][c] @ EP[nm][c] for c in hv]) - noise[hv].mean())), "topline + contrast": float(Vg[hv].mean() / (np.mean([EC[nm][c] @ EC[nm][c] for c in hv]) - (noise[hv] / 3).mean()))} for nm in full}}
v_ = res["party"]["contrast without a model"]; print(f"   [no model] {len(hv)} cells with at least two other questions on the same answer options: topline {v_['topline alone on these cells']:.0f}, topline + mean real contrast of the scale {v_['topline + mean real contrast on the same answer scale']:.0f} | models there: " +
      " ".join(f"{nm.replace('GPT-', '').replace('Claude-', '')} {x['persona']:.0f}/{x['topline + contrast']:.0f}" for nm, x in v_["models on these cells"].items()))
# sensitivity of the noise correction: group size taken as a fifth or a half of the sample
res["party"]["noise sensitivity"] = {}
for frac in (0.2, 1 / 3, 0.5):
    nz = Vg / (ss * frac); res["party"]["noise sensitivity"][f"{frac:.2f}"] = {nm: float(Vg.mean() / (np.mean([e @ e for e in EC[nm]]) - ((1 - 2 * frac) * nz).mean())) for nm in ("Claude-Fable-5.2", "GPT-6-astra", "GPT-5.6-luna") if nm in full}
print("   [noise] topline + contrast with the group taken as 20% / 33% / 50% of the sample: " + " | ".join(f"{k}: " + " ".join(f"{v:.0f}" for v in d.values()) for k, d in res["party"]["noise sensitivity"].items()))
# the groups asked in a call of their own (no population in the same call); the simulator's population answer then comes from the separate topline call
res["party"]["separate calls"] = {}
for nm, tag in STATED:
    S_ = read(f"{SOL}/pollssep_{tag}", r"s\d+\.json"); T_ = read(f"{SOL}/polls_{tag}", r"t\d+\.json")
    if nm in full and all(10 * i + 1 + g in S_ for i, g, key in cells):
        ps_ = np.array([sq(S_[10 * i + 1 + g], hg[c]) for c, (i, g, key) in enumerate(cells)]); cs2 = np.array([sq(norm(hp[i] + S_[10 * i + 1 + g] - T_[i]), hg[c]) for c, (i, g, key) in enumerate(cells)])
        ea = [S_[10 * i + 1 + g] - hg[c] for c, (i, g, key) in enumerate(cells)]; eb = [T_[i] - hp[i] for c, (i, g, key) in enumerate(cells)]
        res["party"]["separate calls"][nm] = {"persona": float(Vg.mean() / (ps_.mean() - noise.mean())), "topline + contrast": float(Vg.mean() / (cs2.mean() - (noise / 3).mean())), "error correlation, group and population": float(np.mean([a @ b for a, b in zip(ea, eb)]) / np.sqrt(np.mean([a @ a for a in ea]) * np.mean([b @ b for b in eb]))),
                                              "same call": {"persona": res["party"]["models"][nm]["persona"]["rate"], "topline + contrast": res["party"]["models"][nm]["contrast"]["rate"]}}
        v_ = res["party"]["separate calls"][nm]; print(f"   [separate calls] {nm}: persona {v_['persona']:.0f} (same call {v_['same call']['persona']:.0f}), topline + contrast {v_['topline + contrast']:.0f} (same call {v_['same call']['topline + contrast']:.0f}), error correlation {v_['error correlation, group and population']:.2f}")
json.dump(res, open(ROOT + "/results/z_groups.json", "w"), indent=1)

# ================================================================== groups of the General Social Survey (loader as in screen/x_grp_analyze.py)
sys.path.insert(0, SCREEN); sys.argv = [sys.argv[0]]
from t4_run import ITEMS, PARTS
PN = list(PARTS); GK = {pn: [g[0] for g in PARTS[pn][1]] for pn in PN}
d = pd.read_parquet((_REPO + "/data/gss/gss_sub.parquet")); d = d[d.year.isin([2016, 2018, 2021, 2022])].copy(); d["w"] = d.wtssps.fillna(1.0)
d["g_party"] = d.partyid.map({0: "dem", 1: "dem", 2: "ind", 3: "ind", 4: "ind", 5: "rep", 6: "rep"}); d["g_ideo"] = d.polviews.map({1: "lib", 2: "lib", 3: "lib", 4: "mod", 5: "con", 6: "con", 7: "con"})
d["g_degree"] = d.degree.map({0: "nocoll", 1: "nocoll", 2: "nocoll", 3: "coll", 4: "coll"}); d["g_age"] = pd.cut(d.age, [17, 29, 49, 64, 200], labels=["18-29", "30-49", "50-64", "65+"]).astype(object)
d["g_sex"] = d.sex.map({1: "male", 2: "female"}); d["g_race"] = d.race.map({1: "white", 2: "black", 3: "other"})
d["g_attend"] = d.attend.map({0: "rare", 1: "rare", 2: "some", 3: "some", 4: "some", 5: "some", 6: "weekly", 7: "weekly", 8: "weekly"})
keys = json.load(open(f"{SOL}/grp_keys.json")); ITN = keys["items"]; GR = [tuple(x) for x in keys["groups"]]
wd = lambda y, w, k: np.bincount(y, w, minlength=k) / w.sum()
truth = []; Vc = []; noise = []; pi = []; top = []; qcl = []; cell = []
for ii, it in enumerate(ITN):
    codes = ITEMS[it][2]; s = d[d[it].isin(codes)]; y = s[it].map({c: j for j, c in enumerate(codes)}).values.astype(int); w = s.w.values.astype(float); k = len(codes); t = wd(y, w, k)
    for gj, (pn, g) in enumerate(GR):
        m = (s["g_" + pn] == g).values; hh = wd(y[m], w[m], k); neff = w[m].sum() ** 2 / (w[m] ** 2).sum(); v = float((hh * (1 - hh)).sum())
        truth.append(hh); Vc.append(v); noise.append(v / neff); pi.append(w[m].sum() / w.sum()); top.append(t); qcl.append(ii); cell.append((ii, gj))
Vc = np.array(Vc); noise = np.array(noise); pi = np.array(pi); qcl = np.array(qcl); C = len(cell)
print(f"General Social Survey: {C} cells from {len(ITN)} items; effective respondents per cell {int(np.min(Vc / noise))} to {int(np.max(Vc / noise))}")
res["gss"] = {"cells": C, "items": len(ITN), "models": {}}; GEP = {}; GEC = {}
TG = [.5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
for nm, path in (("OLMo-3-7B", SCREEN + "/x_grp_out/olmo_fwd.json"), ("Qwen3.5-9B", SCREEN + "/t4_llm.json")):
    FM = {(r["item"], r["part"], r["key"]): np.asarray(r["lp"], float) for r in json.load(open(path)) if r["kind"] == "fwd"}
    # one temperature for the read-out, chosen on the other items by the persona's squared error
    E = np.array([[sq(soft(FM[(ITN[ii], GR[gj][0], GR[gj][1])] / T), truth[c]) for T in TG] for c, (ii, gj) in enumerate(cell)]); per = [None] * C; mod = [None] * C
    for ii in range(len(ITN)):
        T = TG[int(np.argmin(E[qcl != ii].mean(0)))]
        for c in np.where(qcl == ii)[0]:
            per[c] = soft(FM[(ITN[ii], GR[cell[c][1]][0], GR[cell[c][1]][1])] / T); mod[c] = soft(FM[(ITN[ii], "pop", "pop")] / T)
    res["gss"]["models"][nm], _ = price(nm, truth, Vc, noise, pi, top, per, mod, qcl)
for nm, tag in STATED:
    dd = f"{SOL}/grp" + ("" if tag == "gpt-6.1-sol" else "_" + tag)
    if not os.path.isdir(dd) or len([f for f in os.listdir(dd) if re.match(r"i\d+\.json", f)]) < len(ITN):
        continue
    per = [None] * C; mod = [None] * C
    for ii in range(len(ITN)):
        r = {int(a): np.asarray(b, float) / np.sum(b) for a, b in json.load(open(f"{dd}/i{ii:02d}.json"))["items"].items()}
        for c in np.where(qcl == ii)[0]:
            per[c] = r[cell[c][1] + 1]; mod[c] = r[0]
    res["gss"]["models"][nm], _ = price(nm, truth, Vc, noise, pi, top, per, mod, qcl)
    GEP[nm] = [per[c] - truth[c] for c in range(C)]; GEC[nm] = [norm(top[c] + per[c] - mod[c]) - truth[c] for c in range(C)]
res["gss"]["across simulators"] = {"persona": across("GSS persona", GEP, Vc, qcl, noise), "topline + contrast": across("GSS topline + contrast", GEC, Vc, qcl, (1 - 2 * pi) * noise)}
json.dump(res, open(ROOT + "/results/z_groups.json", "w"), indent=1)
