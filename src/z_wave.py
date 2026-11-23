"""Paper 3 (fifth round): an earlier wave of the same question as a source without a model (General Social Survey).
Target: the answers of the 2021 and 2022 waves pooled. Earlier wave: 2016 and 2018 pooled. 36 questions, 20 groups.
  population   rate of the earlier wave's answer for the later wave's answer; rate of each simulator; what the pair is worth
  groups       later topline alone; later topline + the earlier wave's real contrast between group and population;
               later topline + the simulator's contrast; the earlier wave's answer of the group; the simulator's persona
Rates are in respondents of the target (mean variance of one respondent / mean squared error); the sampling noise of the target's
observed answer is removed (for estimates that contain the later topline: (1 - 2 pi) V / m). Bootstrap over questions.
Usage: python z_wave.py -> results/z_wave.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, re, sys, json
import numpy as np, pandas as pd
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); SCREEN = ROOT + "/outputs/screen"; rng = np.random.default_rng(0)
STATED = [("GPT-6-luna", "gpt-6-luna"), ("GPT-5.6-luna", "gpt-5.6-luna"), ("GPT-5.6-sol", "gpt-5.6-sol"), ("GPT-6-sol", "gpt-6-sol"), ("GPT-6.1-terra", "gpt-6.1-terra"), ("GPT-6.1-sol", "gpt-6.1-sol"), ("GPT-6-astra", "gpt-6-astra"),
          ("Claude-Haiku-5.5", "claude-haiku-5-5"), ("Claude-Sonnet-5.5", "claude-sonnet-5-5"), ("Claude-Opus-5.5", "claude-opus-5-5"), ("Claude-Fable-5.2", "claude-fable-5-2")]
sys.path.insert(0, SCREEN); sys.argv = [sys.argv[0]]
from t4_run import ITEMS, PARTS
norm = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum(); sq = lambda a, b: float((a - b) @ (a - b))
d = pd.read_parquet((_REPO + "/data/gss/gss_sub.parquet")); d = d[d.year.isin([2016, 2018, 2021, 2022])].copy(); d["w"] = d.wtssps.fillna(1.0)
d["g_party"] = d.partyid.map({0: "dem", 1: "dem", 2: "ind", 3: "ind", 4: "ind", 5: "rep", 6: "rep"}); d["g_ideo"] = d.polviews.map({1: "lib", 2: "lib", 3: "lib", 4: "mod", 5: "con", 6: "con", 7: "con"})
d["g_degree"] = d.degree.map({0: "nocoll", 1: "nocoll", 2: "nocoll", 3: "coll", 4: "coll"}); d["g_age"] = pd.cut(d.age, [17, 29, 49, 64, 200], labels=["18-29", "30-49", "50-64", "65+"]).astype(object)
d["g_sex"] = d.sex.map({1: "male", 2: "female"}); d["g_race"] = d.race.map({1: "white", 2: "black", 3: "other"})
d["g_attend"] = d.attend.map({0: "rare", 1: "rare", 2: "some", 3: "some", 4: "some", 5: "some", 6: "weekly", 7: "weekly", 8: "weekly"})
keys = json.load(open(f"{SOL}/grp_keys.json")); ITN = keys["items"]; GR = [tuple(x) for x in keys["groups"]]
wd = lambda y, w, k: np.bincount(y, w, minlength=k) / w.sum(); neff = lambda w: w.sum() ** 2 / (w ** 2).sum()
late = d.year.isin([2021, 2022]); early = d.year.isin([2016, 2018])
pop = []; cells = []
for ii, it in enumerate(ITN):
    codes = ITEMS[it][2]; k = len(codes); sl = d[late & d[it].isin(codes)]; se = d[early & d[it].isin(codes)]
    if len(sl) < 300 or len(se) < 300:
        continue
    yl = sl[it].map({c: j for j, c in enumerate(codes)}).values.astype(int); wl = sl.w.values.astype(float); ye = se[it].map({c: j for j, c in enumerate(codes)}).values.astype(int); we = se.w.values.astype(float)
    tl = wd(yl, wl, k); te = wd(ye, we, k); pop.append({"ii": ii, "late": tl, "early": te, "V": float((tl * (1 - tl)).sum()), "m": neff(wl)})
    for gj, (pn, g) in enumerate(GR):
        ml = (sl["g_" + pn] == g).values; me = (se["g_" + pn] == g).values
        if ml.sum() < 60 or me.sum() < 60:
            continue
        hl = wd(yl[ml], wl[ml], k); he = wd(ye[me], we[me], k); cells.append({"ii": ii, "gj": gj, "late": hl, "early": he, "top": tl, "etop": te, "V": float((hl * (1 - hl)).sum()), "m": neff(wl[ml]), "pi": wl[ml].sum() / wl.sum()})
print(f"{len(pop)} questions asked in both periods; {len(cells)} group cells; later-wave effective respondents per question {int(min(p['m'] for p in pop))} to {int(max(p['m'] for p in pop))}, per cell {int(min(c['m'] for c in cells))} to {int(max(c['m'] for c in cells))}")
res = {"questions": len(pop), "cells": len(cells)}
SIM = {}
for nm, tag in STATED:
    dd = f"{SOL}/grp" + ("" if tag == "gpt-6.1-sol" else "_" + tag)
    if os.path.isdir(dd) and len([f for f in os.listdir(dd) if re.match(r"i\d+\.json", f)]) >= len(ITN):
        SIM[nm] = {ii: {int(a): np.asarray(b, float) / np.sum(b) for a, b in json.load(open(f"{dd}/i{ii:02d}.json"))["items"].items()} for ii in range(len(ITN))}
# ---- population: the earlier wave, each simulator, and the pair
Vp = np.array([p["V"] for p in pop]); nz = np.array([p["V"] / p["m"] for p in pop]); ee = [p["early"] - p["late"] for p in pop]; B = 2000; dr = rng.integers(0, len(pop), (B, len(pop)))
def rate(e2, idx=None):
    idx = np.arange(len(pop)) if idx is None else idx; return float(Vp[idx].mean() / max(e2[idx].mean() - nz[idx].mean(), 1e-9))
e_early = np.array([e @ e for e in ee]); res["population"] = {"earlier wave": [rate(e_early), float(np.percentile([rate(e_early, r) for r in dr], 2.5)), float(np.percentile([rate(e_early, r) for r in dr], 97.5))], "simulators": {}}
print(f"population: the earlier wave is worth {res['population']['earlier wave'][0]:.0f} [{res['population']['earlier wave'][1]:.0f}, {res['population']['earlier wave'][2]:.0f}] respondents of the later wave")
for nm, S_ in SIM.items():
    es = [S_[p["ii"]][0] - p["late"] for p in pop]; g11 = np.array([e @ e for e in es]); g12 = np.array([a @ b for a, b in zip(es, ee)])
    def pair(idx):
        a = g11[idx].mean() - nz[idx].mean(); b = e_early[idx].mean() - nz[idx].mean(); c = g12[idx].mean() - nz[idx].mean(); lam = min(1.0, max(0.0, (a - c) / (a + b - 2 * c))); return float(Vp[idx].mean() / ((1 - lam) ** 2 * a + lam ** 2 * b + 2 * lam * (1 - lam) * c)), lam
    full = np.arange(len(pop)); pr, lam = pair(full); bs = [pair(r)[0] for r in dr[:1000]]
    res["population"]["simulators"][nm] = {"rate": rate(g11), "pair with the earlier wave": [pr, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "weight of the earlier wave": lam,
                                           "error correlation with the earlier wave": float((g12.mean() - nz.mean()) / np.sqrt((g11.mean() - nz.mean()) * (e_early.mean() - nz.mean())))}
    v = res["population"]["simulators"][nm]; print(f"   {nm:18s} alone {v['rate']:.0f} | with the earlier wave {pr:.0f} [{v['pair with the earlier wave'][1]:.0f}, {v['pair with the earlier wave'][2]:.0f}] (weight of the wave {lam:.2f}) | error correlation {v['error correlation with the earlier wave']:.2f}")
# ---- groups
C = len(cells); Vc = np.array([c["V"] for c in cells]); nzc = np.array([c["V"] / c["m"] for c in cells]); pic = np.array([c["pi"] for c in cells]); qc = np.array([c["ii"] for c in cells]); uq = np.unique(qc); byq = [np.where(qc == q)[0] for q in uq]
drc = [np.concatenate([byq[j] for j in rng.integers(0, len(uq), len(uq))]) for _ in range(1000)]
def grate(e2, corr, idx=None):
    idx = np.arange(C) if idx is None else idx; return float(Vc[idx].mean() / max(e2[idx].mean() - corr[idx].mean(), 1e-9))
def ci(e2, corr):
    b = [grate(e2, corr, r) for r in drc]; return [grate(e2, corr), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
tcorr = (1 - 2 * pic) * nzc
arms = {"later topline": (np.array([sq(c["top"], c["late"]) for c in cells]), tcorr), "earlier wave's answer of the group": (np.array([sq(c["early"], c["late"]) for c in cells]), nzc),
        "later topline + earlier wave's contrast": (np.array([sq(norm(c["top"] + c["early"] - c["etop"]), c["late"]) for c in cells]), tcorr)}
res["groups"] = {k: ci(*v) for k, v in arms.items()}; res["groups"]["simulators"] = {}
print("groups: " + " | ".join(f"{k} {v[0]:.0f} [{v[1]:.0f}, {v[2]:.0f}]" for k, v in res["groups"].items() if isinstance(v, list)))
for nm, S_ in SIM.items():
    per = np.array([sq(S_[c["ii"]][c["gj"] + 1], c["late"]) for c in cells]); con = np.array([sq(norm(c["top"] + S_[c["ii"]][c["gj"] + 1] - S_[c["ii"]][0]), c["late"]) for c in cells])
    both = np.array([sq(norm(c["top"] + 0.5 * (c["early"] - c["etop"]) + 0.5 * (S_[c["ii"]][c["gj"] + 1] - S_[c["ii"]][0])), c["late"]) for c in cells])
    res["groups"]["simulators"][nm] = {"persona": ci(per, nzc), "later topline + simulator's contrast": ci(con, tcorr), "later topline + mean of the two contrasts": ci(both, tcorr)}
    v = res["groups"]["simulators"][nm]; print(f"   {nm:18s} persona {v['persona'][0]:.0f} | topline + its contrast {v[chr(108) + 'ater topline + simulator' + chr(39) + 's contrast'][0]:.0f} [{v[chr(108) + 'ater topline + simulator' + chr(39) + 's contrast'][1]:.0f}, {v[chr(108) + 'ater topline + simulator' + chr(39) + 's contrast'][2]:.0f}] | topline + mean of both contrasts {v['later topline + mean of the two contrasts'][0]:.0f}")
json.dump(res, open(ROOT + "/results/z_wave.json", "w"), indent=1)

# ---- round 6: the best fixed mixture of the real and the simulated contrast (weight fitted on the other half of the questions)
fold = (np.arange(len(uq)) % 2); fq = {q: fold[j] for j, q in enumerate(uq)}; fc = np.array([fq[q] for q in qc]); res["groups"]["mixture of contrasts"] = {}
real_c = [c["early"] - c["etop"] for c in cells]; base_e = arms["later topline + earlier wave's contrast"][0]
for nm, S_ in SIM.items():
    sim_c = [S_[c["ii"]][c["gj"] + 1] - S_[c["ii"]][0] for c in cells]; e2 = np.zeros(C); lam_ = []
    for f in (0, 1):
        tr = np.where(fc != f)[0]; te = np.where(fc == f)[0]
        y = np.concatenate([cells[c]["late"] - cells[c]["top"] - real_c[c] for c in tr]); x = np.concatenate([sim_c[c] - real_c[c] for c in tr]); lam = float(np.clip(x @ y / (x @ x), 0, 1)); lam_.append(lam)
        for c in te:
            e2[c] = sq(norm(cells[c]["top"] + (1 - lam) * real_c[c] + lam * sim_c[c]), cells[c]["late"])
    v = ci(e2, tcorr); own = res["groups"]["simulators"][nm]["later topline + simulator's contrast"][0]; res["groups"]["mixture of contrasts"][nm] = {"rate": v, "weight of the simulated contrast": lam_, "better of the two alone": max(own, res["groups"]["later topline + earlier wave's contrast"][0])}
    print(f"   [mixture] {nm:18s} weight of the simulated contrast {lam_[0]:.2f}/{lam_[1]:.2f}: {v[0]:.0f} [{v[1]:.0f}, {v[2]:.0f}] against {res['groups']['mixture of contrasts'][nm]['better of the two alone']:.0f} for the better of the two")
json.dump(res, open(ROOT + "/results/z_wave.json", "w"), indent=1)
