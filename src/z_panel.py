"""Panels of sixteen simulated individuals for two frontier models (x_ladder.py with PANEL=k): the panel's answer is the mean of the
sixteen stated probability vectors, used as given. Compared with the model asked directly for the population's shares (main run),
on the common items of all fifty batches.  python z_panel.py -> results/z_panel.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, re, json, pickle
import numpy as np
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); Z = pickle.load(open(ROOT + "/results/y_cov_cache.pkl", "rb")); ids = Z["ids"]; h = Z["h"]; V = Z["V"]; C = Z["C"]; pos = {int(i): k for k, i in enumerate(ids)}
def rd(d):
    P = {}
    for f in os.listdir(d):
        if re.match(r"b\d+\.json", f):
            for i, p in json.load(open(f"{d}/{f}"))["items"].items():
                p = np.asarray(p, float); P[int(i)] = p / p.sum() if p.sum() > 0 else None
    return P
PAN = {}
for nm, tag in (("Claude-Fable-5.2", "claude-fable-5-2"), ("GPT-6.1-sol", "gpt-6.1-sol")):
    runs = [rd(f"{SOL}/survey_{tag}_panel{k}") for k in range(16)]; com = set.intersection(*[set(r) for r in runs]) & set(pos); PAN[nm] = {i: np.mean([r[i] for r in runs], 0) for i in com if all(r[i] is not None for r in runs)}
    one = {i: runs[0][i] for i in PAN[nm]}; PAN[nm + " (one persona)"] = one
com = sorted(set(PAN["Claude-Fable-5.2"]) & set(PAN["GPT-6.1-sol"])); kk = [pos[i] for i in com]; v = V[kk].mean(); res = {"items": len(com)}
err = lambda P: [P[i] - h[pos[i]] for i in com]; direct = lambda nm: {i: C[nm][pos[i]] for i in com}
def stats(A, B):
    ea, eb = err(A), err(B); g11 = np.mean([x @ x for x in ea]); g22 = np.mean([x @ x for x in eb]); g12 = np.mean([x @ y for x, y in zip(ea, eb)]); lam = float(np.clip((g11 - g12) / (g11 + g22 - 2 * g12), 0, 1)) if g11 <= g22 else float(np.clip((g22 - g12) / (g11 + g22 - 2 * g12), 0, 1))
    s, w = (g11, g22) if g11 <= g22 else (g22, g11); pair = v / ((1 - lam) ** 2 * s + lam ** 2 * w + 2 * lam * (1 - lam) * g12); return {"rates": [float(v / g11), float(v / g22)], "error correlation": float(g12 / np.sqrt(g11 * g22)), "pair": float(pair), "gain over the stronger": float(pair / max(v / g11, v / g22) - 1)}
for nm in ("Claude-Fable-5.2", "GPT-6.1-sol"):
    res[nm] = {"panel against the direct answer": stats(PAN[nm], direct(nm)), "one persona": float(v / np.mean([e @ e for e in err(PAN[nm + ' (one persona)'])]))}
    s = res[nm]["panel against the direct answer"]; print(f"{nm}: panel of sixteen {s['rates'][0]:.1f}, asked directly {s['rates'][1]:.1f}, one persona {res[nm]['one persona']:.1f}; errors of panel and direct answer correlate {s['error correlation']:.3f}; together {s['pair']:.1f} (+{100 * s['gain over the stronger']:.1f}%)")
res["between the two panels"] = stats(PAN["Claude-Fable-5.2"], PAN["GPT-6.1-sol"]); res["between the two direct answers"] = stats(direct("Claude-Fable-5.2"), direct("GPT-6.1-sol")); res["panel of one, direct answer of the other"] = stats(PAN["Claude-Fable-5.2"], direct("GPT-6.1-sol"))
for k in ("between the two panels", "between the two direct answers", "panel of one, direct answer of the other"):
    s = res[k]; print(f"{k}: error correlation {s['error correlation']:.3f}; pair {s['pair']:.1f} (+{100 * s['gain over the stronger']:.1f}%)")
json.dump(res, open(ROOT + "/results/z_panel.json", "w"), indent=1)
