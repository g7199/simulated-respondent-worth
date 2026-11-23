"""Held-out replication. The two weekly Economist/YouGov reports after the main data (fieldwork 9-11 and 13-16 November 2026) were kept out of every analysis until the
protocol was fixed, and then run once with the same parser, prompts and scoring on the eleven GPT and Claude simulators.  ->  results/w_fresh.json"""
import os as _os; _os.environ["W_SET"] = "fresh"; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, re, sys, numpy as np
sys.path.insert(0, _REPO + "/src")
from w_groups import load, norm, GROUPS
from w_util import read, MODELS
SOL = _REPO + "/outputs/sol"; T = load(); N = len(T); rng = np.random.default_rng(0); nz = lambda p: np.asarray(p, float) / np.sum(p); sq = lambda a, b: float((a - b) @ (a - b)); NG = len(GROUPS)
th = [nz(t["percent"]) for t in T]; V = np.array([float((x * (1 - x)).sum()) for x in th]); noise = V / np.array([t["n"] for t in T]); hg = [[nz(t["groups"][g]["p"]) for g in GROUPS] for t in T]
_b = {}; BLOCK = np.array([_b.setdefault((t["wave"], norm(t["text"])), len(_b)) for t in T]); bl = np.unique(BLOCK); by = [np.where(BLOCK == b)[0] for b in bl]; draws = [np.concatenate([by[j] for j in rng.integers(0, len(bl), len(bl))]) for _ in range(2000)]
key = lambda t: (norm(t["title"]), norm(t["text"]), tuple(norm(o) for o in t["options"])); old = {key(t) for t in json.load(open(_REPO + "/data/polls_2026/econ_groups.json"))}; is_new = np.array([key(t) not in old for t in T])
rate = lambda P, x: float(V[x].mean() / max(np.mean([sq(P[int(i)], th[int(i)]) for i in x]) - noise[x].mean(), 1e-9)); all_ = np.arange(N); new_ = np.where(is_new)[0]
ci = lambda f: [f(all_), *np.percentile([f(x) for x in draws], [2.5, 97.5]).tolist()]


def set_rate(G, v):
    keep = list(range(len(G)))
    while True:
        w = np.linalg.solve(G[np.ix_(keep, keep)], np.ones(len(keep)))
        if (w >= 0).all():
            return float(v * w.sum())
        keep.pop(int(np.argmin(w)))


def pair_gain(G, v):
    out = []
    for a in range(len(G)):
        for b in range(a + 1, len(G)):
            out.append(set_rate(G[np.ix_([a, b], [a, b])], v) / (v / min(G[a, a], G[b, b])) - 1)
    return np.array(out)


res = {"questions": N, "questions not in the main data": int(is_new.sum()), "reports": 2, "models": {}}; A = {}; B = {}
for nm, tag in MODELS:
    a = read(f"{SOL}/egplain_fresh_{tag}"); b = read(f"{SOL}/egctx_fresh_{tag}")
    if len(a) == N and len(b) == N:
        A[nm] = a; B[nm] = b; res["models"][nm] = {"cold": ci(lambda x: rate(a, x)), "shown": ci(lambda x: rate(b, x)), "shown over cold": ci(lambda x: rate(b, x) / rate(a, x)), "new questions": {"cold": rate(a, new_), "shown": rate(b, new_)}}
names = list(A); E = {m: np.concatenate([A[m][i] - th[i] for i in range(N)]) for m in names}; cut = np.cumsum([0] + [len(x) for x in th]); Gi = np.stack([np.stack([[float(E[x][cut[i]:cut[i + 1]] @ E[y][cut[i]:cut[i + 1]]) for y in names] for x in names]) for i in range(N)])
G = Gi.mean(0) - noise.mean(); v = V.mean(); r = v / np.diag(G); g = pair_gain(G, v); d = np.diag(G); S = {"simulators": len(names), "sum of rates": float(r.sum()), "rate of the set": set_rate(G, v), "best single": float(r.max()), "pair gain: median, largest": [float(np.median(g)), float(g.max())], "pooled G_ij / min": float(G[np.triu_indices(len(G), 1)].sum() / np.minimum.outer(d, d)[np.triu_indices(len(G), 1)].sum())}
S["set over best"] = [S["rate of the set"] / S["best single"] - 1, *np.percentile([set_rate(Gi[x].mean(0) - noise[x].mean(), V[x].mean()) / (V[x].mean() / np.diag(Gi[x].mean(0) - noise[x].mean()).min()) - 1 for x in draws[:600]], [2.5, 97.5]).tolist()]; res["set"] = S
for nm, tag in MODELS:
    J = read(f"{SOL}/egjoint_fresh_{tag}"); W = read(f"{SOL}/egshown_fresh_{tag}"); qs = [i for i in range(N) if all(100 * i + k in J for k in range(21)) and all(100 * i + k in W for k in range(1, 21))]
    if nm not in res["models"] or len(qs) < 0.9 * N:
        continue
    qa = np.array(qs); Vc = np.array([[float((hg[i][k] * (1 - hg[i][k])).sum()) for k in range(NG)] for i in qs]); m = np.array([[T[i]["groups"][GROUPS[k]]["n"] for k in range(NG)] for i in qs], float); pi = m / np.array([[T[i]["n"]] * NG for i in qs]); nzc = Vc / m; tc = (1 - 2 * pi) * nzc
    ec = np.array([[sq(J[100 * i + 1 + k], hg[i][k]) for k in range(NG)] for i in qs]); es = np.array([[sq(W[100 * i + 1 + k], hg[i][k]) for k in range(NG)] for i in qs]); et = np.array([[sq(th[i], hg[i][k]) for k in range(NG)] for i in qs]); bq = BLOCK[qa]; ub = np.unique(bq); byq = [np.where(bq == u)[0] for u in ub]; dq = [np.concatenate([byq[j] for j in rng.integers(0, len(ub), len(ub))]) for _ in range(1000)]
    wr = lambda e, c, x: float(Vc[x].mean() / max(e[x].mean() - c[x].mean(), 1e-9)); full = np.arange(len(qs))
    res["models"][nm]["groups"] = {"questions": len(qs), "overall result": wr(et, tc, full), "cold": [wr(ec, nzc, full), *np.percentile([wr(ec, nzc, x) for x in dq], [2.5, 97.5]).tolist()], "shown": [wr(es, tc, full), *np.percentile([wr(es, tc, x) for x in dq], [2.5, 97.5]).tolist()], "shown over cold": [wr(es, tc, full) / wr(ec, nzc, full), *np.percentile([wr(es, tc, x) / wr(ec, nzc, x) for x in dq], [2.5, 97.5]).tolist()]}
json.dump(res, open(_REPO + "/results/w_fresh.json", "w"), indent=1)
print(N, "questions;", int(is_new.sum()), "not in the main data"); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items()})
for m, v in res["models"].items():
    g_ = v.get("groups"); print(f"{m:18s} cold {v['cold'][0]:.1f} [{v['cold'][1]:.0f}, {v['cold'][2]:.0f}] shown {v['shown'][0]:.1f} ratio {v['shown over cold'][0]:.2f} [{v['shown over cold'][1]:.2f}, {v['shown over cold'][2]:.2f}]" + (f" | groups: result {g_['overall result']:.0f} cold {g_['cold'][0]:.0f} shown {g_['shown'][0]:.0f} x{g_['shown over cold'][0]:.1f} [{g_['shown over cold'][1]:.1f}, {g_['shown over cold'][2]:.1f}] (q={g_['questions']})" if g_ else ""))
