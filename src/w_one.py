"""Same-survey answers in the prompt when the target is asked alone (one question per call) instead of in a batch of twenty.  ->  results/w_one.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, numpy as np
sys.path.insert(0, _REPO + "/src")
from w_groups import load, norm
from w_util import read
SOL = _REPO + "/outputs/sol"; T = load(); N = len(T); rng = np.random.default_rng(0); nz = lambda p: np.asarray(p, float) / np.sum(p); th = [nz(t["percent"]) for t in T]; V = np.array([float((x * (1 - x)).sum()) for x in th]); noise = V / np.array([t["n"] for t in T])
_b = {}; BLOCK = np.array([_b.setdefault((t["wave"], norm(t["text"])), len(_b)) for t in T]); bl = np.unique(BLOCK); by = [np.where(BLOCK == b)[0] for b in bl]; draws = [np.concatenate([by[j] for j in rng.integers(0, len(bl), len(bl))]) for _ in range(2000)]
rate = lambda P, x: float(V[x].mean() / max(np.mean([((P[int(i)] - th[int(i)]) ** 2).sum() for i in x]) - noise[x].mean(), 1e-9)); out = {}; all_ = np.arange(N)
for nm, tag in (("Claude-Fable-5.2", "claude-fable-5-2"), ("GPT-6.1-sol", "gpt-6.1-sol")):
    A, B, A1, B1 = (read(f"{SOL}/eg{m}_{tag}") for m in ("plain", "ctx", "plainone", "ctxone")); assert all(len(x) == N for x in (A, B, A1, B1)), nm
    f = lambda c, s: [rate(s, all_) / rate(c, all_), *np.percentile([rate(s, x) / rate(c, x) for x in draws], [2.5, 97.5]).tolist()]
    out[nm] = {"batch of twenty": {"cold": rate(A, all_), "shown": rate(B, all_), "shown over cold": f(A, B)}, "one question per call": {"cold": rate(A1, all_), "shown": rate(B1, all_), "shown over cold": f(A1, B1)}}
    for k, v in out[nm].items():
        print(f"{nm:18s} {k:24s} cold {v['cold']:.1f} shown {v['shown']:.1f} ratio {v['shown over cold'][0]:.2f} [{v['shown over cold'][1]:.2f}, {v['shown over cold'][2]:.2f}]")
json.dump(out, open(_REPO + "/results/w_one.json", "w"), indent=1)
