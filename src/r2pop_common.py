"""Round-2 shared loader: item set, cached LLM logits, embeddings (reuses t4_confirm.build)."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys, re, numpy as np, pandas as pd
D = (_REPO + '/outputs/screen/')
sys.path.insert(0, D)
sys.argv = sys.argv if len(sys.argv) > 1 else sys.argv + ['none']
from t4_confirm import build, prompt, L
EPS = 1e-3
def soft(x): x = np.asarray(x, float); e = np.exp(x - x.max()); return e / e.sum()
def fl(p, e): return (1 - e) * p + e / len(p)
def load():
    I = build().reset_index(drop=True); raw = json.load(open(D + 't4c_llm.json')); N = len(I)
    uq = json.load(open(D + 't4c_embq.json')); E = np.load(D + 't4c_emb.npy') if os.path.exists(D + 't4c_emb.npy') else None; qi = I.q.map({q: j for j, q in enumerate(uq)}).values
    P0 = [soft(raw[f'{i}_0']) for i in range(N)]; P1 = [soft(raw[f'{i}_1']) for i in range(N)]
    L0 = [np.log(fl(p, 1e-4)) for p in P0]; L1 = [np.log(fl(p, 1e-4)) for p in P1]; L2 = [np.log(fl((p + q) / 2, 1e-4)) for p, q in zip(P0, P1)]
    return I, E, qi, L0, L1, L2
def klv(h, q):
    q = fl(q, EPS); m = h > 0; return float((h[m] * np.log(h[m] / q[m])).sum())
def boot(v, cl, B=5000, seed=0):
    """mean and 95% CI, cluster bootstrap over cluster ids cl (question texts)"""
    rng = np.random.default_rng(seed); s = pd.DataFrame({'v': v, 'q': cl}).groupby('q').v.agg(['sum', 'count']).values
    r = rng.integers(0, len(s), (B, len(s))); bm = s[r, 0].sum(1) / s[r, 1].sum(1)
    return float(np.mean(v)), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
