"""T4 confirmatory test of "anchor human distribution x shrunken LLM log-ratio" on an automatic item set.
  python t4_confirm.py collect   # LLM option log-probs, 2 option orders (~6.1k score_options calls) -> t4c_llm.json
  python t4_confirm.py embed     # bge-large-en-v1.5 question embeddings (CPU) -> t4c_emb.npy
  python t4_confirm.py analyze   # all estimators, cross-fitted, cluster bootstrap by question -> t4c_results.json
Items: 41 GSS items (2016-22 micro-data) + all rows of the 6 survey datasets in SimBenchPop (3,000 rows).
An "item" is one (question, population) row; anchors must have a DIFFERENT question text and the SAME response scale."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, re, ast, time, numpy as np, pandas as pd
D = (_REPO + '/outputs/screen/')
sys.path.insert(0, D)
from t4_run import ITEMS as G36, HEAD
L = 'ABCDEFGHIJKLMNOPQRST'
GX = {
 'eqwlth': ("Some people think that the government in Washington ought to reduce the income differences between the rich and the poor, perhaps by raising the taxes of wealthy families or by giving income assistance to the poor. Others think that the government should not concern itself with reducing this income difference between the rich and the poor. Here is a scale from 1 to 7. Think of a score of 1 as meaning that the government ought to reduce the income differences between rich and poor, and a score of 7 meaning that the government should not concern itself with reducing income differences. What score between 1 and 7 comes closest to the way you feel?",
            ['1 (government should reduce income differences)', '2', '3', '4', '5', '6', '7 (government should not concern itself with reducing income differences)'], [1, 2, 3, 4, 5, 6, 7]),
 'helppoor': ("Some people think that the government in Washington should do everything possible to improve the standard of living of all poor Americans; they are at Point 1 on this scale. Other people think it is not the government's responsibility, and that each person should take care of himself; they are at Point 5. Where would you place yourself on this scale?",
              ['1 (government should do everything possible)', '2', '3 (agree with both)', '4', '5 (people should take care of themselves)'], [1, 2, 3, 4, 5]),
 'polviews': ("We hear a lot of talk these days about liberals and conservatives. Where would you place yourself on a seven-point scale from extremely liberal to extremely conservative?",
              ['Extremely liberal', 'Liberal', 'Slightly liberal', 'Moderate, middle of the road', 'Slightly conservative', 'Conservative', 'Extremely conservative'], [1, 2, 3, 4, 5, 6, 7]),
 'partyid': ("Generally speaking, do you usually think of yourself as a Republican, Democrat, Independent, or what?",
             ['Strong Democrat', 'Not very strong Democrat', 'Independent, close to Democrat', 'Independent', 'Independent, close to Republican', 'Not very strong Republican', 'Strong Republican', 'Other party'], [0, 1, 2, 3, 4, 5, 6, 7]),
 'attend': ("How often do you attend religious services?",
            ['Never', 'Less than once a year', 'About once or twice a year', 'Several times a year', 'About once a month', '2-3 times a month', 'Nearly every week', 'Every week', 'Several times a week'], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
}
SURV = ['OpinionQA', 'ESS', 'Afrobarometer', 'ISSP', 'LatinoBarometro', 'GlobalOpinionQA']


def build():
    """-> DataFrame: src, pop (population prompt), q (question text), opts (list), h (human dist), n (sample size or -1), hA/hB (GSS real halves)"""
    rows = []
    d = pd.read_parquet((_REPO + '/data/gss/gss_sub.parquet')); d = d[d.year.isin([2016, 2018, 2021, 2022])]; w = d.wtssps.fillna(1.0).values
    rng = np.random.default_rng(0)
    for it, (q, o, codes) in {**G36, **GX}.items():
        x = d[it].values; m = np.isin(x, codes); half = rng.random(len(x)) < 0.5
        f = lambda mm: np.array([w[mm & (x == c)].sum() for c in codes]) / w[mm].sum()
        rows.append(dict(src='GSS', pop=HEAD, q=q, opts=list(o), h=f(m), n=int(m.sum()), hA=f(m & half), hB=f(m & ~half)))
    p = pd.read_csv((_REPO + '/data/simbench/SimBenchPop.csv')); p = p[p.dataset_name.isin(SURV)]
    for r in p.itertuples():
        q, _, ob = r.input_template.partition('\n\nOptions:\n')
        opts = re.findall(r'^\(([A-Z])\): (.*)$', ob, flags=re.M); ha = ast.literal_eval(r.human_answer)
        if not opts or [a for a, _ in opts] != list(ha): continue
        h = np.array([ha[a] for a, _ in opts], float)
        if h.sum() <= 0 or len(opts) > 15: continue
        rows.append(dict(src=r.dataset_name, pop=r.group_prompt_template.strip(), q=q.strip(), opts=[t.strip() for _, t in opts], h=h / h.sum(), n=int(r.group_size), hA=None, hB=None))
    I = pd.DataFrame(rows)
    I['scale'] = I.opts.map(lambda o: '|'.join(re.sub(r'[^a-z0-9 ]', '', x.lower()).strip() for x in o)); I['K'] = I.opts.map(len)
    return I


def prompt(pop, q, opts):
    return f"{pop}\n\n{q}\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + "\nAnswer with a single letter."


if __name__ == '__main__' and sys.argv[1] == 'collect':
    from llm import score_options
    I = build(); print('items', len(I), I.src.value_counts().to_dict(), flush=True)
    P, K = [], []
    for i, r in enumerate(I.itertuples()):
        P.append(prompt(r.pop, r.q, r.opts)); K.append((i, 0)); P.append(prompt(r.pop, r.q, r.opts[::-1])); K.append((i, 1))
    print('calls', len(P), 'max chars', max(map(len, P)), flush=True); t0 = time.time(); out = {}
    for k in sorted(I.K.unique()):
        idx = [j for j, (i, o) in enumerate(K) if I.K.iloc[i] == k]
        for j, r in zip(idx, score_options([P[j] for j in idx], list(L[:k]), workers=96)):
            lp = [r[c] for c in L[:k]]; out[f'{K[j][0]}_{K[j][1]}'] = lp if K[j][1] == 0 else lp[::-1]     # reversed order mapped back to original option order
        print(k, len(idx), round(time.time() - t0, 1), flush=True)
    json.dump(out, open(D + 't4c_llm.json', 'w'))

if __name__ == '__main__' and sys.argv[1] == 'embed':
    from sentence_transformers import SentenceTransformer
    I = build(); uq = sorted(I.q.unique()); t0 = time.time()
    E = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu').encode(uq, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(D + 't4c_emb.npy', E); json.dump(uq, open(D + 't4c_embq.json', 'w')); print('embedded', len(uq), E.shape, round(time.time() - t0, 1))

# ------------------------------------------------------------------ analyze ------------------------------------------------------------------
if __name__ == '__main__' and sys.argv[1] == 'analyze':
    from scipy.optimize import minimize
    rng = np.random.default_rng(0); EPS = 1e-3; NF = 10; B = 5000
    I = build(); N = len(I); raw = json.load(open(D + 't4c_llm.json'))
    uq = json.load(open(D + 't4c_embq.json')); E = np.load(D + 't4c_emb.npy'); qi = I.q.map({q: j for j, q in enumerate(uq)}).values
    def soft(x): x = np.asarray(x, float); e = np.exp(x - x.max()); return e / e.sum()
    def fl(p, e): return (1 - e) * p + e / len(p)
    h = list(I.h.values); Kn = I.K.values
    H = [np.log(fl(x, EPS)) for x in h]                                              # log human (anchor side)
    P0 = [soft(raw[f'{i}_0']) for i in range(N)]; P1 = [soft(raw[f'{i}_1']) for i in range(N)]
    FEAT = {'1order': [np.log(fl(p, 1e-4)) for p in P0], '2order': [np.log(fl((p + q) / 2, 1e-4)) for p, q in zip(P0, P1)]}
    print('items', N, I.src.value_counts().to_dict(), '| questions', len(uq), '| scales', I.scale.nunique())
    print('LLM letter mass (median sum exp lp, order 0): %.3f | top-1 = human mode: %.3f' % (np.median([np.exp(raw[f"{i}_0"]).sum() for i in range(N)]), np.mean([np.argmax(P0[i]) == np.argmax(h[i]) for i in range(N)])))
    def kl(i, logq):
        q = fl(soft(logq), EPS); m = h[i] > 0; return float((h[i][m] * np.log(h[i][m] / q[m])).sum())
    qfold = rng.permutation(len(uq)) % NF; fold = qfold[qi]                         # folds by question text
    # ---- anchors: same scale, different question, nearest by cosine; one row per anchor question (same population preferred)
    def anchors(cross):
        A, S, ALL = [None] * N, [None] * N, [None] * N
        for _, g in I.groupby('scale'):
            idx = g.index.values
            if len(idx) < 2: continue
            sim = E[qi[idx]] @ E[qi[idx]].T
            for a, t in enumerate(idx):
                ok = (qi[idx] != qi[t]) & ((I.src.values[idx] != I.src.values[t]) if cross else True)
                if not ok.any(): continue
                c = idx[ok]; s = sim[a][ok] + 1e-3 * (I['pop'].values[c] == I['pop'].values[t])
                o = np.argsort(-s); seen, keep = set(), []
                for j in o:
                    if qi[c[j]] in seen: continue
                    seen.add(qi[c[j]]); keep.append(j)
                    if len(keep) == 5: break
                A[t] = c[keep]; S[t] = sim[a][ok][keep]; ALL[t] = c
        return A, S, ALL
    def cf(M, valid, groups=None, min_g=10):
        """cross-fitted choice of a grid column: for fold f pick the column with the lowest mean over valid items outside f (optionally per group)."""
        out = np.full(N, np.nan); ch = []
        for f in range(NF):
            tr = valid & (fold != f); te = np.where(valid & (fold == f))[0]; g = int(np.nanargmin(M[tr].mean(0))); ch.append(g)
            if groups is None: out[te] = M[te, g]
            else:
                for i in te:
                    trg = tr & (groups == groups[i]); gi = int(np.argmin(M[trg].mean(0))) if len(set(qi[trg])) >= min_g else g; out[i] = M[i, gi]
        return out, ch
    WG = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0, 1.25, 1.5]); TG = np.array([.5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16])
    qsum = lambda v: pd.Series(v).groupby(qi).agg(['sum', 'count'])
    def boot(v, mask=None):
        """mean and 95% CI, cluster bootstrap over question texts"""
        m = np.ones(N, bool) if mask is None else mask; s = pd.DataFrame({'v': v[m], 'q': qi[m]}).groupby('q').v.agg(['sum', 'count']).values
        r = rng.integers(0, len(s), (B, len(s))); bm = s[r, 0].sum(1) / s[r, 1].sum(1)
        return v[m].mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)
    def mle_fit(tr, Lf, a0, lam0):
        k = len(h[tr[0]]); X = np.array([Lf[j] for j in tr]); Y = np.array([h[j] for j in tr]); lam = lam0 / len(tr)
        def f(th):
            z = th[0] * X + th[1:]; z = z - z.max(1, keepdims=True); q = np.exp(z); q /= q.sum(1, keepdims=True)
            loss = -(Y * np.log(q + 1e-12)).sum(1).mean() + lam * ((th[1:] ** 2).sum() + (th[0] - a0) ** 2)
            return loss, np.r_[((q - Y) * X).sum(1).mean() + 2 * lam * (th[0] - a0), (q - Y).mean(0) + 2 * lam * th[1:]]
        return minimize(f, np.r_[a0, np.zeros(k)], jac=True, method='L-BFGS-B', options=dict(maxiter=60)).x

    def run(cross, tag):
        A, S, ALL = anchors(cross); has = np.array([a is not None for a in A]); res = {}; info = {}
        print(f'\n################ {tag}: items with >=1 same-scale other-question anchor: {has.sum()} / {N}', I.src[has].value_counts().to_dict())
        res['uniform'] = np.array([kl(i, np.zeros(Kn[i])) for i in range(N)])
        for k in (1, 3, 5):
            res[f'kNN mean of anchors, k={k} (no LLM)'] = np.array([kl(i, np.log(np.mean([h[j] for j in A[i][:k]], 0) + 1e-12)) if has[i] else res['uniform'][i] for i in range(N)])
        res['mean of ALL same-scale items (no LLM)'] = np.array([kl(i, np.log(np.mean([h[j] for j in ALL[i]], 0) + 1e-12)) if has[i] else res['uniform'][i] for i in range(N)])
        for fn, Lf in FEAT.items():
            MT = np.array([[kl(i, Lf[i] / T) for T in TG] for i in range(N)]); llmT, chT = cf(MT, np.ones(N, bool)); res[f'LLM + T [{fn}]'] = llmT
            info[f'T [{fn}]'] = [float(TG[c]) for c in chT]
            # calibration map, closed form: per-option bias = mean log-residual over ALL same-scale other-question items; slope global or per scale
            mH = [np.mean([H[j] for j in ALL[i]], 0) if has[i] else None for i in range(N)]; mL = [np.mean([Lf[j] for j in ALL[i]], 0) if has[i] else None for i in range(N)]
            MC = np.array([[kl(i, mH[i] + a * (Lf[i] - mL[i])) if has[i] else np.nan for a in WG] for i in range(N)])
            c1, ch1 = cf(MC, has); res[f'calib map: global slope + per-scale option bias [{fn}]'] = np.where(has, c1, llmT)
            c2, _ = cf(MC, has, groups=I.scale.values); res[f'calib map: per-scale slope + option bias [{fn}]'] = np.where(has, c2, llmT)
            info[f'calib slope [{fn}]'] = [float(WG[c]) for c in ch1]
            # calibration map, KL-optimal (L-BFGS) per scale: temperature + option bias, ridge lam0/n
            for lam0 in (0.2, 2.0):
                v = llmT.copy(); a0 = 1 / np.median(info[f'T [{fn}]'])
                for sc, g in I[has].groupby('scale'):
                    idx = g.index.values
                    for key in (sorted(set(I.src.values[idx])) if cross else range(NF)):
                        te = idx[(I.src.values[idx] == key) if cross else (fold[idx] == key)]
                        if not len(te): continue
                        tr = np.array([j for j in np.where(I.scale.values == sc)[0] if ((I.src.values[j] != key) if cross else (fold[j] != key))])
                        if len(tr) < 2: continue
                        th = mle_fit(tr, Lf, a0, lam0)
                        for i in te: v[i] = kl(i, th[0] * Lf[i] + th[1:])
                res[f'calib map: KL-optimal per-scale T + bias, ridge {lam0} [{fn}]'] = v
            # METHOD: anchor human x (LLM target / LLM anchor)^w, log-space average over k nearest anchors
            for k in (1, 3, 5):
                mHk = [np.mean([H[j] for j in A[i][:k]], 0) if has[i] else None for i in range(N)]; mLk = [np.mean([Lf[j] for j in A[i][:k]], 0) if has[i] else None for i in range(N)]
                MM = np.array([[kl(i, mHk[i] + w * (Lf[i] - mLk[i])) if has[i] else np.nan for w in WG] for i in range(N)])
                m, chw = cf(MM, has); res[f'METHOD k={k} [{fn}]'] = np.where(has, m, llmT); info[f'w k={k} [{fn}]'] = [float(WG[c]) for c in chw]
                if k == 3: info[f'fixed-w curve k=3 [{fn}] (anchor items only)'] = dict(zip(WG.tolist(), np.round(MM[has].mean(0), 4).tolist()))
        # human split-half: GSS real halves; SimBench parametric multinomial halves of size n/2
        sh = np.full(N, np.nan)
        for i, r in enumerate(I.itertuples()):
            if r.hA is not None: sh[i] = float((r.hB[r.hB > 0] * np.log(r.hB[r.hB > 0] / fl(r.hA, EPS)[r.hB > 0])).sum())
            elif r.n > 20:
                a = rng.multinomial(r.n // 2, r.h) / (r.n // 2); b = rng.multinomial(r.n // 2, r.h) / (r.n // 2); sh[i] = float((b[b > 0] * np.log(b[b > 0] / fl(a, EPS)[b > 0])).sum())
        base = [k for k in res if not k.startswith('METHOD')]; best = min(base, key=lambda k: res[k].mean()); bestA = min(base, key=lambda k: res[k][has].mean())
        print('chosen hyper-parameters per fold:', {k: v for k, v in info.items() if 'curve' not in k}); print({k: v for k, v in info.items() if 'curve' in k})
        print(f'\n-- {tag}: FULL SET n={N} (items without an anchor fall back to LLM+T; kNN falls back to uniform). KL nats, 95% CI cluster-bootstrap by question')
        print(f'best baseline (full set) = {best}')
        tab = []
        for k, v in res.items():
            m, lo, hi = boot(v); dm, dlo, dhi = boot(v - res[best]); ma = v[has].mean(); da, dalo, dahi = boot(v - res[bestA], has)
            tab.append(dict(arm=k, kl=m, lo=lo, hi=hi, d=dm, dlo=dlo, dhi=dhi, rel=100 * dm / res[best].mean(), kl_anch=ma, d_anch=da, d_anch_lo=dalo, d_anch_hi=dahi, rel_anch=100 * da / res[bestA][has].mean()))
            print(f'{k[:66]:66s} {m:.4f} [{lo:.4f},{hi:.4f}] d {dm:+.4f} [{dlo:+.4f},{dhi:+.4f}] {100 * dm / res[best].mean():+6.1f}% | anchored-only {ma:.4f} d {da:+.4f} [{dalo:+.4f},{dahi:+.4f}] {100 * da / res[bestA][has].mean():+6.1f}%')
        ok = ~np.isnan(sh); print('human split-half (GSS real halves; SimBench parametric multinomial from n): %.4f over %d items; GSS only %.4f' % (sh[ok].mean(), ok.sum(), sh[(I.src == 'GSS').values].mean()))
        print(f'(anchored-only best baseline = {bestA})')
        meth = 'METHOD k=3 [2order]'
        print(f'\n-- {tag}: by source ({meth} vs {best}; anchored items only)')
        for s in I.src.unique():
            m = has & (I.src == s).values
            if m.sum() < 5: print(f'  {s:16s} n={m.sum()} (too few)'); continue
            d, lo, hi = boot(res[meth] - res[best], m); print(f'  {s:16s} n={m.sum():4d} uniform {res["uniform"][m].mean():.4f}  best-baseline {res[best][m].mean():.4f}  method {res[meth][m].mean():.4f}  diff {d:+.4f} [{lo:+.4f},{hi:+.4f}] {100 * d / res[best][m].mean():+.1f}%')
        s1 = np.array([S[i][0] if has[i] else np.nan for i in range(N)]); cuts = np.nanpercentile(s1, [100 / 3, 200 / 3])
        print(f'-- {tag}: by top-1 anchor cosine similarity tercile (cuts {cuts.round(3)}), {meth} vs {best} and vs kNN mean k=3')
        for name, m in [('low', has & (s1 <= cuts[0])), ('mid', has & (s1 > cuts[0]) & (s1 <= cuts[1])), ('high', has & (s1 > cuts[1]))]:
            d, lo, hi = boot(res[meth] - res[best], m); d2, lo2, hi2 = boot(res[meth] - res['kNN mean of anchors, k=3 (no LLM)'], m)
            print(f'  {name:5s} n={m.sum():4d} best-baseline {res[best][m].mean():.4f} kNN3 {res["kNN mean of anchors, k=3 (no LLM)"][m].mean():.4f} method {res[meth][m].mean():.4f} | vs best {d:+.4f} [{lo:+.4f},{hi:+.4f}] {100 * d / res[best][m].mean():+.1f}% | vs kNN3 {d2:+.4f} [{lo2:+.4f},{hi2:+.4f}]')
        return dict(table=tab, best=best, bestA=bestA, n_anchor=int(has.sum()), info={k: (v if isinstance(v, dict) else list(v)) for k, v in info.items()}, split_half=float(sh[ok].mean()))

    out = {'all_sources': run(False, 'ANCHORS FROM ANY SOURCE'), 'cross_source': run(True, 'ANCHORS ONLY FROM A DIFFERENT SOURCE')}
    json.dump(out, open(D + 't4c_results.json', 'w'), indent=1)
