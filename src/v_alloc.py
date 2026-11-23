"""Direction B, design 3 (no model calls): how to spend a fixed budget of interviews when a simulator is at hand.
Budget: B respondents per question on average.  Designs:
  uniform        every question gets B respondents; pooled with the cold simulator by the m + r rule, or adaptively
  two-stage      a probe of m0 respondents on every question; the worth of the simulator on each question is estimated from its distance to the probe;
                 the rest of the budget goes where n_i + r_i is smallest (water-filling); final estimate by adaptive pooling
  oracle         the same water-filling with the true error of the simulator on each question
All constants are fitted on the other reports.  -> results/v_alloc.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json
import numpy as np
sys.path.insert(0, (_REPO + "/src"))
from w_groups import load, norm
from w_util import read, MODELS
T = load(); N = len(T); SOL = (_REPO + "/outputs/sol"); waves = sorted({t["wave"] for t in T}); wave = np.array([t["wave"] for t in T])
th = [np.asarray(t["percent"], float) / sum(t["percent"]) for t in T]; V = np.array([(x * (1 - x)).sum() for x in th])
rng = np.random.default_rng(1)


def waterfill(r, Vq, budget):
    """n_i >= 0 with sum n_i = budget minimising sum V_i / (n_i + r_i): n_i + r_i = c sqrt(V_i) where positive"""
    s = np.sqrt(Vq); lo, hi = 0.0, 1e7
    for _ in range(80):
        c = (lo + hi) / 2; n = np.maximum(c * s - r, 0)
        if n.sum() > budget:
            hi = c
        else:
            lo = c
    return np.maximum(lo * s - r, 0)


def run(C, B, m0, reps=40):
    e2 = np.array([(C[i] - th[i]) @ (C[i] - th[i]) for i in range(N)]); out = {k: [] for k in ("respondents alone", "uniform, m + r rule", "uniform, adaptive pooling", "two-stage, adaptive", "oracle allocation, adaptive", "two-stage, share of budget on the worst tenth")}
    for rep in range(reps):
        err = {k: np.zeros(N) for k in out}
        for w in waves:
            tr = np.where(wave != w)[0]; te = np.where(wave == w)[0]; Vm = V[tr].mean(); r = Vm / e2[tr].mean()
            def dist(x, i, n):
                return max(float((x - C[i]) @ (x - C[i])) - float((x * (1 - x)).sum()) / max(n, 1), 0.0)
            def adaptive(x, i, n, a, b):
                ri = Vm / (a + b * dist(x, i, n)); return (n * x + ri * C[i]) / (n + ri), ri
            # constants of the adaptive rule: fitted on the other reports with n = B and n = m0 respondents
            def fit(n):
                X = {i: rng.multinomial(n, th[i]) / n for i in tr}; grid = [(a * Vm / r, b) for a in (.25, .5, 1, 2) for b in (0, .25, .5, 1, 2, 4, 8)]
                return min(grid, key=lambda ab: np.mean([(adaptive(X[i], i, n, *ab)[0] - th[i]) @ (adaptive(X[i], i, n, *ab)[0] - th[i]) for i in tr]))
            abB = fit(B); ab0 = fit(m0)
            for i in te:
                x = rng.multinomial(B, th[i]) / B; err["respondents alone"][i] = (x - th[i]) @ (x - th[i]); p = (B * x + r * C[i]) / (B + r); err["uniform, m + r rule"][i] = (p - th[i]) @ (p - th[i])
                p, _ = adaptive(x, i, B, *abB); err["uniform, adaptive pooling"][i] = (p - th[i]) @ (p - th[i])
            # two-stage: probe, estimate worth per question, water-fill the rest, pool adaptively with all answers
            probe = {i: rng.multinomial(m0, th[i]) for i in te}; rhat = np.array([adaptive(probe[i] / m0, i, m0, *ab0)[1] for i in te]); Vh = np.array([max(float(((probe[i] / m0) * (1 - probe[i] / m0)).sum()), 0.05) for i in te])
            for key, rr in (("two-stage, adaptive", rhat), ("oracle allocation, adaptive", V[te] / np.maximum(e2[te], 1e-6))):
                extra = waterfill(rr + m0, Vh if key.startswith("two") else V[te], (B - m0) * len(te)); extra = np.floor(extra).astype(int)
                for j, i in enumerate(te):
                    cnt = probe[i] + (rng.multinomial(extra[j], th[i]) if extra[j] > 0 else 0); n = m0 + extra[j]; p, _ = adaptive(cnt / n, i, n, *abB); err[key][i] = (p - th[i]) @ (p - th[i])
                if key.startswith("two"):
                    worst = te[np.argsort(-e2[te])[:max(1, len(te) // 10)]]; err["two-stage, share of budget on the worst tenth"][te] = extra[np.isin(te, worst)].sum() / max(extra.sum(), 1)
        for k in out:
            out[k].append(err[k].mean())
    res = {k: float(V.mean() / np.mean(v)) for k, v in out.items() if "share" not in k}; res["share of the extra budget that reaches the worst tenth of questions"] = float(np.mean(out["two-stage, share of budget on the worst tenth"])); return res


if __name__ == "__main__":
    ALL = {}
    for nm, tag in MODELS:
        C = read(f"{SOL}/egplain_{tag}")
        if len(C) < N or nm not in ("Claude-Fable-5.2", "GPT-6-astra", "Claude-Opus-5.5", "GPT-5.6-luna", "Qwen3.8-27B"):
            continue
        ALL[nm] = {}
        for B, m0 in ((30, 10), (50, 15), (100, 25)):
            o = run(C, B, m0); ALL[nm][f"B={B}, probe {m0}"] = o
            print(f"{nm:18s} B={B:3d} probe {m0:2d} | alone {o['respondents alone']:.0f} | uniform m+r {o['uniform, m + r rule']:.0f} | uniform adaptive {o['uniform, adaptive pooling']:.0f} | two-stage {o['two-stage, adaptive']:.0f} "
                  f"(x{o['two-stage, adaptive'] / o['uniform, adaptive pooling']:.2f} over uniform adaptive) | oracle allocation {o['oracle allocation, adaptive']:.0f} | extra budget reaching the worst tenth {100 * o['share of the extra budget that reaches the worst tenth of questions']:.0f}%", flush=True)
    json.dump(ALL, open((_REPO + "/results/v_alloc.json"), "w"), indent=1)
