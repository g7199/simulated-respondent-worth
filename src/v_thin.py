"""Direction B, design 2: every question is asked of only m respondents, and the simulator reads the whole thin survey.
Is the simulator's answer worth more than the optimal pooling rule of Proposition 1 (m + r) applied to the same m respondents?
  python v_thin.py collect <model> <m> [own|others]   -> <repo>/outputs/sol/egthin<m>[o]_<model>/
       own (default): the pilot results of all questions of the report are shown, the target's own included
       others: for the targets of a call, their own pilot results are withheld (only the other questions' are shown)
  python v_thin.py score                               -> results/v_thin.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json
import numpy as np
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); sys.path.insert(0, ROOT + "/src")
from w_groups import load, norm, MONTH
T = load(); N = len(T); waves = sorted({t["wave"] for t in T}); wave = np.array([t["wave"] for t in T])
th = [np.asarray(t["percent"], float) / sum(t["percent"]) for t in T]; V = np.array([(x * (1 - x)).sum() for x in th]); noise = V / np.array([t["n"] for t in T])
draw = lambda i, m: np.random.default_rng(5000 + 97 * m + i).multinomial(m, th[i])            # the answers of m respondents to question i

cmd = sys.argv[1]
if cmd == "collect":
    model, m = sys.argv[2], int(sys.argv[3]); mode = sys.argv[4] if len(sys.argv) > 4 else "own"
    os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + model
    from sol_common import run, CLOSED
    when = lambda t: f"{MONTH[int(t['wave'][5:7]) - 1]} {t['wave'][:4]}"
    base = (f"You are estimating the results of a real opinion survey. So far only a small pilot has been run: each question was answered by {m} randomly drawn respondents. "
            "The pilot counts are given below. They are noisy: with so few respondents a count can be far from the share in the full survey. "
            "For each item, estimate the percentage of all respondents of the full survey who chose each option (percentages over the listed options, summing to 100), using the pilot counts and your own knowledge as you see fit. ")
    tail = CLOSED + " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options."
    pil = lambda i: f"- {T[i]['question']}\n  " + "; ".join(f"{o}: {c} of {m}" for o, c in zip(T[i]["options"], draw(i, m)))
    item = lambda i: f"### Item {i}\nQuestion: {T[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(T[i]["options"]))
    jobs = {}
    for w in waves:
        I = [int(i) for i in np.where(wave == w)[0]]; pop = f"The survey: {T[I[0]]['population']}, surveyed in {when(T[I[0]])}.\n"
        for a in range(0, len(I), 20):
            ch = I[a:a + 20]; sh = [i for i in I if mode == "own" or i not in ch]
            jobs[f"t{w[5:7]}{w[8:]}_{a // 20:02d}"] = (base + tail + "\n\n" + pop + f"\nPilot results ({m} respondents per question):\n" + "\n".join(pil(i) for i in sh) + "\n\nItems to estimate:\n\n" + "\n\n".join(item(i) for i in ch), {i: len(T[i]["options"]) for i in ch})
    print(next(iter(jobs.values()))[0][:1800], flush=True); print(len(jobs), "calls", flush=True)
    run(f"egthin{m}" + ("o" if mode == "others" else ""), jobs, 3, "medium")
else:
    from w_util import read, MODELS
    rng = np.random.default_rng(0); bk = {}; BLOCK = np.array([bk.setdefault((t["wave"], norm(t["text"])), len(bk)) for t in T]); ALL = np.arange(N)
    def worth(E2, ii=ALL):
        return float(V[ii].mean() / max(E2[ii].mean() - noise[ii].mean(), 1e-9))
    def ci_ratio(a, b, B=1000):
        out = []; bl = np.unique(BLOCK)
        for _ in range(B):
            ii = np.concatenate([np.where(BLOCK == x)[0] for x in rng.choice(bl, len(bl))]); out.append(worth(a, ii) / worth(b, ii))
        return [worth(a) / worth(b), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
    sq = lambda P: np.array([(P[i] - th[i]) @ (P[i] - th[i]) for i in range(N)]); res = {}
    for nm, tag in MODELS:
        C = read(f"{SOL}/egplain_{tag}")
        if len(C) < N:
            continue
        e_c = sq(C); o = {"cold": worth(e_c)}
        for m in (20, 50, 100):
            P = read(f"{SOL}/egthin{m}_{tag}"); Po = read(f"{SOL}/egthin{m}o_{tag}")
            if len(P) < N and len(Po) < N:
                continue
            X = {i: draw(i, m) / m for i in range(N)}; e_x = sq(X)
            # pooling rule of Proposition 1 with the worth of the cold simulator measured on the other reports
            pool = {}; adapt = {}
            for w in waves:
                tr = np.where(wave != w)[0]; te = np.where(wave == w)[0]; r = V[tr].mean() / (e_c[tr].mean() - noise[tr].mean())
                for i in te:
                    pool[int(i)] = (m * X[i] + r * C[i]) / (m + r)
                # adaptive pooling: the weight of the simulator falls with its distance from the pilot (beyond sampling noise); two constants fitted on the other reports
                def est(i, a, b):
                    d = max(float((X[i] - C[i]) @ (X[i] - C[i])) - float((X[i] * (1 - X[i])).sum()) / m, 0.0); ri = V[tr].mean() / (a + b * d); return (m * X[i] + ri * C[i]) / (m + ri)
                grid = [(a, b) for a in np.array([.25, .5, 1, 2]) * (V[tr].mean() / r) for b in (0, .25, .5, 1, 2, 4)]; best = min(grid, key=lambda ab: np.mean([(est(i, *ab) - th[i]) @ (est(i, *ab) - th[i]) for i in tr]))
                for i in te:
                    adapt[int(i)] = est(i, *best)
            e_p = sq(pool); e_a = sq(adapt); om = {"respondents alone": worth(e_x), "pooled with the cold simulator (m + r rule)": worth(e_p), "adaptive pooling": worth(e_a)}
            if len(P) >= N:
                e_l = sq(P); om["simulator reads the thin survey"] = worth(e_l); om["reads / m + r rule"] = ci_ratio(e_l, e_p); om["reads / adaptive pooling"] = ci_ratio(e_l, e_a)
            if len(Po) >= N:
                e_o = sq(Po); om["simulator reads the other questions only"] = worth(e_o)
                two = {}
                for w in waves:
                    tr = np.where(wave != w)[0]; r2 = V[tr].mean() / (e_o[tr].mean() - noise[tr].mean())
                    for i in np.where(wave == w)[0]:
                        two[int(i)] = (m * X[i] + r2 * Po[i]) / (m + r2)
                om["reads the others, then pooled with the own pilot"] = worth(sq(two))
            o[str(m)] = om
            print(f"{nm:18s} m={m:3d} | cold {o['cold']:.0f} | respondents alone {om['respondents alone']:.0f} | m + r rule {om['pooled with the cold simulator (m + r rule)']:.0f} | adaptive pooling {om['adaptive pooling']:.0f}"
                  + (f" | reads the thin survey {om['simulator reads the thin survey']:.0f} (x{om['reads / m + r rule'][0]:.2f} [{om['reads / m + r rule'][1]:.2f}, {om['reads / m + r rule'][2]:.2f}] over m + r; x{om['reads / adaptive pooling'][0]:.2f} [{om['reads / adaptive pooling'][1]:.2f}, {om['reads / adaptive pooling'][2]:.2f}] over adaptive)" if "simulator reads the thin survey" in om else "")
                  + (f" | reads others only {om['simulator reads the other questions only']:.0f}, then pooled {om['reads the others, then pooled with the own pilot']:.0f}" if "simulator reads the other questions only" in om else ""))
        if len(o) > 1:
            res[nm] = o
    json.dump(res, open(ROOT + "/results/v_thin.json", "w"), indent=1)
