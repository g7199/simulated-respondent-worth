"""The earlier wave in the prompt: for the 44 questions that the weekly poll asked in more than one report, the simulator is shown the
result of the same question in the earlier report and asked for the later one.  (Is a simulator that reads the earlier answer worth more than the earlier answer?)
  python w_wave.py collect <model>      -> <repo>/outputs/sol/econshown_<model>/
  python w_wave.py score                -> results/w_wave.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json
import numpy as np
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); sys.path.insert(0, ROOT + "/src")
T = json.load(open((_REPO + "/data/polls_2026/econ_targets.json"))); MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
day = lambda s: f"{int(s[8:])} {MONTH[int(s[5:7]) - 1]} {s[:4]}"
if sys.argv[1] == "collect":
    model = sys.argv[2]; os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + model
    from sol_common import run, CLOSED
    head = ("You are estimating the results of real opinion surveys. Each item below is a question from one weekly survey. The same pollster asked the same question of the same population in an earlier week, "
            "and that earlier result is given with the item. Estimate the percentage of respondents who chose each option in the later survey (percentages over the listed options, summing to 100). " + CLOSED +
            " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options.")
    txt = lambda i: (f"### Item {i}\nRespondents: {T[i]['population']}\nQuestion: {T[i]['question']}\nOptions, with the result of the survey that began on {day(T[i]['earlier'])}:\n" +
                     "\n".join(f"{j + 1}. {o}: {p}%" for j, (o, p) in enumerate(zip(T[i]["options"], T[i]["earlier_percent"]))) + f"\nEstimate the result of the survey that began on {day(T[i]['later'])}.")
    jobs = {f"w{a:03d}": (head + "\n\n" + "\n\n".join(txt(i) for i in range(a, min(a + 15, len(T)))), {i: len(T[i]["options"]) for i in range(a, min(a + 15, len(T)))}) for a in range(0, len(T), 15)}
    print(next(iter(jobs.values()))[0][:1500], flush=True); run("econshown", jobs, 3, "medium")
else:
    from w_util import read, MODELS
    nz_ = lambda p: np.asarray(p, float) / np.sum(p); Q = len(T); th = [nz_(t["percent"]) for t in T]; te = [nz_(t["earlier_percent"]) for t in T]; V = np.array([float((x * (1 - x)).sum()) for x in th]); noise = V / np.array([t["n"] for t in T]); rng = np.random.default_rng(0); dr = rng.integers(0, Q, (2000, Q))
    rate = lambda P, idx=slice(None): float(V[idx].mean() / max(np.array([float((P[i] - th[i]) @ (P[i] - th[i])) for i in range(Q)])[idx].mean() - noise[idx].mean(), 1e-9))
    E = {i: te[i] for i in range(Q)}; res = {"questions": Q, "earlier wave alone": [rate(E), *np.percentile([rate(E, r) for r in dr], [2.5, 97.5])], "simulators": {}}
    print(f"{Q} questions; the earlier wave alone is worth {res['earlier wave alone'][0]:.0f} [{res['earlier wave alone'][1]:.0f}, {res['earlier wave alone'][2]:.0f}]")
    for nm, tag in MODELS:
        S = read(f"{SOL}/econshown_{tag}"); C = read(f"{SOL}/econ_{tag}")
        if len(S) < Q:
            continue
        ratio = [rate(S, r) / rate(E, r) for r in dr]; moved = float(np.mean([np.abs(S[i] - te[i]).sum() / 2 for i in range(Q)]))
        res["simulators"][nm] = {"cold": rate(C) if len(C) >= Q else None, "shown the earlier wave": [rate(S), *np.percentile([rate(S, r) for r in dr], [2.5, 97.5])], "over the earlier wave alone": [rate(S) / rate(E), *np.percentile(ratio, [2.5, 97.5])], "mean total variation between its answer and the earlier wave": moved}
        v = res["simulators"][nm]; print(f"   {nm:18s} cold {v['cold'] or float('nan'):.0f} | shown the earlier wave {v['shown the earlier wave'][0]:.0f} [{v['shown the earlier wave'][1]:.0f}, {v['shown the earlier wave'][2]:.0f}] | over the earlier wave alone x{v['over the earlier wave alone'][0]:.2f} [{v['over the earlier wave alone'][1]:.2f}, {v['over the earlier wave alone'][2]:.2f}] | moves the earlier answer by {100 * moved:.1f} points")
    json.dump(res, open(ROOT + "/results/w_wave.json", "w"), indent=1)
