"""Other pollsters: the 190 new polls that are not from The Economist/YouGov come from twenty releases of eight pollsters. Within each
release the questions are split at random into two halves; a simulator estimates one half with and without the real answers of the other.
  python w_rel.py collect <model> plain|ctx [workers]   -> <repo>/outputs/sol/rl<mode>_<model>/
  python w_rel.py score                                  -> results/w_rel.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, collections
import numpy as np
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); sys.path.insert(0, ROOT + "/src")
D = json.load(open((_REPO + "/data/polls_2026/items.json"))); MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
rel = collections.defaultdict(list)
for i, d in enumerate(D):
    if d["source"] != "The Economist/YouGov":
        rel[(d["source"], d["fieldwork_start"], d["population"])].append(i)
rel = {k: v for k, v in rel.items() if len(v) >= 4}; rng = np.random.default_rng(3); half = {}
for k in sorted(rel):
    for j, i in enumerate(rng.permutation(rel[k])):
        half[int(i)] = j % 2
targets = sorted(half); when = lambda d: f"{MONTH[int(d['fieldwork_start'][5:7]) - 1]} {d['fieldwork_start'][:4]}"
cmd = sys.argv[1]
if cmd == "collect":
    model, mode = sys.argv[2], sys.argv[3]; workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + (model.split(":")[2] + "-stated" if model.startswith("local:") else model)
    from sol_common import run, CLOSED
    base = ("You are estimating the results of real opinion surveys. For each item below, a question from one survey is given with its answer options. Estimate the percentage of the "
            "respondents who chose each option in the actual survey (percentages over the listed options, summing to 100). ")
    tail = CLOSED + " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options."
    item = lambda i: f"### Item {i}\nQuestion: {D[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(D[i]["options"]))
    known = lambda i: f"- {D[i]['question']}\n  " + "; ".join(f"{o}: {p}%" for o, p in zip(D[i]["options"], D[i]["percent"]))
    jobs = {}
    for n, k in enumerate(sorted(rel)):
        for h in (0, 1):
            tg = [i for i in rel[k] if half[i] == h]; kn = [i for i in rel[k] if half[i] != h]; pop = f"The survey: {k[2]}, surveyed in {when(D[tg[0]])}.\n"
            if mode == "plain":
                prompt = base + tail + "\n\n" + pop + "\n" + "\n\n".join(item(i) for i in tg)
            else:
                prompt = (base + "The actual results of other questions asked in the same survey are given first; use them as you see fit. " + tail + "\n\n" + pop +
                          "\nActual results of other questions in the same survey:\n" + "\n".join(known(i) for i in kn) + "\n\nItems to estimate:\n\n" + "\n\n".join(item(i) for i in tg))
            jobs[f"r{n:02d}_{h}"] = (prompt, {i: len(D[i]["options"]) for i in tg})
    print(list(jobs.values())[3][0][:2500], flush=True); print(len(jobs), "calls,", len(targets), "targets,", len(rel), "releases", flush=True)
    run("rl" + mode, jobs, workers, "medium")
elif cmd == "score":
    from w_util import read, MODELS
    nz_ = lambda p: np.asarray(p, float) / np.sum(p); th = {i: nz_(D[i]["percent"]) for i in targets}; V = np.array([float((th[i] * (1 - th[i])).sum()) for i in targets]); noise = V / np.array([D[i].get("sample_size") or 1000 for i in targets], float)
    rid = {i: n for n, k in enumerate(sorted(rel)) for i in rel[k]}; rr = np.array([rid[i] for i in targets]); rng = np.random.default_rng(0); res = {"targets": len(targets), "releases": len(rel), "pollsters": len({k[0] for k in rel}), "countries": len({D[i]["country"] for i in targets}), "models": {}}
    print(f"{len(targets)} polls in {len(rel)} releases of {res['pollsters']} pollsters, {res['countries']} countries; shown per target: median {np.median([sum(half[j] != half[i] for j in rel[k]) for k in rel for i in rel[k]]):.0f}")
    OUT = {}
    for nm, tag in MODELS:
        A = read(f"{SOL}/rlplain_{tag}"); B = read(f"{SOL}/rlctx_{tag}")
        if len(A) < len(targets) or len(B) < len(targets):
            if A or B:
                print(nm, "incomplete", len(A), len(B))
            continue
        OUT[nm] = (A, B); ea = np.array([float((A[i] - th[i]) @ (A[i] - th[i])) for i in targets]); eb = np.array([float((B[i] - th[i]) @ (B[i] - th[i])) for i in targets]); bs = []
        for _ in range(2000):
            r = np.concatenate([np.where(rr == q)[0] for q in rng.integers(0, len(rel), len(rel))]); bs.append((V[r].mean() / (ea[r].mean() - noise[r].mean()), V[r].mean() / (eb[r].mean() - noise[r].mean())))
        bs = np.array(bs); f = lambda e: float(V.mean() / (e.mean() - noise.mean())); us = np.array([D[i]["country"] == "United States" for i in targets])
        o = {"plain": [f(ea), *np.percentile(bs[:, 0], [2.5, 97.5])], "shown": [f(eb), *np.percentile(bs[:, 1], [2.5, 97.5])], "ratio": [f(eb) / f(ea), *np.percentile(bs[:, 1] / bs[:, 0], [2.5, 97.5])],
             "United States": [float(V[us].mean() / (ea[us].mean() - noise[us].mean())), float(V[us].mean() / (eb[us].mean() - noise[us].mean())), int(us.sum())], "elsewhere": [float(V[~us].mean() / (ea[~us].mean() - noise[~us].mean())), float(V[~us].mean() / (eb[~us].mean() - noise[~us].mean())), int((~us).sum())]}
        res["models"][nm] = o
        print(f"{nm:18s} plain {o['plain'][0]:.1f} [{o['plain'][1]:.1f}, {o['plain'][2]:.1f}] | shown the rest of the release {o['shown'][0]:.1f} [{o['shown'][1]:.1f}, {o['shown'][2]:.1f}] | x{o['ratio'][0]:.2f} [{o['ratio'][1]:.2f}, {o['ratio'][2]:.2f}] | US {o['United States'][0]:.0f} -> {o['United States'][1]:.0f} ({o['United States'][2]}), elsewhere {o['elsewhere'][0]:.0f} -> {o['elsewhere'][1]:.0f} ({o['elsewhere'][2]})")
    json.dump(res, open(ROOT + "/results/w_rel.json", "w"), indent=1)
