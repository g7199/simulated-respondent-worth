"""The older surveys (961 items, answers the models may have seen): does showing real answers to other questions asked of the same
population raise a simulator's worth?  Targets: the 961 common items; shown: up to 30 other questions of the same population from the
remaining items of the seven programmes (never a target).  The control repeats the request in the same layout without them.
  python w_old.py collect <model> plain|ctx [workers]     -> <repo>/outputs/sol/og<mode>_<model>/
  python w_old.py score                                    -> results/w_old.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, pickle, collections
import numpy as np
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); cmd = sys.argv[1]; argv = sys.argv
sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/outputs/screen"); sys.argv = [sys.argv[0], "none"]
from r2pop_common import load
sys.argv = argv
I, E, qi, L0, L1, LQ = load(); c = pickle.load(open(ROOT + "/results/y_cov_cache.pkl", "rb")); ids = [int(i) for i in c["ids"]]; tset = set(ids)
bypop = collections.defaultdict(list)
for i in range(len(I)):
    bypop[I.iloc[i]["pop"]].append(i)
rng = np.random.default_rng(5); KNOWN = {}
for p, ii in bypop.items():
    rest = [i for i in ii if i not in tset]
    KNOWN[p] = [int(x) for x in rng.permutation(rest)[:30]]
targets = [i for i in ids if len(KNOWN[I.iloc[i]["pop"]]) >= 3]
# calls: targets grouped by population, at most 20 targets per call
tp = collections.defaultdict(list)
for i in targets:
    tp[I.iloc[i]["pop"]].append(i)
calls = []; cur = []
for p in sorted(tp, key=lambda p: -len(tp[p])):
    for a in range(0, len(tp[p]), 20):
        ch = tp[p][a:a + 20]
        if len(ch) >= 12:
            calls.append([(p, ch)])
        else:
            if sum(len(x[1]) for x in cur) + len(ch) > 20:
                calls.append(cur); cur = []
            cur.append((p, ch))
if cur:
    calls.append(cur)

if cmd == "collect":
    model, mode = sys.argv[2], sys.argv[3]; workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + (model.split(":")[2] + "-stated" if model.startswith("local:") else model)
    from sol_common import run, CLOSED
    base = ("You are estimating the results of real opinion surveys. Below are one or more groups of respondents; for each, questions are given with their answer options. Estimate the percentage of "
            "that group who chose each option in the actual survey (percentages over the listed options, summing to 100). ")
    tail = CLOSED + " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options."
    item = lambda i: f"### Item {i}\nQuestion: {I.iloc[i].q}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(I.iloc[i].opts))
    def known(i):
        r = I.iloc[i]; h = np.asarray(r.h, float); h = 100 * h / h.sum(); return f"- {r.q}\n  " + "; ".join(f"{o}: {v:.0f}%" for o, v in zip(r.opts, h))
    jobs = {}
    for k, call in enumerate(calls):
        parts = []
        for p, ch in call:
            s = f"## Respondents: {p}\n"
            if mode == "ctx":
                s += "Actual results of other questions asked of these respondents:\n" + "\n".join(known(j) for j in KNOWN[p]) + "\n\nItems to estimate:\n\n"
            parts.append(s + "\n\n".join(item(i) for i in ch))
        prompt = base + ("For each group, the actual results of other questions asked of the same respondents are given first; use them as you see fit. " if mode == "ctx" else "") + tail + "\n\n" + "\n\n".join(parts)
        jobs[f"o{k:03d}"] = (prompt, {i: int(I.iloc[i].K) for p, ch in call for i in ch})
    print(next(iter(jobs.values()))[0][:2500], flush=True); print(len(jobs), "calls,", len(targets), "targets", flush=True)
    run("og" + mode, jobs, workers, "medium")

elif cmd == "score":
    from w_util import read, MODELS
    pos = {i: k for k, i in enumerate(ids)}; h = c["h"]; V = c["V"]; sid = c["sid"]; S = c["S"]; qid = c["qid"]; rng = np.random.default_rng(0); res = {"targets": len(targets), "calls": len(calls), "models": {}}
    tk = np.array([pos[i] for i in targets]); sq = lambda a, b: float((a - b) @ (a - b)); uq = np.unique(qid[tk]); OUT = {}
    print(f"{len(targets)} of the 961 items have at least three other questions of the same population to show; {len(calls)} calls")
    for nm, tag in MODELS:
        A = read(f"{SOL}/ogplain_{tag}"); B = read(f"{SOL}/ogctx_{tag}")
        if len(A) < len(targets) or len(B) < len(targets):
            if A or B:
                print(nm, "incomplete", len(A), len(B))
            continue
        ea = np.array([sq(A[i], h[pos[i]]) for i in targets]); eb = np.array([sq(B[i], h[pos[i]]) for i in targets]); e0 = np.array([sq(c["C"][nm][pos[i]], h[pos[i]]) for i in targets]) if nm in c["C"] else None; Vt = V[tk]; OUT[nm] = (A, B)
        bs = []
        for _ in range(1000):
            r = np.concatenate([np.where(qid[tk] == q)[0] for q in rng.choice(uq, len(uq))]); bs.append((Vt[r].mean() / ea[r].mean(), Vt[r].mean() / eb[r].mean()))
        bs = np.array(bs); o = {"plain": [float(Vt.mean() / ea.mean()), *np.percentile(bs[:, 0], [2.5, 97.5])], "shown": [float(Vt.mean() / eb.mean()), *np.percentile(bs[:, 1], [2.5, 97.5])], "ratio": [float(ea.mean() / eb.mean()), *np.percentile(bs[:, 1] / bs[:, 0], [2.5, 97.5])],
                                "in the original batches": None if e0 is None else float(Vt.mean() / e0.mean()), "by programme": {S[s]: [float(Vt[sid[tk] == s].mean() / ea[sid[tk] == s].mean()), float(Vt[sid[tk] == s].mean() / eb[sid[tk] == s].mean()), int((sid[tk] == s).sum())] for s in range(len(S)) if (sid[tk] == s).sum() > 0}}
        res["models"][nm] = o
        print(f"{nm:18s} plain {o['plain'][0]:.1f} [{o['plain'][1]:.1f}, {o['plain'][2]:.1f}] (original batches {o['in the original batches'] or float('nan'):.1f}) | shown other answers of the population {o['shown'][0]:.1f} [{o['shown'][1]:.1f}, {o['shown'][2]:.1f}] | x{o['ratio'][0]:.2f} [{o['ratio'][1]:.2f}, {o['ratio'][2]:.2f}]")
        print("      by programme (plain / shown, items): " + " | ".join(f"{k} {v[0]:.0f}/{v[1]:.0f} ({v[2]})" for k, v in o["by programme"].items()))
    json.dump(res, open(ROOT + "/results/w_old.json", "w"), indent=1)
