"""Paper 2 (revision): polls fielded after the models were trained (fieldwork from 2026-10-01; <repo>/data/polls_2026).
  python y_polls.py collect <codex model>     stated shares for the 250 toplines (20 per call) and, for the questions with a
                                              split by party identification, for all respondents and three party groups (5 per call)
  python y_polls.py score                     every constant (calibration rule, smoothing, exchange rate) comes from the seven
                                              survey programmes; the polls are only evaluated  ->  results/y_polls.json
"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, re, json
import numpy as np, pandas as pd
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol")
D = json.load(open((_REPO + "/data/polls_2026/items.json"))); NP = len(D)
NA = r"not sure|don.?t know|no opinion|refus|skip|unsure|prefer not|undecided|can.?t say|dont know"
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
PARTY = [i for i, d in enumerate(D) if d.get("by_party") and {"Democrats", "Republicans"} <= set(d["by_party"])]
ind = lambda d: next(k for k in d["by_party"] if k.startswith("Independents"))
cmd = sys.argv[1]

if cmd == "collect":
    model = sys.argv[2]; os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + (model.split(":")[2] + "-stated" if model.startswith("local:") else model)
    sys.path.insert(0, ROOT + "/src")
    from sol_common import run, CLOSED
    when = lambda d: f"{MONTH[int(d['fieldwork_start'][5:7]) - 1]} {d['fieldwork_start'][:4]}"
    head = ("You are estimating the results of real opinion surveys. For each item below, a group of respondents is described and a question is given with its answer "
            "options. Estimate the percentage of that group who chose each option in the actual survey (percentages over the listed options, summing to 100). " + CLOSED +
            " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options.")
    txt = lambda i: f"### Item {i}\nRespondents: {D[i]['population']}, surveyed in {when(D[i])}\nQuestion: {D[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(D[i]["options"]))
    jobs = {f"t{a:03d}": (head + "\n\n" + "\n\n".join(txt(i) for i in range(a, min(a + 20, NP))), {i: len(D[i]["options"]) for i in range(a, min(a + 20, NP))}) for a in range(0, NP, 20)}
    run("polls", jobs, 5, "medium")
    head2 = ("You are estimating the results of real opinion surveys. For each question below, estimate the percentage who chose each option in the actual survey, first among all "
             "respondents and then among the respondents who identify as Democrats, as Republicans and as Independents. " + CLOSED + " Return JSON with four entries per question, "
             "with ids 10*q, 10*q+1, 10*q+2, 10*q+3 for all respondents, Democrats, Republicans and Independents of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
    txt2 = lambda i: f"### Question {i}\nRespondents: {D[i]['population']}, surveyed in {when(D[i])}\nQuestion: {D[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(D[i]["options"]))
    jobs = {}
    for a in range(0, len(PARTY), 5):
        ch = PARTY[a:a + 5]; jobs[f"p{a // 5:03d}"] = (head2 + "\n\n" + "\n\n".join(txt2(i) for i in ch), {10 * i + g: len(D[i]["options"]) for i in ch for g in range(4)})
    run("pollsparty", jobs, 5, "medium")

elif cmd == "given":
    # the poll's real topline is shown to the model, which is asked for the three party groups (a baseline for adding the model's contrast to the topline)
    model = sys.argv[2]; os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + (model.split(":")[2] + "-stated" if model.startswith("local:") else model)
    sys.path.insert(0, ROOT + "/src")
    from sol_common import run, CLOSED
    when = lambda d: f"{MONTH[int(d['fieldwork_start'][5:7]) - 1]} {d['fieldwork_start'][:4]}"
    head3 = ("You are estimating the results of real opinion surveys. For each question below, the percentage of all respondents who chose each option in the actual survey is given. "
             "Estimate the percentage who chose each option among the respondents who identify as Democrats, as Republicans and as Independents. " + CLOSED + " Return JSON with three entries per question, "
             "with ids 10*q+1, 10*q+2, 10*q+3 for Democrats, Republicans and Independents of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
    txt3 = lambda i: (f"### Question {i}\nRespondents: {D[i]['population']}, surveyed in {when(D[i])}\nQuestion: {D[i]['question']}\nOptions, with the percentage of all respondents who chose each:\n" +
                      "\n".join(f"{j + 1}. {o}: {p}%" for j, (o, p) in enumerate(zip(D[i]["options"], D[i]["percent"]))))
    jobs = {}
    for a in range(0, len(PARTY), 5):
        ch = PARTY[a:a + 5]; jobs[f"g{a // 5:03d}"] = (head3 + "\n\n" + "\n\n".join(txt3(i) for i in ch), {10 * i + g: len(D[i]["options"]) for i in ch for g in (1, 2, 3)})
    print(next(iter(jobs.values()))[0][:1500], flush=True)
    run("pollsgiven", jobs, 5, "medium")

elif cmd == "sep":
    # the three party groups asked without the population in the same call (is the contrast an artefact of asking both together?)
    model = sys.argv[2]; os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + (model.split(":")[2] + "-stated" if model.startswith("local:") else model)
    sys.path.insert(0, ROOT + "/src")
    from sol_common import run, CLOSED
    when = lambda d: f"{MONTH[int(d['fieldwork_start'][5:7]) - 1]} {d['fieldwork_start'][:4]}"
    head4 = ("You are estimating the results of real opinion surveys. For each question below, estimate the percentage who chose each option in the actual survey among the respondents "
             "who identify as Democrats, as Republicans and as Independents. " + CLOSED + " Return JSON with three entries per question, "
             "with ids 10*q+1, 10*q+2, 10*q+3 for Democrats, Republicans and Independents of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
    txt4 = lambda i: f"### Question {i}\nRespondents: {D[i]['population']}, surveyed in {when(D[i])}\nQuestion: {D[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(D[i]["options"]))
    jobs = {}
    for a in range(0, len(PARTY), 5):
        ch = PARTY[a:a + 5]; jobs[f"s{a // 5:03d}"] = (head4 + "\n\n" + "\n\n".join(txt4(i) for i in ch), {10 * i + g: len(D[i]["options"]) for i in ch for g in (1, 2, 3)})
    run("pollssep", jobs, 5, "medium")

elif cmd == "score":
    sys.path.insert(0, ROOT + "/outputs/screen"); sys.argv = [sys.argv[0], "none"]
    from r2pop_common import load, soft, fl
    I, E, qi, L0, L1, LQ = load(); NO = len(I); ho = [np.asarray(x, float) / np.sum(x) for x in I.h.values]; Ko = I.K.values
    nso = [np.array([bool(re.search(r"refus|don.?t know|dont know|can.?t choose|cant choose|no answer|not sure|undecided|haven.?t heard", o.lower())) for o in os_]) for os_ in I.opts]
    hp = [np.asarray(d["percent"], float) / np.sum(d["percent"]) for d in D]; Kp = np.array([len(x) for x in hp]); nsp = [np.array([bool(re.search(NA, o.lower())) for o in d["options"]]) for d in D]
    EPS = 1e-3; rng = np.random.default_rng(0); R = 8; NS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]; NE = [1, 5, 10, 20, 50, 100]; AG = [.1, .25, .5, 1, 2, 3, 4, 6, 8]
    TG = [.6, .8, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20, 30]; DG = [0, .5, 1, 1.5, 2, 2.5, 3, 4, 5]
    kl = lambda hv, Q: (hv[hv > 0] * np.log(hv[hv > 0] / ((1 - EPS) * Q + EPS / Q.shape[-1])[..., hv > 0])).sum(-1)

    def crossing(curve, target, grid=NS):
        if curve[0] <= target:
            return float(grid[0]) * 0.999
        for a in range(1, len(grid)):
            if curve[a] <= target:
                return float(np.exp(np.log(grid[a - 1]) + (curve[a - 1] - target) / (curve[a - 1] - curve[a]) * (np.log(grid[a]) - np.log(grid[a - 1]))))
        return float(grid[-1]) * 1.001

    def read(d, pat):
        P = {}
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if re.match(pat, f):
                    for i, p in json.load(open(f"{d}/{f}"))["items"].items():
                        p = np.asarray(p, float); P[int(i)] = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
        return P

    # base estimate: the smoothing constant per n is fitted on the old programmes and applied to the polls
    CO = {n: [rng.multinomial(n, ho[i], size=4).astype(float) for i in range(NO)] for n in NS}; a_n = {}; base_old = np.zeros((len(NS), NO))
    for x, n in enumerate(NS):
        G = np.array([[kl(ho[i], (CO[n][i] + a / Ko[i]) / (n + a)).mean() for a in AG] for i in range(NO)]); j = int(np.argmin(G.mean(0))); a_n[n] = AG[j]; base_old[x] = G[:, j]
    CP = {n: [rng.multinomial(n, hp[i], size=R).astype(float) for i in range(NP)] for n in NS}
    base = np.array([[kl(hp[i], (CP[n][i] + a_n[n] / Kp[i]) / (n + a_n[n])).mean() for i in range(NP)] for n in NS])
    us = np.array([d["country"] == "United States" for d in D]); trk = np.array([bool(d.get("tracker")) for d in D])

    def boot(f, B=1000):
        xs = [f(rng.integers(0, NP, NP)) for _ in range(B)]; return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

    res = {"polls": NP, "fieldwork": [min(d["fieldwork_start"] for d in D), max(d["fieldwork_end"] for d in D)], "base": [float(v) for v in base.mean(1)], "smoothing": a_n, "models": {},
           "uniform": float(np.mean([kl(hp[i], np.ones((1, Kp[i])) / Kp[i])[0] for i in range(NP)]))}
    print(f"{NP} polls, fieldwork {res['fieldwork']}; base estimate: " + " ".join(f"{n}: {base[x].mean():.4f}" for x, n in enumerate(NS)) + f" | uniform {res['uniform']:.4f}")
    MODELS = {"GPT-5.6-luna": "gpt-5.6-luna", "GPT-6-luna": "gpt-6-luna", "GPT-5.6-sol": "gpt-5.6-sol", "GPT-6-sol": "gpt-6-sol", "GPT-6.1-terra": "gpt-6.1-terra", "GPT-6.1-sol": "gpt-6.1-sol", "GPT-6-astra": "gpt-6-astra"}
    OPEN = {"Qwen3.5-9B": (f"{ROOT}/screen/r3_out/polls_qwen.json", {i: LQ[i] for i in range(NO)})}
    for name, tag in list(MODELS.items()) + [(k, None) for k in OPEN]:
        if tag:
            new = read(f"{SOL}/polls_{tag}", r"t\d+\.json"); old = read(f"{SOL}/survey" + ("" if tag == "gpt-6.1-sol" else "_" + tag), r"b\d+\.json")
            Ln = {i: np.log(fl(p, 1e-4)) for i, p in new.items()}; Lo = {i: np.log(fl(p, 1e-4)) for i, p in old.items()}
        else:
            if not os.path.exists(OPEN[name][0]):
                continue
            raw = json.load(open(OPEN[name][0])); Ln = {i: np.log(fl((soft(raw[f"{i}_0"]) + soft(raw[f"{i}_1"])) / 2, 1e-4)) for i in range(NP)}; Lo = OPEN[name][1]
        if len(Ln) < NP:
            print(name, "incomplete", len(Ln)); continue
        oi = sorted(Lo); M = np.array([[[kl(ho[i], soft(Lo[i] / T - d * nso[i])[None])[0] for d in DG] for T in TG] for i in oi]); a, b = np.unravel_index(np.argmin(M.mean(0)), (len(TG), len(DG))); T, dl = TG[a], DG[b]
        r_old = crossing(base_old[:, oi].mean(1), float(M[:, a, b].mean()))                                   # rate on the old programmes (in-sample rule; for reference)
        cen = [soft(Ln[i] / T - dl * nsp[i]) for i in range(NP)]; err = np.array([kl(hp[i], cen[i][None])[0] for i in range(NP)]); raw_err = np.array([kl(hp[i], soft(Ln[i])[None])[0] for i in range(NP)])
        rate = crossing(base.mean(1), float(err.mean())); ci = boot(lambda r: crossing(base[:, r].mean(1), float(err[r].mean())))
        o = {"rule from the old programmes": [T, dl], "error": float(err.mean()), "error as stated": float(raw_err.mean()), "rate": rate, "rate_ci": ci, "rate on the old programmes": r_old,
             "rate, United States": crossing(base[:, us].mean(1), float(err[us].mean())), "rate, other countries": crossing(base[:, ~us].mean(1), float(err[~us].mean())),
             "rate without trackers": crossing(base[:, ~trk].mean(1), float(err[~trk].mean())), "plurality": float(np.mean([np.argmax(np.where(~nsp[i], cen[i], -1)) == np.argmax(np.where(~nsp[i], hp[i], -1)) for i in range(NP)])), "by_n": {}}
        for x, n in enumerate(NS):
            if n in NE:
                pool = np.array([kl(hp[i], (CP[n][i] + 1.0 / Kp[i] + r_old * cen[i]) / (n + 1.0 + r_old)).mean() for i in range(NP)])
                o["by_n"][n] = {"base": float(base[x].mean()), "rate pooling": float(pool.mean()), "worth": crossing(base.mean(1), float(pool.mean())), "n plus rate": n + r_old}
        res["models"][name] = o
        print(f"{name:14s} error {err.mean():.4f} (as stated {raw_err.mean():.4f}) | rate on the new polls {rate:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | on the old programmes {r_old:.1f} | US {o['rate, United States']:.1f}, elsewhere {o['rate, other countries']:.1f} | "
              f"plurality {o['plurality']:.2f} | pooled with n respondents, worth (n + old rate): " + " ".join(f"{n}: {v['worth']:.0f} ({v['n plus rate']:.0f})" for n, v in o["by_n"].items()), flush=True)
    # ---- party groups
    gi = [(i, g, key) for i in PARTY for g, key in enumerate(("Democrats", "Republicans", ind(D[i])))]; hg = [np.asarray(D[i]["by_party"][key], float) for i, g, key in gi]; hg = [x / x.sum() for x in hg]; NG = len(gi)
    CG = {n: [rng.multinomial(n, hg[c], size=R).astype(float) for c in range(NG)] for n in NS}
    bg = np.array([[kl(hg[c], (CG[n][c] + a_n[n] / len(hg[c])) / (n + a_n[n])).mean() for c in range(NG)] for n in NS]); top = np.array([kl(hg[c], hp[i][None])[0] for c, (i, g, key) in enumerate(gi)])
    res["party"] = {"cells": NG, "questions": len(PARTY), "base": [float(v) for v in bg.mean(1)], "topline": {"error": float(top.mean()), "rate": crossing(bg.mean(1), float(top.mean()))}, "models": {}}
    print(f"party groups: {NG} cells from {len(PARTY)} questions | the poll's own topline as the estimate for each group: error {top.mean():.4f}, worth {res['party']['topline']['rate']:.1f} respondents of the group")
    qcl = np.array([i for i, g, key in gi]); uq = np.unique(qcl)

    def bootg(f, B=1000):
        xs = []
        for _ in range(B):
            r = np.concatenate([np.where(qcl == q)[0] for q in rng.choice(uq, len(uq))]); xs.append(f(r))
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

    for name, tag in MODELS.items():
        P = read(f"{SOL}/pollsparty_{tag}", r"p\d+\.json")
        if len(P) < 4 * len(PARTY):
            continue
        per = np.array([kl(hg[c], fl(P[10 * i + 1 + g], 1e-4)[None])[0] for c, (i, g, key) in enumerate(gi)])
        WG = [0, .25, .5, .75, 1, 1.25, 1.5]; C = np.array([[kl(hg[c], soft(np.log(fl(hp[i], 1e-4)) + w * (np.log(fl(P[10 * i + 1 + g], 1e-3)) - np.log(fl(P[10 * i], 1e-3))))[None])[0] for w in WG] for c, (i, g, key) in enumerate(gi)])
        fold = (qcl % 2); con = np.zeros(NG)
        for f in (0, 1):
            con[fold == f] = C[fold == f, int(np.argmin(C[fold != f].mean(0)))]
        o = {"persona alone": {"error": float(per.mean()), "rate": crossing(bg.mean(1), float(per.mean())), "rate_ci": bootg(lambda r: crossing(bg[:, r].mean(1), float(per[r].mean())))},
             "contrast on the topline": {"error": float(con.mean()), "rate": crossing(bg.mean(1), float(con.mean())), "rate_ci": bootg(lambda r: crossing(bg[:, r].mean(1), float(con[r].mean()))), "power": [WG[int(np.argmin(C[fold != f].mean(0)))] for f in (0, 1)],
                                         "relative to topline": float(con.mean() / top.mean() - 1)}}
        res["party"]["models"][name] = o
        print(f"   {name:14s} persona alone {per.mean():.4f} = {o['persona alone']['rate']:.0f} [{o['persona alone']['rate_ci'][0]:.0f}, {o['persona alone']['rate_ci'][1]:.0f}] | contrast on the topline {con.mean():.4f} = {o['contrast on the topline']['rate']:.0f} "
              f"[{o['contrast on the topline']['rate_ci'][0]:.0f}, {o['contrast on the topline']['rate_ci'][1]:.0f}] respondents of the group ({100 * o['contrast on the topline']['relative to topline']:+.0f}% against the topline; power {o['contrast on the topline']['power']})", flush=True)
    json.dump(res, open(ROOT + "/results/y_polls.json", "w"), indent=1)
