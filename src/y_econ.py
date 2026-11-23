"""Earlier waves after the training cutoff: the five weekly Economist/YouGov reports from which 60 of the 250 new polls were taken
(fieldwork 2-5, 10-12, 19-20, 23-26 October and 4-6 November 2026) are parsed in full; every question is read with its topline,
its split by party identification and the group sizes. Questions asked in more than one report give an earlier wave of the same
question, same pollster, one to five weeks before.
  python y_econ.py parse    -> <repo>/data/polls_2026/econ_waves.json  (and a check against the 60 collected records)
"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import re, json, sys, os, collections
DIR = (_REPO + "/data/polls_2026/yougov_pdf"); OUT = (_REPO + "/data/polls_2026/econ_waves.json")
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
HEAD = re.compile(r"^(\d+[A-Z]?)\. (\S.*)$")


def parse(path):
    L = open(path).read().split("\n"); starts = [i for i, l in enumerate(L) if HEAD.match(l)]; out = []
    for a, b in zip(starts, starts[1:] + [len(L)]):
        m = HEAD.match(L[a]); blk = L[a + 1:b]; tot = [i for i, l in enumerate(blk) if re.match(r"\s+Total\s", l) and "Total" in l.split()[:1]]
        if len(tot) < 2:
            continue
        text = " ".join(x.strip() for x in blk[:tot[0] - 1] if x.strip()); halves = []
        for h_i, t in enumerate(tot[:2]):
            rows = []; n_row = None
            for l in blk[t + 1:]:
                if re.match(r"\s*Unweighted N", l):
                    n_row = [int(x.replace(",", "")) for x in re.findall(r"\(([\d,]+)\)", l)]; break
                if not l.strip() or re.match(r"\s*Totals\b", l):
                    continue
                vals = re.findall(r"(\d+)%", l); lab = re.split(r"\s{2,}\d+%", l.strip())[0].strip()
                if vals and lab and not re.match(r"Total\s", lab):
                    rows.append((lab, [int(v) for v in vals]))
            halves.append((blk[t], rows, n_row))
        (h1, r1, n1), (h2, r2, n2) = halves
        if not r1 or [x[0] for x in r1] != [x[0] for x in r2] or not n1 or not n2 or not re.search(r"Dem\s+Ind\s+Rep\s*$", h2) or any(len(v) < 4 for _, v in r2):
            continue
        tl = [v[0] for _, v in r1]
        if not 96 <= sum(tl) <= 104:
            continue
        out.append({"number": m.group(1), "title": m.group(2).strip(), "text": text, "options": [x[0] for x in r1], "percent": tl, "by_party": {"Democrats": [v[-3] for _, v in r2], "Independents": [v[-2] for _, v in r2], "Republicans": [v[-1] for _, v in r2]},
                    "n": n1[0], "n_party": {"Democrats": n2[-3], "Independents": n2[-2], "Republicans": n2[-1]}})
    return out


if sys.argv[1] == "parse":
    W = {}
    for f in sorted(os.listdir(DIR)):
        if f.endswith(".txt"):
            W[f[5:15]] = parse(f"{DIR}/{f}"); print(f[5:15], len(W[f[5:15]]), "questions parsed")
    json.dump(W, open(OUT, "w"), indent=0)
    # check against the collected records
    D = json.load(open((_REPO + "/data/polls_2026/items.json"))); ok = bad = miss = 0
    for i, d in enumerate(D):
        if d["source"] != "The Economist/YouGov":
            continue
        c = [q for q in W[d["fieldwork_start"]] if norm(q["text"]).endswith(norm(d["question"])[-60:]) or norm(d["question"]) in norm(q["text"]) or norm(q["title"]) == norm(d.get("topic") or "")]
        c = [q for q in c if [norm(o) for o in q["options"]] == [norm(o) for o in d["options"]]]
        if not c:
            miss += 1
        elif any(q["percent"] == d["percent"] for q in c):
            ok += 1
        else:
            bad += 1; print("   differs:", i, d["question"][:60], d["percent"], [q["percent"] for q in c])
    print(f"collected records found in the parsed reports with identical toplines: {ok}; found with different numbers: {bad}; not matched: {miss}")
    key = collections.defaultdict(list)
    for w, qs in W.items():
        for q in qs:
            key[(norm(q["title"]), norm(q["text"]), tuple(norm(o) for o in q["options"]))].append(w)
    rep = {k: sorted(v) for k, v in key.items() if len(set(v)) > 1}; print("questions asked in more than one report:", len(rep), "| in all five:", sum(len(set(v)) == 5 for v in rep.values()), "| distinct questions in total:", len(key))

MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
TGT = (_REPO + "/data/polls_2026/econ_targets.json")
if sys.argv[1] in ("targets", "collect", "score"):
    W = json.load(open(OUT)); D = json.load(open((_REPO + "/data/polls_2026/items.json"))); popn = {d["fieldwork_start"]: d["population"] for d in D if d["source"] == "The Economist/YouGov"}
if sys.argv[1] == "targets":
    # every question asked in more than one report: the last time it was asked is the target, the time before is the earlier wave
    key = collections.defaultdict(list)
    for w in sorted(W):
        for q in W[w]:
            key[(norm(q["title"]), norm(q["text"]), tuple(norm(o) for o in q["options"]))].append((w, q))       # the title names the item of a battery
    T = []
    for k, v in key.items():
        if len({w for w, _ in v}) > 1:
            (we, qe), (wl, ql) = v[-2], v[-1]; T.append({"title": ql["title"], "question": ql["text"] + (" — " + ql["title"].split("—")[-1].strip() if "—" in ql["title"] else ""), "options": ql["options"], "later": wl, "earlier": we, "population": popn[wl], "percent": ql["percent"], "by_party": ql["by_party"], "n": ql["n"], "n_party": ql["n_party"],
                                                         "earlier_percent": qe["percent"], "earlier_by_party": qe["by_party"], "earlier_n": qe["n"], "earlier_n_party": qe["n_party"], "waves": sorted({w for w, _ in v})})
    json.dump(T, open(TGT, "w"), indent=0); print(len(T), "recurring questions; gaps (later, earlier):", collections.Counter((t["later"], t["earlier"]) for t in T).most_common(8)); print(T[0]["question"][:200], T[0]["options"], T[0]["percent"], T[0]["earlier_percent"])
elif sys.argv[1] == "collect":
    model = sys.argv[2]; os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + model
    sys.path.insert(0, (_REPO + "/src"))
    from sol_common import run, CLOSED
    T = json.load(open(TGT)); when = lambda t: f"{MONTH[int(t['later'][5:7]) - 1]} {t['later'][:4]}"
    head = ("You are estimating the results of real opinion surveys. For each item below, a group of respondents is described and a question is given with its answer "
            "options. Estimate the percentage of that group who chose each option in the actual survey (percentages over the listed options, summing to 100). " + CLOSED +
            " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options.")
    txt = lambda i: f"### Item {i}\nRespondents: {T[i]['population']}, surveyed in {when(T[i])}\nQuestion: {T[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(T[i]["options"]))
    jobs = {f"t{a:03d}": (head + "\n\n" + "\n\n".join(txt(i) for i in range(a, min(a + 20, len(T)))), {i: len(T[i]["options"]) for i in range(a, min(a + 20, len(T)))}) for a in range(0, len(T), 20)}
    run("econ", jobs, 5, "medium")
    head2 = ("You are estimating the results of real opinion surveys. For each question below, estimate the percentage who chose each option in the actual survey, first among all "
             "respondents and then among the respondents who identify as Democrats, as Republicans and as Independents. " + CLOSED + " Return JSON with four entries per question, "
             "with ids 10*q, 10*q+1, 10*q+2, 10*q+3 for all respondents, Democrats, Republicans and Independents of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
    txt2 = lambda i: f"### Question {i}\nRespondents: {T[i]['population']}, surveyed in {when(T[i])}\nQuestion: {T[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(T[i]["options"]))
    jobs = {}
    for a in range(0, len(T), 5):
        ch = list(range(a, min(a + 5, len(T)))); jobs[f"p{a // 5:03d}"] = (head2 + "\n\n" + "\n\n".join(txt2(i) for i in ch), {10 * i + g: len(T[i]["options"]) for i in ch for g in range(4)})
    run("econparty", jobs, 5, "medium")
elif sys.argv[1] == "score":
    import numpy as np
    ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); rng = np.random.default_rng(0); T = json.load(open(TGT)); Q = len(T)
    STATED = [("GPT-6-luna", "gpt-6-luna"), ("GPT-5.6-luna", "gpt-5.6-luna"), ("GPT-5.6-sol", "gpt-5.6-sol"), ("GPT-6-sol", "gpt-6-sol"), ("GPT-6.1-terra", "gpt-6.1-terra"), ("GPT-6.1-sol", "gpt-6.1-sol"), ("GPT-6-astra", "gpt-6-astra"),
              ("Claude-Haiku-5.5", "claude-haiku-5-5"), ("Claude-Sonnet-5.5", "claude-sonnet-5-5"), ("Claude-Opus-5.5", "claude-opus-5-5"), ("Claude-Fable-5.2", "claude-fable-5-2")]
    nz_ = lambda p: np.asarray(p, float) / np.sum(p); cl = lambda x: np.clip(x, 1e-4, None) / np.clip(x, 1e-4, None).sum(); sq = lambda a, b: float((a - b) @ (a - b))
    def read(d, pat):
        P = {}
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if re.match(pat, f):
                    for i, p in json.load(open(f"{d}/{f}"))["items"].items():
                        p = np.asarray(p, float); P[int(i)] = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
        return P
    th = [nz_(t["percent"]) for t in T]; te = [nz_(t["earlier_percent"]) for t in T]; V = np.array([float((x * (1 - x)).sum()) for x in th]); noise = V / np.array([t["n"] for t in T]); dr = rng.integers(0, Q, (2000, Q))
    rate = lambda e2, idx=slice(None): float(V[idx].mean() / max(e2[idx].mean() - noise[idx].mean(), 1e-9)); ci = lambda e2: [rate(e2), float(np.percentile([rate(e2, r) for r in dr], 2.5)), float(np.percentile([rate(e2, r) for r in dr], 97.5))]
    ee = [a - b for a, b in zip(te, th)]; e_w = np.array([e @ e for e in ee]); uni = np.array([sq(np.ones(len(x)) / len(x), x) for x in th])
    gaps = [(int(t["later"][5:7]) - int(t["earlier"][5:7])) * 31 + int(t["later"][8:]) - int(t["earlier"][8:]) for t in T]
    res = {"questions": Q, "days between the waves: min, median, max": [int(min(gaps)), float(np.median(gaps)), int(max(gaps))], "population": {"earlier wave": ci(e_w), "uniform answer": ci(uni), "simulators": {}}}
    print(f"{Q} recurring questions (waves {min(gaps)} to {max(gaps)} days apart, median {np.median(gaps):.0f}); the earlier wave is worth {res['population']['earlier wave'][0]:.0f} [{res['population']['earlier wave'][1]:.0f}, {res['population']['earlier wave'][2]:.0f}] respondents of the later one; uniform {res['population']['uniform answer'][0]:.1f}")
    SIM = {}
    for nm, tag in STATED:
        P = read(f"{SOL}/econ_{tag}", r"t\d+\.json"); PP = read(f"{SOL}/econparty_{tag}", r"p\d+\.json")
        if len(P) < Q or len(PP) < 4 * Q:
            print(nm, "incomplete", len(P), len(PP)); continue
        SIM[nm] = (P, PP); es = [P[i] - th[i] for i in range(Q)]; g11 = np.array([e @ e for e in es]); g12 = np.array([a @ b for a, b in zip(es, ee)])
        def pair(idx):
            a = g11[idx].mean() - noise[idx].mean(); b = e_w[idx].mean() - noise[idx].mean(); c = g12[idx].mean() - noise[idx].mean(); lam = min(1.0, max(0.0, (a - c) / (a + b - 2 * c))); return float(V[idx].mean() / ((1 - lam) ** 2 * a + lam ** 2 * b + 2 * lam * (1 - lam) * c)), lam
        pr, lam = pair(slice(None)); bs = [pair(r)[0] for r in dr[:1000]]; r_s = ci(g11); strong = max(r_s[0], res["population"]["earlier wave"][0])
        res["population"]["simulators"][nm] = {"rate": r_s, "pair with the earlier wave": [pr, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "gain over the stronger": pr / strong - 1, "weight of the earlier wave": lam,
                                               "error correlation with the earlier wave": float((g12.mean() - noise.mean()) / np.sqrt((g11.mean() - noise.mean()) * (e_w.mean() - noise.mean())))}
        v = res["population"]["simulators"][nm]; print(f"   {nm:18s} alone {r_s[0]:.0f} [{r_s[1]:.0f}, {r_s[2]:.0f}] | with the earlier wave {pr:.0f} (+{100 * v['gain over the stronger']:.0f}% over the stronger; weight of the wave {lam:.2f}) | error correlation {v['error correlation with the earlier wave']:.2f}")
    # simulators with each other on these questions
    nm_ = list(SIM); Gs = np.array([[np.mean([(SIM[a][0][i] - th[i]) @ (SIM[b][0][i] - th[i]) for i in range(Q)]) - noise.mean() for b in nm_] for a in nm_]); iu = np.triu_indices(len(nm_), 1); cr = Gs / np.sqrt(np.outer(np.diag(Gs), np.diag(Gs)))
    res["population"]["between simulators"] = {"error correlation: min, median, max": [float(cr[iu].min()), float(np.median(cr[iu])), float(cr[iu].max())]}
    print(f"   between simulators on these questions: error correlations {cr[iu].min():.2f}/{np.median(cr[iu]):.2f}/{cr[iu].max():.2f}")
    # ---- party groups with the real group sizes
    G3 = ("Democrats", "Republicans", "Independents"); cells = [(i, g) for i in range(Q) for g in range(3)]; C = len(cells)
    hl = [nz_(T[i]["by_party"][G3[g]]) for i, g in cells]; he = [nz_(T[i]["earlier_by_party"][G3[g]]) for i, g in cells]; Vc = np.array([float((x * (1 - x)).sum()) for x in hl]); m = np.array([T[i]["n_party"][G3[g]] for i, g in cells], float); pi = m / np.array([T[i]["n"] for i, g in cells]); nzc = Vc / m; tc = (1 - 2 * pi) * nzc
    qc = np.array([i for i, g in cells]); drc = [np.concatenate([np.where(qc == q)[0] for q in rng.integers(0, Q, Q)]) for _ in range(1000)]
    gr = lambda e2, corr, idx=slice(None): float(Vc[idx].mean() / max(e2[idx].mean() - corr[idx].mean(), 1e-9)); gci = lambda e2, corr: [gr(e2, corr), float(np.percentile([gr(e2, corr, r) for r in drc], 2.5)), float(np.percentile([gr(e2, corr, r) for r in drc], 97.5))]
    top = [th[i] for i, g in cells]; etop = [te[i] for i, g in cells]
    arms = {"later topline": (np.array([sq(top[c], hl[c]) for c in range(C)]), tc), "earlier wave's answer of the group": (np.array([sq(he[c], hl[c]) for c in range(C)]), nzc), "later topline + earlier wave's contrast": (np.array([sq(cl(top[c] + he[c] - etop[c]), hl[c]) for c in range(C)]), tc)}
    res["groups"] = {"cells": C, "group sizes: min, median, max": [int(m.min()), float(np.median(m)), int(m.max())], **{k: gci(*v) for k, v in arms.items()}, "simulators": {}}
    print(f"party groups: {C} cells of {int(m.min())} to {int(m.max())} respondents | " + " | ".join(f"{k} {res['groups'][k][0]:.0f} [{res['groups'][k][1]:.0f}, {res['groups'][k][2]:.0f}]" for k in arms))
    fold = np.array([i % 2 for i, g in cells]); real_c = [he[c] - etop[c] for c in range(C)]
    for nm, (P, PP) in SIM.items():
        per = np.array([sq(PP[10 * i + 1 + g], hl[c]) for c, (i, g) in enumerate(cells)]); sim_c = [PP[10 * i + 1 + g] - PP[10 * i] for i, g in cells]; con = np.array([sq(cl(top[c] + sim_c[c]), hl[c]) for c in range(C)]); mix = np.zeros(C); lam_ = []
        for f in (0, 1):
            tr = np.where(fold != f)[0]; y = np.concatenate([hl[c] - top[c] - real_c[c] for c in tr]); x = np.concatenate([sim_c[c] - real_c[c] for c in tr]); lam = float(np.clip(x @ y / (x @ x), 0, 1)); lam_.append(lam)
            for c in np.where(fold == f)[0]:
                mix[c] = sq(cl(top[c] + (1 - lam) * real_c[c] + lam * sim_c[c]), hl[c])
        res["groups"]["simulators"][nm] = {"persona": gci(per, nzc), "later topline + simulator's contrast": gci(con, tc), "mixture of the two contrasts": gci(mix, tc), "weight of the simulated contrast": lam_}
        v = res["groups"]["simulators"][nm]; print(f"   {nm:18s} persona {v['persona'][0]:.0f} | topline + its contrast {gci(con, tc)[0]:.0f} [{gci(con, tc)[1]:.0f}, {gci(con, tc)[2]:.0f}] | mixture with the real contrast {v['mixture of the two contrasts'][0]:.0f} (weight of the simulated one {lam_[0]:.2f}/{lam_[1]:.2f})")
    # ---- the collected Economist/YouGov records: asked before (within the five reports) or not; real group sizes against a third of the sample
    W = json.load(open(OUT)); keyc = collections.Counter()
    for w, qs in W.items():
        for q in qs:
            keyc[(norm(q["title"]), norm(q["text"]), tuple(norm(o) for o in q["options"]))] += 1
    D = json.load(open((_REPO + "/data/polls_2026/items.json"))); flag = {}
    for i, d in enumerate(D):
        if d["source"] != "The Economist/YouGov":
            continue
        c = [q for q in W[d["fieldwork_start"]] if (norm(d["question"]) in norm(q["text"]) or norm(q["text"]).endswith(norm(d["question"])[-60:]) or norm(q["title"]) == norm(d.get("topic") or "")) and [norm(o) for o in q["options"]] == [norm(o) for o in d["options"]] and q["percent"] == d["percent"]]
        if c:
            flag[i] = (keyc[(norm(c[0]["title"]), norm(c[0]["text"]), tuple(norm(o) for o in c[0]["options"]))] > 1, c[0])
    rec = [i for i, (r, q) in flag.items() if r]; one = [i for i, (r, q) in flag.items() if not r]; res["collected records"] = {"asked in more than one report": len(rec), "asked once": len(one), "rates": {}}
    hp = {i: nz_(D[i]["percent"]) for i in flag}
    for nm, tag in STATED:
        P = read(f"{SOL}/polls_{tag}", r"t\d+\.json")
        if len(P) >= len(D):
            r_ = lambda idx: float(np.mean([(hp[i] * (1 - hp[i])).sum() for i in idx]) / np.mean([sq(P[i], hp[i]) for i in idx])); res["collected records"]["rates"][nm] = {"asked in more than one report": r_(rec), "asked once": r_(one)}
    print(f"collected Economist/YouGov records matched: {len(rec)} asked in more than one of the five reports, {len(one)} asked once | rates (recurring | once): " + " ".join(f"{nm.replace('GPT-', '').replace('Claude-', '')} {v['asked in more than one report']:.0f}|{v['asked once']:.0f}" for nm, v in res["collected records"]["rates"].items()))
    json.dump(res, open(ROOT + "/results/y_econ.json", "w"), indent=1)
