"""Twenty demographic and political groups on questions asked after the training cutoff.
The five weekly Economist/YouGov reports (fieldwork 2 October to 6 November 2026) publish, for every question, the answers of all
respondents and of twenty groups with their sizes: sex (2), race (3), age (4), education (2), 2024 vote (2), ideology (3),
MAGA supporters (1), party identification (3).
  python w_groups.py parse                       -> <repo>/data/polls_2026/econ_groups.json (every distinct question, last time asked)
  python w_groups.py collect <model> joint [nq]  the model states everyone and the twenty groups in one answer (no real number shown)
  python w_groups.py collect <model> shown [nq]  the real answer of everyone is shown; the model states the twenty groups
  python w_groups.py collect <model> ctx   [nq]  population only: the real answers to the other questions of the same report are shown
Outputs: <repo>/outputs/sol/eg<mode>_<model>/."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import re, json, sys, os, collections
DIR = (_REPO + "/data/polls_2026/yougov_pdf"); OUT = (_REPO + "/data/polls_2026/econ_groups.json"); ROOT = (_REPO + "")
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
HEAD = re.compile(r"^(\d+[A-Z]?)\. (\S.*)$")
G1 = ["Male", "Female", "White", "Black", "Hispanic", "18-29", "30-44", "45-64", "65+", "No degree", "College grad"]
G2 = ["Harris", "Trump", "Lib", "Mod", "Con", "MAGA", "Dem", "Ind", "Rep"]
GROUPS = G1 + G2
KIND = dict(zip(GROUPS, ["sex"] * 2 + ["race"] * 3 + ["age"] * 4 + ["education"] * 2 + ["2024 vote"] * 2 + ["ideology"] * 3 + ["MAGA"] + ["party"] * 3))
LABEL = dict(zip(GROUPS, ["men", "women", "White respondents", "Black respondents", "Hispanic respondents", "respondents aged 18 to 29", "respondents aged 30 to 44", "respondents aged 45 to 64",
                          "respondents aged 65 and over", "respondents without a college degree", "college graduates", "respondents who voted for Kamala Harris in 2024",
                          "respondents who voted for Donald Trump in 2024", "liberals", "moderates", "conservatives", "respondents who call themselves MAGA supporters", "Democrats", "Independents", "Republicans"]))
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def parse(path):
    L = open(path).read().split("\n"); starts = [i for i, l in enumerate(L) if HEAD.match(l)]; out = []; pop = None
    for l in L[:400]:
        m = re.search(r"\d{4} - [\d,]+ (U\.S\. [A-Za-z ]+)$", l.strip())
        if m:
            pop = m.group(1).strip(); break
    for a, b in zip(starts, starts[1:] + [len(L)]):
        m = HEAD.match(L[a]); blk = L[a + 1:b]; tot = [i for i, l in enumerate(blk) if re.match(r"\s+Total\s", l)]
        if len(tot) < 2:
            continue
        text = " ".join(x.strip() for x in blk[:tot[0] - 1] if x.strip()); halves = []
        for t in tot[:2]:
            rows = []; n_row = None
            for l in blk[t + 1:]:
                if re.match(r"\s*Unweighted N", l):
                    n_row = [int(x.replace(",", "")) for x in re.findall(r"\(([\d,]+)\)", l)]; break
                if not l.strip() or re.match(r"\s*Totals\b", l):
                    continue
                vals = re.findall(r"(\d+)%", l); lab = re.split(r"\s{2,}\d+%", l.strip())[0].strip()
                if vals and lab:
                    rows.append((lab, [int(v) for v in vals]))
            halves.append((blk[t], rows, n_row))
        (h1, r1, n1), (h2, r2, n2) = halves
        if not r1 or [x[0] for x in r1] != [x[0] for x in r2] or not n1 or not n2 or h1.split()[1:] != ["Male", "Female", "White", "Black", "Hispanic", "18-29", "30-44", "45-64", "65+", "No", "degree", "College", "grad"]:
            continue
        if not re.search(r"Harris\s+Trump\s+Lib\s+Mod\s+Con\s+Supporter\s+Dem\s+Ind\s+Rep\s*$", h2) or any(len(v) != 12 for _, v in r1) or len(n1) != 12 or any(len(v) < 10 for _, v in r2) or len(n2) < 10:
            continue
        tl = [v[0] for _, v in r1]
        if not 96 <= sum(tl) <= 104 or len(tl) < 2:
            continue
        g = {}
        for k, name in enumerate(G1):
            g[name] = {"p": [v[1 + k] for _, v in r1], "n": n1[1 + k]}
        for k, name in enumerate(G2):
            g[name] = {"p": [v[len(v) - 9 + k] for _, v in r2], "n": n2[len(n2) - 9 + k]}
        if any(not 90 <= sum(x["p"]) <= 110 for x in g.values()):
            continue
        title = m.group(2).strip()
        out.append({"number": m.group(1), "title": title, "text": text, "question": text + (" — " + title.split("—")[-1].strip() if "—" in title else ""), "options": [x[0] for x in r1], "percent": tl, "n": n1[0], "groups": g, "population": "registered voters in the United States"})
    return out


def load():
    return json.load(open(OUT))


if __name__ == "__main__" and sys.argv[1] == "parse":
    key = collections.OrderedDict(); per = {}
    for f in sorted(os.listdir(DIR)):
        if f.endswith(".txt"):
            w = f[5:15]; qs = parse(f"{DIR}/{f}"); per[w] = len(qs)
            for q in qs:
                key.setdefault((norm(q["title"]), norm(q["text"]), tuple(norm(o) for o in q["options"])), []).append((w, q))
    T = []
    for k, v in key.items():
        w, q = v[-1]; e = v[-2] if len(v) > 1 else None
        T.append({**q, "wave": w, "earlier": None if e is None else {"wave": e[0], "percent": e[1]["percent"], "n": e[1]["n"], "groups": e[1]["groups"]}})
    json.dump(T, open(OUT, "w"), indent=0)
    print("questions parsed per report:", per, "| distinct:", len(T), "| with an earlier wave:", sum(t["earlier"] is not None for t in T), "| populations:", collections.Counter(t["population"] for t in T))
    import numpy as np
    print("options per question:", collections.Counter(len(t["options"]) for t in T).most_common(), "| group sizes (median):", {g: int(np.median([t["groups"][g]["n"] for t in T])) for g in GROUPS})
    print(json.dumps(T[5], ensure_ascii=False)[:900])

if __name__ == "__main__" and sys.argv[1] == "collect":
    model, mode = sys.argv[2], sys.argv[3]; nq = int(sys.argv[4]) if len(sys.argv) > 4 else None; workers = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + (model.split(":")[2] + "-stated" if model.startswith("local:") else model); sys.path.insert(0, ROOT + "/src")
    from sol_common import run, CLOSED
    T = load(); idx = list(range(len(T)))[:nq] if nq else list(range(len(T)))
    when = lambda t: f"{MONTH[int(t['wave'][5:7]) - 1]} {t['wave'][:4]}"
    glist = "\n".join(f"{k + 1}. {LABEL[g]}" for k, g in enumerate(GROUPS))
    if mode in ("joint", "shown"):
        if mode == "joint":
            head = ("You are estimating the results of real opinion surveys. For each question below, estimate the percentage who chose each option in the actual survey, first among all respondents "
                    "and then within each of the twenty groups of respondents listed here:\n" + glist + "\n" + CLOSED + " Return JSON with twenty-one entries per question: id 100*q for all respondents and "
                    "id 100*q+g for group g (1 to 20) of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
            txt = lambda i: f"### Question {i}\nRespondents: {T[i]['population']}, surveyed in {when(T[i])}\nQuestion: {T[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(T[i]["options"]))
            want = lambda ch: {100 * i + g: len(T[i]["options"]) for i in ch for g in range(21)}
        else:
            head = ("You are estimating the results of real opinion surveys. For each question below, the percentage of all respondents who chose each option in the actual survey is given. "
                    "Estimate the percentage who chose each option within each of the twenty groups of respondents listed here:\n" + glist + "\n" + CLOSED + " Return JSON with twenty entries per question: "
                    "id 100*q+g for group g (1 to 20) of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
            txt = lambda i: (f"### Question {i}\nRespondents: {T[i]['population']}, surveyed in {when(T[i])}\nQuestion: {T[i]['question']}\nOptions, with the percentage of all respondents who chose each:\n" +
                             "\n".join(f"{j + 1}. {o}: {p}%" for j, (o, p) in enumerate(zip(T[i]["options"], T[i]["percent"]))))
            want = lambda ch: {100 * i + g: len(T[i]["options"]) for i in ch for g in range(1, 21)}
        B = int(os.environ.get("W_B", "4")); jobs = {}
        for a in range(0, len(idx), B):
            ch = idx[a:a + B]; jobs[f"q{a // B:03d}"] = (head + "\n\n" + "\n\n".join(txt(i) for i in ch), want(ch))
        print(next(iter(jobs.values()))[0][:2500], flush=True)
        run("eg" + mode, jobs, workers, "medium")
    elif mode == "shownx":
        # the real answer of everyone is shown, and so are the full results (everyone and the twenty groups) of five other questions of the same report
        import numpy as np
        head = ("You are estimating the results of real opinion surveys. For each question below, the percentage of all respondents who chose each option in the actual survey is given. "
                "Estimate the percentage who chose each option within each of the twenty groups of respondents listed here:\n" + glist + "\nThe actual results of some other questions asked in the same survey, "
                "for all respondents and for each of the twenty groups, are given first; use them as you see fit. " + CLOSED + " Return JSON with twenty entries per question: "
                "id 100*q+g for group g (1 to 20) of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
        txt = lambda i: (f"### Question {i}\nQuestion: {T[i]['question']}\nOptions, with the percentage of all respondents who chose each:\n" +
                         "\n".join(f"{j + 1}. {o}: {p}%" for j, (o, p) in enumerate(zip(T[i]["options"], T[i]["percent"]))))
        ref = lambda j: (f"- Question: {T[j]['question']}\n  Options: " + " | ".join(T[j]["options"]) + "\n  all respondents: " + "/".join(str(p) for p in T[j]["percent"]) + "\n  " +
                         "\n  ".join(f"{LABEL[g]}: " + "/".join(str(p) for p in T[j]["groups"][g]["p"]) for g in GROUPS))
        rng = np.random.default_rng(11); half = {}
        for w_ in sorted({t["wave"] for t in T}):
            ii = [i for i in range(len(T)) if T[i]["wave"] == w_]; perm = rng.permutation(ii)
            for k, i in enumerate(perm):
                half[int(i)] = k % 2
        jobs = {}
        for w_ in sorted({t["wave"] for t in T}):
            for h in (0, 1):
                tg = [i for i in idx if T[i]["wave"] == w_ and half[i] == h]; kn = [i for i in range(len(T)) if T[i]["wave"] == w_ and half[i] != h and len(T[i]["options"]) <= 7]
                kn = sorted(int(x) for x in np.random.default_rng(17 + 7 * h + int(w_[5:7]) * 100 + int(w_[8:])).permutation(kn)[:5])
                for a in range(0, len(tg), 4):
                    ch = tg[a:a + 4]
                    jobs[f"x{w_[5:7]}{w_[8:]}_{h}_{a // 4:02d}"] = (head + f"\n\nThe survey: {T[ch[0]]['population']}, surveyed in {when(T[ch[0]])}.\n\nActual results of other questions in the same survey (percentages in the order of the options):\n" +
                                                                "\n".join(ref(j) for j in kn) + "\n\nQuestions to estimate:\n\n" + "\n\n".join(txt(i) for i in ch), {100 * i + g: len(T[i]["options"]) for i in ch for g in range(1, 21)})
        print(next(iter(jobs.values()))[0][:3500], flush=True); print(len(jobs), "calls", flush=True)
        run("egshownx", jobs, workers, "medium")
    elif mode == "peers":
        # no real answer is shown; instead the simulator reads the answers of two other simulators (the cold answers of GPT-6-astra and GPT-6.1-sol) to the same question
        import numpy as np
        from w_util import read as _read
        PA = _read((_REPO + "/outputs/sol/egplain_gpt-6-astra")); PB = _read((_REPO + "/outputs/sol/egplain_gpt-6.1-sol"))
        rng = np.random.default_rng(11); half = {}
        for w_ in sorted({t["wave"] for t in T}):
            ii = [i for i in range(len(T)) if T[i]["wave"] == w_]; perm = rng.permutation(ii)
            for k, i in enumerate(perm):
                half[int(i)] = k % 2
        base = ("You are estimating the results of real opinion surveys. For each item below, a question from one survey is given with its answer options. Estimate the percentage of the "
                "respondents who chose each option in the actual survey (percentages over the listed options, summing to 100). With each item you are given the estimates of two other forecasters, "
                "who had no access to the results either; use them as you see fit. ")
        tail = CLOSED + " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options."
        pc = lambda p: "/".join(f"{100 * x:.0f}" for x in p)
        item = lambda i: (f"### Item {i}\nQuestion: {T[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(T[i]["options"])) +
                          f"\nForecaster A (percentages in the order of the options): {pc(PA[i])}\nForecaster B: {pc(PB[i])}")
        jobs = {}
        for w_ in sorted({t["wave"] for t in T}):
            for h in (0, 1):
                tg = [i for i in idx if T[i]["wave"] == w_ and half[i] == h]
                if not tg:
                    continue
                pop = f"The survey: {T[tg[0]]['population']}, surveyed in {when(T[tg[0]])}.\n"
                for a in range(0, len(tg), 20):
                    ch = tg[a:a + 20]; jobs[f"c{w_[5:7]}{w_[8:]}_{h}_{a // 20:02d}"] = (base + tail + "\n\n" + pop + "\n" + "\n\n".join(item(i) for i in ch), {i: len(T[i]["options"]) for i in ch})
        print(next(iter(jobs.values()))[0][:1800], flush=True); print(len(jobs), "calls", flush=True)
        run("egpeers", jobs, workers, "medium")
    elif mode in ("ctxweek", "ctxoth"):
        # controls for the same-survey context: (ctxprev) results of OTHER questions from the report of the week before, same pollster;
        # (ctxoth) results of questions asked by other pollsters in the United States in the same five weeks. Same targets and calls as ctx.
        import numpy as np
        rng = np.random.default_rng(11); half = {}
        for w_ in sorted({t["wave"] for t in T}):
            ii = [i for i in range(len(T)) if T[i]["wave"] == w_]; perm = rng.permutation(ii)
            for k, i in enumerate(perm):
                half[int(i)] = k % 2
        key = lambda q: (norm(q["title"]), norm(q["text"]), tuple(norm(o) for o in q["options"])); waves = sorted({t["wave"] for t in T}); PW = {}
        for f in sorted(os.listdir(DIR)):
            if f.endswith(".txt"):
                PW[f[5:15]] = parse(f"{DIR}/{f}")
        tkeys = {key(t) for t in T}
        OTH = [d for d in json.load(open((_REPO + "/data/polls_2026/items.json"))) if d["source"] != "The Economist/YouGov" and d["country"] == "United States"]
        base = ("You are estimating the results of real opinion surveys. For each item below, a question from one survey is given with its answer options. Estimate the percentage of the "
                "respondents who chose each option in the actual survey (percentages over the listed options, summing to 100). ")
        tail = CLOSED + " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options."
        item = lambda i: f"### Item {i}\nQuestion: {T[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(T[i]["options"]))
        jobs = {}
        for wi, w_ in enumerate(waves):
            for h in (0, 1):
                tg = [i for i in idx if T[i]["wave"] == w_ and half[i] == h]
                if not tg or (mode == "ctxweek" and wi == 0):
                    continue
                r3 = np.random.default_rng(29 + 7 * h + int(w_[5:7]) * 100 + int(w_[8:]))
                if mode == "ctxweek":
                    now = {key(q) for q in PW[w_]}; pool = [q for q in PW[waves[wi - 1]] if key(q) not in now]          # never a question that is asked again this week (no earlier wave of a target)
                    kn = [pool[int(x)] for x in r3.permutation(len(pool))[:29]]
                    lead = "The actual results of other questions that the same pollster asked of the same population in its survey of the week before are given first; use them as you see fit. "
                    block = "Actual results of other questions in the same pollster's survey of the week before:\n" + "\n".join(f"- {q['question']}\n  " + "; ".join(f"{o}: {p}%" for o, p in zip(q["options"], q["percent"])) for q in kn)
                else:
                    kn = [OTH[int(x)] for x in r3.permutation(len(OTH))[:29]]
                    lead = "The actual results of questions that other pollsters asked in the United States in the same weeks are given first; use them as you see fit. "
                    block = "Actual results of questions from other recent polls in the United States (other pollsters, October and November 2026):\n" + "\n".join(f"- ({d['population']}) {d['question']}\n  " + "; ".join(f"{o}: {p}%" for o, p in zip(d["options"], d["percent"])) for d in kn)
                pop = f"The survey: {T[tg[0]]['population']}, surveyed in {when(T[tg[0]])}.\n"
                for a in range(0, len(tg), 20):
                    ch = tg[a:a + 20]
                    jobs[f"c{w_[5:7]}{w_[8:]}_{h}_{a // 20:02d}"] = (base + lead + tail + "\n\n" + pop + "\n" + block + "\n\nItems to estimate:\n\n" + "\n\n".join(item(i) for i in ch), {i: len(T[i]["options"]) for i in ch})
        print(next(iter(jobs.values()))[0][:2200], flush=True); print(len(jobs), "calls", sum(len(v[1]) for v in jobs.values()), "targets", flush=True)
        run("eg" + mode, jobs, workers, "medium")
    elif mode in ("ctx", "plain"):
        # population answers. plain: no real number shown. ctx: every other question of the same report, in a different half, is shown with its real answer
        import numpy as np
        if os.environ.get("W_FILL"):                              # only the items still missing (after malformed answers), in smaller calls with other file names
            from w_util import read as _read
            have = set(_read(f"{_REPO}/outputs/sol/eg{mode}{os.environ.get('W_K', '')}{os.environ['SOL_SUFFIX']}")); idx = [i for i in idx if i not in have]; print("to fill:", len(idx), flush=True)
        rng = np.random.default_rng(int(os.environ.get("W_SEED", "11"))); half = {}
        for w in sorted({t["wave"] for t in T}):
            ii = [i for i in range(len(T)) if T[i]["wave"] == w]; perm = rng.permutation(ii)
            for k, i in enumerate(perm):
                half[int(i)] = k % 2
        base = ("You are estimating the results of real opinion surveys. For each item below, a question from one survey is given with its answer options. Estimate the percentage of the "
                "respondents who chose each option in the actual survey (percentages over the listed options, summing to 100). ")
        tail = CLOSED + " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options."
        item = lambda i: f"### Item {i}\nQuestion: {T[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(T[i]["options"]))
        NZ = int(os.environ.get("W_NOISE", "0")); _r = np.random.default_rng(77)
        def known(i):
            p = np.asarray(T[i]["percent"], float) / sum(T[i]["percent"])
            if NZ:
                p = np.random.default_rng(1000 + i).multinomial(NZ, p) / NZ
            return f"- {T[i]['question']}\n  " + "; ".join(f"{o}: {100 * x:.0f}%" for o, x in zip(T[i]["options"], p))
        jobs = {}
        for w in sorted({t["wave"] for t in T}):
            for h in (0, 1):
                tg = [i for i in idx if T[i]["wave"] == w and half[i] == h]; kn = [i for i in range(len(T)) if T[i]["wave"] == w and half[i] != h]
                if os.environ.get("W_K"):                        # only a random K of the other half are shown (how much real data is needed?)
                    kn = sorted(int(x) for x in np.random.default_rng(13 + 7 * h + int(w[5:7]) * 100 + int(w[8:])).permutation(kn)[:int(os.environ["W_K"])])
                if not tg:
                    continue
                pop = f"The survey: {T[tg[0]]['population']}, surveyed in {when(T[tg[0]])}.\n"
                BC = int(os.environ.get("W_BC", "20"))
                for a in range(0, len(tg), BC):
                    ch = tg[a:a + BC]
                    if mode == "plain":
                        prompt = base + tail + "\n\n" + pop + "\n" + "\n\n".join(item(i) for i in ch)
                    else:
                        prompt = (base + ("The results of other questions asked in the same survey are given first" + (f", from a random sample of only {NZ} of its respondents" if NZ else "") + "; use them as you see fit. ") + tail + "\n\n" + pop +
                                  ("\nActual results of other questions in the same survey:\n" if not NZ else f"\nResults of other questions in the same survey among {NZ} randomly drawn respondents:\n") + "\n".join(known(i) for i in kn) + "\n\nItems to estimate:\n\n" + "\n\n".join(item(i) for i in ch))
                    jobs[f"{os.environ.get('W_KEY', 'f') if os.environ.get('W_FILL') else 'c'}{w[5:7]}{w[8:]}_{h}_{a // BC:02d}"] = (prompt, {i: len(T[i]["options"]) for i in ch})
        print(next(iter(jobs.values()))[0][:3000], flush=True); print(len(jobs), "calls", flush=True)
        run("eg" + mode + os.environ.get("W_K", "") + ("s" + os.environ["W_SEED"] if os.environ.get("W_SEED") else "") + ("n" + os.environ["W_NOISE"] if os.environ.get("W_NOISE") else ""), jobs, workers, "medium")
