"""Which questions to field?  For each weekly report, k questions are fielded (their real results go into the prompt) and the simulator
answers all the others.  The k questions are chosen at random, or greedily so that every other question has a similar fielded question
(facility location on the TF-IDF similarity of the question texts; no answers are used for the choice).
  python w_design.py collect <model> <greedy|random> <k|half> [seed]   -> <repo>/outputs/sol/egdesign-<method><k>[s<seed>]_<model>/
  python w_design.py score                                              -> results/w_design.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json
import numpy as np
ROOT = (_REPO + ""); SOL = (_REPO + "/outputs/sol"); sys.path.insert(0, ROOT + "/src")
from w_groups import load, norm, MONTH
from sklearn.feature_extraction.text import TfidfVectorizer
T = load(); N = len(T); waves = sorted({t["wave"] for t in T}); wave = np.array([t["wave"] for t in T])
X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([t["title"] + " . " + t["question"] + " . " + " ; ".join(t["options"]) for t in T]); SIM = (X @ X.T).toarray()


def choose(w, method, k, seed=0):
    I = np.where(wave == w)[0]; kk = len(I) // 2 if k == "half" else int(k)
    if method == "random":
        return sorted(int(x) for x in np.random.default_rng(100 * seed + int(w[5:7]) * 100 + int(w[8:])).permutation(I)[:kk])
    S = []; cover = np.zeros(len(I))                             # greedy facility location: maximise the sum over questions of the similarity to the closest fielded question
    for _ in range(kk):
        gain = [(np.maximum(cover, SIM[I, j]).sum() - cover.sum()) if j not in S else -1 for j in I]; j = int(I[int(np.argmax(gain))]); S.append(j); cover = np.maximum(cover, SIM[I, j])
    return sorted(S)


def batteries(w):
    I = np.where(wave == w)[0]; g = {}
    for i in I:
        g.setdefault((norm(T[i]["text"]), tuple(T[i]["options"])), []).append(int(i))
    return [v for v in g.values() if len(v) > 1], [v[0] for v in g.values() if len(v) == 1]


_choose0 = choose


def choose(w, method, k, seed=0, model=None):
    """adds two designs: 'battery' (one random item of every battery, then random other questions) and 'llm' (the simulator's own choice, stored per model)"""
    if method == "battery":
        r = np.random.default_rng(700 * seed + int(w[5:7]) * 100 + int(w[8:])); I = np.where(wave == w)[0]; kk = len(I) // 2 if k == "half" else int(k); bats, singles = batteries(w)
        bats = sorted(bats, key=lambda v: -len(v))[:kk]; S = [int(r.choice(v)) for v in bats]; rest = [int(x) for x in r.permutation([i for i in I if i not in S and i in singles])] + [int(x) for x in r.permutation([i for i in I if i not in S and i not in singles])]
        return sorted(S + rest[:kk - len(S)])
    if method == "llm":
        f = f"{SOL}/egpick{k}_{model}/{w}.json"; return sorted(json.load(open(f))["ids"])
    return _choose0(w, method, k, seed)


def name(method, k, seed):
    return f"egdesign-{method}{k}" + (f"s{seed}" if method in ("random", "battery") else "")


def pick_llm(model, k):
    """the simulator chooses the k questions whose real results it wants to see"""
    from sol_common import call
    out = f"{SOL}/egpick{k}_{model}"; os.makedirs(out, exist_ok=True)
    for w in waves:
        f = f"{out}/{w}.json"
        if os.path.exists(f):
            continue
        I = [int(i) for i in np.where(wave == w)[0]]
        prompt = (f"A survey of registered voters in the United States will ask the {len(I)} questions listed below. You will later have to estimate the results of all of them. Before you do, you may see the real results of {k} of these questions, "
                  f"asked of the full sample. Choose the {k} questions whose real results would help you most to estimate the results of all the others. Answer from your own knowledge only: do not run commands, read files or search. "
                  f"Return JSON with exactly {k} entries, one per chosen question: its id and the list \"p\" equal to [1].\n\n" + "\n".join(f"Question {i}: {T[i]['question']} [" + " | ".join(T[i]["options"]) + "]" for i in I))
        for attempt in range(4):
            try:
                tmp = f"{out}/tmp_{w}.json"; call(prompt, tmp, "medium"); ids = sorted({int(x["id"]) for x in json.load(open(tmp))["items"]}); assert len(ids) == int(k) and all(i in I for i in ids), (len(ids), [i for i in ids if i not in I][:3])
                json.dump({"ids": ids}, open(f, "w")); os.remove(tmp); break
            except Exception as ex:
                print("pick", w, "attempt", attempt, type(ex).__name__, str(ex)[:120], flush=True)
        print("picked", w, json.load(open(f))["ids"], flush=True)


cmd = sys.argv[1]
if cmd == "collect":
    model, method, k = sys.argv[2], sys.argv[3], sys.argv[4]; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = "_" + model
    from sol_common import run, CLOSED
    when = lambda t: f"{MONTH[int(t['wave'][5:7]) - 1]} {t['wave'][:4]}"
    base = ("You are estimating the results of real opinion surveys. For each item below, a question from one survey is given with its answer options. Estimate the percentage of the "
            "respondents who chose each option in the actual survey (percentages over the listed options, summing to 100). The actual results of other questions asked in the same survey are given first; use them as you see fit. ")
    tail = CLOSED + " Return JSON with one entry per item, in the order given: its id and a list \"p\" with one percentage per option, in the order of the options."
    item = lambda i: f"### Item {i}\nQuestion: {T[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(T[i]["options"]))
    known = lambda i: f"- {T[i]['question']}\n  " + "; ".join(f"{o}: {p}%" for o, p in zip(T[i]["options"], T[i]["percent"]))
    jobs = {}
    if method == "llm":
        pick_llm(model, k)
    for w in waves:
        S = choose(w, method, k, seed, model); tg = [int(i) for i in np.where(wave == w)[0] if i not in S]; pop = f"The survey: {T[tg[0]]['population']}, surveyed in {when(T[tg[0]])}.\n"
        for a in range(0, len(tg), 20):
            ch = tg[a:a + 20]
            jobs[f"d{w[5:7]}{w[8:]}_{a // 20:02d}"] = (base + tail + "\n\n" + pop + "\nActual results of other questions in the same survey:\n" + "\n".join(known(i) for i in S) + "\n\nItems to estimate:\n\n" + "\n\n".join(item(i) for i in ch), {i: len(T[i]["options"]) for i in ch})
    print(len(jobs), "calls,", sum(len(v[1]) for v in jobs.values()), "targets", flush=True); run(name(method, k, seed), jobs, 3, "medium")
else:
    from w_util import read, MODELS
    th = [np.asarray(t["percent"], float) / sum(t["percent"]) for t in T]; V = np.array([(x * (1 - x)).sum() for x in th]); noise = V / np.array([t["n"] for t in T]); rng = np.random.default_rng(0)
    bk = {}; BLOCK = np.array([bk.setdefault((t["wave"], norm(t["text"])), len(bk)) for t in T])
    def rate(P, ii):
        ii = np.asarray(ii); return float(V[ii].mean() / max(np.mean([(P[i] - th[i]) @ (P[i] - th[i]) for i in ii]) - noise[ii].mean(), 1e-9))
    def ci(P, ii, B=600):
        ii = np.asarray(ii); bl = np.unique(BLOCK[ii]); out = []
        for _ in range(B):
            pick = rng.choice(bl, len(bl)); out.append(rate(P, np.concatenate([ii[BLOCK[ii] == b] for b in pick])))
        return [rate(P, ii), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
    res = {"designs": {}, "models": {}}
    DES = [(mth, k, sd) for k in ("half", "10", "5") for mth, sds in (("greedy", (0,)), ("llm", (0,)), ("battery", (1, 2, 3)), ("random", (1, 2, 3, 4, 5))) for sd in sds]
    def describe(S):
        tg = [i for i in range(N) if i not in set(S)]
        sibl = np.mean([any(norm(T[j]["text"]) == norm(T[i]["text"]) and tuple(T[j]["options"]) == tuple(T[i]["options"]) and wave[j] == wave[i] for j in S) for i in tg]); near = np.mean([max(SIM[i, j] for j in S if wave[j] == wave[i]) for i in tg])
        return {"fielded": len(S), "simulated": len(tg), "share of simulated questions with a fielded sibling": float(sibl), "mean similarity to the closest fielded question": float(near)}
    for nm, tag in MODELS:
        cold = read(f"{SOL}/egplain_{tag}"); o = {}
        for method, k, seed in DES:
            P = read(f"{SOL}/{name(method, k, seed)}_{tag}")
            if not len(P):
                continue
            try:
                S = set(j for w in waves for j in choose(w, method, k, seed, tag))
            except FileNotFoundError:
                continue
            tg = [i for i in range(N) if i not in S]
            if all(i in P for i in tg):
                err_all = float(np.sum([(P[i] - th[i]) @ (P[i] - th[i]) - noise[i] for i in tg]) / N)        # mean squared error over ALL questions of the reports, fielded ones counted as known
                o[name(method, k, seed)] = {"worth on the simulated questions": ci(P, tg, 300), "cold on the same questions": rate(cold, tg) if len(cold) >= N else None, "mean squared error over all questions (fielded = 0)": err_all, **describe(sorted(S))}
        if o:
            res["models"][nm] = o
            for k in ("half", "10", "5"):
                sub = {d: v for d, v in o.items() if d.split("-")[1].replace("greedy", "").replace("random", "").replace("battery", "").replace("llm", "").split("s")[0] == k}
                if not sub:
                    continue
                rnd = [v["mean squared error over all questions (fielded = 0)"] for d, v in sub.items() if "random" in d]
                print(f"{nm} k={k}: random draws error x1e4: " + " ".join(f"{1e4 * x:.0f}" for x in rnd) + (f" (mean {1e4 * np.mean(rnd):.0f})" if rnd else "") + " | " +
                      " | ".join(f"{d.split('-')[1]}: {1e4 * v['mean squared error over all questions (fielded = 0)']:.0f}" + (f" ({100 * (v['mean squared error over all questions (fielded = 0)'] / np.mean(rnd) - 1):+.0f}%)" if rnd else "") + f" worth {v['worth on the simulated questions'][0]:.0f}, sibling share {v['share of simulated questions with a fielded sibling']:.2f}" for d, v in sub.items() if "random" not in d))
    json.dump(res, open(ROOT + "/results/w_design.json", "w"), indent=1)
