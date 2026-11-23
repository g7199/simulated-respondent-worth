"""Stated shares from a local model behind an OpenAI-compatible server (vLLM), with the prompt, the batches and the JSON schema
used for the proprietary models: the read-out control (the same open model read through next-token probabilities elsewhere).
  python x_ladder_local.py <tag> <port> <served model name> [survey|polls]
Output: <repo>/outputs/sol/survey_<tag>/bNNNN.json, polls_<tag>/tNNN.json (same format as sol_common.run)."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, time, urllib.request, concurrent.futures as cf
ROOT = (_REPO + ""); tag, port, served = sys.argv[1], sys.argv[2], sys.argv[3]; what = sys.argv[4] if len(sys.argv) > 4 else "survey"
from sol_common import ANSWER_SCHEMA as SCHEMA
BASE = (_REPO + "/outputs/sol")
HEAD = ("You are estimating the results of real opinion surveys. For each item below, a group of respondents is described and a question is given with its answer "
        "options. Estimate the percentage of that group who chose each option in the actual survey (percentages over the listed options, summing to 100). "
        "Answer from your own knowledge only: do not run commands, read files or search. Return JSON with one entry per item, in the order given: its id and a "
        "list \"p\" with one percentage per option, in the order of the options.")
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
if what == "survey":
    sys.path.insert(0, ROOT + "/outputs/screen"); argv = sys.argv; sys.argv = [sys.argv[0], "none"]
    from r2pop_common import load
    sys.argv = argv
    I, E, qi, L0, L1, LQ = load(); K = I.K.values
    def text(i):
        r = I.iloc[i]; return f"### Item {i}\nRespondents: {r['pop']}\nQuestion: {r.q}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(r.opts))
    batches = json.load(open(f"{BASE}/survey/batches.json"))[:50]; jobs = {f"b{k:04d}": [int(i) for i in b] for k, b in enumerate(batches)}; want = lambda i: int(K[i])
else:
    D = json.load(open((_REPO + "/data/polls_2026/items.json"))); when = lambda d: f"{MONTH[int(d['fieldwork_start'][5:7]) - 1]} {d['fieldwork_start'][:4]}"
    text = lambda i: f"### Item {i}\nRespondents: {D[i]['population']}, surveyed in {when(D[i])}\nQuestion: {D[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(D[i]["options"]))
    jobs = {f"t{a:03d}": list(range(a, min(a + 20, len(D)))) for a in range(0, len(D), 20)}; want = lambda i: len(D[i]["options"])
CLOSED = "Answer from your own knowledge only: do not run commands, read files or search."
if what in ("party", "given"):                                   # party groups of the new polls, with the prompts of y_polls.py (collect, given)
    PARTY = [i for i, d in enumerate(D) if d.get("by_party") and {"Democrats", "Republicans"} <= set(d["by_party"])]
    if what == "party":
        HEAD = ("You are estimating the results of real opinion surveys. For each question below, estimate the percentage who chose each option in the actual survey, first among all "
                "respondents and then among the respondents who identify as Democrats, as Republicans and as Independents. " + CLOSED + " Return JSON with four entries per question, "
                "with ids 10*q, 10*q+1, 10*q+2, 10*q+3 for all respondents, Democrats, Republicans and Independents of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
        text = lambda i: f"### Question {i}\nRespondents: {D[i]['population']}, surveyed in {when(D[i])}\nQuestion: {D[i]['question']}\nOptions:\n" + "\n".join(f"{j + 1}. {o}" for j, o in enumerate(D[i]["options"])); G = (0, 1, 2, 3)
    else:
        HEAD = ("You are estimating the results of real opinion surveys. For each question below, the percentage of all respondents who chose each option in the actual survey is given. "
                "Estimate the percentage who chose each option among the respondents who identify as Democrats, as Republicans and as Independents. " + CLOSED + " Return JSON with three entries per question, "
                "with ids 10*q+1, 10*q+2, 10*q+3 for Democrats, Republicans and Independents of question q, each with a list \"p\" of percentages in the order of the options (summing to 100).")
        text = lambda i: (f"### Question {i}\nRespondents: {D[i]['population']}, surveyed in {when(D[i])}\nQuestion: {D[i]['question']}\nOptions, with the percentage of all respondents who chose each:\n" +
                          "\n".join(f"{j + 1}. {o}: {p}%" for j, (o, p) in enumerate(zip(D[i]["options"], D[i]["percent"])))); G = (1, 2, 3)
    jobs = {f"{'p' if what == 'party' else 'g'}{a // 5:03d}": PARTY[a:a + 5] for a in range(0, len(PARTY), 5)}; want = lambda j: len(D[j // 10]["options"]); expand = lambda items: [10 * i + g for i in items for g in G]
else:
    expand = lambda items: list(items)
out = f"{BASE}/{ {'survey': 'survey', 'polls': 'polls', 'party': 'pollsparty', 'given': 'pollsgiven'}[what]}_{tag}"; os.makedirs(out, exist_ok=True)


def ask(items, temperature):
    body = {"model": served, "messages": [{"role": "user", "content": HEAD + "\n\n" + "\n\n".join(text(i) for i in items)}], "temperature": temperature, "max_tokens": 2500,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}}, "chat_template_kwargs": {"enable_thinking": False}}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(rq, timeout=600)); got = {int(x["id"]): x["p"] for x in json.loads(r["choices"][0]["message"]["content"])["items"]}
    assert set(got) == set(expand(items)), "ids"; assert all(len(got[i]) == want(i) and min(got[i]) >= 0 and sum(got[i]) > 0 for i in expand(items)), "lengths"
    return got, r.get("usage", {})


def one(k):
    f = f"{out}/{k}.json"
    if os.path.exists(f):
        return "cached"
    t0 = time.time(); got = {}; use = []; err = ""
    for chunk in ([jobs[k]], [jobs[k][:10], jobs[k][10:]], [jobs[k][a:a + 5] for a in range(0, len(jobs[k]), 5)], [[i] for i in jobs[k]]):       # 20 per call as for the other models; smaller calls only if the answer is malformed
        try:
            got = {}; use = []
            for c in chunk:
                if c:
                    for attempt, temp in enumerate((0.0, 0.3, 0.6)):
                        try:
                            g, u = ask(c, temp); got.update(g); use.append(u); break
                        except Exception as ex:
                            err = f"{type(ex).__name__}: {str(ex)[:100]}"
                            if attempt == 2:
                                raise
            json.dump({"items": {str(i): got[i] for i in expand(jobs[k])}, "usage": use, "seconds": time.time() - t0, "model": served, "calls": len(chunk)}, open(f, "w")); return f"ok ({len(chunk)} calls)"
        except Exception as ex:
            err = f"{type(ex).__name__}: {str(ex)[:100]}"
    return "FAILED " + err


with cf.ThreadPoolExecutor(8) as ex:
    for k, r in zip(jobs, ex.map(one, list(jobs))):
        if r != "cached":
            print(k, r, flush=True)
