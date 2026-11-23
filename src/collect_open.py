"""Next-token read-out of a local open model, with the prompts used for the other open models.
  LLM_MODEL=<served> LLM_PORTS=<port> python collect_open.py polls <tag>   -> outputs/screen/r3_out/polls_<tag>.json
  LLM_MODEL=<served> LLM_PORTS=<port> python collect_open.py gss <tag>     -> outputs/screen/x_grp_out/<tag>_fwd.json"""
import json, os, sys, time, concurrent.futures as cf
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
what, tag = sys.argv[1], sys.argv[2]; sys.argv = [sys.argv[0], "none"]
import llm
if what == "polls":
    from llm import score_options
    D = json.load(open(REPO + "/data/polls_2026/items.json")); L = "ABCDEFGHIJKL"
    MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    pop = lambda d: f"It is {MONTH[int(d['fieldwork_start'][5:7]) - 1]} {d['fieldwork_start'][:4]}. You are a respondent in a survey of {d['population']}."
    prompt = lambda d, opts: f"{pop(d)}\n\n{d['question']}\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + "\nAnswer with a single letter."
    out = {}
    for k in sorted({len(d["options"]) for d in D}):
        idx = [i for i, d in enumerate(D) if len(d["options"]) == k]
        for o in (0, 1):
            r = score_options([prompt(D[i], D[i]["options"][::-1] if o else D[i]["options"]) for i in idx], list(L[:k]), workers=32)
            for i, x in zip(idx, r):
                lp = [x[c] for c in L[:k]]; out[f"{i}_{o}"] = lp[::-1] if o else lp
    json.dump(out, open(f"{REPO}/outputs/screen/r3_out/polls_{tag}.json", "w")); print("scored", len(out) // 2, "polls")
else:
    from t4_run import ITEMS, PARTS, fwd, L
    OUT = f"{REPO}/outputs/screen/x_grp_out/{tag}_fwd.json"; port = llm.PORTS[0]

    def one(job):
        key, prompt, k = job
        body = {"model": llm.MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            top = llm._post(port, body, timeout=300)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception as e:
            return key, None, repr(e)
        d = {}
        for t in top:
            d.setdefault(t["token"].strip(), t["logprob"])
        return key, dict(kind=key[0], item=key[1], part=key[2], key=key[3], lp=[d.get(c) for c in L[:k]], lp20=min(t["logprob"] for t in top), top1=top[0]["token"]), None
    jobs = []
    for it, (q, opts, _) in ITEMS.items():
        jobs.append((("fwd", it, "pop", "pop"), fwd(q, opts), len(opts)))
        for pn, (desc, gs) in PARTS.items():
            for g, sent, _lab in gs:
                jobs.append((("fwd", it, pn, g), fwd(q, opts, sent), len(opts)))
    done = {}
    for attempt in range(3):
        todo = [j for j in jobs if j[0] not in done]
        if not todo:
            break
        with cf.ThreadPoolExecutor(16) as ex:
            for key, rec, err in ex.map(one, todo):
                if rec is not None:
                    done[key] = rec
                else:
                    print("fail", key, err, flush=True)
    out = [done[j[0]] for j in jobs if j[0] in done]; json.dump(out, open(OUT, "w"))
    print("prompts", len(jobs), "collected", len(out), "with an option outside the top-20:", sum(any(x is None for x in r["lp"]) for r in out))
