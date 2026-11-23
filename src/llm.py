"""Tiny client for the shared local LLM servers (OpenAI-compatible vLLM).
  from llm import chat, score_options
  outs = chat(["prompt 1", "prompt 2"], max_tokens=64)                  # list[str], greedy, thinking off
  outs = chat(prompts, max_tokens=512, think=True)                       # let the model reason first (returns final text)
  outs = chat(prompts, n=5, temperature=0.8)                             # list[list[str]]
  lps  = score_options(prompts, ["A","B","C","D"])                       # list[dict option->logprob] for the next token
Servers: port 8001 (GPU1) and, when up, port 8000 (GPU0); requests are spread over the servers that answer."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, re, concurrent.futures as cf, urllib.request, itertools, os

PORTS = [int(p) for p in os.environ.get("LLM_PORTS", "8001,8000").split(",")]
MODEL = os.environ.get("LLM_MODEL", "qwen9b")


def _alive():
    ok = []
    for p in PORTS:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{p}/v1/models", timeout=3).read(); ok.append(p)
        except Exception:
            pass
    return ok


def _post(port, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


def chat(prompts, max_tokens=64, temperature=0.0, n=1, think=False, system=None, workers=48, seed=0):
    ports = _alive()
    assert ports, "no LLM server is up"
    cyc = itertools.cycle(ports)
    jobs = [(next(cyc), p) for p in prompts]

    def one(job):
        port, p = job
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
        body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "n": n, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": bool(think)}}
        for _ in range(3):
            try:
                r = _post(port, body)
                outs = [_strip_think(c["message"].get("content") or "") for c in r["choices"]]
                return outs[0] if n == 1 else outs
            except Exception as e:
                err = e
        return "" if n == 1 else [""] * n

    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def score_options(prompts, options, workers=48):
    """Log-probability of each option string's first token as the next token (thinking off). Missing -> -30."""
    ports = _alive(); assert ports, "no LLM server is up"
    cyc = itertools.cycle(ports); jobs = [(next(cyc), p) for p in prompts]

    def one(job):
        port, p = job
        body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 1, "temperature": 0.0,
                "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            top = _post(port, body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception:
            return {o: -30.0 for o in options}
        d = {}
        for t in top:
            d.setdefault(t["token"].strip(), t["logprob"])
        return {o: d.get(o, -30.0) for o in options}

    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))
