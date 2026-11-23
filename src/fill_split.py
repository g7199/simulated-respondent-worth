"""Re-asks only the batches that never returned a valid answer, one question per call, with the same instructions and question text.
  python fill_split.py <party|given|sep> <model or local:port:name>
The result is stored under the batch's usual file name, with "split": true."""
import os, re, sys, json, time, runpy, urllib.request, concurrent.futures as cf
what, model = sys.argv[1], sys.argv[2]
suffix = "_" + (model.split(":")[2] + "-stated" if model.startswith("local:") else model)
os.environ["SOL_MODEL"] = model; os.environ["SOL_SUFFIX"] = suffix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sol_common
NAME = {"party": "pollsparty", "given": "pollsgiven", "sep": "pollssep"}[what]; CMD = {"party": "collect", "given": "given", "sep": "sep"}[what]


def ask_local(prompt, want, temp):
    port, served = model.split(":")[1:3]; ids = sorted(want)
    entries = [{"type": "object", "properties": {"id": {"const": i}, "p": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 100}, "minItems": want[i], "maxItems": want[i]}}, "required": ["id", "p"], "additionalProperties": False} for i in ids]
    schema = {"type": "object", "properties": {"items": {"type": "array", "prefixItems": entries, "items": False, "minItems": len(ids), "maxItems": len(ids)}}, "required": ["items"], "additionalProperties": False}
    body = {"model": served, "messages": [{"role": "user", "content": prompt}], "temperature": temp, "max_tokens": 1200, "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}, "chat_template_kwargs": {"enable_thinking": False}}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(rq, timeout=600)); return json.loads(r["choices"][0]["message"]["content"]), r.get("usage", {})


def ask_cli(prompt, out, effort):
    tmp = f"{out}/tmp_split_{os.getpid()}_{time.time_ns()}.json"
    try:
        _, usage, _ = sol_common.call(prompt, tmp, effort); return json.load(open(tmp)), usage
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def split_run(name, jobs, workers=5, effort="medium", check=None):
    if name != NAME:
        return
    out = f"{sol_common.BASE}/{name}{suffix}"; os.makedirs(out, exist_ok=True)
    todo = [k for k in jobs if not os.path.exists(f"{out}/{k}.json")]

    def one(k):
        prompt, want = jobs[k]; parts = re.split(r"\n\n(?=### Question )", prompt); head, blocks = parts[0], parts[1:]; got = {}; use = {}; t0 = time.time()
        for b in blocks:
            q = int(re.match(r"### Question (\d+)", b).group(1)); w = {i: n for i, n in want.items() if i // 10 == q}; ok = False
            for temp in (0.0, 0.3, 0.6, 0.8, 1.0):
                try:
                    r, u = ask_local(head + "\n\n" + b, w, temp) if model.startswith("local:") else ask_cli(head + "\n\n" + b, out, effort)
                    g = {int(x["id"]): x["p"] for x in r["items"]}
                    assert set(g) == set(w) and all(len(g[i]) == w[i] and min(g[i]) >= 0 and sum(g[i]) > 0 for i in w)
                    got.update(g); ok = True
                    for key, v in (u or {}).items():
                        if isinstance(v, (int, float)):
                            use[key] = use.get(key, 0) + v
                    break
                except Exception as ex:
                    err = f"{type(ex).__name__}: {str(ex)[:120]}"
            if not ok:
                return f"{k}: FAILED q{q} {err}"
        json.dump({"items": {str(i): got[i] for i in want}, "usage": use, "seconds": time.time() - t0, "effort": effort, "model": sol_common.MODEL, "split": True}, open(f"{out}/{k}.json", "w"))
        return f"{k}: ok"
    with cf.ThreadPoolExecutor(3 if not model.startswith("local:") else 6) as ex:
        for r in ex.map(one, todo):
            print(name, suffix, r, flush=True)
    print(name, suffix, "complete" if all(os.path.exists(f"{out}/{k}.json") for k in jobs) else "INCOMPLETE", flush=True)


sol_common.run = split_run; sys.argv = ["y_polls.py", CMD, model]; runpy.run_path("y_polls.py", run_name="__main__")
