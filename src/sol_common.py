"""Model calls for the data collection, closed-book and with one JSON schema for the answer.
SOL_BACKEND=cli (default, used for the paper): the Codex and Claude command-line clients with all tools disabled, in an empty directory.
SOL_BACKEND=api: the OpenAI Responses API (OPENAI_API_KEY) or the Claude Messages API (ANTHROPIC_API_KEY), with no tools.
SOL_MODEL=local:<port>:<name> uses a local OpenAI-compatible server under either backend.
run(name, jobs, workers, effort): jobs = {key: (prompt, {id: number of values expected})}; results are cached per key in
<repo>/outputs/sol/<name>/<key>.json as {"items": {id: [values]}, "usage": ..., "seconds": ...}."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, json, time, subprocess, concurrent.futures as cf

MODEL = os.environ.get("SOL_MODEL", "gpt-6.1-sol"); BASE = (_REPO + "/outputs/sol"); WORK = (_REPO + "/outputs/.cli_work/w"); SCHEMA = (_REPO + "/outputs/.cli_work/schema.json")
BACKEND = os.environ.get("SOL_BACKEND", "cli").lower()
if BACKEND not in ("cli", "api"):
    raise SystemExit("SOL_BACKEND must be cli or api")
ANSWER_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}, "p": {"type": "array", "items": {"type": "number"}}}, "required": ["id", "p"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
os.makedirs(WORK, exist_ok=True)
if not os.path.exists(SCHEMA):
    json.dump(ANSWER_SCHEMA, open(SCHEMA, "w"))
CLOSED = "Answer from your own knowledge only: do not run commands, read files or search."


def call_claude(prompt, out_file, effort):
    """the same request through the Claude command-line client: no tools, no settings or memory, a one-line system prompt, the same JSON schema"""
    os.makedirs((_REPO + "/outputs/.cli_work/cw"), exist_ok=True)
    p = subprocess.run(["claude", "-p", "--model", MODEL, "--tools", "", "--system-prompt", "You are a helpful assistant.", "--effort", effort, "--output-format", "json", "--json-schema", open(SCHEMA).read(),
                        "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"], input=prompt, capture_output=True, text=True, timeout=1200, cwd=(_REPO + "/outputs/.cli_work/cw"))
    r = json.loads(p.stdout); assert not r.get("is_error"), str(r.get("result"))[:160]
    json.dump(r["structured_output"], open(out_file, "w")); u = r.get("usage", {}); st = u.get("server_tool_use", {})
    kinds = (["web"] if st.get("web_search_requests") or st.get("web_fetch_requests") else []) + (["denied_tool"] if r.get("permission_denials") else [])
    usage = {"input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0), "output_tokens": u.get("output_tokens", 0), "cost_usd": r.get("total_cost_usd"), "served_by": list(r.get("modelUsage", {}))}
    return kinds, usage, p.stderr[-200:]


def call_local(prompt, out_file, effort, _state={"n": 0}):
    """an open model behind a local OpenAI-compatible server (vLLM): SOL_MODEL=local:<port>:<served name>; same prompt and JSON schema"""
    import urllib.request
    _, port, served = MODEL.split(":"); _state["n"] += 1; temp = (0.0, 0.3, 0.6)[_state["n"] % 3] if os.environ.get("LOCAL_RETRY_TEMP") else 0.0
    body = {"model": served, "messages": [{"role": "user", "content": prompt}], "temperature": temp, "max_tokens": int(os.environ.get("LOCAL_MAX_TOKENS", "4500")),
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": json.load(open(SCHEMA))}}, "chat_template_kwargs": {"enable_thinking": False}}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(rq, timeout=900)); json.dump(json.loads(r["choices"][0]["message"]["content"]), open(out_file, "w"))
    return [], r.get("usage", {}), ""


def call_openai_api(prompt, out_file, effort, _client=[]):
    """the same request through the OpenAI Responses API: no tools, nothing stored"""
    from openai import OpenAI
    if not _client:
        _client.append(OpenAI(timeout=1200, max_retries=4))
    r = _client[0].responses.create(model=MODEL, input=prompt, reasoning={"effort": effort}, store=False, text={"format": {"type": "json_schema", "name": "answer", "schema": ANSWER_SCHEMA, "strict": True}})
    if r.status != "completed":
        raise RuntimeError(f"response status {r.status}")
    json.dump(json.loads(r.output_text), open(out_file, "w"))
    return [], {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}, ""


def call_claude_api(prompt, out_file, effort, _client=[]):
    """the same request through the Claude Messages API. No refusal fallback: an answer from another model would be a different simulator."""
    from anthropic import Anthropic
    if not _client:
        _client.append(Anthropic(timeout=1200, max_retries=4))
    m = _client[0].messages.create(model=MODEL, max_tokens=16000, system="You are a helpful assistant.", messages=[{"role": "user", "content": prompt}],
                                   output_config={"effort": effort, "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}})
    if m.stop_reason != "end_turn":
        raise RuntimeError(f"stop reason {m.stop_reason}")
    text = next(b.text for b in m.content if b.type == "text"); json.dump(json.loads(text), open(out_file, "w"))
    return [], {"input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens}, ""


def call(prompt, out_file, effort):
    if MODEL.startswith("local:"):
        return call_local(prompt, out_file, effort)
    if BACKEND == "api":
        return call_claude_api(prompt, out_file, effort) if MODEL.startswith("claude") else call_openai_api(prompt, out_file, effort)
    if MODEL.startswith("claude"):
        return call_claude(prompt, out_file, effort)
    p = subprocess.run(["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-C", WORK, "-s", "read-only", "-m", MODEL, "--disable", "shell_tool", "--disable", "unified_exec",
                        "--disable", "apps", "--disable", "browser_use", "--disable", "computer_use", "-c", 'web_search="disabled"', "-c", f'model_reasoning_effort="{effort}"',
                        "--output-schema", SCHEMA, "--json", "-o", out_file, "-"], input=prompt, capture_output=True, text=True, timeout=1200)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    kinds = [e.get("item", {}).get("type") for e in ev if e.get("type") == "item.completed"]; usage = next((e["usage"] for e in ev if e.get("type") == "turn.completed"), {})
    return kinds, usage, p.stderr[-200:]


def run(name, jobs, workers=6, effort="medium", check=None):
    out = f"{BASE}/{name}{os.environ.get('SOL_SUFFIX', '')}"; os.makedirs(out, exist_ok=True); keys = list(jobs)

    def one(k):
        f = f"{out}/{k}.json"
        if os.path.exists(f):
            return "cached"
        prompt, want = jobs[k]; err = ""
        for attempt in range(3):
            t0 = time.time(); o = f"{out}/tmp_{k}.json"
            try:
                kinds, usage, stderr = call(prompt, o, effort)
                r = json.load(open(o)); got = {int(x["id"]): x["p"] for x in r["items"]}
                assert set(got) == set(want), f"ids {sorted(set(want) ^ set(got))[:5]}"
                assert all(len(got[i]) == want[i] and min(got[i]) >= 0 for i in want), "lengths"
                if check:
                    check(got)
                tools = [x for x in kinds if x not in ("agent_message", "reasoning")]
                assert not tools, f"tool use {tools}"
                json.dump({"items": {str(i): got[i] for i in want}, "usage": usage, "seconds": time.time() - t0, "effort": effort, "model": MODEL}, open(f, "w")); os.remove(o)
                return f"ok {time.time() - t0:.0f}s"
            except Exception as ex:
                err = f"{type(ex).__name__}: {str(ex)[:160]}"
        return "FAILED " + err

    t0 = time.time(); done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, r in zip(keys, ex.map(one, keys)):
            done += 1
            if not r.startswith("cached") and (r.startswith("FAILED") or done % 10 == 0 or done == len(keys)):
                print(f"{name} {done}/{len(keys)} {k}: {r} ({time.time() - t0:.0f}s elapsed)", flush=True)


def collect(name):
    out = f"{BASE}/{name}"; P = {}; use = []
    for f in sorted(os.listdir(out)):
        if f.endswith(".json") and not f.startswith("tmp_") and f != "batches.json":
            d = json.load(open(f"{out}/{f}")); use.append(d["usage"])
            for i, p in d["items"].items():
                P[int(i)] = p
    return P, {"calls": len(use), "input_tokens": sum(u.get("input_tokens", 0) for u in use), "output_tokens": sum(u.get("output_tokens", 0) for u in use)}
