"""Shared by the w_* scripts: the list of simulators with their output tags and the reader of stored answers."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, json
import numpy as np
MODELS = [("Qwen3.5-9B", "qwen9b-stated"), ("Mistral-Small-24B", "mistral24b-stated"), ("Qwen3.8-27B", "qwen27b-stated"), ("GPT-5.6-luna", "gpt-5.6-luna"), ("GPT-6-luna", "gpt-6-luna"), ("GPT-5.6-sol", "gpt-5.6-sol"), ("GPT-6-sol", "gpt-6-sol"),
          ("GPT-6.1-terra", "gpt-6.1-terra"), ("GPT-6.1-sol", "gpt-6.1-sol"), ("GPT-6-astra", "gpt-6-astra"), ("Claude-Haiku-5.5", "claude-haiku-5-5"), ("Claude-Sonnet-5.5", "claude-sonnet-5-5"), ("Claude-Opus-5.5", "claude-opus-5-5"), ("Claude-Fable-5.2", "claude-fable-5-2")]


def read(dn):
    P = {}
    if os.path.isdir(dn):
        for f in sorted(os.listdir(dn)):
            if f.endswith(".json") and not f.startswith("tmp_") and f != "batches.json":
                for i, p in json.load(open(f"{dn}/{f}"))["items"].items():
                    p = np.asarray(p, float); P[int(i)] = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
    return P
