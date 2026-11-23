#!/usr/bin/env python
"""Converts the original public files in data/raw/ into the files the analysis reads.

Download the files listed in data/download_list.tsv into data/raw/, then run:  python scripts/prepare_data.py
Run it with --list to see which files are still missing. Nothing is downloaded by this script. Needs `pdftotext` (poppler-utils)."""
import argparse, hashlib, html, json, os, re, shutil, subprocess, sys, zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = REPO + "/data"; RAW = DATA + "/raw"; POLLS = DATA + "/polls_2026"
TOKEN = re.compile(r"(?<![\d.,/-])(?:<\s*1%?|\d{1,3}(?:\.\d+)?%?|\*|-|–)(?![\d/])")
GAP = re.compile(r"[\s­￾\x02]+")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_index(cache={}):
    """checksums of the files in data/raw/, so that a downloaded file can keep any name"""
    if not cache:
        for name in os.listdir(RAW):
            path = f"{RAW}/{name}"
            if os.path.isfile(path) and not name.endswith((".dta", ".part")):
                cache[sha256(path)] = path
    return cache


def wanted():
    S = json.load(open(DATA + "/sources.json")); recipe = json.load(open(POLLS + "/recipe.json"))
    files = [("SimBenchPop.csv", S["simbench"]["url"], S["simbench"]["sha256"]), ("GSS_stata.zip", S["gss"]["url"], S["gss"]["sha256"])]
    files += [(s["file"], s["url"], s["sha256"]) for s in recipe["sources"]]
    files += [(f"flourish_{c}.html", f"https://public.flourish.studio/visualisation/{c}/embed", None) for c in recipe["charts"]]
    files += [(page["file"], page["url"], None) for page in recipe["pages"]]
    return files


def locate(strict):
    found, missing, changed = {}, [], []
    index = raw_index()
    for name, url, digest in wanted():
        path = f"{RAW}/{name}"
        if digest and digest in index:
            found[name] = index[digest]
        elif os.path.exists(path):
            found[name] = path
            if digest:
                changed.append(name)
        else:
            missing.append((name, url))
    if missing:
        print(f"{len(missing)} files are missing from data/raw/:")
        for name, url in missing:
            print(f"  {name}\n    {url}")
        raise SystemExit("Download them (see README), then run this script again.")
    if changed:
        message = "these files differ from the versions used in the paper: " + ", ".join(changed)
        if strict:
            raise SystemExit(message + ". Use --allow-changed to continue.")
        print("WARNING:", message)
    return found


def number(token):
    token = token.strip()
    if token in ("", "*", "-", "–") or token.startswith("<"):
        return 0
    value = float(token.rstrip("%")); return int(value) if value.is_integer() else value


def pdf_pages(path, cache={}):
    if path not in cache:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(path); cache[path] = [doc[i].get_textpage().get_text_bounded() for i in range(len(doc))]
    return cache[path]


def pdf_numbers(path, cache={}):
    if path not in cache:
        cache[path] = [[number(t) for t in TOKEN.findall(page)] for page in pdf_pages(path)]
    return cache[path]


def pdf_text(path, cache={}):
    if path not in cache:
        cache[path] = GAP.sub("", "".join(pdf_pages(path)))
    return cache[path]


def chart_json(path, cache={}):
    if path not in cache:
        text = open(path, encoding="utf-8", errors="ignore").read(); out = {}
        for var in ("_Flourish_data_column_names", "_Flourish_data", "_Flourish_state"):
            match = re.search(var + r"\s*=\s*", text)
            if match:
                out[var], _ = json.JSONDecoder().raw_decode(text[match.end():])
        cache[path] = out
    return cache[path]


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


def chart_text(path, cache={}):
    if path not in cache:
        cache[path] = GAP.sub("", "".join(strings(chart_json(path))))
    return cache[path]


def page_text(path, cache={}):
    if path not in cache:
        text = open(path, encoding="utf-8", errors="ignore").read(); text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text); start = text.find("entry-content")
        cache[path] = GAP.sub("", html.unescape(re.sub(r"<[^>]+>", "", text[max(start, 0):])))
    return cache[path]


def text_of(name, path):
    return chart_text(path) if name.startswith("flourish_") else page_text(path) if name.endswith(".html") else pdf_text(path)


def chart_series(path):
    return {row["label"]: [number(str(v)) for v in row["value"]] for row in chart_json(path)["_Flourish_data"]["data"]}


def read_values(where, files):
    if "literal" in where:
        return where["literal"]
    if "chart" in where:
        series = chart_series(files[f"flourish_{where['chart']}.html"])
        if "rows" in where:
            return [series[label][where["column"]] for label in where["rows"]]
        values = series[where["series"]]; return [values[i] for i in where["index"]]
    pages = pdf_numbers(files[where["source"]]); return [pages[p][i] for p, i in where["at"]]


def read_text(where, files):
    """a piece of text is stored as positions in the source document, plus the spaces that the extraction removes"""
    sources = [text_of(name, files[name]) for name in where["from"]]
    text = "".join(sources[p[0]][p[1]:p[1] + p[2]] if isinstance(p, list) else p for p in where["parts"])
    for position, gap in reversed(where["gaps"]):
        text = text[:position] + gap + text[position:]
    return text


def build_polls(files):
    recipe = json.load(open(POLLS + "/recipe.json")); items = []
    for r in recipe["items"]:
        item = {k: v for k, v in r.items() if not k.endswith("_from")}; item["question"] = read_text(r["question_from"], files); item["options"] = [read_text(o, files) for o in r["options_from"]]
        item["percent"] = read_values(r["percent_from"], files)
        if r.get("by_party_from") is not None:
            item["by_party"] = {g: read_values(w, files) for g, w in r["by_party_from"].items()}
        items.append(item)
    json.dump(items, open(POLLS + "/items.json", "w"), indent=0)
    pdf_dir = POLLS + "/yougov_pdf"; os.makedirs(pdf_dir, exist_ok=True)
    for wave, name in recipe["economist_waves"].items():
        pdf = f"{pdf_dir}/econ_{wave}.pdf"; shutil.copyfile(files[name], pdf); subprocess.run(["pdftotext", "-layout", pdf, pdf[:-4] + ".txt"], check=True)
    for script, step in (("w_groups.py", "parse"), ("y_econ.py", "parse"), ("y_econ.py", "targets")):
        subprocess.run([sys.executable, script, step], cwd=REPO + "/src", check=True, stdout=subprocess.DEVNULL)
    fresh_dir = POLLS + "/fresh_pdf"; os.makedirs(fresh_dir, exist_ok=True)               # two later weekly reports, held out for the replication
    for wave, name in recipe["economist_fresh"].items():
        pdf = f"{fresh_dir}/econ_{wave}.pdf"; shutil.copyfile(files[name], pdf); subprocess.run(["pdftotext", "-layout", pdf, pdf[:-4] + ".txt"], check=True)
    subprocess.run([sys.executable, "w_groups.py", "parse"], cwd=REPO + "/src", check=True, stdout=subprocess.DEVNULL, env={**os.environ, "W_SET": "fresh"})
    print("polls:", len(items), "questions")


def build_gss(source, path):
    import pyreadstat
    os.makedirs(DATA + "/gss", exist_ok=True); dta = RAW + "/" + source["member"]
    if not os.path.exists(dta):
        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist() if n.endswith(source["member"]))
            with z.open(member) as r, open(dta, "wb") as w:
                shutil.copyfileobj(r, w)
    frame, meta = pyreadstat.read_dta(dta, usecols=source["columns"], apply_value_formats=False, encoding="latin1"); frame = frame[source["columns"]]; frame.to_parquet(DATA + "/gss/gss_sub.parquet")
    json.dump({c: {str(k): v for k, v in meta.variable_value_labels.get(c, {}).items()} for c in source["columns"]}, open(DATA + "/gss/labels.json", "w"))
    print("GSS:", frame.shape)


def build_question_index():
    os.makedirs(REPO + "/outputs/screen", exist_ok=True)
    code = "import json, sys; out = sys.argv[1]; sys.argv = ['x', 'none']; from t4_confirm import build; json.dump(sorted(build().q.unique()), open(out, 'w'))"
    subprocess.run([sys.executable, "-c", code, REPO + "/outputs/screen/t4c_embq.json"], cwd=REPO + "/src", check=True)


def canonical(path):
    if path.endswith(".json"):
        return hashlib.sha256(json.dumps(json.load(open(path)), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if path.endswith(".parquet"):
        import pandas as pd
        frame = pd.read_parquet(path); frame = frame[sorted(frame.columns)]; return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    return sha256(path)


def check(strict):
    want = json.load(open(DATA + "/sources.json"))["built"]; bad = [rel for rel, digest in want.items() if not os.path.exists(f"{REPO}/{rel}") or canonical(f"{REPO}/{rel}") != digest]
    if bad:
        message = "these files differ from the versions used in the paper: " + ", ".join(bad)
        if strict:
            raise SystemExit(message)
        print("WARNING:", message)
    else:
        print("all", len(want), "built files match the versions used in the paper")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); parser.add_argument("--allow-changed", action="store_true", help="continue if a source file has changed since the paper"); parser.add_argument("--list", action="store_true", help="show which source files are present or missing"); args = parser.parse_args(); strict = not args.allow_changed
    os.makedirs(RAW, exist_ok=True)
    if args.list:
        index = raw_index(); rows = [(n, u, (d in index) if d else os.path.exists(f"{RAW}/{n}")) for n, u, d in wanted()]
        for n, u, ok in rows:
            print("ok     " if ok else "MISSING", n, "" if ok else u)
        print(sum(r[2] for r in rows), "of", len(rows), "files present"); return
    files = locate(strict); S = json.load(open(DATA + "/sources.json")); os.makedirs(DATA + "/simbench", exist_ok=True)
    shutil.copyfile(files["SimBenchPop.csv"], DATA + "/simbench/SimBenchPop.csv")
    build_gss(S["gss"], files["GSS_stata.zip"]); build_polls(files); build_question_index(); check(strict); print("data/ is ready")


if __name__ == "__main__":
    main()
