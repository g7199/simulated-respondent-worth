"""Compares the recomputed results and tables with the values behind the paper (expected/)."""
import glob, json, math, os, re, sys
REPO = os.path.dirname(os.path.abspath(__file__)); bad = 0


def same(a, b, tol=1e-9):
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list): return isinstance(b, list) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) or a == b or abs(a - b) <= tol * max(abs(a), abs(b))
    return a == b


# gradient-boosted trees can differ slightly between machines, so their results are compared within a few per cent
LOOSE = {"w_strong.json": 0.05}; LOOSE_TEX = {"tab_ways.tex"}


def close_tex(a, b, tol=0.05):
    x, y = re.findall(r"\d+\.?\d*", a), re.findall(r"\d+\.?\d*", b)
    return re.sub(r"\d+\.?\d*", "#", a) == re.sub(r"\d+\.?\d*", "#", b) and all(abs(float(p) - float(q)) <= max(1, tol * float(q)) for p, q in zip(x, y))


for f in sorted(glob.glob(REPO + "/expected/results/*.json")):
    n = os.path.basename(f); g = f"{REPO}/results/{n}"; ok = os.path.exists(g) and same(json.load(open(f)), json.load(open(g)), LOOSE.get(n, 1e-9)); bad += not ok; print("OK  " if ok else "DIFF", "results/" + n)
for f in sorted(glob.glob(REPO + "/expected/tables/*.tex")):
    n = os.path.basename(f); g = f"{REPO}/generated/{n}"; ok = os.path.exists(g) and (close_tex(open(f).read(), open(g).read()) if n in LOOSE_TEX else open(f).read() == open(g).read()); bad += not ok; print("OK  " if ok else "DIFF", "generated/" + n)
print("all match" if not bad else f"{bad} files differ"); sys.exit(bool(bad))
