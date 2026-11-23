"""Direction B, step 1 (no new calls): how far does a real answer carry?  From every stored run in which a simulator was shown real
answers of the same report, the error on a target is related to how similar the closest shown question is.
  python v_transfer.py   -> results/v_transfer.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, os
import numpy as np
sys.path.insert(0, (_REPO + "/src"))
from w_groups import load, norm
from w_util import read, MODELS
from sklearn.feature_extraction.text import TfidfVectorizer
T = load(); N = len(T); SOL = (_REPO + "/outputs/sol"); waves = sorted({t["wave"] for t in T}); wave = np.array([t["wave"] for t in T])
th = [np.asarray(t["percent"], float) / sum(t["percent"]) for t in T]; V = np.array([(x * (1 - x)).sum() for x in th]); noise = V / np.array([t["n"] for t in T])
X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([t["title"] + " . " + t["question"] + " . " + " ; ".join(t["options"]) for t in T]); SIM = (X @ X.T).toarray()
opt = lambda i: tuple(norm(o) for o in T[i]["options"])


def halves(seed):
    r = np.random.default_rng(seed); half = {}
    for w in waves:
        ii = [i for i in range(N) if wave[i] == w]; perm = r.permutation(ii)
        for k, i in enumerate(perm):
            half[int(i)] = k % 2
    return half


def choose(w, method, k, seed=0):
    I = np.where(wave == w)[0]; kk = len(I) // 2 if k == "half" else int(k)
    if method == "random":
        return sorted(int(x) for x in np.random.default_rng(100 * seed + int(w[5:7]) * 100 + int(w[8:])).permutation(I)[:kk])
    S = []; cover = np.zeros(len(I))
    for _ in range(kk):
        gain = [(np.maximum(cover, SIM[I, j]).sum() - cover.sum()) if j not in S else -1 for j in I]; j = int(I[int(np.argmax(gain))]); S.append(j); cover = np.maximum(cover, SIM[I, j])
    return sorted(S)


def shown_sets():
    """run name -> {target: list of shown questions}"""
    R = {}; h11 = halves(11); h12 = halves(12)
    R["egctx"] = {i: [j for j in range(N) if wave[j] == wave[i] and h11[j] != h11[i]] for i in range(N)}
    R["egctxs12"] = {i: [j for j in range(N) if wave[j] == wave[i] and h12[j] != h12[i]] for i in range(N)}
    for K in (3, 10):
        d = {}
        for w in waves:
            for h in (0, 1):
                kn = [i for i in range(N) if wave[i] == w and h11[i] != h]; kn = sorted(int(x) for x in np.random.default_rng(13 + 7 * h + int(w[5:7]) * 100 + int(w[8:])).permutation(kn)[:K])
                for i in range(N):
                    if wave[i] == w and h11[i] == h:
                        d[i] = kn
        R[f"egctx{K}"] = d
    for method, k, seed in (("greedy", "half", 0), ("random", "half", 1), ("random", "half", 2), ("greedy", "10", 0), ("random", "10", 1), ("random", "10", 2)):
        d = {}
        for w in waves:
            S = choose(w, method, k, seed)
            for i in np.where(wave == w)[0]:
                if i not in S:
                    d[int(i)] = S
        R[f"egdesign-{method}{k}" + (f"s{seed}" if method == "random" else "")] = d
    return R


if __name__ == "__main__":
    RUNS = shown_sets(); rows = []
    for nm, tag in MODELS:
        A = read(f"{SOL}/egplain_{tag}")
        if len(A) < N:
            continue
        for run, d in RUNS.items():
            B = read(f"{SOL}/{run}_{tag}")
            if not all(i in B for i in d):
                continue
            for i, S in d.items():
                smax = max(SIM[i, j] for j in S); sib = any(opt(j) == opt(i) and norm(T[j]["text"]) == norm(T[i]["text"]) for j in S)
                rows.append((nm, run, i, float(smax), int(sib), len(S), float((A[i] - th[i]) @ (A[i] - th[i]) - noise[i]), float((B[i] - th[i]) @ (B[i] - th[i]) - noise[i])))
    import pandas as pd
    D = pd.DataFrame(rows, columns=["model", "run", "i", "smax", "sib", "k", "e0", "e1"]); print(len(D), "target-run-model rows;", D.run.nunique(), "runs;", D.model.nunique(), "models")
    strong = ["Claude-Fable-5.2", "GPT-6-astra", "GPT-6.1-sol", "GPT-6.1-terra", "Claude-Opus-5.5"]; D["grp"] = np.where(D.model.isin(strong), "strong", "other")
    bins = [0, .15, .25, .35, .5, .7, 1.01]; D["bin"] = pd.cut(D.smax, bins, right=False); res = {"bins": bins, "curve": {}}
    for g, sub in D.groupby("grp"):
        t = sub.groupby("bin", observed=True).agg(n=("e0", "size"), e0=("e0", "mean"), e1=("e1", "mean"), sib=("sib", "mean")); t["ratio"] = t.e1 / t.e0
        res["curve"][g] = {str(k): {"n": int(v.n), "error left": float(v.ratio), "share with a sibling": float(v.sib)} for k, v in t.iterrows()}
        print(f"\n{g}: error left after reading (shown / cold), by similarity of the closest shown question"); print(t[["n", "sib", "ratio"]].round(2).to_string())
    for m in strong + ["Claude-Haiku-5.5", "GPT-5.6-luna"]:
        sub = D[D.model == m]; t = sub.groupby("bin", observed=True).agg(e0=("e0", "mean"), e1=("e1", "mean")); print(f"{m:18s} " + " ".join(f"{b.left:.2f}-{b.right:.2f}: {r:.2f}" for b, r in zip(t.index, t.e1 / t.e0)))
    # does the number shown matter beyond the closest one?  within similarity bins, runs with k <= 10 against runs with k >= 20
    for g, sub in D[D.grp == "strong"].groupby(pd.cut(D[D.grp == "strong"].k, [0, 5, 12, 100])):
        t = sub.groupby("bin", observed=True).agg(e0=("e0", "mean"), e1=("e1", "mean")); print(f"strong, shown {g}: " + " ".join(f"{b.left:.2f}-{b.right:.2f}: {r:.2f}" for b, r in zip(t.index, t.e1 / t.e0)))
    D.to_pickle((_REPO + "/results/v_transfer_rows.pkl")); json.dump(res, open((_REPO + "/results/v_transfer.json"), "w"), indent=1)
