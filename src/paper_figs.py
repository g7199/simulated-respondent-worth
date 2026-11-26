"""The two main-text figures of the paper.
  generated/figs/models.pdf  (a) what each simulator is worth, (b) what a second simulator adds, (c) what sets are worth
  generated/figs/real.pdf    (a) more simulators or real answers, (b) groups by simulator, (c) groups by kind for the strongest simulator
Reads results/y_cov.json, results/y_cov_cache.pkl, results/w_groups.json and results/w_curve.json."""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, OUT = ROOT + "/results/", ROOT + "/generated/figs/"
BLACK, GRAY, LIGHT, GRID = "#000000", "#666666", "#c8c8c8", "#ececec"
BLUE, ORANGE, GREEN = "#2f6fb7", "#e0702f", "#16a07a"
COL = {"open": ORANGE, "GPT": BLUE, "Claude": GREEN}
FONT = "Liberation Serif" if any(f.name == "Liberation Serif" for f in font_manager.fontManager.ttflist) else "DejaVu Serif"
plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": BLACK, "axes.linewidth": 0.5, "axes.labelsize": 8, "axes.labelpad": 2, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "legend.frameon": False, "legend.fontsize": 7, "legend.handletextpad": 0.35, "legend.columnspacing": 0.9, "legend.labelspacing": 0.15, "legend.borderpad": 0, "legend.borderaxespad": 0, "legend.handlelength": 1.3,
})
C = json.load(open(R + "y_cov.json")); W = json.load(open(R + "w_groups.json")); ZC = pickle.load(open(R + "y_cov_cache.pkl", "rb"))
names = C["models"]; rc = C["rate_mse_ci"]; pr = C["polls"]["rate_mse_ci"]; order = sorted(names, key=lambda m: rc[m][0])


def fam(m):
    return "GPT" if m.startswith("GPT") else "Claude" if m.startswith("Claude") else "open"


def short(m):
    return m.replace("Claude-", "").replace("-Small", "")


def pair_gain(g):
    out = []
    for x in range(len(g)):
        for y in range(x + 1, len(g)):
            s, w = (x, y) if g[x, x] <= g[y, y] else (y, x)
            lam = min(1.0, max(0.0, (g[s, s] - g[s, w]) / (g[s, s] + g[w, w] - 2 * g[s, w])))
            risk = (1 - lam) ** 2 * g[s, s] + lam ** 2 * g[w, w] + 2 * lam * (1 - lam) * g[s, w]
            out.append((g[s, s] / g[w, w], g[s, s] / risk - 1))
    return np.array(out)


def panels(height, top, bottom):
    """three axes of exactly the same size"""
    fig = plt.figure(figsize=(6.75, height)); w = 0.232
    return fig, [fig.add_axes([x, bottom, w, top - bottom]) for x in (0.092, 0.426, 0.760)]


def title(a, text):
    a.text(0, 1.155, text, transform=a.transAxes, fontsize=8.5, ha="left", va="bottom")


def key(a, handles, labels, ncol=1):
    a.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.012), ncol=ncol)


def mark(marker, size=3.6, **kw):
    return Line2D([], [], ls="none", marker=marker, ms=size, **kw)


def rows(a, labels, lim, ticks):
    a.set_yticks(range(len(labels))); a.set_yticklabels(labels); a.set_ylim(-0.7, len(labels) - 0.3); a.tick_params(axis="y", length=0)
    a.set_xscale("log"); a.set_xlim(*lim); a.xaxis.set_major_locator(FixedLocator(ticks)); a.set_xticklabels([str(t) for t in ticks]); a.xaxis.set_minor_locator(NullLocator()); a.grid(True, axis="x", color=GRID, lw=0.5)


def save(fig, name):
    fig.savefig(OUT + name + ".pdf", metadata={"Creator": None, "Producer": None}); fig.savefig(OUT + name + ".png", dpi=300); plt.close(fig)


# ------------------------------------------------------------------ figure 1
fig, ax = panels(2.85, 0.835, 0.125)
a = ax[0]; y = 0; pos, labels = [], []
for f, name in (("open", "Open models"), ("GPT", "GPT"), ("Claude", "Claude")):
    for m in sorted([m for m in names if fam(m) == f], key=lambda m: rc[m][0]):
        col = COL[f]; r = rc[m]; pos.append(y); labels.append(short(m)); a.plot(r[1:], [y - 0.17] * 2, color=col, lw=0.7, solid_capstyle="butt", zorder=2); a.plot([r[0]], [y - 0.17], marker="o", ms=2.7, color=col, mew=0, zorder=3)
        if m in pr:
            p = pr[m]; a.plot(p[1:], [y + 0.19] * 2, color=col, lw=0.7, solid_capstyle="butt", zorder=2); a.plot([p[0]], [y + 0.19], marker="D", ms=2.3, mfc="white", mec=col, mew=0.6, zorder=3)
        y += 1
    a.text(-0.014, y - 0.05, name, transform=a.get_yaxis_transform(), fontsize=6.6, fontweight="bold", ha="right", va="center", clip_on=False)
    if f != "Claude":
        a.axhline(y + 0.72, color=LIGHT, lw=0.45, zorder=1)
    y += 1.45
rows(a, labels, (4.3, 70), [5, 10, 20, 40]); a.set_yticks(pos); a.set_yticklabels(labels); a.set_ylim(-0.7, y - 0.85)
a.tick_params(axis="y", labelsize=6.2); a.set_xlabel("worth (real respondents)")
key(a, [mark("o", 3, color=GRAY, mew=0), mark("D", 2.6, mfc="white", mec=GRAY, mew=0.6)], ["961 older survey questions", "250 new-poll questions"]); title(a, "(a) What each simulator is worth")

b = ax[1]; hh = ZC["h"]
Gs = np.array([[np.mean([(ZC["C"][x][k] - hh[k]) @ (ZC["C"][y][k] - hh[k]) for k in range(len(hh))]) for y in names] for x in names]); ps = pair_gain(Gs)
hp = ZC["polls"]["hp"]; pn = ZC["polls"]["models"]
Gp = np.array([[np.mean([(ZC["polls"]["PM"][x][i] - hp[i]) @ (ZC["polls"]["PM"][y][i] - hp[i]) for i in range(len(hp))]) for y in pn] for x in pn]); pp = pair_gain(Gp)
b.plot([0, 1], [0, 1], color=BLACK, lw=0.7, zorder=2); b.plot([0, 1], [0, 0], color=BLACK, lw=0.7, ls=(0, (3, 2)), zorder=2)
b.scatter(pp[:, 0], pp[:, 1], s=4.5, facecolors="white", edgecolors=BLUE, linewidths=0.4, zorder=3); b.scatter(ps[:, 0], ps[:, 1], s=4.5, color=BLUE, linewidths=0, zorder=4)
b.set_xlim(0, 1.03); b.set_ylim(-0.04, 1.03); b.set_xticks([0, .5, 1]); b.set_xticklabels(["0", "+50%", "+100%"]); b.set_yticks([0, .5, 1]); b.set_yticklabels(["0", "+50%", "+100%"]); b.grid(True, color=GRID, lw=0.5)
b.set_xlabel("gain if errors were unrelated"); b.set_ylabel("actual gain of the pair", labelpad=1)
b.text(0.47, 0.53, "unrelated errors", rotation=45, rotation_mode="anchor", transform_rotates_text=True, fontsize=6.8, ha="center", va="bottom"); b.text(0.98, 0.235, "same errors", fontsize=6.8, ha="right", va="center")
key(b, [mark("o", 2.8, color=BLUE, mew=0), mark("o", 2.7, mfc="white", mec=BLUE, mew=0.5)], [f"{len(ps)} pairs, older surveys", f"{len(pp)} pairs, new polls"]); title(b, "(b) What a second simulator adds")

c = ax[2]; S = C["sets"]; Pa = C["polls"]["all"]


def best(ms):
    return max(rc[m][0] for m in ms)


def count(f):
    return sum(fam(m) == f for m in names)


sets = [(f"{count('open')} open", S["open models"]["sum of rates"], S["open models"]["rate of the set"], best([m for m in names if fam(m) == "open"])),
        (f"{count('GPT')} GPT", S["seven GPT models"]["sum of rates"], S["seven GPT models"]["rate of the set"], best([m for m in names if fam(m) == "GPT"])),
        (f"{count('Claude')} Claude", S["Claude models"]["sum of rates"], S["Claude models"]["rate of the set"], best([m for m in names if fam(m) == "Claude"])),
        (f"all {len(names)}", S["all simulators"]["sum of rates"], S["all simulators"]["rate of the set"], best(names)),
        (f"all {len(C['polls']['models'])}, new polls", Pa["sum of rates"], Pa["rate of the set"], Pa["best single"])][::-1]
for y, (_, total, together, one) in enumerate(sets):
    c.barh(y + 0.18, total, height=0.32, color=LIGHT, lw=0, zorder=2); c.barh(y - 0.18, together, height=0.32, color=BLUE, lw=0, zorder=2); c.plot([one, one], [y - 0.38, y + 0.02], color=BLACK, lw=0.9, solid_capstyle="butt", zorder=4)
    c.text(total + 6, y + 0.18, f"{total:.0f}", va="center", fontsize=6.8); c.text(max(together, one) + 6, y - 0.18, f"{together:.0f}", va="center", fontsize=6.8)
c.set_yticks(range(len(sets))); c.set_yticklabels([s[0].replace(", new polls", ",\nnew polls") for s in sets]); c.tick_params(axis="y", length=0); c.set_ylim(-0.6, len(sets) - 0.4); c.set_xlim(0, 370); c.set_xticks([0, 100, 200, 300]); c.grid(True, axis="x", color=GRID, lw=0.5); c.set_xlabel("respondents")
key(c, [Rectangle((0, 0), 1, 1, color=LIGHT), Rectangle((0, 0), 1, 1, color=BLUE), Line2D([], [], color=BLACK, lw=0.9)], ["separately", "together", "best one alone"], ncol=2); title(c, "(c) What sets are worth")
save(fig, "models")

# ------------------------------------------------------------------ figure 2
CX = {m: v for m, v in W["context"].items() if isinstance(v, dict) and "all" in v}
GR = {m: v for m, v in W["groups"].items() if isinstance(v, dict) and "all" in v and "topline shown" in v["all"] and v["questions"] >= 220}
fig, ax = panels(2.55, 0.815, 0.14)


def link(a, xs, y, col):
    a.plot([min(xs), max(xs)], [y, y], color=col, alpha=0.45, lw=0.9, solid_capstyle="butt", zorder=2)


def cold(a, x, y, col):
    a.plot([x], [y], marker="o", ms=3.2, mfc="white", mec=col, mew=0.7, zorder=4)


COLD = mark("o", 3.2, mfc="white", mec=GRAY, mew=0.7); FILLED = mark("o", 3.2, color=GRAY, mew=0); SHOWN = mark("s", 3, color=BLACK, mew=0)
CV = json.load(open(R + "w_curve.json"))["curves"]; a = ax[0]; XP = [0, 1, 2, 3, 4]
for m, col in (("Claude-Fable-5.2", GREEN), ("GPT-6.1-sol", BLUE)):
    v = CV[m]; sim = v["more simulators"]; a.plot(XP, [sim[0], sim[1], sim[3], sim[10], sim[-1]], color=col, lw=0.8, ls=(0, (3, 2)), marker="o", ms=2.9, mfc="white", mec=col, mew=0.6, zorder=3)
    a.plot(XP, [v["cold"], *[v["real answers"][k]["mean"] for k in ("1", "3", "10")], v["the other half"]["mean"]], color=col, lw=1.0, marker="o", ms=3.2, mew=0, zorder=5)
    for x, d in zip((1, 2, 3, 4), [v["real answers"][k]["splits"] for k in ("1", "3", "10")] + [v["the other half"]["splits"]]):
        a.plot([x, x], [min(d), max(d)], color=col, alpha=0.55, lw=0.7, solid_capstyle="butt", zorder=4)
a.set_xticks(XP); a.set_xticklabels(["0", "1", "3", "10", "all"]); a.set_xlim(-0.25, 4.25); a.set_ylim(18, 62); a.set_yticks([20, 30, 40, 50, 60]); a.grid(True, axis="y", color=GRID, lw=0.5); a.set_xlabel("how many are added"); a.set_ylabel("worth (real respondents)", labelpad=1)
key(a, [Line2D([], [], color=BLACK, lw=1.0, marker="o", ms=3, mew=0), Line2D([], [], color=BLACK, lw=0.8, ls=(0, (3, 2)), marker="o", ms=2.8, mfc="white", mew=0.6), Line2D([], [], color=GREEN, lw=1.2), Line2D([], [], color=BLUE, lw=1.2)], ["real answers", "other simulators", "Fable-5.2", "GPT-6.1-sol"], ncol=2); title(a, "(a) More models or real answers?")

b = ax[1]; gm = sorted(GR, key=lambda m: GR[m]["all"]["persona"][0]); rows(b, [short(m) for m in gm], (5, 260), [10, 30, 100])
top = np.mean([GR[m]["all"]["topline"][0] for m in gm]); b.axvline(top, color=BLACK, lw=0.7, ls=(0, (3, 2)), zorder=3)
for y, m in enumerate(gm):
    col = COL[fam(m)]; v = GR[m]["all"]; xs = [v["persona"][0], v["topline + contrast"][0], v["topline shown"][0]]; link(b, xs, y, col); cold(b, xs[0], y, col); b.plot([xs[1]], [y], marker="o", ms=3.3, color=col, mew=0, zorder=5); b.plot([xs[2]], [y], marker="s", ms=3, color=BLACK, mew=0, zorder=6)
b.tick_params(axis="y", labelsize=6.5); b.set_xlabel("worth (respondents of the group)")
key(b, [COLD, FILLED, SHOWN, Line2D([], [], color=BLACK, lw=0.7, ls=(0, (3, 2)))], ["asked cold", "anchor", "shown the result", "result alone"], ncol=2); title(b, "(b) A group's answer")

c = ax[2]; bestm = max(GR, key=lambda m: GR[m]["all"]["topline shown"][0]); K = GR[bestm]["by kind"]; kinds = sorted(K, key=lambda k: K[k]["topline shown"]); lab = {}
rows(c, [lab.get(k, k) for k in kinds], (3, 900), [10, 100])
for y, k in enumerate(kinds):
    v = K[k]; col = COL[fam(bestm)]; link(c, [v["topline"], v["persona"], v["topline shown"]], y, col); c.plot([v["topline"]] * 2, [y - 0.26, y + 0.26], color=BLACK, lw=0.9, solid_capstyle="butt", zorder=3); cold(c, v["persona"], y, col); c.plot([v["topline shown"]], [y], marker="s", ms=3, color=BLACK, mew=0, zorder=6)
c.tick_params(axis="y", labelsize=6.5); c.set_xlabel("worth (respondents of the group)")
key(c, [COLD, SHOWN, Line2D([], [], color=BLACK, lw=0.9, marker="|", ms=4, ls="none", mew=0.9)], ["asked cold", "shown the result", "result alone"], ncol=2); title(c, f"(c) By kind of group ({short(bestm)})")
save(fig, "real")
