"""Figures and tables of the paper "Simulated People Share Their Errors"
-> paper3/figs/{ladder,sets}.pdf, paper3/tab_ladder.tex, paper3/tab_prog.tex, paper3/tab_nested.tex
Reads results/y_cov.json, z_nest.json, z_groups.json."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
ROOT = (_REPO + ""); R = ROOT + "/results/"; PAPER = os.environ.get("PAPER", "generated")
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"; BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42})
C = json.load(open(R + "y_cov.json")); Zn = json.load(open(R + "z_nest.json")); Zg = json.load(open(R + "z_groups.json"))
names = C["models"]; rc = C["rate_mse_ci"]; order = sorted(names, key=lambda m: rc[m][0]); pr = C["polls"]["rate_mse_ci"]
fam = lambda m: "GPT" if m.startswith("GPT") else "Claude" if m.startswith("Claude") else "open"; COL = {"open": ORANGE, "GPT": BLUE, "Claude": GREEN}


def clean(a):
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)


# ------------------------------------------------------------------ figure 1: the ladder and the nested law
fig, ax = plt.subplots(1, 2, figsize=(6.75, 3.15), gridspec_kw={"wspace": 0.42, "width_ratios": [1.05, 1]})
a = ax[0]
for y, m in enumerate(order):
    col = COL[fam(m)]; r = rc[m]; a.plot(r[1:], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2); a.plot([r[0]], [y], marker="o", ms=5.2, color=col, mec="white", mew=0.8, zorder=3)
    if m in pr:
        p = pr[m]; a.plot(p[1:], [y + 0.32, y + 0.32], color=col, lw=1.0, alpha=0.7, zorder=2); a.plot([p[0]], [y + 0.32], marker="D", ms=4.3, mfc="white", mec=col, mew=1.2, zorder=3)
a.set_yticks(range(len(order))); a.set_yticklabels(order, fontsize=6.5); a.set_xscale("log"); a.set_xticks([5, 10, 20, 40]); a.set_xticklabels(["5", "10", "20", "40"]); a.minorticks_off(); a.set_xlim(4.2, 66); a.set_ylim(-0.6, len(order) - 0.1)
a.set_xlabel("exchange rate (real respondents)"); clean(a); a.grid(False, axis="y")
a.legend([Line2D([], [], ls="none", marker="o", ms=5.2, color=MUTED, mec="white", mew=0.8), Line2D([], [], ls="none", marker="D", ms=4.3, mfc="white", mec=MUTED, mew=1.2)], ["961 survey questions", "250 polls fielded after training"],
         frameon=False, fontsize=6.5, loc="lower right", handletextpad=0.3, borderaxespad=0.2, labelcolor=INK)
a.set_title("a  What each simulator is worth", fontsize=8.5, loc="left", color=INK)
b = ax[1]; b.plot([0, 1], [0, 1], color=INK, lw=0.9, zorder=1); b.plot([0, 1], [0, 0], color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
import pickle
ZC = pickle.load(open(R + "y_cov_cache.pkl", "rb"))


def pair_rows(g, labels):
    out_ = []
    for x in range(len(g)):
        for y in range(x + 1, len(g)):
            s_, w_ = (x, y) if g[x, x] <= g[y, y] else (y, x); lam = min(1.0, max(0.0, (g[s_, s_] - g[s_, w_]) / (g[s_, s_] + g[w_, w_] - 2 * g[s_, w_])))
            risk = (1 - lam) ** 2 * g[s_, s_] + lam ** 2 * g[w_, w_] + 2 * lam * (1 - lam) * g[s_, w_]
            out_.append({"pair": [labels[s_], labels[w_]], "x": g[s_, s_] / g[w_, w_], "gain": g[s_, s_] / risk - 1, "corr": g[s_, w_] / np.sqrt(g[s_, s_] * g[w_, w_]), "ratio": g[s_, w_] / g[s_, s_]})
    return out_
pair_gain = lambda g: np.array([[r_["x"], r_["gain"]] for r_ in pair_rows(g, list(range(len(g))))])
hh = ZC["h"]; Gs = np.array([[np.mean([(ZC["C"][x][k] - hh[k]) @ (ZC["C"][y][k] - hh[k]) for k in range(len(hh))]) for y in names] for x in names]); ps_ = pair_gain(Gs)
hp_ = ZC["polls"]["hp"]; pn_ = ZC["polls"]["models"]; Gp_ = np.array([[np.mean([(ZC["polls"]["PM"][x][i] - hp_[i]) @ (ZC["polls"]["PM"][y][i] - hp_[i]) for i in range(len(hp_))]) for y in pn_] for x in pn_]); pp_ = pair_gain(Gp_)
b.scatter(pp_[:, 0], pp_[:, 1], s=13, facecolors="white", edgecolors=BLUE, linewidths=0.8, zorder=3); b.scatter(ps_[:, 0], ps_[:, 1], s=13, color=BLUE, edgecolors="white", linewidths=0.4, zorder=4)
rl = []
for m, v in C["mixed sets"].items():
    r0 = v["rates"]; rr = list(r0.values())
    for k_, j_ in (("simulator + same-scale", 1), ("simulator + another population", 2)):
        hi, lo = max(rr[0], rr[j_]), min(rr[0], rr[j_]); rl.append((lo / hi, v[k_]["rate of the set"] / hi - 1))
rl = np.array(rl); b.scatter(rl[:, 0], rl[:, 1], s=16, marker="x", color=ORANGE, linewidths=1.0, zorder=4)
ce = Zn["ceiling"]; st = [(min(ce["rates"]["main"], ce["rates"]["repeat"]) / max(ce["rates"]["main"], ce["rates"]["repeat"]), ce["rate of the pair of runs"] / max(ce["rates"]["main"], ce["rates"]["repeat"]) - 1)]
for v in Zn.get("ceiling more", {}).values():
    r1, r2 = v["rates of the two runs"]; st.append((min(r1, r2) / max(r1, r2), v["rate of the mean of two runs"] / max(r1, r2) - 1))
st = np.array(st); b.scatter(st[:, 0], np.clip(st[:, 1], 0, None), s=60, marker="*", color=INK, edgecolors="white", linewidths=0.5, zorder=5)
ZW = json.load(open(R + "z_wave.json")); rw = ZW["population"]["earlier wave"][0]; wv = np.array([(min(v["rate"], rw) / max(v["rate"], rw), v["pair with the earlier wave"][0] / max(v["rate"], rw) - 1) for v in ZW["population"]["simulators"].values()])
b.scatter(wv[:, 0], wv[:, 1], s=26, marker="^", color=GREEN, edgecolors="white", linewidths=0.5, zorder=5)
b.set_xlim(0, 1.03); b.set_ylim(-0.04, 1.42); clean(b)
b.set_xlabel("rate of the weaker source / rate of the stronger"); b.set_ylabel("gain of the pair over the stronger source")
b.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); b.set_yticklabels(["0", "+25%", "+50%", "+75%", "+100%"])
b.legend([Line2D([], [], color=INK, lw=0.9), Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2))), Line2D([], [], ls="none", marker="o", ms=4.2, color=BLUE, mec="white", mew=0.4), Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=BLUE, mew=0.8),
          Line2D([], [], ls="none", marker="*", ms=7, color=INK, mec="white", mew=0.5), Line2D([], [], ls="none", marker="x", ms=4.5, color=ORANGE, mew=1.0), Line2D([], [], ls="none", marker="^", ms=5, color=GREEN, mec="white", mew=0.5)],
         ["independent errors: rates add", "nested errors: nothing added", "two simulators, surveys", "two simulators, new polls", "one simulator, run twice", "a simulator and other questions\nor another population", "a simulator and an earlier\nwave of the question"],
         frameon=False, fontsize=6.0, loc="upper left", ncol=2, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1, labelcolor=INK, labelspacing=0.35)
b.set_title("b  What a second source adds", fontsize=8.5, loc="left", color=INK)
fig.savefig(ROOT + "/" + PAPER + "/figs/ladder.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(ROOT + "/" + PAPER + "/figs/ladder.png", dpi=170, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------------------------ figure 2: sets, and groups
fig, ax = plt.subplots(1, 2, figsize=(6.75, 2.75), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1.05]})
a = ax[0]; S = C["sets"]; best = lambda ms: max(rc[m][0] for m in ms); An = C["annotators"]; Pa = C["polls"]["all"]
rows = [(f"{sum(fam(m) == chr(111) + chr(112) + chr(101) + chr(110) for m in names)} open models", S["open models"]["sum of rates"], S["open models"]["rate of the set"], best([m for m in names if fam(m) == "open"])),
        ("7 GPT models", S["seven GPT models"]["sum of rates"], S["seven GPT models"]["rate of the set"], best([m for m in names if fam(m) == "GPT"])),
        ("4 Claude models", S["Claude models"]["sum of rates"], S["Claude models"]["rate of the set"], best([m for m in names if fam(m) == "Claude"])),
        (f"all {len(names)} simulators", S["all simulators"]["sum of rates"], S["all simulators"]["rate of the set"], best(names)),
        (f"{len(C['polls']['models'])} simulators,\nnew polls", Pa["sum of rates"], Pa["rate of the set"], Pa["best single"]),
        (f"{len(An['models'])} simulators,\nannotators", An["sum of rates"], An["rate of the set"], An["best single"])]
for y, (nm, sm, st, bs) in enumerate(rows[::-1]):
    a.barh(y + 0.19, sm, height=0.34, color="#d6dbe1", zorder=2); a.barh(y - 0.19, st, height=0.34, color=BLUE, zorder=2); a.plot([bs, bs], [y - 0.42, y + 0.04], color=INK, lw=1.4, zorder=4)
    a.text(sm + 4, y + 0.19, f"{sm:.0f}", va="center", fontsize=6.3, color=MUTED); a.text(st + 4, y - 0.19, f"{st:.0f}", va="center", fontsize=6.3, color=INK)
a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.5); a.set_xlim(0, 330); a.set_xlabel("respondents"); clean(a); a.grid(False, axis="y")
a.legend([plt.Rectangle((0, 0), 1, 1, color="#d6dbe1"), plt.Rectangle((0, 0), 1, 1, color=BLUE), Line2D([], [], color=INK, lw=1.4)], ["sum of their rates", "what the set is worth", "best single simulator"], frameon=False, fontsize=6.3, loc="upper right", borderaxespad=0.0, labelcolor=INK)
a.set_title("a  Simulators do not add up", fontsize=8.5, loc="left", color=INK)
b = ax[1]; PM = Zg["party"]["models"]; pm = sorted(PM, key=lambda m: PM[m]["persona"]["rate"]); top = PM["GPT-6-astra"]["topline"]["rate"]
b.axvline(top, color=INK, lw=1.1, ls=(0, (4, 2)), zorder=2)
for y, m in enumerate(pm):
    v = PM[m]; xs_ = [v["persona"]["rate"], v["contrast"]["rate"]] + ([v["given"]["rate"]] if "given" in v else [])
    b.plot([min(xs_), max(xs_)], [y, y], color="#c9d2dc", lw=1.2, zorder=2)
    b.plot([v["persona"]["rate"]], [y], marker="o", ms=5, mfc="white", mec=BLUE, mew=1.2, zorder=4); b.plot([v["contrast"]["rate"]], [y], marker="o", ms=5, color=BLUE, mec="white", mew=0.7, zorder=4)
    if "given" in v:
        b.plot([v["given"]["rate"]], [y], marker="s", ms=4.6, color=INK, mec="white", mew=0.7, zorder=3)
b.set_yticks(range(len(pm))); b.set_yticklabels(pm, fontsize=6.5); b.set_xscale("log"); b.set_xticks([5, 10, 20, 50, 100]); b.set_xticklabels(["5", "10", "20", "50", "100"]); b.minorticks_off(); b.set_xlim(3, 190); b.set_ylim(-0.6, len(pm) + 2.2)
b.set_xlabel("respondents of the group"); clean(b); b.grid(False, axis="y")
b.legend([Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 2))), Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=BLUE, mew=1.2), Line2D([], [], ls="none", marker="o", ms=5, color=BLUE, mec="white", mew=0.7), Line2D([], [], ls="none", marker="s", ms=4.6, color=INK, mec="white", mew=0.7)],
         ["the poll's topline alone", "persona alone", "topline + persona's contrast", "persona, shown the topline"], frameon=False, fontsize=6.3, loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.1, labelcolor=INK)
b.set_title("b  The shared error cancels against real data", fontsize=8.5, loc="left", color=INK)
fig.savefig(ROOT + "/" + PAPER + "/figs/sets.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(ROOT + "/" + PAPER + "/figs/sets.png", dpi=170, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------------------------ table 1
ci = lambda r, d=1: f"{r[0]:.{d}f} {{\\scriptsize[{r[1]:.0f}, {r[2]:.0f}]}}"
GM = Zg["gss"]["models"]; g = lambda v, k: f"{v[k]['rate']:.0f}" if v and k in v else "--"
out = ["\\begin{table*}[t]", "\\centering\\small", "\\setlength{\\tabcolsep}{3.2pt}", "\\begin{tabular}{lcccccccc}", "\\toprule",
       " & \\multicolumn{2}{c}{Population (respondents)} & & \\multicolumn{3}{c}{Party groups, new polls} & \\multicolumn{2}{c}{20 groups, GSS} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
       "Source & 961 older questions & 250 new polls & & cold & +difference & shown & cold & +difference \\\\", "\\midrule",
       f"\\emph{{Uniform answer}} & {Zn['anchor']['uniform']:.1f} & {Zn['anchor']['new polls by subset']['all']['uniform']:.1f} & & & & & & \\\\",
       f"\\emph{{Same-scale answers}} & {C['real']['same-scale answers']['rate_mse']:.1f} & -- & & & & & & \\\\", f"\\emph{{Another population}} & {C['real']['another population']['rate_mse']:.1f} & -- & & & & & & \\\\",
       f"\\emph{{The overall result alone}} & & & & \\multicolumn{{3}}{{c}}{{{top:.0f} {{\\scriptsize[{PM['GPT-6-astra']['topline']['interval'][0]:.0f}, {PM['GPT-6-astra']['topline']['interval'][1]:.0f}]}}}} & \\multicolumn{{2}}{{c}}{{{GM['Qwen3.5-9B']['topline']['rate']:.0f} {{\\scriptsize[{GM['Qwen3.5-9B']['topline']['interval'][0]:.0f}, {GM['Qwen3.5-9B']['topline']['interval'][1]:.0f}]}}}} \\\\", "\\midrule"]
for m in order:
    v = PM.get(m); w = GM.get(m)
    out.append(f"{m} & {ci(rc[m])} & {ci(pr[m]) if m in pr else '--'} & & {g(v, 'persona')}{'$^\\dagger$' if m == 'Qwen3.5-9B' else ''} & {g(v, 'contrast')}{'$^\\dagger$' if m == 'Qwen3.5-9B' else ''} & {g(v, 'given')}{'$^\\dagger$' if m == 'Qwen3.5-9B' else ''} & {g(w, 'persona')} & {g(w, 'contrast')} \\\\")
out += ["\\bottomrule", "\\end{tabular}",
        "\\caption{What each source is worth, in real respondents (95\\% intervals from a bootstrap over question texts, and over poll releases for the new polls). Left: for a population's answer, on the 961 older survey questions and the 250 new polls. "
        "Right: for a group's answer on two further data sets (Appendix~\\ref{app:groups}), in respondents of the group: the simulator asked for the group cold, its difference between the group and everyone added to the real overall result, and the simulator shown the overall result. The GSS is older than the models. "
        "Stated percentages are used as given; the three smallest open models are read through rescaled next-token probabilities, except $^\\dagger$Qwen3.5-9B on the party groups (stated percentages).}", "\\label{tab:ladder}", "\\end{table*}"]
open(ROOT + "/" + PAPER + "/tab_ladder.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out[9:-5]))

# ------------------------------------------------------------------ appendix: rates by programme and for subsets of the new polls
A = Zn["anchor"]; progs = list(A["uniform by programme"]); short = {"GSS": "GSS", "OpinionQA": "OpinionQA", "ESS": "ESS", "Afrobarometer": "Afrobar.", "ISSP": "ISSP", "LatinoBarometro": "Latinobar.", "GlobalOpinionQA": "GlobalOQA"}
sub = ["United States", "other countries", "not a tracker"]; ps = A["new polls by subset"]
out = ["\\begin{table}[h]", "\\centering\\small", "\\setlength{\\tabcolsep}{3.6pt}", "\\begin{tabular}{l" + "c" * (len(progs) + len(sub) + 1) + "}", "\\toprule",
       f" & \\multicolumn{{{len(progs)}}}{{c}}{{Survey programme}} & all, re- & \\multicolumn{{{len(sub)}}}{{c}}{{New polls}} \\\\", f"\\cmidrule(lr){{2-{len(progs) + 1}}}\\cmidrule(lr){{{len(progs) + 3}-{len(progs) + len(sub) + 2}}}",
       "Source & " + " & ".join(short.get(p, p) for p in progs) + " & scaled & U.S. & other & no tracker \\\\", "\\midrule",
       "\\emph{Items} & " + " & ".join(str(A["items by programme"][p]) for p in progs) + " & & " + " & ".join(str(ps[s]["items"]) for s in sub) + " \\\\",
       "\\emph{Uniform answer} & " + " & ".join(f"{A['uniform by programme'][p]:.1f}" for p in progs) + f" & {A['uniform']:.1f} & " + " & ".join(f"{ps[s]['uniform']:.1f}" for s in sub) + " \\\\", "\\midrule"]
for m in order:
    out.append(f"{m} & " + " & ".join(f"{A['rates by programme'][m][p]:.0f}" for p in progs) + f" & {Zn['calibrated variant']['rates'][m]:.1f} & " + " & ".join(f"{ps[s][m]:.0f}" if m in ps[s] else "--" for s in sub) + " \\\\")
out += ["\\bottomrule", "\\end{tabular}", "\\caption{What each simulator is worth by survey programme (answers as given; rescaled probabilities for the three smallest open models), and on subsets of the 250 new polls. The column ``all, rescaled'' gives its worth on all 961 questions when its answers are rescaled with two constants fitted on the other programmes; for the three smallest open models, whose raw probabilities are worth 1.8, 3.9 and 1.3 respondents, this is the read-out used throughout.}", "\\label{tab:prog}", "\\end{table}"]
open(ROOT + "/" + PAPER + "/tab_prog.tex", "w").write("\n".join(out) + "\n")

# ------------------------------------------------------------------ appendix: pairs of simulators by kind
RS = pair_rows(Gs, names); RP = pair_rows(Gp_, pn_)
def block(rows, a_, b_):
    rr = [x for x in rows if sorted([fam(x["pair"][0]), fam(x["pair"][1])]) == sorted([a_, b_])] if a_ else rows
    if not rr:
        return None
    o = np.array([x["corr"] for x in rr]); q = np.array([x["ratio"] for x in rr]); gn = np.array([x["gain"] for x in rr])
    return f"{len(rr)} & {o.min():.2f}--{o.max():.2f} & {np.median(q):.2f} & {100 * np.median(gn):.1f}\\% & {100 * gn.max():.1f}\\%"
out = ["\\begin{table}[h]", "\\centering\\small", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{llccccc}", "\\toprule",
       " & & & error & $G_{ij}/\\min(G_{ii},G_{jj})$ & \\multicolumn{2}{c}{gain of the pair} \\\\", " & Pairs of simulators & pairs & correlation & median & median & largest \\\\", "\\midrule"]
for rows_, lab in ((RS, "Surveys"), (RP, "New polls")):
    first = True
    for a_, b_, nm in (("open", "open", "open, open"), ("GPT", "GPT", "GPT, GPT"), ("Claude", "Claude", "Claude, Claude"), ("open", "GPT", "open, GPT"), ("open", "Claude", "open, Claude"), ("GPT", "Claude", "GPT, Claude"), (None, None, "all")):
        s_ = block(rows_, a_, b_)
        if s_:
            out.append(f"{lab if first else ''} & {nm} & {s_} \\\\"); first = False
    if lab == "Surveys":
        out.append("\\midrule")
ns_ = Zn["nested"]["surveys"]["G_ij / min(G_ii, G_jj): pooled"]; np_ = Zn["nested"]["new polls"]["G_ij / min(G_ii, G_jj): pooled"]; nc_ = Zn["calibrated variant"]["G_ij / min(G_ii, G_jj): pooled"]
out += ["\\bottomrule", "\\end{tabular}", "\\caption{Pairs of simulators by kind: range of the correlations of their errors, the median ratio $G_{ij}/\\min(G_{ii},G_{jj})$ (nested: 1), and what the pair is worth over its stronger member. "
        f"Pooled over all pairs the ratio is {ns_[0]:.2f} [{ns_[1]:.2f}, {ns_[2]:.2f}] on the surveys and {np_[0]:.2f} [{np_[1]:.2f}, {np_[2]:.2f}] on the new polls; with every simulator rescaled on the other programmes it is {nc_[0]:.2f} [{nc_[1]:.2f}, {nc_[2]:.2f}] on the surveys.}}", "\\label{tab:nested}", "\\end{table}"]
open(ROOT + "/" + PAPER + "/tab_nested.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out[8:-5]))

# ------------------------------------------------------------------ appendix: groups with intervals
cell = lambda v, k: (f"{v[k]['rate']:.0f} {{\\scriptsize[{v[k]['interval'][0]:.0f}, {v[k]['interval'][1]:.0f}]}}" if v and k in v else "--")
SEP = Zg["party"].get("separate calls", {})
out = ["\\begin{table}[h]", "\\centering\\small", "\\setlength{\\tabcolsep}{3.2pt}", "\\begin{tabular}{lcccccccc}", "\\toprule",
       " & \\multicolumn{6}{c}{Party groups, new polls} & \\multicolumn{2}{c}{20 groups, GSS} \\\\", "\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}",
       "Simulator & cold & pooled & +difference & separate calls & shown & $\\rho$ & cold & +difference \\\\", "\\midrule"]
for m in order:
    v = PM.get(m); w = GM.get(m)
    if not v and not w:
        continue
    rho = f"{v[chr(99) + 'orrelation of the simulator' + chr(39) + 's errors for the group and for all respondents']:.2f}" if v else "--"
    out.append(f"{m} & {cell(v, 'persona')} & {cell(v, 'pooled')} & {cell(v, 'contrast')} & {(format(SEP[m]['topline + contrast'], '.0f') if m in SEP else '--')} & {cell(v, 'given')} & {rho} & {cell(w, 'persona')} & {cell(w, 'contrast')} \\\\")
out += ["\\bottomrule", "\\end{tabular}", "\\caption{Groups, with 95\\% intervals from a bootstrap over poll releases (party groups) and over questions (GSS), in respondents of the group, with the noise of the group's observed answer removed. Cold: the simulator asked for the group. Pooled: the best fixed mixture of the overall result and the cold answer. +difference: the overall result plus the simulator's difference between the group and everyone. Separate calls: the same when the group and everyone are asked in different calls. Shown: the simulator shown the overall result. "
        "$\\rho$: correlation of the simulator's errors for the group and for all respondents. The overall result alone is worth 11 [9, 13] (party groups) and 36 [30, 45] (GSS).}", "\\label{tab:groups}", "\\end{table}"]
open(ROOT + "/" + PAPER + "/tab_groups.tex", "w").write("\n".join(out) + "\n")

# ------------------------------------------------------------------ appendix: sets and pairs within each programme
BP = Zn.get("by programme", {})
out = ["\\begin{table}[h]", "\\centering\\small", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{lcccccccc}", "\\toprule",
       " & & sum of & the set & best & set over & $G_{ij}/\\min$ & \\multicolumn{2}{c}{gain of a pair} \\\\", "Programme & items & rates & is worth & single & best & pooled & median & largest \\\\", "\\midrule"]
for p, v in BP.items():
    out.append(f"{short.get(p, p)} & {v['items']} & {v['sum of rates']:.0f} & {v['rate of the set']:.1f} & {v['best single']:.1f} & $+${100 * v['set over best']:.0f}\\% & {v['G_ij / min, pooled']:.2f} & {100 * v['pair gain: median, max'][0]:.1f}\\% & {100 * v['pair gain: median, max'][1]:.0f}\\% \\\\")
for nm_, v in Zn.get("new polls by subset: sets", {}).items():
    out.append(f"New polls, {nm_} & {ps[nm_]['items']} & {v['sum of rates']:.0f} & {v['rate of the set']:.1f} & {v['best single']:.1f} & $+${100 * (v['rate of the set'] / v['best single'] - 1):.0f}\\% & {v['G_ij / min, pooled']:.2f} & & \\\\")
out += ["\\bottomrule", "\\end{tabular}", "\\caption{The sixteen simulators within each survey programme (fourteen on the new polls): the sum of their rates, what the set is worth with weights fitted on that programme, its best member, the pooled ratio $\\sum_{i<j}G_{ij}/\\sum_{i<j}\\min(G_{ii},G_{jj})$, and the gain of a pair over its stronger member.}", "\\label{tab:bypro}", "\\end{table}"]
open(ROOT + "/" + PAPER + "/tab_bypro.tex", "w").write("\n".join(out) + "\n")
