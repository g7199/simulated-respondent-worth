"""Figures and the main-text table of paper4 ("What Is a Simulated Respondent Worth?").
  figs/models.pdf  (a) what each simulator is worth, (b) what a second simulator adds, (c) what sets are worth
  figs/real.pdf    (a) shown the rest of the survey, (b) groups by simulator, (c) groups by kind for the strongest simulator
  tab_ways.tex     the same real answers used next to the simulator and inside the prompt
Reads results/y_cov.json, y_cov_cache.pkl, w_groups.json."""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = (_REPO + ""); R = ROOT + "/results/"; OUT = ROOT + "/generated/"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"; BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"; SOFT = "#c9d2dc"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42})
C = json.load(open(R + "y_cov.json")); W = json.load(open(R + "w_groups.json")); ZC = pickle.load(open(R + "y_cov_cache.pkl", "rb"))
names = C["models"]; rc = C["rate_mse_ci"]; order = sorted(names, key=lambda m: rc[m][0]); pr = C["polls"]["rate_mse_ci"]
fam = lambda m: "GPT" if m.startswith("GPT") else "Claude" if m.startswith("Claude") else "open"; COL = {"open": ORANGE, "GPT": BLUE, "Claude": GREEN}
short = lambda m: m.replace("Claude-", "").replace("-Small", "")


def clean(a):
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)


def pair_gain(g):
    out_ = []
    for x in range(len(g)):
        for y in range(x + 1, len(g)):
            s_, w_ = (x, y) if g[x, x] <= g[y, y] else (y, x); lam = min(1.0, max(0.0, (g[s_, s_] - g[s_, w_]) / (g[s_, s_] + g[w_, w_] - 2 * g[s_, w_])))
            risk = (1 - lam) ** 2 * g[s_, s_] + lam ** 2 * g[w_, w_] + 2 * lam * (1 - lam) * g[s_, w_]; out_.append((g[s_, s_] / g[w_, w_], g[s_, s_] / risk - 1))
    return np.array(out_)


# ------------------------------------------------------------------ figure 1: models
fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.95), gridspec_kw={"wspace": 0.55, "width_ratios": [1.0, 1.0, 0.95]})
a = ax[0]
for y, m in enumerate(order):
    col = COL[fam(m)]; r = rc[m]; a.plot(r[1:], [y, y], color=col, lw=1.5, solid_capstyle="round", zorder=2); a.plot([r[0]], [y], marker="o", ms=4.8, color=col, mec="white", mew=0.8, zorder=3)
    if m in pr:
        p = pr[m]; a.plot(p[1:], [y + 0.34, y + 0.34], color=col, lw=0.9, alpha=0.7, zorder=2); a.plot([p[0]], [y + 0.34], marker="D", ms=3.9, mfc="white", mec=col, mew=1.1, zorder=3)
a.set_yticks(range(len(order))); a.set_yticklabels([short(m) for m in order], fontsize=6.2); a.set_xscale("log"); a.set_xticks([5, 10, 20, 40]); a.set_xticklabels(["5", "10", "20", "40"]); a.minorticks_off(); a.set_xlim(4.2, 70); a.set_ylim(-0.6, len(order) + 2.0)
a.set_xlabel("worth (real respondents)"); clean(a); a.grid(False, axis="y")
a.legend([Line2D([], [], ls="none", marker="o", ms=4.8, color=MUTED, mec="white", mew=0.8), Line2D([], [], ls="none", marker="D", ms=3.9, mfc="white", mec=MUTED, mew=1.1)], ["961 older survey questions", "250 polls after training"],
         frameon=False, fontsize=6.2, loc="upper left", handletextpad=0.2, borderaxespad=0.1, labelcolor=INK)
a.set_title("a  What each simulator is worth", fontsize=8.2, loc="left", color=INK)
b = ax[1]; b.plot([0, 1], [0, 1], color=INK, lw=0.9, zorder=1); b.plot([0, 1], [0, 0], color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
hh = ZC["h"]; Gs = np.array([[np.mean([(ZC["C"][x][k] - hh[k]) @ (ZC["C"][y][k] - hh[k]) for k in range(len(hh))]) for y in names] for x in names]); ps_ = pair_gain(Gs)
hp_ = ZC["polls"]["hp"]; pn_ = ZC["polls"]["models"]; Gp_ = np.array([[np.mean([(ZC["polls"]["PM"][x][i] - hp_[i]) @ (ZC["polls"]["PM"][y][i] - hp_[i]) for i in range(len(hp_))]) for y in pn_] for x in pn_]); pp_ = pair_gain(Gp_)
b.scatter(pp_[:, 0], pp_[:, 1], s=11, facecolors="white", edgecolors=BLUE, linewidths=0.7, zorder=3); b.scatter(ps_[:, 0], ps_[:, 1], s=11, color=BLUE, edgecolors="white", linewidths=0.4, zorder=4)
b.set_xlim(0, 1.03); b.set_ylim(-0.05, 1.22); clean(b); b.set_xticks([0, 0.5, 1.0]); b.set_xticklabels(["0", "+50%", "+100%"]); b.set_yticks([0, 0.5, 1.0]); b.set_yticklabels(["0", "+50%", "+100%"])
b.set_xlabel("gain if mistakes were unrelated"); b.set_ylabel("actual gain of the pair", labelpad=1)
b.text(0.60, 0.70, "unrelated\nmistakes", fontsize=6.2, color=INK, ha="right", va="center"); b.annotate("same mistakes", xy=(0.09, 0.0), xytext=(0.05, 0.27), fontsize=6.2, color=INK, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=1))
b.legend([Line2D([], [], ls="none", marker="o", ms=3.9, color=BLUE, mec="white", mew=0.4), Line2D([], [], ls="none", marker="o", ms=3.9, mfc="white", mec=BLUE, mew=0.7)], [f"{len(ps_)} pairs, older surveys", f"{len(pp_)} pairs, new polls"],
         frameon=False, fontsize=6.2, loc="upper left", handletextpad=0.2, borderaxespad=0.1, labelcolor=INK)
b.set_title("b  What a second simulator adds", fontsize=8.2, loc="left", color=INK)
c = ax[2]; S = C["sets"]; best = lambda ms: max(rc[m][0] for m in ms); Pa = C["polls"]["all"]; nfam = lambda f: sum(fam(m) == f for m in names)
rows = [(f"{nfam('open')} open", S["open models"]["sum of rates"], S["open models"]["rate of the set"], best([m for m in names if fam(m) == "open"])),
        (f"{nfam('GPT')} GPT", S["seven GPT models"]["sum of rates"], S["seven GPT models"]["rate of the set"], best([m for m in names if fam(m) == "GPT"])),
        (f"{nfam('Claude')} Claude", S["Claude models"]["sum of rates"], S["Claude models"]["rate of the set"], best([m for m in names if fam(m) == "Claude"])),
        (f"all {len(names)}", S["all simulators"]["sum of rates"], S["all simulators"]["rate of the set"], best(names)),
        (f"all {len(C['polls']['models'])},\nnew polls", Pa["sum of rates"], Pa["rate of the set"], Pa["best single"])]
for y, (nm, sm, st, bs) in enumerate(rows[::-1]):
    c.barh(y + 0.19, sm, height=0.34, color="#d6dbe1", zorder=2); c.barh(y - 0.19, st, height=0.34, color=BLUE, zorder=2); c.plot([bs, bs], [y - 0.42, y + 0.04], color=INK, lw=1.4, zorder=4)
    c.text(sm + 6, y + 0.19, f"{sm:.0f}", va="center", fontsize=6.2, color=MUTED); c.text(st + 6, y - 0.19, f"{st:.0f}", va="center", fontsize=6.2, color=INK)
c.set_yticks(range(len(rows))); c.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.2); c.set_xlim(0, max(r[1] for r in rows) * 1.22); c.set_ylim(-0.6, len(rows) + 1.25); c.set_xlabel("respondents"); clean(c); c.grid(False, axis="y")
c.legend([plt.Rectangle((0, 0), 1, 1, color="#d6dbe1"), plt.Rectangle((0, 0), 1, 1, color=BLUE), Line2D([], [], color=INK, lw=1.4)], ["sum of what each is worth", "worth together", "best one alone"], frameon=False, fontsize=6.2, loc="upper right", borderaxespad=0.0, labelcolor=INK, labelspacing=0.3)
c.set_title("c  What sets are worth", fontsize=8.2, loc="left", color=INK)
fig.savefig(OUT + "figs/models.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT + "figs/models.png", dpi=170, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------------------------ figure 2: real answers in the prompt
CX = {m: v for m, v in W["context"].items() if isinstance(v, dict) and "all" in v}; GR = {m: v for m, v in W["groups"].items() if isinstance(v, dict) and "all" in v and "topline shown" in v["all"] and v["questions"] >= 220}
fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.75), gridspec_kw={"wspace": 0.62, "width_ratios": [1.0, 1.0, 0.92]})
a = ax[0]; cm = sorted(CX, key=lambda m: CX[m]["all"]["plain"][0])
for y, m in enumerate(cm):
    col = COL[fam(m)]; p0 = CX[m]["all"]["plain"][0]; p1 = CX[m]["all"]["shown"][0]; a.plot([p0, p1], [y, y], color=SOFT, lw=1.3, zorder=2)
    a.plot([p0], [y], marker="o", ms=4.8, mfc="white", mec=col, mew=1.2, zorder=4); a.plot([p1], [y], marker="o", ms=4.8, color=col, mec="white", mew=0.7, zorder=4)
a.set_yticks(range(len(cm))); a.set_yticklabels([short(m) for m in cm], fontsize=6.2); a.set_xscale("log"); a.set_xticks([10, 20, 40]); a.set_xticklabels(["10", "20", "40"]); a.minorticks_off(); a.set_xlim(7.5, 70); a.set_ylim(-0.6, len(cm) + 1.7)
a.set_xlabel("worth (real respondents)"); clean(a); a.grid(False, axis="y")
a.legend([Line2D([], [], ls="none", marker="o", ms=4.8, mfc="white", mec=MUTED, mew=1.2), Line2D([], [], ls="none", marker="o", ms=4.8, color=MUTED, mec="white", mew=0.7)], ["asked cold", "shown the rest of the survey"], frameon=False, fontsize=6.2, loc="upper left", handletextpad=0.2, borderaxespad=0.1, labelcolor=INK)
a.set_title("a  A population's answer", fontsize=8.2, loc="left", color=INK)
b = ax[1]; gm = sorted(GR, key=lambda m: GR[m]["all"]["persona"][0]); top = np.mean([GR[m]["all"]["topline"][0] for m in gm]); b.plot([top, top], [-0.6, len(gm) - 0.45], color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
for y, m in enumerate(gm):
    col = COL[fam(m)]; v = GR[m]["all"]; xs_ = [v["persona"][0], v["topline + contrast"][0], v["topline shown"][0]]; b.plot([min(xs_), max(xs_)], [y, y], color=SOFT, lw=1.3, zorder=2)
    b.plot([xs_[0]], [y], marker="o", ms=4.8, mfc="white", mec=col, mew=1.2, zorder=4); b.plot([xs_[1]], [y], marker="o", ms=4.8, color=col, mec="white", mew=0.7, zorder=4); b.plot([xs_[2]], [y], marker="s", ms=4.6, color=INK, mec="white", mew=0.7, zorder=5)
b.set_yticks(range(len(gm))); b.set_yticklabels([short(m) for m in gm], fontsize=6.2); b.set_xscale("log"); b.set_xticks([10, 30, 100]); b.set_xticklabels(["10", "30", "100"]); b.minorticks_off(); b.set_xlim(5, 260); b.set_ylim(-0.6, len(gm) + 2.6)
b.set_xlabel("worth (respondents of the group)"); clean(b); b.grid(False, axis="y")
b.legend([Line2D([], [], ls="none", marker="o", ms=4.8, mfc="white", mec=MUTED, mew=1.2), Line2D([], [], ls="none", marker="o", ms=4.8, color=MUTED, mec="white", mew=0.7), Line2D([], [], ls="none", marker="s", ms=4.6, color=INK, mec="white", mew=0.7), Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2)))],
         ["asked cold", "overall result + difference", "shown the overall result", "overall result alone"], frameon=False, fontsize=6.0, loc="upper left", handletextpad=0.2, borderaxespad=0.05, labelcolor=INK, labelspacing=0.25, ncol=1)
b.set_title("b  A group's answer", fontsize=8.2, loc="left", color=INK)
c = ax[2]; bestm = max(GR, key=lambda m: GR[m]["all"]["topline shown"][0]); K = GR[bestm]["by kind"]; kinds = sorted(K, key=lambda k: K[k]["topline shown"]); lab = {"2024 vote": "2024 vote", "MAGA": "MAGA supporters"}
for y, k in enumerate(kinds):
    v = K[k]; col = COL[fam(bestm)]; xs_ = [v["topline"], v["persona"], v["topline shown"]]; c.plot([min(xs_), max(xs_)], [y, y], color=SOFT, lw=1.3, zorder=2)
    c.plot([v["topline"]], [y], marker="|", ms=8, color=INK, mew=1.4, zorder=3); c.plot([v["persona"]], [y], marker="o", ms=4.8, mfc="white", mec=col, mew=1.2, zorder=4); c.plot([v["topline shown"]], [y], marker="s", ms=4.6, color=INK, mec="white", mew=0.7, zorder=5)
c.set_yticks(range(len(kinds))); c.set_yticklabels([lab.get(k, k) for k in kinds], fontsize=6.2); c.set_xscale("log"); c.set_xticks([10, 100]); c.set_xticklabels(["10", "100"]); c.minorticks_off(); c.set_xlim(2.5, 900); c.set_ylim(-0.6, len(kinds) + 2.3)
c.set_xlabel("worth (respondents of the group)"); clean(c); c.grid(False, axis="y")
c.legend([Line2D([], [], ls="none", marker="|", ms=8, color=INK, mew=1.4), Line2D([], [], ls="none", marker="o", ms=4.8, mfc="white", mec=MUTED, mew=1.2), Line2D([], [], ls="none", marker="s", ms=4.6, color=INK, mec="white", mew=0.7)],
         ["overall result alone", "asked cold", "shown the overall result"], frameon=False, fontsize=6.0, loc="upper left", handletextpad=0.2, borderaxespad=0.05, labelcolor=INK, labelspacing=0.25)
c.set_title(f"c  By kind of group ({short(bestm)})", fontsize=8.2, loc="left", color=INK)
fig.savefig(OUT + "figs/real.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT + "figs/real.png", dpi=170, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------------------------ table: the same real answers, next to the simulator and inside the prompt
rows_pop = [m for m in ("GPT-5.6-luna", "Claude-Haiku-5.5", "Claude-Opus-5.5", "GPT-6.1-sol", "GPT-6-astra", "Claude-Fable-5.2") if m in CX]; rows_grp = [m for m in ("GPT-5.6-luna", "Claude-Haiku-5.5", "Claude-Opus-5.5", "GPT-6.1-sol", "GPT-6-astra", "Claude-Fable-5.2") if m in GR]
out = ["\\begin{tabular}{lccc}", "\\toprule", " & simulator & real answers & real answers \\\\", " & alone & next to it & in the prompt \\\\", "\\midrule", "\\multicolumn{4}{l}{\\emph{A population's answer}} \\\\"]
for m in rows_pop:
    v = CX[m]; ph = max(v["post hoc"]["all"].values()); out.append(f"{m} & {v['all']['plain'][0]:.0f} & {ph:.0f} & \\textbf{{{v['all']['shown'][0]:.0f}}} \\\\")
out += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\emph{{A group's answer}} (overall result alone: {top:.0f})}} \\\\"]
for m in rows_grp:
    v = GR[m]["all"]; out.append(f"{m} & {v['persona'][0]:.0f} & {max(v['topline + contrast'][0], v['free coefficients'][0], v['multiplicative anchor'][0]):.0f} & \\textbf{{{v['topline shown'][0]:.0f}}} \\\\")
out += ["\\bottomrule", "\\end{tabular}"]
open(OUT + "tab_ways.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out))

# ------------------------------------------------------------------ appendix tables
WO = json.load(open(R + "w_old.json")); WR = json.load(open(R + "w_rel.json")); ALLM = [m for m in order if m in CX] + [m for m in CX if m not in order]
sub = {"sib": "a sibling item of the same battery shown", "none": "no question with the same options shown"}
out = ["\\begin{table}[h]", "\\centering\\small", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{lccccccccc}", "\\toprule",
       " & \\multicolumn{3}{c}{All 246 questions} & \\multicolumn{2}{c}{Sibling item shown (61)} & \\multicolumn{2}{c}{No shared options (140)} & \\multicolumn{2}{c}{Fewer shown} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
       "Simulator & cold & shown & ratio & cold & shown & cold & shown & 3 & 10 \\\\", "\\midrule"]
for m in sorted(CX, key=lambda m: CX[m]["all"]["plain"][0]):
    v = CX[m]; d = v.get("dose", {})
    out.append(f"{m} & {v['all']['plain'][0]:.0f} & {v['all']['shown'][0]:.0f} & {v['all']['ratio'][0]:.2f} {{\\scriptsize[{v['all']['ratio'][1]:.2f}, {v['all']['ratio'][2]:.2f}]}} & {v[sub['sib']]['plain'][0]:.0f} & {v[sub['sib']]['shown'][0]:.0f} & {v[sub['none']]['plain'][0]:.0f} & {v[sub['none']]['shown'][0]:.0f} & {(format(d['3'], '.0f') if '3' in d else '--')} & {(format(d['10'], '.0f') if '10' in d else '--')} \\\\")
out += ["\\bottomrule", "\\end{tabular}", "\\caption{A population's answer with the real results of the other half of the same weekly report in the prompt (29 questions on average), in respondents. Ratio: shown over cold, with a 95\\% interval from a bootstrap over batteries (questions of one report that share a stem). Sibling item shown: the question belongs to a battery and another item of the battery is among the answers shown. No shared options: no shown question has the same answer options. Fewer shown: only three or ten of the other answers are shown.}", "\\label{tab:ctx}", "\\end{table}"]
open(OUT + "tab_ctx.tex", "w").write("\n".join(out) + "\n")
GA = {m: v for m, v in W["groups"].items() if isinstance(v, dict) and "by kind" in v and "topline shown" in v["all"] and v["questions"] >= 220}; cols = [m for m in ("GPT-5.6-luna", "Claude-Haiku-5.5", "GPT-6.1-sol", "Claude-Fable-5.2") if m in GA]
out = ["\\begin{table}[h]", "\\centering\\small", "\\setlength{\\tabcolsep}{3.4pt}", "\\begin{tabular}{lc" + "ccc" * len(cols) + "}", "\\toprule", " & overall & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in cols) + " \\\\",
       "".join(f"\\cmidrule(lr){{{3 + 3 * k}-{5 + 3 * k}}}" for k in range(len(cols))), "Kind of group & result & " + " & ".join("cold & +diff. & shown" for m in cols) + " \\\\", "\\midrule"]
for kd, nm_ in (("sex", "Sex (2 groups)"), ("education", "Education (2)"), ("age", "Age (4)"), ("race", "Race (3)"), ("party", "Party (3)"), ("ideology", "Ideology (3)"), ("2024 vote", "2024 vote (2)"), ("MAGA", "MAGA supporters (1)")):
    out.append(f"{nm_} & {GA[cols[-1]]['by kind'][kd]['topline']:.0f} & " + " & ".join(f"{GA[m]['by kind'][kd]['persona']:.0f} & {GA[m]['by kind'][kd]['topline + contrast']:.0f} & {GA[m]['by kind'][kd]['topline shown']:.0f}" for m in cols) + " \\\\")
out += ["\\midrule", f"All twenty groups & {GA[cols[-1]]['all']['topline'][0]:.0f} & " + " & ".join(f"{GA[m]['all']['persona'][0]:.0f} & {GA[m]['all']['topline + contrast'][0]:.0f} & {GA[m]['all']['topline shown'][0]:.0f}" for m in cols) + " \\\\",
        "\\bottomrule", "\\end{tabular}", "\\caption{A group's answer by kind of group, in respondents of the group (246 questions asked after training). Overall result: the question's real overall result used as the estimate for the group. Cold: the simulator asked for the group. +diff.: the simulator's difference between the group and everyone, added to the real overall result. Shown: the simulator asked for the group with the overall result in the prompt.}", "\\label{tab:grp20}", "\\end{table}"]
open(OUT + "tab_grp20.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out[8:-5]))

# groups by simulator with intervals
out = ["\\begin{table}[h]", "\\centering\\small", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{lcccccc}", "\\toprule", "Simulator & asked cold & result + difference & free coefficients & result $\\times$ ratio & shown the result & shown over cold \\\\", "\\midrule"]
ci3 = lambda r: f"{r[0]:.0f} {{\\scriptsize[{r[1]:.0f}, {r[2]:.0f}]}}"
for m in sorted(GA, key=lambda m: GA[m]["all"]["persona"][0]):
    v = GA[m]; out.append(f"{m} & {ci3(v['all']['persona'])} & {ci3(v['all']['topline + contrast'])} & {ci3(v['all']['free coefficients'])} & {ci3(v['all']['multiplicative anchor'])} & {ci3(v['all']['topline shown'])} & {v['shown over persona'][0]:.1f} {{\\scriptsize[{v['shown over persona'][1]:.1f}, {v['shown over persona'][2]:.1f}]}} \\\\")
tl = GA[cols[-1]]["all"]["topline"]
out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{A group's answer over twenty groups and 246 questions (4,920 group answers), in respondents of the group, with 95\\% intervals from a bootstrap over batteries of questions. Result + difference: the real overall result plus the simulator's cold difference between the group and everyone. Free coefficients: the same three numbers combined with coefficients fitted on the other half of the questions. Result $\\times$ ratio: the simulator's cold answer for the group times the ratio of the real to the simulated overall result. The question's real overall result alone is worth {tl[0]:.0f} [{tl[1]:.0f}, {tl[2]:.0f}].}}", "\\label{tab:grpci}", "\\end{table}"]
open(OUT + "tab_grpci.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out[7:-5]))

# controls for the same-survey context
WC = json.load(open(R + "w_ctl.json")); ks = ("other pollsters, same weeks", "same pollster, the week before", "same survey"); kn = "no shared options in the same-survey condition"
out = ["\\begin{table}[h]", "\\centering\\small", "\\setlength{\\tabcolsep}{4.5pt}", "\\begin{tabular}{lcccccccc}", "\\toprule",
       f" & \\multicolumn{{4}}{{c}}{{All {WC['targets']} questions}} & \\multicolumn{{4}}{{c}}{{No shared options ({WC['subsets'][kn]})}} \\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
       "Real answers shown & none & other pollsters & week before & same survey & none & other pollsters & week before & same survey \\\\", "\\midrule"]
for m in sorted(WC["models"], key=lambda m: WC["models"][m]["cold"]):
    v = WC["models"][m]; b = v["by subset"][kn]
    out.append(f"{m} & {v['cold']:.0f} & {v[ks[0]][0]:.0f} & {v[ks[1]][0]:.0f} & {v[ks[2]][0]:.0f} & {b['cold']:.0f} & {b[ks[0]]:.0f} & {b[ks[1]]:.0f} & {b[ks[2]]:.0f} \\\\")
out += ["\\bottomrule", "\\end{tabular}", "\\caption{What the simulator reads. A population's answer, in respondents, on the questions of the second to fifth weekly report, with 29 real answers in the prompt that come from other pollsters' polls in the United States in the same weeks, from other questions in the same pollster's report of the week before, or from the other half of the same report. Right: the questions for which the same-report condition shows no question with the same answer options.}", "\\label{tab:ctl}", "\\end{table}"]
open(OUT + "tab_ctl.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out[9:-5]))
