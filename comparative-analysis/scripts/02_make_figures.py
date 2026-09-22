"""
Figures for the MMAD (ICLR 2025) vs this-dissertation comparative analysis.

Palette: validated categorical slots (blue / orange / aqua / yellow / magenta).
Every low-contrast fill carries a direct value label, and each figure has a
matching CSV in results/tables/ as the table view.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
CA = ROOT / "comparative-analysis"
DATA, FIGS = CA / "data", CA / "figures"
TABLES = CA / "results" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# --- validated palette (light surface #fcfcfb) -----------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET, RED = "#4a3aa7", "#e34948"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8f8e88"
GRID = "#e6e5e1"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = RED, "#f0efec", BLUE

FAMILY_COLOR = {
    "this_work": AQUA,
    "commercial": ORANGE,
    "open_source": BLUE,
    "human": MUTED,
    "baseline": "#c9c8c2",
}
FAMILY_LABEL = {
    "this_work": "This work — open MSLM 2–4B, 0-shot, local RTX 4060",
    "commercial": "MMAD paper — commercial API",
    "open_source": "MMAD paper — open-source MLLM (7–76B)",
    "human": "MMAD paper — human reference",
    "baseline": "Random chance",
}

PAPER_COLS = ["anomaly_discrimination", "defect_classification", "defect_localization",
              "defect_description", "defect_analysis", "object_classification",
              "object_analysis"]
NICE = {
    "anomaly_discrimination": "Anomaly\nDiscrimination",
    "defect_classification": "Defect\nClassification",
    "defect_localization": "Defect\nLocalization",
    "defect_description": "Defect\nDescription",
    "defect_analysis": "Defect\nAnalysis",
    "object_classification": "Object\nClassification",
    "object_analysis": "Object\nAnalysis",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax, xgrid=False, ygrid=False):
    ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.tick_params(length=0)


def title(ax, head, sub=None):
    n = sub.count("\n") + 1 if sub else 0
    ax.set_title(head, loc="left", fontsize=13.5, fontweight="bold",
                 color=INK, pad=10 + 13 * n)
    if sub:
        ax.text(0, 1.008, sub, transform=ax.transAxes, fontsize=9.5,
                color=INK2, va="bottom", linespacing=1.45)


def save(fig, name):
    path = FIGS / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}")


merged = pd.read_csv(DATA / "merged_leaderboard.csv")
tw = pd.read_csv(DATA / "thiswork_mmad_protocol.csv")
paper = pd.read_csv(DATA / "mmad_paper_table2.csv")
sub9 = pd.read_csv(DATA / "thiswork_subtask_9.csv")
shot = pd.read_csv(DATA / "mmad_paper_table3_shot_setting.csv")
rob = pd.read_csv(DATA / "thiswork_robustness_5k_by_subtask.csv")
rob_all = pd.read_csv(DATA / "thiswork_robustness_5k_overall.csv")

print("rendering figures:")

# =========================================================== FIG 1 leaderboard
d = merged.sort_values("average").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10.5, 10.2))
colors = [FAMILY_COLOR[f] for f in d["family"]]
bars = ax.barh(np.arange(len(d)), d["average"], height=0.68, color=colors)
for b in bars:
    b.set_joinstyle("round")

for i, (v, fam) in enumerate(zip(d["average"], d["family"])):
    ax.text(v + 0.7, i, f"{v:.1f}", va="center", ha="left", fontsize=9,
            color=INK if fam == "this_work" else INK2,
            fontweight="bold" if fam == "this_work" else "normal")

labels = [f"{m}  ·  {s}" if s != "-" else m for m, s in zip(d["model"], d["scale"])]
ax.set_yticks(np.arange(len(d)))
ax.set_yticklabels(labels, fontsize=9.5)
for tick, fam in zip(ax.get_yticklabels(), d["family"]):
    if fam == "this_work":
        tick.set_fontweight("bold")
        tick.set_color(INK)

gpt = float(paper.loc[paper.model == "GPT-4o", "average"].iloc[0])
ax.axvline(gpt, color=ORANGE, lw=1.4, ls=(0, (4, 3)), zorder=0)
ax.text(gpt + 0.8, len(d) + 0.45, f"GPT-4o {gpt:.1f}", color=ORANGE, fontsize=9,
        va="center", ha="left")

ax.set_ylim(-0.8, len(d) + 1.0)
ax.set_xlim(0, 100)
ax.set_xlabel("MMAD 7-task average accuracy (%)")
style(ax, xgrid=True)
title(ax, "A 2B open model lands inside the MMAD leaderboard's upper half",
      "Qwen3-VL-2B scores 68.4% — above every open-source model ≤13B in the MMAD paper, 2.4 pts under InternVL2-76B, 6.5 under GPT-4o.\n"
      "Averages are the unweighted mean of MMAD's 7 task columns. Paper rows: 1-shot, full benchmark. This work: 0-shot, 2,500-question sample.")
ax.legend(handles=[Patch(facecolor=FAMILY_COLOR[k], label=FAMILY_LABEL[k])
                   for k in ["this_work", "commercial", "open_source", "human", "baseline"]],
          loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=3,
          frameon=False, fontsize=9)
save(fig, "fig1_overall_leaderboard.png")
d.to_csv(TABLES / "fig1_overall_leaderboard.csv", index=False)

# ==================================================== FIG 2 accuracy by size band
# A log-parameter scatter piles 10 of the paper's 13 open models onto two
# x-values, so size is binned into bands and each band gets its own column.
BANDS = ["2-4B\n(this work)", "7B", "8B", "13B", "34B", "76B"]
BAND_X = [0.0, 1.9, 2.9, 3.9, 4.9, 5.9]
BAND_OF = {2.2: 0, 2.3: 0, 3.1: 0, 4.4: 0, 7: 1, 8: 2, 13: 3, 34: 4, 76: 5}

fig, ax = plt.subplots(figsize=(10.4, 6.6))
op = paper[(paper.family == "open_source") & paper.params_b.notna()].copy()
op["band"] = op.params_b.map(BAND_OF)
op["bx"] = op.band.map(lambda b: BAND_X[b])
tw2 = tw.copy()
tw2["band"] = 0

for name, col, dash in [("GPT-4o", ORANGE, (0, (4, 3))),
                        ("Gemini-1.5-pro", ORANGE, (0, (1, 3))),
                        ("Human (expert)", MUTED, (0, (6, 3)))]:
    v = float(paper.loc[paper.model == name, "average"].iloc[0])
    ax.axhline(v, color=col, lw=1.3, ls=dash, zorder=1)
    ax.text(6.42, v + 0.7, f"{name} {v:.1f}", color=col, fontsize=8.5, ha="right")

ax.scatter(op.bx, op.average, s=95, color=BLUE, zorder=3, edgecolor=SURFACE,
           linewidth=2, label="MMAD paper - open-source (1-shot, full benchmark)")
ax.scatter(tw2.band, tw2.average, s=190, color=AQUA, marker="D", zorder=4,
           edgecolor=SURFACE, linewidth=2,
           label="This work - open MSLM (0-shot, 2,500-question sample)")

# Within a band, labels alternate sides in y order so no two collide.
SIDE_OVERRIDE = {"Qwen-VL-Chat": -1}
for band, grp in op.groupby("band"):
    grp = grp.sort_values("average").reset_index(drop=True)
    for i, r in grp.iterrows():
        if band == 1:
            side = SIDE_OVERRIDE.get(r.model, -1 if i % 2 == 0 else 1)
        elif band == 2:
            side = 1
        else:
            side = 0
        bx = BAND_X[band]
        if side == 0:
            ax.annotate(r.model, (bx, r.average), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=8.2, color=MUTED)
        else:
            ax.annotate(r.model, (bx, r.average), textcoords="offset points",
                        xytext=(11 * side, 0), ha="left" if side > 0 else "right",
                        va="center", fontsize=8.2, color=MUTED)

for _, r in tw2.sort_values("average").iterrows():
    ax.annotate(f"{r.model}  {r.average:.1f}", (0, r.average),
                textcoords="offset points", xytext=(-13, 0), ha="right",
                va="center", fontsize=8.8, color=INK, fontweight="bold")

ax.set_xticks(BAND_X)
ax.set_xticklabels(BANDS, fontsize=9.5)
ax.set_xlim(-1.75, 6.5)
ax.set_ylim(33, 91)
ax.set_xlabel("Model size band (billion parameters)")
ax.set_ylabel("MMAD 7-task average accuracy (%)")
style(ax, ygrid=True)
title(ax, "Parameter count is not what buys MMAD accuracy",
      "Qwen3-VL-2B (68.4) lands between the paper's 34B and 76B systems - and runs on a single 8 GB laptop GPU.")
ax.legend(loc="lower right", frameon=False, fontsize=9)
save(fig, "fig2_accuracy_vs_scale.png")

# ============================================== FIG 3 seven-task profile bars
picks = [("Qwen3-VL-2B", AQUA, "This work · Qwen3-VL-2B (2.2B, 0-shot)"),
         ("GPT-4o", ORANGE, "MMAD · GPT-4o (commercial)"),
         ("InternVL2 (76B)", BLUE, "MMAD · InternVL2-76B"),
         ("MiniCPM-V2.6", YELLOW, "MMAD · MiniCPM-V2.6 (8B, best ≤8B)")]
fig, ax = plt.subplots(figsize=(11.2, 5.9))
x = np.arange(len(PAPER_COLS))
w = 0.2
for i, (name, col, lab) in enumerate(picks):
    row = merged[merged.model == name].iloc[0]
    vals = [row[c] for c in PAPER_COLS]
    off = (i - 1.5) * w
    ax.bar(x + off, vals, w * 0.88, color=col, label=lab, zorder=3)
    for xx, v in zip(x + off, vals):
        ax.text(xx, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.8, color=INK2)

rc = paper[paper.model == "Random Chance"].iloc[0]
ax.plot(x, [rc[c] for c in PAPER_COLS], marker="_", ms=26, mew=2.2, ls="none",
        color=MUTED, zorder=4, label="Random chance")

ax.set_xticks(x)
ax.set_xticklabels([NICE[c] for c in PAPER_COLS], fontsize=9)
ax.set_ylim(0, 108)
ax.set_ylabel("Accuracy (%)")
style(ax, ygrid=True)
title(ax, "Where the 2B model holds its own against GPT-4o - and where it does not",
      "Level on Object Classification (94.9 vs 95.0) and within 2 pts on three further tasks, but 25 pts behind on Defect Classification.")
ax.legend(loc="upper left", frameon=False, fontsize=8.8, ncol=2)
save(fig, "fig3_seven_task_profile.png")

# ================================================== FIG 4 delta vs GPT-4o heat
ref = paper[paper.model == "GPT-4o"].iloc[0]
delta = pd.DataFrame({c: tw[c] - ref[c] for c in PAPER_COLS})
delta.index = tw["model"]
delta["7-task\naverage"] = tw["average"].values - ref["average"]

fig, ax = plt.subplots(figsize=(10.4, 3.6))
lim = float(np.abs(delta.values).max())
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", [DIV_NEG, DIV_MID, DIV_POS])
im = ax.imshow(delta.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(delta.shape[1]))
ax.set_xticklabels([NICE.get(c, c) for c in delta.columns], fontsize=8.8)
ax.set_yticks(range(len(delta)))
ax.set_yticklabels(delta.index, fontsize=9.5)
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        v = delta.values[i, j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                color="#ffffff" if abs(v) > lim * 0.55 else INK)
ax.set_xticks(np.arange(-0.5, delta.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, delta.shape[0], 1), minor=True)
ax.grid(which="minor", color=SURFACE, lw=2.5)
ax.tick_params(which="both", length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.028)
cb.set_label("accuracy points vs GPT-4o", fontsize=9, color=INK2)
cb.outline.set_visible(False)
title(ax, "Per-task gap to GPT-4o (blue = this work ahead, red = behind)",
      "No local model leads GPT-4o on any task. Qwen3-VL-2B is within 2 pts on four of the seven; Defect Classification and Localization carry its deficit.")
save(fig, "fig4_delta_vs_gpt4o.png")
delta.round(2).to_csv(TABLES / "fig4_delta_vs_gpt4o.csv")

# =========================================== FIG 5 defect-localization ranking
d5 = merged[merged.family.isin(["open_source", "commercial", "this_work"])] \
    .sort_values("defect_localization").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.8, 8.4))
cols = [FAMILY_COLOR[f] for f in d5["family"]]
ax.barh(np.arange(len(d5)), d5["defect_localization"], height=0.66, color=cols)
for i, (v, fam) in enumerate(zip(d5["defect_localization"], d5["family"])):
    ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=9,
            color=INK if fam == "this_work" else INK2,
            fontweight="bold" if fam == "this_work" else "normal")
ax.set_yticks(np.arange(len(d5)))
ax.set_yticklabels(d5["model"], fontsize=9.5)
for tick, fam in zip(ax.get_yticklabels(), d5["family"]):
    if fam == "this_work":
        tick.set_fontweight("bold")
        tick.set_color(INK)
ax.axvline(25, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=0)
ax.text(25.8, len(d5) + 0.45, "random chance 25", color=MUTED, fontsize=9, va="center")
ax.set_ylim(-0.8, len(d5) + 1.0)
ax.set_xlim(0, 100)
ax.set_xlabel("Defect Localization accuracy (%)")
style(ax, xgrid=True)
title(ax, "Defect Localization separates vision encoders, not model sizes",
      "Qwen's dynamic-patch models (51.3, 47.5) sit mid-field among the paper's 7–76B systems, above GPT-4o-mini.\n"
      "Gemma's pooled-token models (31.4, 26.4) land in the bottom four of the whole table — 6.4 and 1.4 pts off random chance.")
ax.legend(handles=[Patch(facecolor=FAMILY_COLOR[k], label=FAMILY_LABEL[k])
                   for k in ["this_work", "commercial", "open_source"]],
          loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3,
          frameon=False, fontsize=9)
save(fig, "fig5_defect_localization_gap.png")

# ================================================ FIG 6 robustness (unmeasured)
order = ["clean", "corr_motion_blur", "tta_motion_blur",
         "corr_gaussian_noise", "tta_gaussian_noise"]
lab = {"clean": "Clean", "corr_motion_blur": "Motion blur (sev 4)",
       "tta_motion_blur": "Motion blur + TTA-IR",
       "corr_gaussian_noise": "Gaussian noise (sev 4)",
       "tta_gaussian_noise": "Gaussian noise + TTA-IR"}
cmap5 = {"clean": BLUE, "corr_motion_blur": ORANGE, "tta_motion_blur": YELLOW,
         "corr_gaussian_noise": AQUA, "tta_gaussian_noise": MAGENTA}

piv = (rob[rob.strategy.isin(["none", "tta"]) | rob.condition.isin(order)]
       .pivot_table(index="subtask", columns="condition", values="accuracy"))
piv = piv[[c for c in order if c in piv.columns]]
piv = piv.sort_values("clean", ascending=False)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.4, 6.6),
                               gridspec_kw={"width_ratios": [1, 2.5]})
fig.subplots_adjust(top=0.82, wspace=0.16)

ro = rob_all.set_index("condition").loc[order]
axA.bar(range(5), ro["accuracy"], color=[cmap5[c] for c in order], width=0.66, zorder=3)
for i, v in enumerate(ro["accuracy"]):
    axA.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=9.5, color=INK)
SHORT = {"clean": "Clean", "corr_motion_blur": "Blur", "tta_motion_blur": "Blur\n+TTA",
         "corr_gaussian_noise": "Noise", "tta_gaussian_noise": "Noise\n+TTA"}
axA.set_xticks(range(5))
axA.set_xticklabels([SHORT[c] for c in order], fontsize=9)
axA.set_ylim(0, 80)
axA.set_ylabel("Accuracy (%)")
axA.axhline(gpt, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
axA.text(4.4, gpt + 0.7, f"GPT-4o clean {gpt:.1f}", color=MUTED, fontsize=8.5, ha="right")
style(axA, ygrid=True)
title(axA, "Overall (N=5,000)", "25,000 inferences, recomputed from the raw logs")

x = np.arange(len(piv))
w = 0.16
for i, c in enumerate(piv.columns):
    axB.bar(x + (i - 2) * w, piv[c], w * 0.88, color=cmap5[c], label=lab[c], zorder=3)
axB.set_xticks(x)
axB.set_xticklabels([s.replace(" ", "\n") for s in piv.index], fontsize=8.6)
axB.set_ylim(0, 100)
axB.set_ylabel("Accuracy (%)")
style(axB, ygrid=True)
title(axB, "Per subtask", "Object tasks collapse under blur; defect tasks drift upward via majority-class bias")
axB.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=2)

fig.text(0.008, 0.995, "The axis MMAD never measured: accuracy under factory-floor image corruption",
         ha="left", va="top", fontsize=14.5, fontweight="bold", color=INK)
fig.text(0.008, 0.925, "MMAD scores only pristine laboratory captures. Under severity-4 conveyor motion blur Qwen3-VL-2B loses 3.8 pts overall — "
                       "but Object Classification alone loses 19.4.",
         ha="left", va="top", fontsize=9.5, color=INK2)
save(fig, "fig6_robustness_gap.png")
piv.round(2).to_csv(TABLES / "fig6_robustness_by_subtask.csv")

# ================================================= FIG 7 shot-setting control
w0 = shot[shot.setting == "0-shot"].set_index("model")["average"]
w1 = shot[shot.setting == "1-shot+"].set_index("model")["average"]
dd = (w1 - w0).sort_values()
fig, ax = plt.subplots(figsize=(8.8, 4.6))
cols = [BLUE if v >= 0 else DIV_NEG for v in dd]
ax.barh(np.arange(len(dd)), dd.values, height=0.6, color=cols, zorder=3)
for i, v in enumerate(dd.values):
    ax.text(v + (0.06 if v >= 0 else -0.06), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
ax.set_yticks(np.arange(len(dd)))
ax.set_yticklabels(dd.index, fontsize=9.5)
ax.axvline(0, color=INK2, lw=1)
ax.set_xlim(-2.0, 2.6)
ax.set_xlabel("Average accuracy change from 0-shot to 1-shot+ (percentage points)")
style(ax, xgrid=True)
title(ax, "Validity control: how much does the shot setting actually matter?",
      "This work is 0-shot; MMAD's Table 2 is 1-shot. The paper's own Table 3 puts that effect between −1.3 and +1.9 pts — smaller than the gaps discussed here.")
ax.legend(handles=[Patch(facecolor=BLUE, label="1-shot+ helps"),
                   Patch(facecolor=DIV_NEG, label="1-shot+ hurts")],
          loc="lower right", frameon=False, fontsize=9)
save(fig, "fig7_shot_setting_control.png")

# ============================================ FIG 8 deployment cost (this work)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.8))
fig.subplots_adjust(top=0.74, wspace=0.22)
t = tw.sort_values("average", ascending=False).reset_index(drop=True)
# Colour follows the model, not its position in the ranking.
MODEL_COLOR = {"Qwen3-VL-2B": AQUA, "Qwen2.5-VL-3B": BLUE,
               "Gemma-4-E4B": ORANGE, "Gemma-4-E2B": YELLOW}
mc = [MODEL_COLOR[m] for m in t.model]

a1.scatter(t.vram_gb, t.average, s=200, marker="D",
           color=mc, edgecolor=SURFACE, lw=2, zorder=3)
for i, r in t.iterrows():
    a1.annotate(f"{r.model}\n{r.precision}", (r.vram_gb, r.average),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=8.5, color=INK)
a1.axvline(8.19, color=RED, lw=1.4, ls=(0, (4, 3)), zorder=1)
a1.text(8.15, 57.2, "RTX 4060 ceiling 8.19 GB ", color=RED, fontsize=8.5, ha="right")
a1.set_xlim(2.4, 8.7)
a1.set_ylim(56, 72.5)
a1.set_xlabel("Peak VRAM (GB)")
a1.set_ylabel("MMAD 7-task average (%)")
style(a1, ygrid=True)
title(a1, "Accuracy per GB of VRAM", "4-bit NF4 + CPU-offloaded embeddings put a 4.4B model in the least memory")

a2.scatter(t.throughput_fps, t.average, s=200, marker="D",
           color=mc, edgecolor=SURFACE, lw=2, zorder=3)
for i, r in t.iterrows():
    a2.annotate(f"{r.model}\n{r.throughput_fps:.1f} fps", (r.throughput_fps, r.average),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=8.5, color=INK)
a2.set_xlim(0.4, 7.4)
a2.set_ylim(56, 72.5)
a2.set_xlabel("Throughput (inferences / second)")
a2.set_ylabel("MMAD 7-task average (%)")
style(a2, ygrid=True)
title(a2, "Accuracy per unit of latency", "The most accurate model is also the fastest — 5.2× the throughput of Gemma-4-E4B")

fig.text(0.008, 0.995, "Deployment envelope — the dimension the MMAD leaderboard omits entirely",
         ha="left", va="top", fontsize=14.5, fontweight="bold", color=INK)
fig.text(0.008, 0.930, "MMAD reports accuracy and parameter count only. For an on-premise inspection line, VRAM and throughput decide what is actually deployable.",
         ha="left", va="top", fontsize=9.5, color=INK2)
save(fig, "fig8_deployment_envelope.png")
t.to_csv(TABLES / "fig8_deployment_envelope.csv", index=False)

print(f"\n{len(list(FIGS.glob('*.png')))} figures -> {FIGS}")
print(f"tables -> {TABLES}")
