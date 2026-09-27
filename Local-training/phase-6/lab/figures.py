"""Publication figures (PNG, 300 dpi) for a finished run, written to results/<run_id>/."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiments import load_executions, run_detail  # noqa: E402
from .metrics import answer_rows  # noqa: E402
from .paths import RESULTS_DIR  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _series_name(a, multi_variant):
    if not multi_variant:
        return a["label"]
    ps = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in a["params"].items())
    return f"{a['label']} ({ps or a['variant']})"


def make_figures(run_id):
    det = run_detail(run_id)
    m, cfg = det["metrics"], det["config"]
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    answers = m["answer"][:8]
    multi = len(m["variants"]) > 1

    # 1. per-subtask accuracy, grouped by arm/variant
    if answers:
        subtasks = sorted({s for a in answers for s in a["per_subtask"]})
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(subtasks) + 3), 4.8), dpi=300)
        width = 0.8 / len(answers)
        x = np.arange(len(subtasks))
        for i, a in enumerate(answers):
            vals = [a["per_subtask"].get(s, {}).get("acc", np.nan) * 100 for s in subtasks]
            ax.bar(x + (i - (len(answers) - 1) / 2) * width, vals, width * 0.92, color=SERIES[i],
                   label=f"{_series_name(a, multi)} - {a['acc'] * 100:.1f}%")
        ax.set_xticks(x, subtasks, rotation=28, ha="right")
        ax.set_ylabel("Accuracy (%)", color=INK)
        ax.set_ylim(0, 100)
        ax.set_title(f"{cfg.get('name') or run_id}: accuracy by subtask (N={m['n_executions']})", color=INK, fontsize=11)
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out_dir / "subtask_accuracy.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p)

    # 2. paired delta per subtask (between arms)
    arms = [pr for pr in m["paired"] if pr["kind"] == "between arms"][:4]
    for j, pr in enumerate(arms):
        items = sorted(pr["per_subtask_delta"].items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(items) + 1.6), dpi=300)
        vals = [v * 100 for _, v in items]
        ax.barh([k for k, _ in items], vals, color=["#e34948" if v < 0 else "#2a78d6" for v in vals], height=0.6)
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_xlabel(f"Δ accuracy, {pr['b']} − {pr['a']} (pp)", color=INK)
        ax.set_title(f"Overall Δ {pr['delta'] * 100:+.2f} pp · McNemar p={pr['p_mcnemar']:.3g} · "
                     f"fixes {pr['fixes']} / breaks {pr['breaks']}", color=INK, fontsize=10)
        _style(ax)
        ax.xaxis.grid(True, color=GRID)
        ax.yaxis.grid(False)
        fig.tight_layout()
        p = out_dir / f"paired_delta_{j}.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p)

    # 3. sweep curves
    by_axis = {}
    for s in m["sweep"]:
        by_axis.setdefault(s["axis"], []).append(s)
    for j, (axis, series) in enumerate(by_axis.items()):
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
        for i, s in enumerate(series[:8]):
            xs = [str(p["value"]) for p in s["points"]]
            ax.plot(xs, [p["acc"] * 100 for p in s["points"]], marker="o", lw=2, ms=6, color=SERIES[i], label=s["label"])
        ax.set_xlabel(axis, color=INK)
        ax.set_ylabel("Accuracy (%)", color=INK)
        _style(ax)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out_dir / f"sweep_{j}.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p)

    # 4. reliability diagram (letter-logits runs)
    rows = [r for r in answer_rows(load_executions(run_id)) if r.get("confidence") is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=300)
        labels = sorted({r["label"] for r in rows})
        edges = np.linspace(0, 1, 11)
        for i, lab in enumerate(labels[:8]):
            rr = [r for r in rows if r["label"] == lab]
            c = np.array([r["confidence"] for r in rr])
            y = np.array([r["correct"] for r in rr], float)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mk = (c > lo) & (c <= hi)
                if mk.sum() >= 3:
                    xs.append(c[mk].mean())
                    ys.append(y[mk].mean())
            ax.plot(xs, ys, marker="o", lw=2, ms=6, color=SERIES[i], label=lab)
        ax.plot([0, 1], [0, 1], ls="--", color="#c3c2b7", lw=1)
        ax.set_xlabel("Confidence", color=INK)
        ax.set_ylabel("Accuracy", color=INK)
        ax.set_title("Reliability", color=INK, fontsize=11)
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out_dir / "reliability.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"config": {k: v for k, v in cfg.items() if k != "graph"}, "metrics": m}, f, indent=2, default=str)
    with open(out_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(cfg.get("graph"), f, indent=2)
    return [str(p.relative_to(RESULTS_DIR)) for p in files] + [f"{run_id}/manifest.json", f"{run_id}/graph.json"]
