"""Plots from saved validation results; this module never calls Jev."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.helpsteer2 import DIMENSIONS

METRICS = (
    ("pearson", "Pearson correlation", "Higher is better"),
    ("spearman", "Spearman correlation", "Higher is better"),
    ("mae", "Mean absolute error", "Lower is better"),
    ("pairwise", "Pairwise preference accuracy", "Higher is better"),
)
COLORS = ("#3973ac", "#26968a", "#d38a35")
SECTION = "\n## Validation plots\n"

def _number(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None

def _metric_panel(ax, summary, key, title, direction):
    for i, dimension in enumerate(DIMENSIONS):
        metric = summary["metrics"][dimension]
        pair = metric.get("pairwise")
        value = _number(pair.get("mean") if pair else None) if key == "pairwise" else _number(metric.get(key))
        if value is None:
            ax.text(i, .03, "N/A", ha="center", va="bottom", color="#777777")
            continue
        ax.bar(i, value, width=.56, color=COLORS[i], zorder=3)
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 6 if value >= 0 else -6),
                    textcoords="offset points", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=10)
        if key == "pairwise" and pair.get("ci95"):
            low, high = pair["ci95"]
            ax.vlines(i, low, high, color="#202b37", linewidth=1.6, zorder=4)
            ax.hlines([low, high], i-.07, i+.07, color="#202b37", linewidth=1.6, zorder=4)
    labels = [d.capitalize() for d in DIMENSIONS]
    if key == "pairwise":
        labels = [f"{d.capitalize()}\npairs={(summary['metrics'][d].get('pairwise') or {}).get('n', 0)}"
                  for d in DIMENSIONS]
        ax.axhline(.5, color="#747d88", linestyle="--", linewidth=1, zorder=2)
    ax.set_xticks(range(len(DIMENSIONS)), labels)
    ax.set_xlim(-.6, len(DIMENSIONS)-.4)
    ax.set_ylim((-1.15, 1.15) if key in ("pearson", "spearman") else (0, 1.15))
    ax.set_title(f"{title}\n{direction}", fontsize=11, pad=12)
    ax.grid(axis="y", alpha=.22, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

def plot_validation(results_dir="results"):
    root = Path(results_dir)
    summary = json.loads((root / "reward_validation.json").read_text())
    df = pd.read_csv(root / "reward_validation.csv")
    if len(df) != summary["n"]:
        raise ValueError("Validation CSV and JSON sample counts differ; regenerate matching results")
    for dimension in DIMENSIONS:
        columns = [f"human_{dimension}", f"jev_{dimension}"]
        values = df[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{dimension}: expected finite normalized scores in [0, 1]")
    out = root / "plots" / "reward_validation"
    out.mkdir(parents=True, exist_ok=True)
    scope = "SMOKE SUBSET" if summary.get("scope") == "smoke_subset" else "FULL VALIDATION"
    caption = (f"{scope} | {len(df)} responses | {summary['prompts']} prompts | "
               + ", ".join(summary.get("models", [])))
    note = "N/A = undefined statistic; it is not zero. Scores normalized to [0, 1]."
    pair_note = "Pairwise: 95% bootstrap CI; human ties excluded; Jev ties receive 0.5."
    saved = []

    def save(fig, name):
        try:
            for extension in ("png", "svg"):
                path = out / f"{name}.{extension}"
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
                saved.append(str(path))
        finally:
            plt.close(fig)

    for key, title, direction in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.7))
        fig.subplots_adjust(bottom=.24, top=.78)
        _metric_panel(ax, summary, key, title, direction)
        fig.suptitle("Jev vs human annotations", fontsize=15, y=.97)
        fig.text(.5, .90, caption, ha="center", fontsize=8.5)
        fig.text(.08, .045, pair_note if key == "pairwise" else note, fontsize=8)
        save(fig, key)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.subplots_adjust(bottom=.13, top=.84, hspace=.55, wspace=.25)
    for ax, (key, title, direction) in zip(axes.flat, METRICS):
        _metric_panel(ax, summary, key, title, direction)
    fig.suptitle("Jev reward validation — metric overview", fontsize=16, y=.97)
    fig.text(.5, .915, caption, ha="center", fontsize=10)
    fig.text(.08, .035, note + "\n" + pair_note, fontsize=8.5)
    save(fig, "overview")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    fig.subplots_adjust(bottom=.20, top=.75, wspace=.30)
    for ax, dimension, color in zip(axes, DIMENSIONS, COLORS):
        # Bubble area represents repeated observations, without jittering the scores.
        counts = df.groupby([f"human_{dimension}", f"jev_{dimension}"]).size().reset_index(name="count")
        ax.scatter(counts.iloc[:, 0], counts.iloc[:, 1], s=counts["count"]*24,
                   alpha=.55, color=color, edgecolors="none")
        ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
        ax.set(xlim=(-.06, 1.06), ylim=(-.06, 1.06), xlabel="Human score",
               ylabel="Jev score", title=dimension.capitalize())
        ax.set_aspect("equal")
        ax.grid(alpha=.18)
    fig.suptitle("Score agreement by dimension", fontsize=16, y=.97)
    fig.text(.5, .88, caption, ha="center", fontsize=10)
    fig.text(.08, .04, "Bubble area = observation count. Dashed line = exact score agreement.", fontsize=9)
    save(fig, "score_agreement")

    markdown = root / "reward_validation_summary.md"
    original = markdown.read_text() if markdown.exists() else "# Jev reward validation\n"
    original = original.split(SECTION, 1)[0].rstrip()
    links = "\n\n".join(f"### {title}\n\n![{title}](plots/reward_validation/{name}.png)"
                         for name, title in [
                             ("overview", "Metric overview"),
                             ("pearson", "Pearson correlation"),
                             ("spearman", "Spearman correlation"),
                             ("mae", "Mean absolute error"),
                             ("pairwise", "Pairwise preference accuracy"),
                             ("score_agreement", "Score agreement"),
                         ])
    markdown.write_text(original + "\n" + SECTION + "\n" + caption + "\n\n" +
                        note + "\n\n" + pair_note + "\n\n" + links + "\n")
    print(f"Saved {len(saved)} plots (PNG + SVG) to {out}", flush=True)
    return saved
