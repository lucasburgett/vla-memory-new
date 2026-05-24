"""Generate bar chart of frozen pi0.5 baseline results across all 16 RoboMME tasks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = Path(__file__).parent.parent / "eval_results" / "seed7" / "log.json"

TASK_SUITES = {
    "Counting":    ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Persistent":  ["ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"],
    "Referential": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "Behavior":    ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}

SUITE_COLORS = {
    "Counting":    "#4C72B0",
    "Persistent":  "#DD8452",
    "Referential": "#55A868",
    "Behavior":    "#C44E52",
}

TASK_LABELS = {
    "BinFill":          "BinFill",
    "PickXtimes":       "PickXtimes",
    "SwingXtimes":      "SwingXtimes",
    "StopCube":         "StopCube",
    "ButtonUnmask":     "ButtonUnmask",
    "VideoUnmask":      "VideoUnmask",
    "VideoUnmaskSwap":  "VideoUnmaskSwap",
    "ButtonUnmaskSwap": "ButtonUnmaskSwap",
    "PickHighlight":    "PickHighlight",
    "VideoRepick":      "VideoRepick",
    "VideoPlaceButton": "VideoPlaceButton",
    "VideoPlaceOrder":  "VideoPlaceOrder",
    "MoveCube":         "MoveCube",
    "InsertPeg":        "InsertPeg",
    "PatternLock":      "PatternLock",
    "RouteStick":       "RouteStick",
}


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    success_rates = data["success_rate"]
    overall = data["total_success_rate"]

    # Build ordered task list grouped by suite
    tasks = []
    colors = []
    suite_boundaries = []  # x positions where suite groups start
    suite_names = []

    for suite, task_list in TASK_SUITES.items():
        suite_boundaries.append(len(tasks))
        suite_names.append(suite)
        for t in task_list:
            tasks.append(t)
            colors.append(SUITE_COLORS[suite])

    rates = [success_rates[t] * 100 for t in tasks]
    x = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(13, 5))

    bars = ax.bar(x, rates, color=colors, width=0.65, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{rate:.0f}%",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # Overall mean line
    ax.axhline(overall * 100, color="black", linestyle="--", linewidth=1.4, zorder=5)
    ax.text(
        len(tasks) - 0.3, overall * 100 + 1.2,
        f"Overall: {overall * 100:.1f}%",
        ha="right", va="bottom", fontsize=9, fontstyle="italic",
    )

    # Suite dividers and group labels
    for i, (boundary, suite) in enumerate(zip(suite_boundaries, suite_names)):
        if i > 0:
            ax.axvline(boundary - 0.5, color="#cccccc", linewidth=1.2, linestyle="-")
        mid = boundary + 1.5
        ax.text(
            mid, 42, suite,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=SUITE_COLORS[suite],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TASK_LABELS[t] for t in tasks],
        rotation=35, ha="right", fontsize=8.5,
    )
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title(
        "Frozen π0.5 Baseline — RoboMME Success Rates (50 episodes/task, seed 7)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_ylim(0, 48)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=SUITE_COLORS[s], label=s) for s in TASK_SUITES
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out = Path(__file__).parent.parent / "eval_results" / "baseline_results.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
