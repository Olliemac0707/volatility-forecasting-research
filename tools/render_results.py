"""Render only frozen aggregate results; no prices, refits or new performance metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "docs/results/summary-v1.json").read_text())
    output = root / "docs/figures"
    output.mkdir(exist_ok=True)
    names = ["historical_63", "ewma_094", "garch_11"]
    labels = ["Historical (63)", "EWMA (0.94)", "GARCH (1,1)"]
    values = [summary["scores"][name]["mean_qlike"] for name in names]
    interval = summary["primary_hac"]
    mean = interval["mean_difference"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
        left.scatter(values, range(3), color=["#7b8794", "#2e748b", "#234a86"], s=65)
        left.set_yticks(range(3), labels)
        left.invert_yaxis()
        left.set_xlabel("Mean QLIKE (lower is better)")
        left.xaxis.set_major_locator(MaxNLocator(5))
        left.set_title("Observed forecast losses")
        left.grid(axis="x", alpha=0.2)
        left.margins(y=0.3)
        right.errorbar(
            [mean],
            [0],
            xerr=[[mean - interval["lower_95"]], [interval["upper_95"] - mean]],
            fmt="o",
            color="#234a86",
            capsize=7,
            markersize=7,
        )
        right.axvline(0, color="#b64933", linestyle="--", linewidth=1)
        right.set_yticks([0], ["GARCH − EWMA"])
        right.set_ylim(-1, 1)
        right.set_xlabel("GARCH − EWMA QLIKE loss (negative favours GARCH)", fontsize=9)
        right.set_title("Primary contrast: Newey–West 95% interval", fontsize=11)
        right.grid(axis="x", alpha=0.2)
        right.text(
            0.5,
            0.15,
            f"Mean: {mean:.6f}\n95% CI: [{interval['lower_95']:.6f}, "
            f"{interval['upper_95']:.6f}]\nInterval includes zero",
            ha="center",
            transform=right.transAxes,
        )
        fig.suptitle("SPY variance forecasts · 2020–2025 · 1,508 paired sessions", fontsize=14)
        fig.text(
            0.5,
            0.015,
            "Retrospective forecast evaluation. No allocation backtest or alpha claim. "
            "GARCH fallbacks: 0.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.93), w_pad=3)
        for suffix in ("svg", "png", "pdf"):
            fig.savefig(output / f"forecast-comparison.{suffix}", dpi=180)
        plt.close(fig)
    print(f"Rendered frozen results in {output}")


if __name__ == "__main__":
    main()
