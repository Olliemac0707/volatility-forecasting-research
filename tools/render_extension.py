"""Plot only the frozen higher-order aggregate results; no refits or new tests."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "docs/results/higher-order-v1/summary.json").read_text())
    output = root / "docs/figures"
    names = ["garch_11", "garch_21", "garch_12"]
    labels = ["Frozen GARCH(1,1)", "GARCH(2,1)", "GARCH(1,2)"]
    means = [result["scores"][name]["mean_qlike"] for name in names]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
        left.scatter(means, range(3), s=60, color=["#234a86", "#2e748b", "#7b8794"])
        left.set_yticks(range(3), labels)
        left.invert_yaxis()
        left.margins(y=0.3, x=0.15)
        left.xaxis.set_major_locator(MaxNLocator(4))
        left.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        left.set_xlabel("Mean QLIKE (lower is better)")
        left.set_title("Observed forecast loss")
        left.grid(axis="x", alpha=0.2)
        for y, name in enumerate(names[1:]):
            interval = result["contrasts"][name]["marginal_hac_95"]
            mean = interval["mean_difference"]
            right.errorbar(
                [mean],
                [y],
                xerr=[[mean - interval["lower_95"]], [interval["upper_95"] - mean]],
                fmt="o",
                capsize=6,
                color=["#2e748b", "#7b8794"][y],
            )
        right.set_yticks([0, 1], labels[1:])
        right.set_ylim(1.8, -0.7)
        right.axvline(0, color="#b64933", linestyle="--", linewidth=1)
        right.xaxis.set_major_locator(MaxNLocator(5))
        right.set_xlabel("Candidate − GARCH(1,1) QLIKE loss", fontsize=9)
        right.set_title("Marginal 95% Newey–West intervals", fontsize=11)
        right.grid(axis="x", alpha=0.2)
        p_values = [result["contrasts"][name]["holm_adjusted_p"] for name in names[1:]]
        right.text(
            0.5,
            0.03,
            f"Holm-adjusted p: {p_values[0]:.3f}, {p_values[1]:.3f}\n"
            "Negative differences favour the candidate",
            transform=right.transAxes,
            ha="center",
            fontsize=9,
        )
        fig.suptitle("Higher-order GARCH · SPY, 2020–2025 · 1,508 paired targets", fontsize=14)
        fig.text(
            0.5,
            0.02,
            "Exploratory extension; known baseline and reused sample. "
            "Family decisions use Holm across two tests.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=3)
        for suffix in ("svg", "png", "pdf"):
            fig.savefig(output / f"higher-order-comparison.{suffix}", dpi=180)
        plt.close(fig)
    print(f"Rendered frozen higher-order results in {output}")


if __name__ == "__main__":
    main()
