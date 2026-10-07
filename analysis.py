"""Analyse the runs of `train.py`: figures, per-run summary and statistical tests

Usage (from the project directory, with the virtual environment activated):

    python analysis.py
    python analysis.py --td3-critic qmin --last 5

Reads every `results/*.json` and writes, in `figures/`:

- `learning_curves.pdf/.png`: evaluation return along training
- `bias.pdf/.png`: critic bias (Q - Monte Carlo return) along training
- `runs.csv`: one row per run (final performance, final and average bias)
- `stats.txt`: LayerNorm vs no LayerNorm tests, for each algorithm

Variability across seeds is shown as a 95% confidence interval of the mean
(Student t, n = number of seeds).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ALGOS = ["ddpg", "td3"]
ALGO_NAMES = {"ddpg": "DDPG", "td3": "TD3"}
# Color follows the LayerNorm condition (same color in every panel); the line
# style and marker repeat it so that the figures also read in grayscale
STYLES = {
    False: dict(color="#2a78d6", linestyle="-", marker="o", label="without LayerNorm"),
    True: dict(color="#eb6834", linestyle="--", marker="s", label="with LayerNorm"),
}
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_runs(results_dir: Path, td3_critic: str) -> list[dict]:
    """One dict per run, with its evaluation and bias curves as arrays"""
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        log = json.loads(path.read_text())
        evaluations = log["evaluations"]
        critic = "q" if log["algo"] == "ddpg" else td3_critic
        runs.append(
            {
                "name": path.stem,
                "algo": log["algo"],
                "layernorm": log["config"]["layernorm"],
                "seed": log["config"]["seed"],
                "steps": np.array([e["step"] for e in evaluations]),
                "reward": np.array([e["mean_reward"] for e in evaluations]),
                "bias": np.array([e["bias"][critic]["mean_bias"] for e in evaluations]),
                "bias_std": np.array([e["bias"][critic]["std_bias"] for e in evaluations]),
                "mean_q": np.array([e["bias"][critic]["mean_q"] for e in evaluations]),
                "return_mc": np.array([e["bias"]["mean_return_mc"] for e in evaluations]),
            }
        )
    return runs


def group(runs: list[dict], algo: str, layernorm: bool) -> list[dict]:
    return [r for r in runs if r["algo"] == algo and r["layernorm"] == layernorm]


def mean_ci(values: np.ndarray, confidence: float = 0.95):
    """Mean over axis 0 and the half-width of its Student-t confidence interval"""
    n = values.shape[0]
    mean = values.mean(axis=0)
    if n < 2:
        return mean, np.zeros_like(mean)
    sem = values.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, sem * stats.t.ppf((1 + confidence) / 2, n - 1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT)


def plot_curves(runs, key: str, ylabel: str, title: str, path: Path, zero_line=False):
    """One panel per algorithm, one line (mean +- 95% CI over seeds) per condition"""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for layernorm in (False, True):
            group_runs = group(runs, algo, layernorm)
            if not group_runs:
                continue
            steps = group_runs[0]["steps"] / 1000
            mean, ci = mean_ci(np.stack([r[key] for r in group_runs]))
            style = STYLES[layernorm]
            ax.fill_between(steps, mean - ci, mean + ci, color=style["color"], alpha=0.15, linewidth=0)
            ax.plot(
                steps, mean,
                color=style["color"], linestyle=style["linestyle"], linewidth=1.8,
                marker=style["marker"], markersize=3.5, markevery=5,
                label=f"{style['label']} (n={len(group_runs)})",
            )
        if zero_line:
            ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_title(ALGO_NAMES[algo], color=TEXT, fontsize=10)
        ax.set_xlabel("environment steps (×1000)", color=TEXT)
        style_axes(ax)
    axes[0].set_ylabel(ylabel, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, color=TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-run summary and tests
# ---------------------------------------------------------------------------


def summarize(runs, last: int) -> list[dict]:
    """One row per run: final performance and bias (mean of the last evaluations)"""
    rows = []
    for r in runs:
        rows.append(
            {
                "run": r["name"],
                "algo": r["algo"],
                "layernorm": r["layernorm"],
                "seed": r["seed"],
                "final_reward": r["reward"][-last:].mean(),
                "final_bias": r["bias"][-last:].mean(),
                "final_bias_std": r["bias_std"][-last:].mean(),
                "mean_bias_over_training": r["bias"].mean(),
            }
        )
    return rows


def compare(rows, algo: str, measure: str) -> str:
    """Welch t-test and Mann-Whitney U test, LayerNorm vs no LayerNorm"""
    without = np.array([r[measure] for r in rows if r["algo"] == algo and not r["layernorm"]])
    with_ln = np.array([r[measure] for r in rows if r["algo"] == algo and r["layernorm"]])
    welch = stats.ttest_ind(with_ln, without, equal_var=False)
    mannwhitney = stats.mannwhitneyu(with_ln, without, alternative="two-sided")
    return (
        f"{ALGO_NAMES[algo]:5} {measure:24} "
        f"without LN {without.mean():8.1f} ± {without.std(ddof=1):6.1f} (n={len(without)})  "
        f"with LN {with_ln.mean():8.1f} ± {with_ln.std(ddof=1):6.1f} (n={len(with_ln)})  "
        f"Welch t={welch.statistic:6.2f} p={welch.pvalue:.4f}  "
        f"Mann-Whitney U={mannwhitney.statistic:4.1f} p={mannwhitney.pvalue:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--td3-critic", choices=["q1", "q2", "qmin"], default="q1",
                        help="Which TD3 critic the bias is reported for")
    parser.add_argument("--last", type=int, default=3,
                        help="Number of final evaluations averaged for the final values")
    args = parser.parse_args()

    runs = load_runs(Path(args.results), args.td3_critic)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs loaded; TD3 bias from critic '{args.td3_critic}'; "
          f"final values = mean of the last {args.last} evaluations")

    plot_curves(runs, "reward", "evaluation return",
                "Performance on LunarLanderContinuous-v3", out / "learning_curves")
    plot_curves(runs, "bias", "bias: Q − Monte Carlo return",
                "Critic bias along training", out / "bias", zero_line=True)

    rows = summarize(runs, args.last)
    with open(out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"LayerNorm vs no LayerNorm (mean ± std over seeds; two-sided tests; "
        f"final = mean of the last {args.last} evaluations; TD3 critic: {args.td3_critic})",
    ]
    for measure in ("final_reward", "final_bias", "mean_bias_over_training"):
        for algo in ALGOS:
            lines.append(compare(rows, algo, measure))
    text = "\n".join(lines)
    (out / "stats.txt").write_text(text + "\n")
    print(text)
    print(f"Figures, runs.csv and stats.txt written to {out}/")


if __name__ == "__main__":
    main()
