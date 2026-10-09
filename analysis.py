"""Analyse the runs of `train.py`: figures, per-run summary and statistical tests

Usage (from the project directory, with the virtual environment activated):

    python analysis.py
    python analysis.py --td3-critic qmin --last 5

Reads every `results/*.json` and writes, in `figures/`:

- `learning_curves.pdf/.png`: evaluation return along training
- `bias.pdf/.png`: critic bias (Q - Monte Carlo return) along training
- `bias_vs_performance.pdf/.png`: final bias vs final performance, one point
  per run
- `runs.csv`: one row per run (final performance, final and average bias)
- `stats.txt`: LayerNorm vs no LayerNorm tests, for each algorithm, TD3 vs
  DDPG tests, for each condition, Holm-corrected p-values, and the Spearman
  correlation between final bias and final performance (all runs, and within
  each algorithm)

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
ALGO_MARKERS = {"ddpg": "o", "td3": "^"}
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
                "learning_starts": log["config"]["learning_starts"],
                "steps": np.array([e["step"] for e in evaluations]),
                "reward": np.array([e["mean_reward"] for e in evaluations]),
                "bias": np.array([e["bias"][critic]["mean_bias"] for e in evaluations]),
                "bias_std": np.array([e["bias"][critic].get("std_bias", np.nan) for e in evaluations]),
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
        ax.set_xlabel("environment steps (x1000)", color=TEXT)
        style_axes(ax)
    axes[0].set_ylabel(ylabel, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    #fig.suptitle(title, color=TEXT, fontsize=11)
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
        trained = r["steps"] > r["learning_starts"]
        rows.append(
            {
                "run": r["name"],
                "algo": r["algo"],
                "layernorm": r["layernorm"],
                "seed": r["seed"],
                "final_reward": r["reward"][-last:].mean(),
                "final_bias": r["bias"][-last:].mean(),
                "final_bias_std": r["bias_std"][-last:].mean(),
                "mean_bias_over_training": r["bias"][trained].mean(),
            }
        )
    return rows

def run_tests(a, b):
    welch = stats.ttest_ind(a, b, equal_var=False)
    mannwhitney = stats.mannwhitneyu(a, b, alternative="two-sided")
    return welch, mannwhitney

def compare(rows, algo: str, measure: str, tests: list) -> str:
    """Welch t-test and Mann-Whitney U test, LayerNorm vs no LayerNorm"""
    without = np.array([r[measure] for r in rows if r["algo"] == algo and not r["layernorm"]])
    with_ln = np.array([r[measure] for r in rows if r["algo"] == algo and r["layernorm"]])
    welch = stats.ttest_ind(with_ln, without, equal_var=False)
    mannwhitney = stats.mannwhitneyu(with_ln, without, alternative="two-sided")
    tests.append((f"{ALGO_NAMES[algo]}: LN vs no LN, {measure}", welch.pvalue))
    return (
        f"{ALGO_NAMES[algo]:5} {measure:24} "
        f"without LN {without.mean():8.1f} ± {without.std(ddof=1):6.1f} (n={len(without)})  "
        f"with LN {with_ln.mean():8.1f} ± {with_ln.std(ddof=1):6.1f} (n={len(with_ln)})  "
        f"Welch t={welch.statistic:6.2f} p={welch.pvalue:.4f}  "
        f"Mann-Whitney U={mannwhitney.statistic:4.1f} p={mannwhitney.pvalue:.4f}"
    )


def compare_algos(rows, layernorm: bool, measure: str, tests: list) -> str:
    """[ADDED] Welch t-test and Mann-Whitney U test, TD3 vs DDPG, same LN condition"""
    ddpg = np.array([r[measure] for r in rows if r["algo"] == "ddpg" and r["layernorm"] == layernorm])
    td3 = np.array([r[measure] for r in rows if r["algo"] == "td3" and r["layernorm"] == layernorm])
    welch, mannwhitney = run_tests(td3, ddpg)
    condition = "with LN" if layernorm else "no LN"
    tests.append((f"TD3 vs DDPG ({condition}), {measure}", welch.pvalue))
    return (
        f"{condition:7} {measure:24} "
        f"DDPG {ddpg.mean():8.1f} ± {ddpg.std(ddof=1):6.1f} (n={len(ddpg)})  "
        f"TD3 {td3.mean():8.1f} ± {td3.std(ddof=1):6.1f} (n={len(td3)})  "
        f"Welch t={welch.statistic:6.2f} p={welch.pvalue:.4f}  "
        f"Mann-Whitney U={mannwhitney.statistic:4.1f} p={mannwhitney.pvalue:.4f}"
    )
 
 
def holm(pvalues: np.ndarray) -> np.ndarray:
    """[ADDED] Holm-Bonferroni adjusted p-values (controls the family-wise error)"""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted

def spearman(rows, algo: str | None) -> tuple[float, float, int]:
    """Spearman correlation between final bias and final performance"""
    selected = [r for r in rows if algo is None or r["algo"] == algo]
    result = stats.spearmanr(
        [r["final_bias"] for r in selected], [r["final_reward"] for r in selected]
    )
    return result.statistic, result.pvalue, len(selected)


def plot_bias_vs_performance(rows, correlations: dict, path: Path):
    """One point per run: final bias (x) vs final performance (y)"""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for algo in ALGOS:
        for layernorm in (False, True):
            selected = [r for r in rows if r["algo"] == algo and r["layernorm"] == layernorm]
            if not selected:
                continue
            ax.scatter(
                [r["final_bias"] for r in selected],
                [r["final_reward"] for r in selected],
                marker=ALGO_MARKERS[algo], s=42, color=STYLES[layernorm]["color"],
                edgecolors="white", linewidths=1.0, zorder=3,
                label=f"{ALGO_NAMES[algo]}, {STYLES[layernorm]['label']}",
            )
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlabel("final bias : Q - Monte Carlo return", color=TEXT)
    ax.set_ylabel("final evaluation return", color=TEXT)
    text = "\n".join(
        f"{name}: ρ = {rho:.2f}, " + ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") + f" (n={n})"
        for name, (rho, p, n) in correlations.items()
    )
    ax.text(0.98, 0.97, "Spearman " + text, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=TEXT_SECONDARY)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5,
              frameon=False)
    #ax.set_title("Final bias vs final performance (one point per run)", color=TEXT, fontsize=10)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    plot_curves(runs, "bias", "bias : Q - Monte Carlo return",
                "Critic bias along training", out / "bias", zero_line=True)

    rows = summarize(runs, args.last)
    with open(out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    measures = ("final_reward", "final_bias", "mean_bias_over_training")
    tests = []

    lines = [
        f"LayerNorm vs no LayerNorm (mean ± std over seeds; two-sided tests; "
        f"final = mean of the last {args.last} evaluations; TD3 critic: {args.td3_critic})",
    ]
    for measure in measures:
        for algo in ALGOS:
            lines.append(compare(rows, algo, measure, tests))

    lines.append("")
    lines.append("TD3 vs DDPG (mean ± std over seeds; two-sided tests)")
    for measure in measures:
        for layernorm in (False, True):
            lines.append(compare_algos(rows, layernorm, measure, tests))
 
    adjusted = holm(np.array([p for _, p in tests]))
    lines.append("")
    lines.append(f"Holm correction over the {len(tests)} Welch tests above")
    for (name, p), p_adj in zip(tests, adjusted):
        lines.append(f"{name:52} p={p:.4f}  p_Holm={p_adj:.4f}")

    # Q2: does a larger bias go with a lower performance?
    correlations = {"all runs": spearman(rows, None)}
    for algo in ALGOS:
        correlations[ALGO_NAMES[algo]] = spearman(rows, algo)
    plot_bias_vs_performance(rows, correlations, out / "bias_vs_performance")
    lines.append("")
    lines.append("Spearman correlation between final bias and final performance")
    for name, (rho, p, n) in correlations.items():
        lines.append(f"{name:9} rho={rho:6.2f}  p={p:.4f}  (n={n})")
    text = "\n".join(lines)
    (out / "stats.txt").write_text(text + "\n")
    print(text)
    print(f"Figures, runs.csv and stats.txt written to {out}/")


if __name__ == "__main__":
    main()
