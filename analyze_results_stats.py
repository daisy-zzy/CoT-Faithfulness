#!/usr/bin/env python3
"""
Comprehensive statistical analysis of CoT Faithfulness experiment results.
Computes per-problem metrics, aggregates with error bars, and performs significance tests.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Paper-quality plot settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_jsonl(path):
    """Load JSONL file."""
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_per_problem_metrics(entries, use_fixed_answer=True):
    """
    Compute metrics PER PROBLEM, aggregating across rollouts.

    For each problem, we compute:
    - match_rate: fraction of rollouts where B = A
    - accuracy: fraction of rollouts where B = ground_truth
    - is_correct_a: whether Model A got it right (for MWC/MWW stratification)

    Returns arrays for statistical analysis.
    """
    problem_match_rates = []
    problem_accuracies = []
    problem_match_when_correct = []  # match rates for problems where A was correct
    problem_match_when_wrong = []  # match rates for problems where A was wrong

    for entry in entries:
        gt = entry.get("ground_truth_normalized")

        # Get Model A's answer
        if use_fixed_answer:
            ans_a = entry.get("fixed_answer_a") or entry.get("answer_a_normalized")
        else:
            ans_a = entry.get("answer_a_normalized")

        a_is_correct = (ans_a == gt) if (ans_a and gt) else None

        # Compute per-rollout metrics for this problem
        rollout_matches = []
        rollout_correct = []

        for rollout in entry.get("rollouts", []):
            ans_b = rollout.get("answer_normalized")

            if not ans_b:
                continue

            # Match with A
            if ans_a:
                rollout_matches.append(1 if ans_b == ans_a else 0)

            # Correct (matches ground truth)
            if gt:
                rollout_correct.append(1 if ans_b == gt else 0)

        # Aggregate for this problem (mean across rollouts)
        if rollout_matches:
            match_rate = np.mean(rollout_matches)
            problem_match_rates.append(match_rate)

            if a_is_correct is True:
                problem_match_when_correct.append(match_rate)
            elif a_is_correct is False:
                problem_match_when_wrong.append(match_rate)

        if rollout_correct:
            problem_accuracies.append(np.mean(rollout_correct))

    return {
        "match_rates": np.array(problem_match_rates),
        "accuracies": np.array(problem_accuracies),
        "mwc_rates": np.array(problem_match_when_correct),
        "mww_rates": np.array(problem_match_when_wrong),
        "n_problems": len(problem_match_rates),
    }


def compute_stats(arr):
    """Compute mean, std, 95% CI for an array."""
    n = len(arr)
    if n == 0:
        return {"mean": 0, "std": 0, "ci_low": 0, "ci_high": 0, "n": 0}

    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    se = std / np.sqrt(n)

    # 95% confidence interval
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se) if n > 1 else (mean, mean)

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "n": n,
    }


def paired_ttest(arr1, arr2):
    """Perform paired t-test between two arrays of per-problem metrics."""
    # Need same problems - assume aligned
    min_len = min(len(arr1), len(arr2))
    if min_len < 2:
        return {"t_stat": 0, "p_value": 1.0, "significant": False}

    t_stat, p_value = stats.ttest_rel(arr1[:min_len], arr2[:min_len])

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def bootstrap_ci(arr, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    if len(arr) == 0:
        return 0, 0, 0

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    ci_low = np.percentile(bootstrap_means, alpha * 100)
    ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return np.mean(arr), ci_low, ci_high


def analyze_with_stats(baselines_dir, interventions_dir):
    """Run full statistical analysis."""

    results = {}

    # Load baseline
    follow_cot_path = baselines_dir / "model_b_follow_cot.jsonl"
    if follow_cot_path.exists():
        baseline_entries = load_jsonl(follow_cot_path)
        baseline_metrics = compute_per_problem_metrics(baseline_entries)
        results["baseline"] = {
            "entries": baseline_entries,
            "metrics": baseline_metrics,
            "match": compute_stats(baseline_metrics["match_rates"]),
            "accuracy": compute_stats(baseline_metrics["accuracies"]),
            "mwc": compute_stats(baseline_metrics["mwc_rates"]),
            "mww": compute_stats(baseline_metrics["mww_rates"]),
        }

    # Load all interventions
    for path in sorted(interventions_dir.glob("*.jsonl")):
        name = path.stem
        entries = load_jsonl(path)
        metrics = compute_per_problem_metrics(entries)

        results[name] = {
            "entries": entries,
            "metrics": metrics,
            "match": compute_stats(metrics["match_rates"]),
            "accuracy": compute_stats(metrics["accuracies"]),
            "mwc": compute_stats(metrics["mwc_rates"]),
            "mww": compute_stats(metrics["mww_rates"]),
        }

        # Compare to baseline with paired t-test
        if "baseline" in results:
            baseline_match = results["baseline"]["metrics"]["match_rates"]
            intervention_match = metrics["match_rates"]
            results[name]["vs_baseline"] = paired_ttest(
                baseline_match, intervention_match
            )

    return results


def print_statistical_summary(results):
    """Print formatted statistical summary."""

    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS - Per-Problem Aggregation with 95% CI")
    print("=" * 100)

    # Header
    print(
        f"\n{'Intervention':<25} {'Match Rate':>15} {'Flip Rate':>15} {'Accuracy':>15} {'MWC':>12} {'MWW':>12} {'p-value':>10}"
    )
    print("-" * 100)

    # Sort by intervention type
    sorted_keys = ["baseline"] + sorted([k for k in results.keys() if k != "baseline"])

    for name in sorted_keys:
        r = results[name]
        match = r["match"]
        acc = r["accuracy"]
        mwc = r["mwc"]
        mww = r["mww"]

        # Format with CI
        match_str = (
            f"{100*match['mean']:.1f}±{100*(match['ci_high']-match['mean']):.1f}"
        )
        flip_str = (
            f"{100*(1-match['mean']):.1f}±{100*(match['ci_high']-match['mean']):.1f}"
        )
        acc_str = f"{100*acc['mean']:.1f}±{100*(acc['ci_high']-acc['mean']):.1f}"
        mwc_str = (
            f"{100*mwc['mean']:.1f}±{100*(mwc['ci_high']-mwc['mean']):.1f}"
            if mwc["n"] > 0
            else "-"
        )
        mww_str = (
            f"{100*mww['mean']:.1f}±{100*(mww['ci_high']-mww['mean']):.1f}"
            if mww["n"] > 0
            else "-"
        )

        # p-value vs baseline
        if name == "baseline":
            p_str = "-"
        else:
            p = r.get("vs_baseline", {}).get("p_value", 1.0)
            if p < 0.001:
                p_str = "<0.001***"
            elif p < 0.01:
                p_str = f"{p:.3f}**"
            elif p < 0.05:
                p_str = f"{p:.3f}*"
            else:
                p_str = f"{p:.3f}"

        display_name = name.replace("_", " ").title()[:24]
        print(
            f"{display_name:<25} {match_str:>15} {flip_str:>15} {acc_str:>15} {mwc_str:>12} {mww_str:>12} {p_str:>10}"
        )


def plot_with_error_bars(results, output_dir):
    """Create plots with proper error bars."""

    # Plot 1: Intervention comparison with error bars
    fig, ax = plt.subplots(figsize=(14, 6))

    # Select key interventions to display
    interventions = [
        ("baseline", "Baseline"),
        ("truncate_first_k1", "Trunc First k=1"),
        ("truncate_first_k3", "Trunc First k=3"),
        ("truncate_last_k1", "Trunc Last k=1"),
        ("truncate_last_k3", "Trunc Last k=3"),
        ("truncate_last_k5", "Trunc Last k=5"),
        ("truncate_percent_p0.3", "Trunc 30%"),
        ("truncate_percent_p0.5", "Trunc 50%"),
        ("filler_replacement_p0.3", "Filler 30%"),
        ("filler_replacement_p0.5", "Filler 50%"),
        ("error_injection", "Error Inject"),
    ]

    labels = []
    match_means = []
    match_errs = []
    flip_means = []
    flip_errs = []
    significance = []

    for key, label in interventions:
        if key in results:
            r = results[key]
            labels.append(label)
            match_means.append(100 * r["match"]["mean"])
            match_errs.append(100 * (r["match"]["ci_high"] - r["match"]["mean"]))
            flip_means.append(100 * (1 - r["match"]["mean"]))
            flip_errs.append(100 * (r["match"]["ci_high"] - r["match"]["mean"]))

            # Check significance
            p = r.get("vs_baseline", {}).get("p_value", 1.0)
            significance.append(p < 0.05)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        match_means,
        width,
        yerr=match_errs,
        label="Match Rate",
        color="tab:blue",
        alpha=0.8,
        capsize=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        flip_means,
        width,
        yerr=flip_errs,
        label="Flip Rate",
        color="tab:orange",
        alpha=0.8,
        capsize=3,
    )

    # Add significance markers
    for i, (sig, bar) in enumerate(zip(significance, bars2)):
        if sig and i > 0:  # Skip baseline
            ax.annotate(
                "*",
                xy=(x[i] + width / 2, flip_means[i] + flip_errs[i] + 2),
                ha="center",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_ylabel("Rate (%)")
    ax.set_title("Intervention Effects on Model B Agreement (Mean ± 95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 85)

    # Add baseline reference line
    baseline_match = results.get("baseline", {}).get("match", {}).get("mean", 0) * 100
    ax.axhline(y=baseline_match, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / "intervention_comparison_with_ci.png")
    plt.savefig(output_dir / "intervention_comparison_with_ci.pdf")
    print(f"Saved: {output_dir / 'intervention_comparison_with_ci.png'}")
    plt.close()

    # Plot 2: Truncation dose-response with error bands
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # K-based truncations
    ax = axes[0]
    for prefix, color, marker, label in [
        ("truncate_first", "tab:blue", "o", "Truncate First"),
        ("truncate_last", "tab:orange", "s", "Truncate Last"),
        ("truncate_contiguous", "tab:green", "^", "Truncate Contiguous"),
    ]:
        ks = []
        means = []
        ci_lows = []
        ci_highs = []

        for k in [1, 2, 3, 5]:
            key = f"{prefix}_k{k}"
            if key in results:
                ks.append(k)
                means.append(100 * (1 - results[key]["match"]["mean"]))  # Flip rate
                ci_lows.append(100 * (1 - results[key]["match"]["ci_high"]))
                ci_highs.append(100 * (1 - results[key]["match"]["ci_low"]))

        if ks:
            ax.errorbar(
                ks,
                means,
                yerr=[
                    np.array(means) - np.array(ci_lows),
                    np.array(ci_highs) - np.array(means),
                ],
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
                capsize=4,
            )

    # Add baseline
    if "baseline" in results:
        baseline_flip = 100 * (1 - results["baseline"]["match"]["mean"])
        ax.axhline(
            y=baseline_flip, color="gray", linestyle="--", alpha=0.7, label="Baseline"
        )

    ax.set_xlabel("Number of Sentences Removed (k)")
    ax.set_ylabel("Flip Rate (%)")
    ax.set_title("Sentence Truncation Effects (Mean ± 95% CI)")
    ax.legend()
    ax.set_ylim(10, 25)  # Adjusted for actual flip rate range
    ax.set_xticks([1, 2, 3, 5])

    # Percentage-based interventions
    ax = axes[1]
    for prefix, color, marker, label in [
        ("truncate_percent", "tab:purple", "D", "Random Truncation"),
        ("filler_replacement", "tab:red", "v", "Filler Replacement"),
    ]:
        ps = []
        means = []
        ci_lows = []
        ci_highs = []

        for p in [0.1, 0.2, 0.3, 0.5]:
            key = f"{prefix}_p{p}"
            if key in results:
                ps.append(p * 100)
                means.append(100 * (1 - results[key]["match"]["mean"]))
                ci_lows.append(100 * (1 - results[key]["match"]["ci_high"]))
                ci_highs.append(100 * (1 - results[key]["match"]["ci_low"]))

        if ps:
            ax.errorbar(
                ps,
                means,
                yerr=[
                    np.array(means) - np.array(ci_lows),
                    np.array(ci_highs) - np.array(means),
                ],
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
                capsize=4,
            )

    if "baseline" in results:
        ax.axhline(
            y=baseline_flip, color="gray", linestyle="--", alpha=0.7, label="Baseline"
        )

    ax.set_xlabel("Intervention Rate (%)")
    ax.set_ylabel("Flip Rate (%)")
    ax.set_title("Percentage-Based Interventions (Mean ± 95% CI)")
    ax.legend()
    ax.set_ylim(10, 25)  # Adjusted for actual flip rate range

    plt.tight_layout()
    plt.savefig(output_dir / "truncation_dose_response.png")
    plt.savefig(output_dir / "truncation_dose_response.pdf")
    print(f"Saved: {output_dir / 'truncation_dose_response.png'}")
    plt.close()

    # Plot 3: Error injection comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    if "error_injection" in results:
        # Load raw data for modified vs unmodified breakdown
        error_path = Path(
            "/home/riyaza/10701-project/outputs/interventions/error_injection.jsonl"
        )
        if error_path.exists():
            entries = load_jsonl(error_path)

            modified = [
                e for e in entries if e.get("original_cot") != e.get("modified_cot")
            ]
            unmodified = [
                e for e in entries if e.get("original_cot") == e.get("modified_cot")
            ]

            mod_metrics = compute_per_problem_metrics(modified)
            unmod_metrics = compute_per_problem_metrics(unmodified)
            baseline_metrics = results["baseline"]["metrics"]

            categories = [
                "Baseline\n(No Intervention)",
                "Unmodified CoTs\n(Extraction Failed)",
                "Modified CoTs\n(Error Injected)",
            ]

            # Match rates with bootstrap CI
            baseline_mean, baseline_low, baseline_high = bootstrap_ci(
                baseline_metrics["match_rates"]
            )
            unmod_mean, unmod_low, unmod_high = bootstrap_ci(
                unmod_metrics["match_rates"]
            )
            mod_mean, mod_low, mod_high = bootstrap_ci(mod_metrics["match_rates"])

            means = [100 * baseline_mean, 100 * unmod_mean, 100 * mod_mean]
            errs_low = [
                100 * (baseline_mean - baseline_low),
                100 * (unmod_mean - unmod_low),
                100 * (mod_mean - mod_low),
            ]
            errs_high = [
                100 * (baseline_high - baseline_mean),
                100 * (unmod_high - unmod_mean),
                100 * (mod_high - mod_mean),
            ]

            colors = ["tab:gray", "tab:blue", "tab:orange"]
            x = np.arange(len(categories))

            bars = ax.bar(
                x,
                means,
                yerr=[errs_low, errs_high],
                color=colors,
                alpha=0.8,
                capsize=5,
                edgecolor="black",
                linewidth=1,
            )

            ax.set_ylabel("Match Rate (%)")
            ax.set_title("Error Injection Effect (Mean ± 95% Bootstrap CI)")
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)

            # Add value labels
            for bar, mean in zip(bars, means):
                ax.annotate(
                    f"{mean:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, mean + 5),
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )

            # Add sample sizes
            n_baseline = len(baseline_metrics["match_rates"])
            n_unmod = len(unmod_metrics["match_rates"])
            n_mod = len(mod_metrics["match_rates"])
            ax.annotate(f"n={n_baseline}", xy=(0, 5), ha="center", fontsize=9)
            ax.annotate(f"n={n_unmod}", xy=(1, 5), ha="center", fontsize=9)
            ax.annotate(f"n={n_mod}", xy=(2, 5), ha="center", fontsize=9)

            # Statistical test
            t_stat, p_val = stats.ttest_ind(
                mod_metrics["match_rates"], baseline_metrics["match_rates"]
            )
            sig_str = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            ax.annotate(
                f"p<0.001{sig_str}",
                xy=(2, means[2] + errs_high[2] + 8),
                ha="center",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "error_injection_stats.png")
    plt.savefig(output_dir / "error_injection_stats.pdf")
    print(f"Saved: {output_dir / 'error_injection_stats.png'}")
    plt.close()


def create_latex_table_with_stats(results, output_dir):
    """Create LaTeX table with statistical notation."""

    latex = r"""
\begin{table*}[t]
\centering
\caption{Intervention Effects on Model B's Answer Agreement with Model A. 
Values are mean $\pm$ 95\% CI across 5000 problems. 
$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ vs.\ baseline (paired t-test).}
\label{tab:interventions}
\small
\begin{tabular}{llcccccc}
\toprule
\textbf{Intervention} & \textbf{Param} & \textbf{Match Rate} & \textbf{Flip Rate} & \textbf{Accuracy} & \textbf{MWC} & \textbf{MWW} & \textbf{n} \\
\midrule
"""

    def format_with_ci(stats_dict, multiply=100):
        mean = stats_dict["mean"] * multiply
        ci_err = (stats_dict["ci_high"] - stats_dict["mean"]) * multiply
        return f"${mean:.1f} \\pm {ci_err:.1f}$"

    def get_sig_marker(r):
        p = r.get("vs_baseline", {}).get("p_value", 1.0)
        if p < 0.001:
            return "$^{***}$"
        elif p < 0.01:
            return "$^{**}$"
        elif p < 0.05:
            return "$^{*}$"
        return ""

    # Baseline
    if "baseline" in results:
        r = results["baseline"]
        latex += f"Baseline & - & {format_with_ci(r['match'])} & {format_with_ci({'mean': 1-r['match']['mean'], 'ci_high': 1-r['match']['ci_low'], 'ci_low': 1-r['match']['ci_high']})} & {format_with_ci(r['accuracy'])} & {format_with_ci(r['mwc'])} & {format_with_ci(r['mww'])} & {r['match']['n']} \\\\\n"
        latex += "\\midrule\n"

    # Group interventions
    groups = [
        ("Truncate First", "truncate_first", ["k1", "k3", "k5"]),
        ("Truncate Last", "truncate_last", ["k1", "k3", "k5"]),
        ("Truncate Contiguous", "truncate_contiguous", ["k1", "k3", "k5"]),
        ("Truncate Percent", "truncate_percent", ["p0.1", "p0.3", "p0.5"]),
        ("Filler Replace", "filler_replacement", ["p0.1", "p0.3", "p0.5"]),
        ("Error Injection", "error_injection", [""]),
    ]

    for group_name, prefix, params in groups:
        first = True
        for param in params:
            key = f"{prefix}_{param}" if param else prefix
            if key in results:
                r = results[key]
                name = group_name if first else ""
                sig = get_sig_marker(r)

                flip_stats = {
                    "mean": 1 - r["match"]["mean"],
                    "ci_high": 1 - r["match"]["ci_low"],
                    "ci_low": 1 - r["match"]["ci_high"],
                }

                latex += f"{name} & {param} & {format_with_ci(r['match'])}{sig} & {format_with_ci(flip_stats)} & {format_with_ci(r['accuracy'])} & {format_with_ci(r['mwc'])} & {format_with_ci(r['mww'])} & {r['match']['n']} \\\\\n"
                first = False
        latex += "\\midrule\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""

    print("\n" + "=" * 80)
    print("LATEX TABLE WITH STATISTICS")
    print("=" * 80)
    print(latex)

    with open(output_dir / "results_table.tex", "w") as f:
        f.write(latex)
    print(f"\nSaved: {output_dir / 'results_table.tex'}")


def main():
    base_dir = Path("/home/riyaza/10701-project")
    baselines_dir = base_dir / "outputs" / "baselines"
    interventions_dir = base_dir / "outputs" / "interventions"
    output_dir = base_dir / "outputs" / "analysis"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("STATISTICAL ANALYSIS OF COT FAITHFULNESS EXPERIMENTS")
    print("=" * 80)

    # Run analysis
    results = analyze_with_stats(baselines_dir, interventions_dir)

    # Print summary
    print_statistical_summary(results)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS WITH ERROR BARS")
    print("=" * 80)
    plot_with_error_bars(results, output_dir)

    # Create LaTeX table
    create_latex_table_with_stats(results, output_dir)

    # Summary statistics
    print("\n" + "=" * 80)
    print("KEY STATISTICAL FINDINGS")
    print("=" * 80)

    if "baseline" in results and "error_injection" in results:
        baseline_flip = 1 - results["baseline"]["match"]["mean"]
        error_flip = 1 - results["error_injection"]["match"]["mean"]

        print(f"\nBaseline Flip Rate: {100*baseline_flip:.1f}%")
        print(f"Error Injection Flip Rate: {100*error_flip:.1f}%")
        print(f"Absolute Increase: {100*(error_flip - baseline_flip):.1f}%")
        print(
            f"Relative Increase: {100*(error_flip - baseline_flip)/baseline_flip:.1f}%"
        )

        p = results["error_injection"].get("vs_baseline", {}).get("p_value", 1.0)
        print(f"p-value vs baseline: {p:.2e}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
