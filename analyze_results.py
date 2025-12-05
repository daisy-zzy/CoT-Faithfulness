#!/usr/bin/env python3
"""
Comprehensive analysis of CoT Faithfulness experiment results.
Generates statistics, tables, and plots for the paper.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving figures

# Set style for paper-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
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


def compute_metrics(entries, use_fixed_answer=True):
    """
    Compute key metrics for a set of entries.

    Returns dict with:
    - accuracy: Pr[B = ground_truth]
    - match_rate (OMR): Pr[B = A]
    - mwc: Pr[B = A | A correct] (Match When Correct)
    - mww: Pr[B = A | A wrong] (Match When Wrong)
    - flip_rate: 1 - match_rate
    """
    total_rollouts = 0
    correct_b = 0
    matches = 0

    # For MWC/MWW
    a_correct_total = 0
    a_correct_matches = 0
    a_wrong_total = 0
    a_wrong_matches = 0

    missing_answers = 0

    for entry in entries:
        gt = entry.get("ground_truth_normalized")

        # Get Model A's answer
        if use_fixed_answer:
            ans_a = entry.get("fixed_answer_a") or entry.get("answer_a_normalized")
        else:
            ans_a = entry.get("answer_a_normalized")

        a_is_correct = (ans_a == gt) if (ans_a and gt) else None

        for rollout in entry.get("rollouts", []):
            ans_b = rollout.get("answer_normalized")

            if not ans_b:
                missing_answers += 1
                continue

            total_rollouts += 1

            # Accuracy
            if gt and ans_b == gt:
                correct_b += 1

            # Match rate
            if ans_a and ans_b == ans_a:
                matches += 1

            # MWC/MWW
            if a_is_correct is True and ans_a:
                a_correct_total += 1
                if ans_b == ans_a:
                    a_correct_matches += 1
            elif a_is_correct is False and ans_a:
                a_wrong_total += 1
                if ans_b == ans_a:
                    a_wrong_matches += 1

    metrics = {
        "total_rollouts": total_rollouts,
        "missing_answers": missing_answers,
        "accuracy": correct_b / total_rollouts if total_rollouts > 0 else 0,
        "match_rate": matches / total_rollouts if total_rollouts > 0 else 0,
        "flip_rate": 1 - (matches / total_rollouts) if total_rollouts > 0 else 0,
        "mwc": a_correct_matches / a_correct_total if a_correct_total > 0 else 0,
        "mww": a_wrong_matches / a_wrong_total if a_wrong_total > 0 else 0,
        "a_correct_total": a_correct_total,
        "a_wrong_total": a_wrong_total,
    }

    return metrics


def analyze_baselines(baselines_dir):
    """Analyze baseline results."""
    print("\n" + "=" * 70)
    print("BASELINE ANALYSIS")
    print("=" * 70)

    results = {}

    # Model A rollouts
    model_a_path = baselines_dir / "model_a_rollouts.jsonl"
    if model_a_path.exists():
        model_a = load_jsonl(model_a_path)

        # Compute Model A accuracy
        total = 0
        correct = 0
        for entry in model_a:
            gt = entry.get("ground_truth_normalized")
            for rollout in entry.get("rollouts", []):
                ans = rollout.get("answer_normalized")
                if ans:
                    total += 1
                    if ans == gt:
                        correct += 1

        results["model_a"] = {
            "n_problems": len(model_a),
            "n_rollouts": total,
            "accuracy": correct / total if total > 0 else 0,
        }
        print(f"\nModel A (Qwen):")
        print(f"  Problems: {len(model_a)}")
        print(f"  Total rollouts: {total}")
        print(f"  Accuracy: {100 * correct / total:.1f}%")

    # Model B no-CoT baseline
    no_cot_path = baselines_dir / "model_b_no_cot.jsonl"
    if no_cot_path.exists():
        no_cot = load_jsonl(no_cot_path)
        metrics = compute_metrics(no_cot, use_fixed_answer=False)
        results["no_cot"] = metrics

        print(f"\nModel B No-CoT Baseline:")
        print(f"  Accuracy: {100 * metrics['accuracy']:.1f}%")
        print(f"  (This is Model B solving problems without any CoT)")

    # Model B follow-CoT baseline
    follow_cot_path = baselines_dir / "model_b_follow_cot.jsonl"
    if follow_cot_path.exists():
        follow_cot = load_jsonl(follow_cot_path)
        metrics = compute_metrics(follow_cot, use_fixed_answer=True)
        results["follow_cot"] = metrics

        print(f"\nModel B Follow-CoT Baseline:")
        print(f"  Accuracy: {100 * metrics['accuracy']:.1f}%")
        print(f"  Match Rate (OMR): {100 * metrics['match_rate']:.1f}%")
        print(f"  MWC (Match When Correct): {100 * metrics['mwc']:.1f}%")
        print(f"  MWW (Match When Wrong): {100 * metrics['mww']:.1f}%")

    return results


def analyze_interventions(interventions_dir, baseline_match_rate=None):
    """Analyze all intervention results."""
    print("\n" + "=" * 70)
    print("INTERVENTION ANALYSIS")
    print("=" * 70)

    results = {}

    # Group interventions by type
    intervention_groups = {
        "truncate_first": [],
        "truncate_last": [],
        "truncate_contiguous": [],
        "truncate_percent": [],
        "filler_replacement": [],
        "error_injection": [],
    }

    for path in sorted(interventions_dir.glob("*.jsonl")):
        name = path.stem
        entries = load_jsonl(path)
        metrics = compute_metrics(entries, use_fixed_answer=True)
        results[name] = metrics

        # Categorize
        for group_name in intervention_groups:
            if name.startswith(group_name):
                # Extract parameter
                param = name.replace(group_name + "_", "")
                intervention_groups[group_name].append((param, metrics))
                break

    # Print results by group
    for group_name, group_results in intervention_groups.items():
        if not group_results:
            continue

        print(f"\n{group_name.upper().replace('_', ' ')}:")
        print("-" * 50)

        for param, metrics in sorted(group_results, key=lambda x: x[0]):
            flip = metrics["flip_rate"]
            match = metrics["match_rate"]
            acc = metrics["accuracy"]
            print(
                f"  {param:12s}: Match={100*match:5.1f}%, Flip={100*flip:5.1f}%, Acc={100*acc:5.1f}%"
            )

    return results, intervention_groups


def create_intervention_comparison_table(baseline_results, intervention_results):
    """Create a LaTeX table comparing all interventions."""
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)

    baseline_match = baseline_results.get("follow_cot", {}).get("match_rate", 0)
    baseline_acc = baseline_results.get("follow_cot", {}).get("accuracy", 0)

    latex = r"""
\begin{table}[h]
\centering
\caption{Intervention Effects on Model B's Answer Agreement with Model A}
\label{tab:interventions}
\begin{tabular}{llccc}
\toprule
\textbf{Intervention} & \textbf{Parameter} & \textbf{Match Rate} & \textbf{Flip Rate} & \textbf{Accuracy} \\
\midrule
Baseline (Follow CoT) & - & %.1f\%% & %.1f\%% & %.1f\%% \\
\midrule
""" % (
        100 * baseline_match,
        100 * (1 - baseline_match),
        100 * baseline_acc,
    )

    # Add intervention rows
    intervention_order = [
        ("truncate_first", "Truncate First"),
        ("truncate_last", "Truncate Last"),
        ("truncate_contiguous", "Truncate Contiguous"),
        ("truncate_percent", "Truncate Percent"),
        ("filler_replacement", "Filler Replacement"),
        ("error_injection", "Error Injection"),
    ]

    for prefix, display_name in intervention_order:
        matching = [
            (k, v) for k, v in intervention_results.items() if k.startswith(prefix)
        ]
        for name, metrics in sorted(matching):
            param = name.replace(prefix + "_", "")
            latex += r"%s & %s & %.1f\%% & %.1f\%% & %.1f\%% \\" % (
                display_name,
                param,
                100 * metrics["match_rate"],
                100 * metrics["flip_rate"],
                100 * metrics["accuracy"],
            )
            latex += "\n"
            display_name = ""  # Only show name once per group
        if matching:
            latex += r"\midrule" + "\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    print(latex)
    return latex


def plot_truncation_effects(intervention_groups, output_dir):
    """Plot flip rate vs truncation severity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: K-based truncations
    ax = axes[0]

    for group_name, color, marker, label in [
        ("truncate_first", "tab:blue", "o", "Truncate First K"),
        ("truncate_last", "tab:orange", "s", "Truncate Last K"),
        ("truncate_contiguous", "tab:green", "^", "Truncate Contiguous K"),
    ]:
        if group_name in intervention_groups and intervention_groups[group_name]:
            data = intervention_groups[group_name]
            # Extract k values
            ks = []
            flips = []
            for param, metrics in data:
                k = int(param.replace("k", ""))
                ks.append(k)
                flips.append(100 * metrics["flip_rate"])

            # Sort by k
            sorted_data = sorted(zip(ks, flips))
            ks, flips = zip(*sorted_data)

            ax.plot(
                ks,
                flips,
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Number of Sentences Removed (k)")
    ax.set_ylabel("Flip Rate (%)")
    ax.set_title("Effect of Sentence Truncation on Answer Deviation")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Plot 2: Percentage-based interventions
    ax = axes[1]

    for group_name, color, marker, label in [
        ("truncate_percent", "tab:purple", "D", "Random Truncation"),
        ("filler_replacement", "tab:red", "v", "Filler Replacement"),
    ]:
        if group_name in intervention_groups and intervention_groups[group_name]:
            data = intervention_groups[group_name]
            # Extract percentages
            pcts = []
            flips = []
            for param, metrics in data:
                p = float(param.replace("p", ""))
                pcts.append(p * 100)
                flips.append(100 * metrics["flip_rate"])

            # Sort by percentage
            sorted_data = sorted(zip(pcts, flips))
            pcts, flips = zip(*sorted_data)

            ax.plot(
                pcts,
                flips,
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Intervention Rate (%)")
    ax.set_ylabel("Flip Rate (%)")
    ax.set_title("Effect of Random Interventions on Answer Deviation")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "truncation_effects.png")
    plt.savefig(output_dir / "truncation_effects.pdf")
    print(f"\nSaved truncation effects plot to {output_dir / 'truncation_effects.png'}")
    plt.close()


def plot_intervention_comparison(baseline_results, intervention_results, output_dir):
    """Bar chart comparing all interventions."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data
    categories = []
    match_rates = []
    flip_rates = []
    accuracies = []

    # Add baseline
    if "follow_cot" in baseline_results:
        categories.append("Baseline\n(Follow CoT)")
        match_rates.append(100 * baseline_results["follow_cot"]["match_rate"])
        flip_rates.append(100 * baseline_results["follow_cot"]["flip_rate"])
        accuracies.append(100 * baseline_results["follow_cot"]["accuracy"])

    # Add interventions in order
    intervention_order = [
        ("truncate_first_k1", "Trunc First\nk=1"),
        ("truncate_first_k3", "Trunc First\nk=3"),
        ("truncate_last_k1", "Trunc Last\nk=1"),
        ("truncate_last_k3", "Trunc Last\nk=3"),
        ("truncate_contiguous_k3", "Trunc Contig\nk=3"),
        ("truncate_percent_p0.3", "Trunc Random\n30%"),
        ("filler_replacement_p0.3", "Filler Replace\n30%"),
        ("error_injection", "Error\nInjection"),
    ]

    for key, label in intervention_order:
        if key in intervention_results:
            categories.append(label)
            match_rates.append(100 * intervention_results[key]["match_rate"])
            flip_rates.append(100 * intervention_results[key]["flip_rate"])
            accuracies.append(100 * intervention_results[key]["accuracy"])

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(
        x - width, match_rates, width, label="Match Rate", color="tab:blue", alpha=0.8
    )
    bars2 = ax.bar(
        x, flip_rates, width, label="Flip Rate", color="tab:orange", alpha=0.8
    )
    bars3 = ax.bar(
        x + width, accuracies, width, label="Accuracy", color="tab:green", alpha=0.8
    )

    ax.set_ylabel("Rate (%)")
    ax.set_title("Comparison of Intervention Effects on Model B Behavior")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha="center")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(
        y=baseline_results.get("follow_cot", {}).get("match_rate", 0) * 100,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Baseline Match",
    )

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "intervention_comparison.png")
    plt.savefig(output_dir / "intervention_comparison.pdf")
    print(
        f"Saved intervention comparison plot to {output_dir / 'intervention_comparison.png'}"
    )
    plt.close()


def plot_mwc_mww(baseline_results, intervention_results, output_dir):
    """Plot MWC vs MWW for different interventions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Collect data points
    points = []

    # Baseline
    if "follow_cot" in baseline_results:
        m = baseline_results["follow_cot"]
        points.append(("Baseline", m["mwc"], m["mww"], "black", 200, "*"))

    # Color maps for different intervention types
    colors = {
        "truncate_first": "tab:blue",
        "truncate_last": "tab:orange",
        "truncate_contiguous": "tab:green",
        "truncate_percent": "tab:purple",
        "filler_replacement": "tab:red",
        "error_injection": "tab:brown",
    }

    for name, metrics in intervention_results.items():
        for prefix, color in colors.items():
            if name.startswith(prefix):
                param = name.replace(prefix + "_", "")
                label = f"{prefix.replace('_', ' ').title()} {param}"
                points.append((label, metrics["mwc"], metrics["mww"], color, 100, "o"))
                break

    # Plot
    for label, mwc, mww, color, size, marker in points:
        ax.scatter(
            mwc * 100,
            mww * 100,
            c=color,
            s=size,
            marker=marker,
            label=label if marker == "*" or "k1" in label or "p0.1" in label else None,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    # Add diagonal line (MWC = MWW)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="MWC = MWW")

    ax.set_xlabel("MWC: Match When Correct (%)")
    ax.set_ylabel("MWW: Match When Wrong (%)")
    ax.set_title("Faithfulness Analysis: MWC vs MWW")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add annotation explaining the quadrants
    ax.annotate(
        "High faithfulness\n(follows correct reasoning)",
        xy=(85, 30),
        fontsize=9,
        ha="center",
        style="italic",
        color="gray",
    )
    ax.annotate(
        "Blind following\n(follows wrong reasoning)",
        xy=(85, 85),
        fontsize=9,
        ha="center",
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "mwc_mww_analysis.png")
    plt.savefig(output_dir / "mwc_mww_analysis.pdf")
    print(f"Saved MWC/MWW analysis plot to {output_dir / 'mwc_mww_analysis.png'}")
    plt.close()


def plot_error_injection_detail(interventions_dir, output_dir):
    """Detailed analysis of error injection results."""
    error_path = interventions_dir / "error_injection.jsonl"
    if not error_path.exists():
        return

    entries = load_jsonl(error_path)

    # Separate modified vs unmodified
    modified = [e for e in entries if e.get("original_cot") != e.get("modified_cot")]
    unmodified = [e for e in entries if e.get("original_cot") == e.get("modified_cot")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Pie chart of modification success
    ax = axes[0]
    sizes = [len(modified), len(unmodified)]
    labels = [f"Modified\n({len(modified)})", f"Unmodified\n({len(unmodified)})"]
    colors = ["tab:orange", "tab:gray"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Error Injection Success Rate")

    # Plot 2: Match rate comparison
    ax = axes[1]

    mod_metrics = compute_metrics(modified, use_fixed_answer=True)
    unmod_metrics = compute_metrics(unmodified, use_fixed_answer=True)

    categories = ["Modified CoTs", "Unmodified CoTs"]
    match_rates = [100 * mod_metrics["match_rate"], 100 * unmod_metrics["match_rate"]]
    flip_rates = [100 * mod_metrics["flip_rate"], 100 * unmod_metrics["flip_rate"]]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, match_rates, width, label="Match Rate", color="tab:blue"
    )
    bars2 = ax.bar(
        x + width / 2, flip_rates, width, label="Flip Rate", color="tab:orange"
    )

    ax.set_ylabel("Rate (%)")
    ax.set_title("Error Injection Effect on Answer Agreement")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "error_injection_detail.png")
    plt.savefig(output_dir / "error_injection_detail.pdf")
    print(
        f"Saved error injection detail plot to {output_dir / 'error_injection_detail.png'}"
    )
    plt.close()

    # Print stats
    print(f"\nError Injection Detailed Stats:")
    print(f"  Modified CoTs: {len(modified)} ({100*len(modified)/len(entries):.1f}%)")
    print(f"    Match Rate: {100*mod_metrics['match_rate']:.1f}%")
    print(f"    Flip Rate: {100*mod_metrics['flip_rate']:.1f}%")
    print(
        f"  Unmodified CoTs: {len(unmodified)} ({100*len(unmodified)/len(entries):.1f}%)"
    )
    print(f"    Match Rate: {100*unmod_metrics['match_rate']:.1f}%")
    print(f"    Flip Rate: {100*unmod_metrics['flip_rate']:.1f}%")


def create_summary_table(baseline_results, intervention_results, output_dir):
    """Create a summary CSV table."""
    import csv

    rows = []

    # Header
    header = [
        "Intervention",
        "Parameter",
        "Match Rate",
        "Flip Rate",
        "Accuracy",
        "MWC",
        "MWW",
        "Total Rollouts",
    ]

    # Baseline
    if "follow_cot" in baseline_results:
        m = baseline_results["follow_cot"]
        rows.append(
            [
                "Baseline (Follow CoT)",
                "-",
                f"{100*m['match_rate']:.1f}",
                f"{100*m['flip_rate']:.1f}",
                f"{100*m['accuracy']:.1f}",
                f"{100*m['mwc']:.1f}",
                f"{100*m['mww']:.1f}",
                m["total_rollouts"],
            ]
        )

    if "no_cot" in baseline_results:
        m = baseline_results["no_cot"]
        rows.append(
            [
                "Baseline (No CoT)",
                "-",
                "-",
                "-",
                f"{100*m['accuracy']:.1f}",
                "-",
                "-",
                m["total_rollouts"],
            ]
        )

    # Interventions
    for name in sorted(intervention_results.keys()):
        m = intervention_results[name]
        # Parse intervention type and parameter
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and (parts[1].startswith("k") or parts[1].startswith("p")):
            int_type = parts[0].replace("_", " ").title()
            param = parts[1]
        else:
            int_type = name.replace("_", " ").title()
            param = "-"

        rows.append(
            [
                int_type,
                param,
                f"{100*m['match_rate']:.1f}",
                f"{100*m['flip_rate']:.1f}",
                f"{100*m['accuracy']:.1f}",
                f"{100*m['mwc']:.1f}",
                f"{100*m['mww']:.1f}",
                m["total_rollouts"],
            ]
        )

    # Write CSV
    csv_path = output_dir / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved summary table to {csv_path}")

    # Also print as formatted table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'Intervention':<25} {'Param':<8} {'Match%':>8} {'Flip%':>8} {'Acc%':>8} {'MWC%':>8} {'MWW%':>8}"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row[0]:<25} {row[1]:<8} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>8} {row[6]:>8}"
        )


def main():
    # Paths
    base_dir = Path("/home/riyaza/10701-project")
    baselines_dir = base_dir / "outputs" / "baselines"
    interventions_dir = base_dir / "outputs" / "interventions"
    output_dir = base_dir / "outputs" / "analysis"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("COT FAITHFULNESS EXPERIMENT ANALYSIS")
    print("=" * 70)

    # Analyze baselines
    baseline_results = analyze_baselines(baselines_dir)

    # Analyze interventions
    intervention_results, intervention_groups = analyze_interventions(interventions_dir)

    # Create plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_truncation_effects(intervention_groups, output_dir)
    plot_intervention_comparison(baseline_results, intervention_results, output_dir)
    plot_mwc_mww(baseline_results, intervention_results, output_dir)
    plot_error_injection_detail(interventions_dir, output_dir)

    # Create tables
    create_intervention_comparison_table(baseline_results, intervention_results)
    create_summary_table(baseline_results, intervention_results, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    for f in output_dir.glob("*"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
