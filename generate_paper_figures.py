#!/usr/bin/env python3
"""
Generate all figures and tables for the paper.
Reports mean ± std across 4 rollouts per problem, aggregated over 5000 problems.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
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
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Paths
BASELINES_DIR = Path("/home/riyaza/10701-project/outputs/baselines")
INTERVENTIONS_DIR = Path("/home/riyaza/10701-project/outputs/interventions")
OUTPUT_DIR = Path("/home/riyaza/10701-project/outputs/analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_jsonl(path):
    """Load JSONL file."""
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_accuracy_per_problem(entries, answer_key="answer_normalized"):
    """
    Compute accuracy per problem (mean of rollouts matching ground truth).
    Returns array of per-problem accuracy rates.
    """
    problem_accuracies = []
    for entry in entries:
        gt = entry.get("ground_truth_normalized")
        if not gt:
            continue

        correct = []
        for rollout in entry.get("rollouts", []):
            ans = rollout.get(answer_key)
            if ans:
                correct.append(1 if ans == gt else 0)

        if correct:
            problem_accuracies.append(np.mean(correct))

    return np.array(problem_accuracies)


def compute_match_rate_per_problem(entries, reference_answer_key="fixed_answer_a"):
    """
    Compute match rate per problem (mean of rollouts matching reference answer).
    Returns array of per-problem match rates.
    """
    problem_match_rates = []
    for entry in entries:
        ref_ans = entry.get(reference_answer_key)
        if not ref_ans:
            continue

        matches = []
        for rollout in entry.get("rollouts", []):
            ans = rollout.get("answer_normalized")
            if ans:
                matches.append(1 if ans == ref_ans else 0)

        if matches:
            problem_match_rates.append(np.mean(matches))

    return np.array(problem_match_rates)


def compute_mwc_mww_per_problem(entries, reference_answer_key="fixed_answer_a"):
    """
    Compute Match When Correct (MWC) and Match When Wrong (MWW) per problem.
    MWC: match rate for problems where Model A was correct
    MWW: match rate for problems where Model A was wrong
    Returns two arrays: mwc_rates, mww_rates
    """
    mwc_rates = []  # Match rates when A is correct
    mww_rates = []  # Match rates when A is wrong
    
    for entry in entries:
        ref_ans = entry.get(reference_answer_key)
        gt = entry.get("ground_truth_normalized")
        if not ref_ans or not gt:
            continue
        
        # Check if Model A was correct
        a_is_correct = (ref_ans == gt)
        
        matches = []
        for rollout in entry.get("rollouts", []):
            ans = rollout.get("answer_normalized")
            if ans:
                matches.append(1 if ans == ref_ans else 0)
        
        if matches:
            match_rate = np.mean(matches)
            if a_is_correct:
                mwc_rates.append(match_rate)
            else:
                mww_rates.append(match_rate)
    
    return np.array(mwc_rates), np.array(mww_rates)


def compute_flip_rate_vs_baseline(intervention_entries, baseline_entries):
    """
    Compute flip rate: how often intervention answer differs from baseline answer.
    Compares rollout-by-rollout.
    """
    problem_flip_rates = []

    for base_entry, interv_entry in zip(baseline_entries, intervention_entries):
        assert base_entry["id"] == interv_entry["id"]

        flips = []
        base_rollouts = base_entry.get("rollouts", [])
        interv_rollouts = interv_entry.get("rollouts", [])

        for base_r, interv_r in zip(base_rollouts, interv_rollouts):
            base_ans = base_r.get("answer_normalized")
            interv_ans = interv_r.get("answer_normalized")
            if base_ans and interv_ans:
                flips.append(1 if base_ans != interv_ans else 0)

        if flips:
            problem_flip_rates.append(np.mean(flips))

    return np.array(problem_flip_rates)


def stats(arr):
    """Compute mean and std error for array."""
    if len(arr) == 0:
        return 0, 0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) / np.sqrt(len(arr))  # Standard error
    return mean, std


# =============================================================================
# LOAD ALL DATA
# =============================================================================
print("Loading data...")

# Baselines
model_a = load_jsonl(BASELINES_DIR / "model_a_rollouts.jsonl")
model_b_no_cot = load_jsonl(BASELINES_DIR / "model_b_no_cot.jsonl")
model_b_with_cot = load_jsonl(BASELINES_DIR / "model_b_follow_cot.jsonl")

# Interventions
interventions = {}
for path in sorted(INTERVENTIONS_DIR.glob("**/*.jsonl")):
    interventions[path.stem] = load_jsonl(path)

print(f"Loaded {len(model_a)} problems")
print(f"Loaded {len(interventions)} intervention types")

# =============================================================================
# COMPUTE ALL METRICS
# =============================================================================
print("\nComputing metrics...")

# Compute Model A accuracy (using fixed_answer_a from model_b_with_cot which has it)
model_a_acc = []
for entry in model_b_with_cot:
    gt = entry.get("ground_truth_normalized")
    ans_a = entry.get("fixed_answer_a")
    if gt and ans_a:
        model_a_acc.append(1 if ans_a == gt else 0)
model_a_acc = np.array(model_a_acc)

# Compute Model B (no CoT) accuracy
model_b_no_cot_acc = compute_accuracy_per_problem(model_b_no_cot)

# Compute Model B (with CoT) accuracy
model_b_with_cot_acc = compute_accuracy_per_problem(model_b_with_cot)

# Compute Model B (with CoT) match rate with Model A (OMR)
model_b_with_cot_match = compute_match_rate_per_problem(model_b_with_cot)

# Compute Model B (with CoT) MWC and MWW
model_b_with_cot_mwc, model_b_with_cot_mww = compute_mwc_mww_per_problem(model_b_with_cot)

# Compute intervention metrics
intervention_metrics = {}
for name, entries in interventions.items():
    acc = compute_accuracy_per_problem(entries)
    match = compute_match_rate_per_problem(entries)
    flip = compute_flip_rate_vs_baseline(entries, model_b_with_cot)
    mwc, mww = compute_mwc_mww_per_problem(entries)
    intervention_metrics[name] = {
        "accuracy": acc,
        "match_rate": match,  # OMR
        "flip_rate": flip,
        "mwc": mwc,  # Match When Correct
        "mww": mww,  # Match When Wrong
    }

# =============================================================================
# FIGURE 1: Baseline Accuracy Comparison
# =============================================================================
print("\nGenerating Figure 1: Baseline Accuracy Comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

# Collect all conditions for accuracy plot
conditions = [
    ("Model A\n(Qwen)", model_a_acc, "tab:blue"),
    ("Model B No CoT\n(Llama)", model_b_no_cot_acc, "tab:orange"),
    ("Model B + CoT\n(Baseline)", model_b_with_cot_acc, "tab:green"),
]

# Add key interventions
key_interventions = [
    ("truncate_last_k5", "Trunc Last k=5"),
    ("truncate_percent_p0.5", "Trunc 50%"),
    ("filler_replacement_p0.5", "Filler 50%"),
    ("error_injection", "Error Inject"),
]

colors = ["tab:red", "tab:purple", "tab:brown", "tab:pink"]
for (key, label), color in zip(key_interventions, colors):
    if key in intervention_metrics:
        conditions.append((label, intervention_metrics[key]["accuracy"], color))

labels = [c[0] for c in conditions]
means = [100 * stats(c[1])[0] for c in conditions]
stds = [100 * stats(c[1])[1] for c in conditions]
colors = [c[2] for c in conditions]

x = np.arange(len(labels))
bars = ax.bar(
    x,
    means,
    yerr=stds,
    color=colors,
    alpha=0.8,
    capsize=4,
    edgecolor="black",
    linewidth=0.5,
)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison (Mean ± SE)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(0, 100)

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    ax.annotate(
        f"{mean:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, mean + std + 2),
        ha="center",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_baseline_accuracy.png")
plt.savefig(OUTPUT_DIR / "fig1_baseline_accuracy.pdf")
print(f"  Saved: fig1_baseline_accuracy.png")
plt.close()

# =============================================================================
# FIGURE 2: Truncation Interventions - OMR, MWC, MWW, Flip Rate
# =============================================================================
print("\nGenerating Figure 2: Truncation Interventions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

baseline_match_mean, baseline_match_std = stats(model_b_with_cot_match)
baseline_mwc_mean, baseline_mwc_std = stats(model_b_with_cot_mwc)
baseline_mww_mean, baseline_mww_std = stats(model_b_with_cot_mww)
baseline_flip_mean, baseline_flip_std = stats(1 - model_b_with_cot_match)

# Top-left: K-based truncations - OMR (Overall Match Rate)
ax = axes[0, 0]
for prefix, color, marker, label in [
    ("truncate_first", "tab:blue", "o", "Truncate First"),
    ("truncate_last", "tab:orange", "s", "Truncate Last"),
    ("truncate_contiguous", "tab:green", "^", "Truncate Contiguous"),
]:
    ks, means, stds = [], [], []
    for k in [1, 2, 3, 5]:
        key = f"{prefix}_k{k}"
        if key in intervention_metrics:
            ks.append(k)
            m, s = stats(intervention_metrics[key]["match_rate"])
            means.append(100 * m)
            stds.append(100 * s)
    if ks:
        ax.errorbar(ks, means, yerr=stds, marker=marker, color=color, label=label, 
                   linewidth=2, markersize=8, capsize=4)

ax.axhline(y=100 * baseline_match_mean, color="gray", linestyle="--", alpha=0.7, label="Baseline")
ax.set_xlabel("Number of Sentences Removed (k)")
ax.set_ylabel("OMR (%)")
ax.set_title("K-Based Truncation: Overall Match Rate (OMR)")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 5])
ax.set_ylim(70, 95)

# Top-right: K-based truncations - MWC (Match When Correct)
ax = axes[0, 1]
for prefix, color, marker, label in [
    ("truncate_first", "tab:blue", "o", "Truncate First"),
    ("truncate_last", "tab:orange", "s", "Truncate Last"),
    ("truncate_contiguous", "tab:green", "^", "Truncate Contiguous"),
]:
    ks, means, stds = [], [], []
    for k in [1, 2, 3, 5]:
        key = f"{prefix}_k{k}"
        if key in intervention_metrics:
            ks.append(k)
            m, s = stats(intervention_metrics[key]["mwc"])
            means.append(100 * m)
            stds.append(100 * s)
    if ks:
        ax.errorbar(ks, means, yerr=stds, marker=marker, color=color, label=label, 
                   linewidth=2, markersize=8, capsize=4)

ax.axhline(y=100 * baseline_mwc_mean, color="gray", linestyle="--", alpha=0.7, label="Baseline")
ax.set_xlabel("Number of Sentences Removed (k)")
ax.set_ylabel("MWC (%)")
ax.set_title("K-Based Truncation: Match When Correct (MWC)")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 5])
ax.set_ylim(70, 100)

# Bottom-left: K-based truncations - MWW (Match When Wrong)
ax = axes[1, 0]
for prefix, color, marker, label in [
    ("truncate_first", "tab:blue", "o", "Truncate First"),
    ("truncate_last", "tab:orange", "s", "Truncate Last"),
    ("truncate_contiguous", "tab:green", "^", "Truncate Contiguous"),
]:
    ks, means, stds = [], [], []
    for k in [1, 2, 3, 5]:
        key = f"{prefix}_k{k}"
        if key in intervention_metrics:
            ks.append(k)
            m, s = stats(intervention_metrics[key]["mww"])
            means.append(100 * m)
            stds.append(100 * s)
    if ks:
        ax.errorbar(ks, means, yerr=stds, marker=marker, color=color, label=label, 
                   linewidth=2, markersize=8, capsize=4)

ax.axhline(y=100 * baseline_mww_mean, color="gray", linestyle="--", alpha=0.7, label="Baseline")
ax.set_xlabel("Number of Sentences Removed (k)")
ax.set_ylabel("MWW (%)")
ax.set_title("K-Based Truncation: Match When Wrong (MWW)")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 5])
ax.set_ylim(40, 80)

# Bottom-right: K-based truncations - Flip Rate
ax = axes[1, 1]
for prefix, color, marker, label in [
    ("truncate_first", "tab:blue", "o", "Truncate First"),
    ("truncate_last", "tab:orange", "s", "Truncate Last"),
    ("truncate_contiguous", "tab:green", "^", "Truncate Contiguous"),
]:
    ks, means, stds = [], [], []
    for k in [1, 2, 3, 5]:
        key = f"{prefix}_k{k}"
        if key in intervention_metrics:
            ks.append(k)
            m, s = stats(intervention_metrics[key]["flip_rate"])
            means.append(100 * m)
            stds.append(100 * s)
    if ks:
        ax.errorbar(ks, means, yerr=stds, marker=marker, color=color, label=label, 
                   linewidth=2, markersize=8, capsize=4)

ax.axhline(y=100 * baseline_flip_mean, color="gray", linestyle="--", alpha=0.7, label="Baseline")
ax.set_xlabel("Number of Sentences Removed (k)")
ax.set_ylabel("Flip Rate (%)")
ax.set_title("K-Based Truncation: Flip Rate")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 5])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_truncation_effects.png")
plt.savefig(OUTPUT_DIR / "fig2_truncation_effects.pdf")
print(f"  Saved: fig2_truncation_effects.png")
plt.close()

# =============================================================================
# FIGURE 3: Filler Replacement Interventions - OMR, MWC, MWW, Flip Rate
# =============================================================================
print("\nGenerating Figure 3: Filler Replacement Effects...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: OMR (Overall Match Rate)
ax = axes[0, 0]
ps, means, stds = [], [], []
for p in [0.1, 0.2, 0.3, 0.5]:
    key = f"filler_replacement_p{p}"
    if key in intervention_metrics:
        ps.append(p * 100)
        m, s = stats(intervention_metrics[key]["match_rate"])
        means.append(100 * m)
        stds.append(100 * s)
if ps:
    ax.errorbar(ps, means, yerr=stds, marker="o", color="tab:blue", label="Filler Replace",
               linewidth=2, markersize=8, capsize=4)
ax.axhline(y=100 * baseline_match_mean, color="gray", linestyle="--", alpha=0.7, label="Baseline")
ax.set_xlabel("Percentage of Sentences Replaced (%)")
ax.set_ylabel("OMR (%)")
ax.set_title("Filler Replacement: Overall Match Rate (OMR)")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(5, 55)
ax.set_ylim(70, 95)

# Top-right: MWC (Match When Correct)
ax = axes[0, 1]
ps, means, stds = [], [], []
for p in [0.1, 0.2, 0.3, 0.5]:
    key = f"filler_replacement_p{p}"
    if key in intervention_metrics:
        ps.append(p * 100)
        m, s = stats(intervention_metrics[key]["mwc"])
        means.append(100 * m)
        stds.append(100 * s)
if ps:
    ax.errorbar(ps, means, yerr=stds, marker="s", color="tab:green", label="Filler Replace",
               linewidth=2, markersize=8, capsize=4)
ax.axhline(y=100 * baseline_mwc_mean, color="gray", linestyle="--", alpha=0.7, label="Baseline")
ax.set_xlabel("Percentage of Sentences Replaced (%)")
ax.set_ylabel("MWC (%)")
ax.set_title("Filler Replacement: Match When Correct (MWC)")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(5, 55)
ax.set_ylim(70, 100)

# Bottom-left: MWW (Match When Wrong)
ax = axes[1, 0]
ps, means, stds = [], [], []
for p in [0.1, 0.2, 0.3, 0.5]:
    key = f"filler_replacement_p{p}"
    if key in intervention_metrics:
        ps.append(p * 100)
        m, s = stats(intervention_metrics[key]["mww"])
        means.append(100 * m)
        stds.append(100 * s)
if ps:
    ax.errorbar(ps, means, yerr=stds, marker="^", color="tab:orange", label="Filler Replace",
               linewidth=2, markersize=8, capsize=4)
ax.axhline(y=100 * baseline_mww_mean, color="gray", linestyle="--", alpha=0.7, label="Baseline")
ax.set_xlabel("Percentage of Sentences Replaced (%)")
ax.set_ylabel("MWW (%)")
ax.set_title("Filler Replacement: Match When Wrong (MWW)")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(5, 55)
ax.set_ylim(40, 80)

# Bottom-right: Flip Rate
ax = axes[1, 1]
ps, means, stds = [], [], []
for p in [0.1, 0.2, 0.3, 0.5]:
    key = f"filler_replacement_p{p}"
    if key in intervention_metrics:
        ps.append(p * 100)
        m, s = stats(intervention_metrics[key]["flip_rate"])
        means.append(100 * m)
        stds.append(100 * s)
if ps:
    ax.errorbar(ps, means, yerr=stds, marker="D", color="tab:red", label="Filler Replace",
               linewidth=2, markersize=8, capsize=4)
ax.axhline(y=100 * baseline_flip_mean, color="gray", linestyle="--", alpha=0.7, label="Baseline")
ax.set_xlabel("Percentage of Sentences Replaced (%)")
ax.set_ylabel("Flip Rate (%)")
ax.set_title("Filler Replacement: Flip Rate")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(5, 55)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_filler_effects.png")
plt.savefig(OUTPUT_DIR / "fig3_filler_effects.pdf")
print(f"  Saved: fig3_filler_effects.png")
plt.close()

# =============================================================================
# FIGURE 4: Error Injection Analysis - OMR, MWC, MWW, Flip Rate
# =============================================================================
print("\nGenerating Figure 4: Error Injection Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

if "error_injection" in interventions:
    error_entries = interventions["error_injection"]

    # Use all error injection data (combined)
    error_match = intervention_metrics["error_injection"]["match_rate"]
    error_mwc = intervention_metrics["error_injection"]["mwc"]
    error_mww = intervention_metrics["error_injection"]["mww"]
    error_flip = intervention_metrics["error_injection"]["flip_rate"]

    categories = ["Baseline\n(No Intervention)", "Error Injection"]
    x = np.arange(len(categories))
    width = 0.5
    colors = ["tab:blue", "tab:red"]

    # Top-left: OMR
    ax = axes[0, 0]
    omr_means = [100 * baseline_match_mean, 100 * stats(error_match)[0]]
    omr_stds = [100 * baseline_match_std, 100 * stats(error_match)[1]]
    bars = ax.bar(x, omr_means, width, yerr=omr_stds, color=colors, alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("OMR (%)")
    ax.set_title("Overall Match Rate (OMR)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    for bar, mean in zip(bars, omr_means):
        ax.annotate(f"{mean:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, mean + 3), ha="center", fontsize=11, fontweight="bold")

    # Top-right: MWC
    ax = axes[0, 1]
    mwc_means = [100 * baseline_mwc_mean, 100 * stats(error_mwc)[0]]
    mwc_stds = [100 * baseline_mwc_std, 100 * stats(error_mwc)[1]]
    bars = ax.bar(x, mwc_means, width, yerr=mwc_stds, color=colors, alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("MWC (%)")
    ax.set_title("Match When Correct (MWC)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    for bar, mean in zip(bars, mwc_means):
        ax.annotate(f"{mean:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, mean + 3), ha="center", fontsize=11, fontweight="bold")

    # Bottom-left: MWW
    ax = axes[1, 0]
    mww_means = [100 * baseline_mww_mean, 100 * stats(error_mww)[0]]
    mww_stds = [100 * baseline_mww_std, 100 * stats(error_mww)[1]]
    bars = ax.bar(x, mww_means, width, yerr=mww_stds, color=colors, alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("MWW (%)")
    ax.set_title("Match When Wrong (MWW)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    for bar, mean in zip(bars, mww_means):
        ax.annotate(f"{mean:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, mean + 3), ha="center", fontsize=11, fontweight="bold")

    # Bottom-right: Flip Rate
    ax = axes[1, 1]
    flip_means = [100 * baseline_flip_mean, 100 * stats(error_flip)[0]]
    flip_stds = [100 * baseline_flip_std, 100 * stats(error_flip)[1]]
    bars = ax.bar(x, flip_means, width, yerr=flip_stds, color=colors, alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Flip Rate (%)")
    ax.set_title("Flip Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 60)
    for bar, mean in zip(bars, flip_means):
        ax.annotate(f"{mean:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, mean + 2), ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(f"Error Injection Analysis (n={len(error_entries)} problems)", fontsize=13)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig4_error_injection.png")
plt.savefig(OUTPUT_DIR / "fig4_error_injection.pdf")
print(f"  Saved: fig4_error_injection.png")
plt.close()

# =============================================================================
# TABLE: Complete Results Summary
# =============================================================================
print("\nGenerating Complete Results Table...")

# Collect all results
all_results = []

# Baselines
all_results.append(
    {
        "name": "Model A (Qwen)",
        "category": "Baseline",
        "omr_mean": "-",
        "omr_se": "-",
        "mwc_mean": "-",
        "mwc_se": "-",
        "mww_mean": "-",
        "mww_se": "-",
        "flip_rate_mean": "-",
        "flip_rate_se": "-",
        "n": len(model_a_acc),
    }
)

all_results.append(
    {
        "name": "Model B No CoT",
        "category": "Baseline",
        "omr_mean": "-",
        "omr_se": "-",
        "mwc_mean": "-",
        "mwc_se": "-",
        "mww_mean": "-",
        "mww_se": "-",
        "flip_rate_mean": "-",
        "flip_rate_se": "-",
        "n": len(model_b_no_cot_acc),
    }
)

all_results.append(
    {
        "name": "Model B + CoT (Baseline)",
        "category": "Baseline",
        "omr_mean": 100 * stats(model_b_with_cot_match)[0],
        "omr_se": 100 * stats(model_b_with_cot_match)[1],
        "mwc_mean": 100 * stats(model_b_with_cot_mwc)[0],
        "mwc_se": 100 * stats(model_b_with_cot_mwc)[1],
        "mww_mean": 100 * stats(model_b_with_cot_mww)[0],
        "mww_se": 100 * stats(model_b_with_cot_mww)[1],
        "flip_rate_mean": 100 * stats(1 - model_b_with_cot_match)[0],
        "flip_rate_se": 100 * stats(1 - model_b_with_cot_match)[1],
        "n": len(model_b_with_cot_acc),
    }
)

# Interventions
intervention_order = [
    ("truncate_first_k1", "Truncate First", "k=1"),
    ("truncate_first_k2", "Truncate First", "k=2"),
    ("truncate_first_k3", "Truncate First", "k=3"),
    ("truncate_first_k5", "Truncate First", "k=5"),
    ("truncate_last_k1", "Truncate Last", "k=1"),
    ("truncate_last_k2", "Truncate Last", "k=2"),
    ("truncate_last_k3", "Truncate Last", "k=3"),
    ("truncate_last_k5", "Truncate Last", "k=5"),
    ("truncate_contiguous_k1", "Truncate Contiguous", "k=1"),
    ("truncate_contiguous_k2", "Truncate Contiguous", "k=2"),
    ("truncate_contiguous_k3", "Truncate Contiguous", "k=3"),
    ("truncate_contiguous_k5", "Truncate Contiguous", "k=5"),
    ("truncate_percent_p0.1", "Truncate Random", "10%"),
    ("truncate_percent_p0.2", "Truncate Random", "20%"),
    ("truncate_percent_p0.3", "Truncate Random", "30%"),
    ("truncate_percent_p0.5", "Truncate Random", "50%"),
    ("filler_replacement_p0.1", "Filler Replace", "10%"),
    ("filler_replacement_p0.2", "Filler Replace", "20%"),
    ("filler_replacement_p0.3", "Filler Replace", "30%"),
    ("filler_replacement_p0.5", "Filler Replace", "50%"),
    ("error_injection", "Error Injection", "-"),
]

for key, category, param in intervention_order:
    if key in intervention_metrics:
        m = intervention_metrics[key]
        all_results.append(
            {
                "name": f"{category} {param}",
                "category": category,
                "omr_mean": 100 * stats(m["match_rate"])[0],
                "omr_se": 100 * stats(m["match_rate"])[1],
                "mwc_mean": 100 * stats(m["mwc"])[0],
                "mwc_se": 100 * stats(m["mwc"])[1],
                "mww_mean": 100 * stats(m["mww"])[0],
                "mww_se": 100 * stats(m["mww"])[1],
                "flip_rate_mean": 100 * stats(m["flip_rate"])[0],
                "flip_rate_se": 100 * stats(m["flip_rate"])[1],
                "n": len(m["accuracy"]),
            }
        )

# Print text table
print("\n" + "=" * 160)
print("COMPLETE RESULTS TABLE")
print("=" * 160)
print(
    f"{'Condition':<30} {'OMR':>12} {'MWC':>12} {'MWW':>12} {'Flip Rate':>12} {'n':>8}"
)
print("-" * 160)

for r in all_results:
    omr_str = (
        f"{r['omr_mean']:.1f} ± {r['omr_se']:.1f}"
        if isinstance(r["omr_se"], float)
        else str(r["omr_mean"])
    )
    mwc_str = (
        f"{r['mwc_mean']:.1f} ± {r['mwc_se']:.1f}"
        if isinstance(r["mwc_se"], float)
        else str(r["mwc_mean"])
    )
    mww_str = (
        f"{r['mww_mean']:.1f} ± {r['mww_se']:.1f}"
        if isinstance(r["mww_se"], float)
        else str(r["mww_mean"])
    )
    flip_str = (
        f"{r['flip_rate_mean']:.1f} ± {r['flip_rate_se']:.1f}"
        if isinstance(r["flip_rate_se"], float)
        else str(r["flip_rate_mean"])
    )
    print(
        f"{r['name']:<30} {omr_str:>12} {mwc_str:>12} {mww_str:>12} {flip_str:>12} {r['n']:>8}"
    )

# Generate LaTeX table
latex = r"""
\begin{table*}[t]
\centering
\caption{Complete Experimental Results. All values are mean $\pm$ standard error across 5000 problems with 4 rollouts each.
OMR = Overall Match Rate P(B=A). MWC = Match When Correct. MWW = Match When Wrong. Flip Rate = P(intervention answer $\neq$ baseline answer).}
\label{tab:complete_results}
\small
\begin{tabular}{llcccc}
\toprule
\textbf{Category} & \textbf{Condition} & \textbf{OMR (\%)} & \textbf{MWC (\%)} & \textbf{MWW (\%)} & \textbf{Flip Rate (\%)} \\
\midrule
"""

current_category = None
for r in all_results:
    cat = r["category"]
    name = r["name"].replace(cat, "").strip()
    if not name:
        name = r["name"]

    if cat != current_category:
        if current_category is not None:
            latex += r"\midrule" + "\n"
        current_category = cat
        cat_display = cat
    else:
        cat_display = ""

    omr_str = (
        f"${r['omr_mean']:.1f} \\pm {r['omr_se']:.1f}$"
        if isinstance(r["omr_se"], float)
        else "-"
    )
    mwc_str = (
        f"${r['mwc_mean']:.1f} \\pm {r['mwc_se']:.1f}$"
        if isinstance(r["mwc_se"], float)
        else "-"
    )
    mww_str = (
        f"${r['mww_mean']:.1f} \\pm {r['mww_se']:.1f}$"
        if isinstance(r["mww_se"], float)
        else "-"
    )
    flip_str = (
        f"${r['flip_rate_mean']:.1f} \\pm {r['flip_rate_se']:.1f}$"
        if isinstance(r["flip_rate_se"], float)
        else "-"
    )

    latex += f"{cat_display} & {name} & {omr_str} & {mwc_str} & {mww_str} & {flip_str} \\\\\n"

latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""

with open(OUTPUT_DIR / "complete_results_table.tex", "w") as f:
    f.write(latex)
print(f"\nSaved: complete_results_table.tex")

# Also save CSV
import csv

with open(OUTPUT_DIR / "complete_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "name",
            "category",
            "omr_mean",
            "omr_se",
            "mwc_mean",
            "mwc_se",
            "mww_mean",
            "mww_se",
            "flip_rate_mean",
            "flip_rate_se",
            "n",
        ],
    )
    writer.writeheader()
    writer.writerows(all_results)
print(f"Saved: complete_results.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF GENERATED FILES")
print("=" * 80)
print(
    f"""
Figures:
  - fig1_baseline_accuracy.png/pdf  : Accuracy comparison (Model A, B no CoT, B+CoT, interventions)
  - fig2_truncation_effects.png/pdf : Truncation dose-response (first/last/contiguous/percent)
  - fig3_filler_effects.png/pdf     : Filler replacement dose-response
  - fig4_error_injection.png/pdf    : Error injection breakdown (modified vs unmodified)

Tables:
  - complete_results_table.tex      : LaTeX table with all results
  - complete_results.csv            : CSV with all results

All files saved to: {OUTPUT_DIR}
"""
)
