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
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

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


def compute_accuracy_per_problem(entries, answer_key='answer_normalized'):
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


def compute_match_rate_per_problem(entries, reference_answer_key='fixed_answer_a'):
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


def compute_flip_rate_vs_baseline(intervention_entries, baseline_entries):
    """
    Compute flip rate: how often intervention answer differs from baseline answer.
    Compares rollout-by-rollout.
    """
    problem_flip_rates = []
    
    for base_entry, interv_entry in zip(baseline_entries, intervention_entries):
        assert base_entry['id'] == interv_entry['id']
        
        flips = []
        base_rollouts = base_entry.get('rollouts', [])
        interv_rollouts = interv_entry.get('rollouts', [])
        
        for base_r, interv_r in zip(base_rollouts, interv_rollouts):
            base_ans = base_r.get('answer_normalized')
            interv_ans = interv_r.get('answer_normalized')
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
for path in sorted(INTERVENTIONS_DIR.glob("*.jsonl")):
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

# Compute Model B (with CoT) match rate with Model A
model_b_with_cot_match = compute_match_rate_per_problem(model_b_with_cot)

# Compute intervention metrics
intervention_metrics = {}
for name, entries in interventions.items():
    acc = compute_accuracy_per_problem(entries)
    match = compute_match_rate_per_problem(entries)
    flip = compute_flip_rate_vs_baseline(entries, model_b_with_cot)
    intervention_metrics[name] = {
        'accuracy': acc,
        'match_rate': match,
        'flip_rate': flip,
    }

# =============================================================================
# FIGURE 1: Baseline Accuracy Comparison
# =============================================================================
print("\nGenerating Figure 1: Baseline Accuracy Comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

# Collect all conditions for accuracy plot
conditions = [
    ("Model A\n(Qwen)", model_a_acc, 'tab:blue'),
    ("Model B No CoT\n(Llama)", model_b_no_cot_acc, 'tab:orange'),
    ("Model B + CoT\n(Baseline)", model_b_with_cot_acc, 'tab:green'),
]

# Add key interventions
key_interventions = [
    ('truncate_last_k5', 'Trunc Last k=5'),
    ('truncate_percent_p0.5', 'Trunc 50%'),
    ('filler_replacement_p0.5', 'Filler 50%'),
    ('error_injection', 'Error Inject'),
]

colors = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
for (key, label), color in zip(key_interventions, colors):
    if key in intervention_metrics:
        conditions.append((label, intervention_metrics[key]['accuracy'], color))

labels = [c[0] for c in conditions]
means = [100 * stats(c[1])[0] for c in conditions]
stds = [100 * stats(c[1])[1] for c in conditions]
colors = [c[2] for c in conditions]

x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison (Mean ± SE)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha='right')
ax.set_ylim(0, 100)

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    ax.annotate(f'{mean:.1f}', xy=(bar.get_x() + bar.get_width()/2, mean + std + 2),
               ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_baseline_accuracy.png')
plt.savefig(OUTPUT_DIR / 'fig1_baseline_accuracy.pdf')
print(f"  Saved: fig1_baseline_accuracy.png")
plt.close()

# =============================================================================
# FIGURE 2: Truncation Interventions - All Metrics
# =============================================================================
print("\nGenerating Figure 2: Truncation Interventions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

baseline_acc_mean, baseline_acc_std = stats(model_b_with_cot_acc)
baseline_match_mean, baseline_match_std = stats(model_b_with_cot_match)
baseline_flip_mean, baseline_flip_std = stats(1 - model_b_with_cot_match)

# Top-left: K-based truncations - Accuracy
ax = axes[0, 0]
for prefix, color, marker, label in [
    ('truncate_first', 'tab:blue', 'o', 'Truncate First'),
    ('truncate_last', 'tab:orange', 's', 'Truncate Last'),
    ('truncate_contiguous', 'tab:green', '^', 'Truncate Contiguous'),
]:
    ks, means, stds = [], [], []
    for k in [1, 2, 3, 5]:
        key = f"{prefix}_k{k}"
        if key in intervention_metrics:
            ks.append(k)
            m, s = stats(intervention_metrics[key]['accuracy'])
            means.append(100 * m)
            stds.append(100 * s)
    if ks:
        ax.errorbar(ks, means, yerr=stds, marker=marker, color=color, label=label, 
                   linewidth=2, markersize=8, capsize=4)

ax.axhline(y=100*baseline_acc_mean, color='gray', linestyle='--', alpha=0.7, label='Baseline')
ax.set_xlabel('Number of Sentences Removed (k)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('K-Based Truncation: Accuracy')
ax.legend(loc='lower left', fontsize=8)
ax.set_xlim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 5])
ax.set_ylim(0, 100)

# Top-right: K-based truncations - Match Rate
ax = axes[0, 1]
for prefix, color, marker, label in [
    ('truncate_first', 'tab:blue', 'o', 'Truncate First'),
    ('truncate_last', 'tab:orange', 's', 'Truncate Last'),
    ('truncate_contiguous', 'tab:green', '^', 'Truncate Contiguous'),
]:
    ks, means, stds = [], [], []
    for k in [1, 2, 3, 5]:
        key = f"{prefix}_k{k}"
        if key in intervention_metrics:
            ks.append(k)
            m, s = stats(intervention_metrics[key]['match_rate'])
            means.append(100 * m)
            stds.append(100 * s)
    if ks:
        ax.errorbar(ks, means, yerr=stds, marker=marker, color=color, label=label, 
                   linewidth=2, markersize=8, capsize=4)

ax.axhline(y=100*baseline_match_mean, color='gray', linestyle='--', alpha=0.7, label='Baseline')
ax.set_xlabel('Number of Sentences Removed (k)')
ax.set_ylabel('Match Rate (%)')
ax.set_title('K-Based Truncation: Match Rate (B=A)')
ax.legend(loc='lower left', fontsize=8)
ax.set_xlim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 5])
ax.set_ylim(0, 100)

# Bottom-left: K-based truncations - Flip Rate
ax = axes[1, 0]
for prefix, color, marker, label in [
    ('truncate_first', 'tab:blue', 'o', 'Truncate First'),
    ('truncate_last', 'tab:orange', 's', 'Truncate Last'),
    ('truncate_contiguous', 'tab:green', '^', 'Truncate Contiguous'),
]:
    ks, means, stds = [], [], []
    for k in [1, 2, 3, 5]:
        key = f"{prefix}_k{k}"
        if key in intervention_metrics:
            ks.append(k)
            m, s = stats(intervention_metrics[key]['flip_rate'])
            means.append(100 * m)
            stds.append(100 * s)
    if ks:
        ax.errorbar(ks, means, yerr=stds, marker=marker, color=color, label=label, 
                   linewidth=2, markersize=8, capsize=4)

ax.axhline(y=100*baseline_flip_mean, color='gray', linestyle='--', alpha=0.7, label='Baseline')
ax.set_xlabel('Number of Sentences Removed (k)')
ax.set_ylabel('Flip Rate (%)')
ax.set_title('K-Based Truncation: Flip Rate')
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 5])

# Bottom-right: Percentage-based truncations - All metrics
ax = axes[1, 1]
for metric_name, metric_key, color, marker in [
    ('Accuracy', 'accuracy', 'tab:blue', 'o'),
    ('Match Rate', 'match_rate', 'tab:green', 's'),
    ('Flip Rate', 'flip_rate', 'tab:red', '^'),
]:
    ps, means, stds = [], [], []
    for p in [0.1, 0.2, 0.3, 0.5]:
        key = f"truncate_percent_p{p}"
        if key in intervention_metrics:
            ps.append(p * 100)
            m, s = stats(intervention_metrics[key][metric_key])
            means.append(100 * m)
            stds.append(100 * s)
    if ps:
        ax.errorbar(ps, means, yerr=stds, marker=marker, color=color, label=metric_name,
                   linewidth=2, markersize=8, capsize=4)

ax.set_xlabel('Percentage of Sentences Removed (%)')
ax.set_ylabel('Rate (%)')
ax.set_title('Random Truncation: All Metrics')
ax.legend(loc='center right', fontsize=8)
ax.set_xlim(5, 55)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_truncation_effects.png')
plt.savefig(OUTPUT_DIR / 'fig2_truncation_effects.pdf')
print(f"  Saved: fig2_truncation_effects.png")
plt.close()

# =============================================================================
# FIGURE 3: Filler Replacement Interventions - All Metrics
# =============================================================================
print("\nGenerating Figure 3: Filler Replacement Effects...")

fig, ax = plt.subplots(figsize=(10, 6))

for metric_name, metric_key, color, marker in [
    ('Accuracy', 'accuracy', 'tab:blue', 'o'),
    ('Match Rate', 'match_rate', 'tab:green', 's'),
    ('Flip Rate', 'flip_rate', 'tab:red', '^'),
]:
    ps, means, stds = [], [], []
    for p in [0.1, 0.2, 0.3, 0.5]:
        key = f"filler_replacement_p{p}"
        if key in intervention_metrics:
            ps.append(p * 100)
            m, s = stats(intervention_metrics[key][metric_key])
            means.append(100 * m)
            stds.append(100 * s)
    if ps:
        ax.errorbar(ps, means, yerr=stds, marker=marker, color=color, label=metric_name,
                   linewidth=2, markersize=8, capsize=4)

# Add baseline references
ax.axhline(y=100*baseline_acc_mean, color='tab:blue', linestyle='--', alpha=0.5, label='Baseline Acc')
ax.axhline(y=100*baseline_match_mean, color='tab:green', linestyle='--', alpha=0.5, label='Baseline Match')
ax.axhline(y=100*baseline_flip_mean, color='tab:red', linestyle='--', alpha=0.5, label='Baseline Flip')

ax.set_xlabel('Percentage of Sentences Replaced with Filler (%)')
ax.set_ylabel('Rate (%)')
ax.set_title('Filler Replacement Effects: All Metrics (Mean ± SE)')
ax.legend(loc='center right')
ax.set_xlim(5, 55)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_filler_effects.png')
plt.savefig(OUTPUT_DIR / 'fig3_filler_effects.pdf')
print(f"  Saved: fig3_filler_effects.png")
plt.close()

# =============================================================================
# FIGURE 4: Error Injection Analysis - All Metrics
# =============================================================================
print("\nGenerating Figure 4: Error Injection Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

if 'error_injection' in interventions:
    error_entries = interventions['error_injection']
    
    # Split by modified vs unmodified
    modified = [e for e in error_entries if e.get('original_cot') != e.get('modified_cot')]
    unmodified = [e for e in error_entries if e.get('original_cot') == e.get('modified_cot')]
    
    # Get corresponding baseline entries for flip rate calculation
    baseline_lookup = {e['id']: e for e in model_b_with_cot}
    
    # Compute flip rates for modified CoTs
    mod_flip_rates = []
    for entry in modified:
        base_entry = baseline_lookup.get(entry['id'])
        if base_entry:
            flips = []
            for base_r, interv_r in zip(base_entry['rollouts'], entry['rollouts']):
                base_ans = base_r.get('answer_normalized')
                interv_ans = interv_r.get('answer_normalized')
                if base_ans and interv_ans:
                    flips.append(1 if base_ans != interv_ans else 0)
            if flips:
                mod_flip_rates.append(np.mean(flips))
    
    # Compute flip rates for unmodified CoTs
    unmod_flip_rates = []
    for entry in unmodified:
        base_entry = baseline_lookup.get(entry['id'])
        if base_entry:
            flips = []
            for base_r, interv_r in zip(base_entry['rollouts'], entry['rollouts']):
                base_ans = base_r.get('answer_normalized')
                interv_ans = interv_r.get('answer_normalized')
                if base_ans and interv_ans:
                    flips.append(1 if base_ans != interv_ans else 0)
            if flips:
                unmod_flip_rates.append(np.mean(flips))
    
    mod_flip_rates = np.array(mod_flip_rates)
    unmod_flip_rates = np.array(unmod_flip_rates)
    
    # Compute accuracy and match rate
    mod_acc = compute_accuracy_per_problem(modified)
    unmod_acc = compute_accuracy_per_problem(unmodified)
    mod_match = compute_match_rate_per_problem(modified)
    unmod_match = compute_match_rate_per_problem(unmodified)
    
    categories = ['Baseline', 'Unmodified\n(No Error)', 'Modified\n(Error Injected)']
    x = np.arange(len(categories))
    width = 0.6
    
    # Plot 1: Accuracy
    ax = axes[0]
    acc_means = [100*baseline_acc_mean, 100*stats(unmod_acc)[0], 100*stats(mod_acc)[0]]
    acc_stds = [100*baseline_acc_std, 100*stats(unmod_acc)[1], 100*stats(mod_acc)[1]]
    colors = ['tab:gray', 'tab:blue', 'tab:red']
    bars = ax.bar(x, acc_means, width, yerr=acc_stds, color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    for bar, mean in zip(bars, acc_means):
        ax.annotate(f'{mean:.1f}%', xy=(bar.get_x() + bar.get_width()/2, mean + 3), ha='center', fontsize=10)
    
    # Plot 2: Match Rate
    ax = axes[1]
    match_means = [100*baseline_match_mean, 100*stats(unmod_match)[0], 100*stats(mod_match)[0]]
    match_stds = [100*baseline_match_std, 100*stats(unmod_match)[1], 100*stats(mod_match)[1]]
    bars = ax.bar(x, match_means, width, yerr=match_stds, color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Match Rate (%)')
    ax.set_title('Match Rate (B = A)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    for bar, mean in zip(bars, match_means):
        ax.annotate(f'{mean:.1f}%', xy=(bar.get_x() + bar.get_width()/2, mean + 3), ha='center', fontsize=10)
    
    # Plot 3: Flip Rate
    ax = axes[2]
    flip_means = [100*baseline_flip_mean, 100*stats(unmod_flip_rates)[0], 100*stats(mod_flip_rates)[0]]
    flip_stds = [100*baseline_flip_std, 100*stats(unmod_flip_rates)[1], 100*stats(mod_flip_rates)[1]]
    bars = ax.bar(x, flip_means, width, yerr=flip_stds, color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Flip Rate (%)')
    ax.set_title('Flip Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 70)
    for bar, mean in zip(bars, flip_means):
        ax.annotate(f'{mean:.1f}%', xy=(bar.get_x() + bar.get_width()/2, mean + 3), ha='center', fontsize=10)
    
    fig.suptitle(f'Error Injection Analysis (n_modified={len(modified)}, n_unmodified={len(unmodified)})', fontsize=13)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_error_injection.png')
plt.savefig(OUTPUT_DIR / 'fig4_error_injection.pdf')
print(f"  Saved: fig4_error_injection.png")
plt.close()

# =============================================================================
# TABLE: Complete Results Summary
# =============================================================================
print("\nGenerating Complete Results Table...")

# Collect all results
all_results = []

# Baselines
all_results.append({
    'name': 'Model A (Qwen)',
    'category': 'Baseline',
    'accuracy_mean': 100 * np.mean(model_a_acc),
    'accuracy_se': 100 * np.std(model_a_acc, ddof=1) / np.sqrt(len(model_a_acc)),
    'match_rate_mean': '-',
    'match_rate_se': '-',
    'flip_rate_mean': '-',
    'flip_rate_se': '-',
    'n': len(model_a_acc),
})

all_results.append({
    'name': 'Model B No CoT',
    'category': 'Baseline',
    'accuracy_mean': 100 * stats(model_b_no_cot_acc)[0],
    'accuracy_se': 100 * stats(model_b_no_cot_acc)[1],
    'match_rate_mean': '-',
    'match_rate_se': '-',
    'flip_rate_mean': '-',
    'flip_rate_se': '-',
    'n': len(model_b_no_cot_acc),
})

all_results.append({
    'name': 'Model B + CoT (Baseline)',
    'category': 'Baseline',
    'accuracy_mean': 100 * stats(model_b_with_cot_acc)[0],
    'accuracy_se': 100 * stats(model_b_with_cot_acc)[1],
    'match_rate_mean': 100 * stats(model_b_with_cot_match)[0],
    'match_rate_se': 100 * stats(model_b_with_cot_match)[1],
    'flip_rate_mean': 100 * stats(1 - model_b_with_cot_match)[0],
    'flip_rate_se': 100 * stats(1 - model_b_with_cot_match)[1],
    'n': len(model_b_with_cot_acc),
})

# Interventions
intervention_order = [
    ('truncate_first_k1', 'Truncate First', 'k=1'),
    ('truncate_first_k2', 'Truncate First', 'k=2'),
    ('truncate_first_k3', 'Truncate First', 'k=3'),
    ('truncate_first_k5', 'Truncate First', 'k=5'),
    ('truncate_last_k1', 'Truncate Last', 'k=1'),
    ('truncate_last_k2', 'Truncate Last', 'k=2'),
    ('truncate_last_k3', 'Truncate Last', 'k=3'),
    ('truncate_last_k5', 'Truncate Last', 'k=5'),
    ('truncate_contiguous_k1', 'Truncate Contiguous', 'k=1'),
    ('truncate_contiguous_k2', 'Truncate Contiguous', 'k=2'),
    ('truncate_contiguous_k3', 'Truncate Contiguous', 'k=3'),
    ('truncate_contiguous_k5', 'Truncate Contiguous', 'k=5'),
    ('truncate_percent_p0.1', 'Truncate Random', '10%'),
    ('truncate_percent_p0.2', 'Truncate Random', '20%'),
    ('truncate_percent_p0.3', 'Truncate Random', '30%'),
    ('truncate_percent_p0.5', 'Truncate Random', '50%'),
    ('filler_replacement_p0.1', 'Filler Replace', '10%'),
    ('filler_replacement_p0.2', 'Filler Replace', '20%'),
    ('filler_replacement_p0.3', 'Filler Replace', '30%'),
    ('filler_replacement_p0.5', 'Filler Replace', '50%'),
    ('error_injection', 'Error Injection', '-'),
]

for key, category, param in intervention_order:
    if key in intervention_metrics:
        m = intervention_metrics[key]
        all_results.append({
            'name': f'{category} {param}',
            'category': category,
            'accuracy_mean': 100 * stats(m['accuracy'])[0],
            'accuracy_se': 100 * stats(m['accuracy'])[1],
            'match_rate_mean': 100 * stats(m['match_rate'])[0],
            'match_rate_se': 100 * stats(m['match_rate'])[1],
            'flip_rate_mean': 100 * stats(m['flip_rate'])[0],
            'flip_rate_se': 100 * stats(m['flip_rate'])[1],
            'n': len(m['accuracy']),
        })

# Print text table
print("\n" + "=" * 120)
print("COMPLETE RESULTS TABLE")
print("=" * 120)
print(f"{'Condition':<30} {'Accuracy':>15} {'Match Rate':>15} {'Flip Rate':>15} {'n':>8}")
print("-" * 120)

for r in all_results:
    acc_str = f"{r['accuracy_mean']:.1f} ± {r['accuracy_se']:.1f}" if isinstance(r['accuracy_se'], float) else f"{r['accuracy_mean']:.1f}"
    match_str = f"{r['match_rate_mean']:.1f} ± {r['match_rate_se']:.1f}" if isinstance(r['match_rate_se'], float) else str(r['match_rate_mean'])
    flip_str = f"{r['flip_rate_mean']:.1f} ± {r['flip_rate_se']:.1f}" if isinstance(r['flip_rate_se'], float) else str(r['flip_rate_mean'])
    print(f"{r['name']:<30} {acc_str:>15} {match_str:>15} {flip_str:>15} {r['n']:>8}")

# Generate LaTeX table
latex = r"""
\begin{table*}[t]
\centering
\caption{Complete Experimental Results. All values are mean $\pm$ standard error across 5000 problems with 4 rollouts each.
Match Rate = P(Model B answer = Model A answer). Flip Rate = P(intervention answer $\neq$ baseline answer).}
\label{tab:complete_results}
\small
\begin{tabular}{llccc}
\toprule
\textbf{Category} & \textbf{Condition} & \textbf{Accuracy (\%)} & \textbf{Match Rate (\%)} & \textbf{Flip Rate (\%)} \\
\midrule
"""

current_category = None
for r in all_results:
    cat = r['category']
    name = r['name'].replace(cat, '').strip()
    if not name:
        name = r['name']
    
    if cat != current_category:
        if current_category is not None:
            latex += r"\midrule" + "\n"
        current_category = cat
        cat_display = cat
    else:
        cat_display = ""
    
    acc_str = f"${r['accuracy_mean']:.1f} \\pm {r['accuracy_se']:.1f}$" if isinstance(r['accuracy_se'], float) else f"${r['accuracy_mean']:.1f}$"
    match_str = f"${r['match_rate_mean']:.1f} \\pm {r['match_rate_se']:.1f}$" if isinstance(r['match_rate_se'], float) else "-"
    flip_str = f"${r['flip_rate_mean']:.1f} \\pm {r['flip_rate_se']:.1f}$" if isinstance(r['flip_rate_se'], float) else "-"
    
    latex += f"{cat_display} & {name} & {acc_str} & {match_str} & {flip_str} \\\\\n"

latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""

with open(OUTPUT_DIR / 'complete_results_table.tex', 'w') as f:
    f.write(latex)
print(f"\nSaved: complete_results_table.tex")

# Also save CSV
import csv
with open(OUTPUT_DIR / 'complete_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'category', 'accuracy_mean', 'accuracy_se', 
                                            'match_rate_mean', 'match_rate_se', 'flip_rate_mean', 'flip_rate_se', 'n'])
    writer.writeheader()
    writer.writerows(all_results)
print(f"Saved: complete_results.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF GENERATED FILES")
print("=" * 80)
print(f"""
Figures:
  - fig1_baseline_accuracy.png/pdf  : Accuracy comparison (Model A, B no CoT, B+CoT, interventions)
  - fig2_truncation_effects.png/pdf : Truncation dose-response (first/last/contiguous/percent)
  - fig3_filler_effects.png/pdf     : Filler replacement dose-response
  - fig4_error_injection.png/pdf    : Error injection breakdown (modified vs unmodified)

Tables:
  - complete_results_table.tex      : LaTeX table with all results
  - complete_results.csv            : CSV with all results

All files saved to: {OUTPUT_DIR}
""")
