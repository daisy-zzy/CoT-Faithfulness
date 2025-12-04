"""Statistical analysis utilities for experiment results."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class MetricStats:
    """Statistics for a single metric across rollouts."""
    mean: float
    std: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n_samples: int
    raw_values: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_samples": self.n_samples,
        }
    
    def __repr__(self) -> str:
        return f"{self.mean:.3f} ± {self.std:.3f} (95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}])"


def compute_metric_stats(values: List[float], confidence: float = 0.95) -> MetricStats:
    """
    Compute mean, std, and confidence interval for a list of values.
    
    Args:
        values: List of metric values (one per rollout)
        confidence: Confidence level for CI (default 0.95)
    
    Returns:
        MetricStats object with computed statistics
    """
    values = [v for v in values if v is not None and not np.isnan(v)]
    
    if not values:
        return MetricStats(
            mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0,
            n_samples=0, raw_values=[]
        )
    
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    
    # Compute confidence interval using t-distribution
    if len(arr) > 1:
        try:
            from scipy import stats
            t_val = stats.t.ppf((1 + confidence) / 2, df=len(arr) - 1)
            margin = t_val * std / np.sqrt(len(arr))
            ci_lower = mean - margin
            ci_upper = mean + margin
        except ImportError:
            # Fallback if scipy not available
            ci_lower = mean - 1.96 * std / np.sqrt(len(arr))
            ci_upper = mean + 1.96 * std / np.sqrt(len(arr))
    else:
        ci_lower = ci_upper = mean
    
    return MetricStats(
        mean=mean,
        std=std,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_samples=len(arr),
        raw_values=list(values)
    )


def compute_agreement_rate(predictions: List[str], reference: str) -> float:
    """Compute fraction of predictions that match reference."""
    if not predictions:
        return 0.0
    matches = sum(1 for p in predictions if p == reference)
    return matches / len(predictions)


def compute_accuracy_from_rollouts(
    rollout_predictions: List[List[str]],  # [n_examples][n_rollouts]
    ground_truths: List[str]
) -> Tuple[MetricStats, List[float]]:
    """
    Compute accuracy statistics across rollouts.
    
    Args:
        rollout_predictions: For each example, list of predictions across rollouts
        ground_truths: Ground truth for each example
    
    Returns:
        Tuple of (MetricStats for accuracy, per-example agreement rates)
    """
    if not rollout_predictions or not rollout_predictions[0]:
        return MetricStats(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n_samples=0, raw_values=[]), []
    
    # For each rollout, compute accuracy
    n_rollouts = len(rollout_predictions[0])
    rollout_accuracies = []
    
    for rollout_idx in range(n_rollouts):
        correct = 0
        total = 0
        for ex_idx, gt in enumerate(ground_truths):
            if ex_idx < len(rollout_predictions) and rollout_idx < len(rollout_predictions[ex_idx]):
                pred = rollout_predictions[ex_idx][rollout_idx]
                if pred == gt:
                    correct += 1
                total += 1
        rollout_accuracies.append(correct / total if total > 0 else 0.0)
    
    # Also compute per-example agreement rates
    per_example_agreement = []
    for ex_idx, gt in enumerate(ground_truths):
        if ex_idx < len(rollout_predictions):
            preds = rollout_predictions[ex_idx]
            agreement = compute_agreement_rate(preds, gt)
            per_example_agreement.append(agreement)
    
    return compute_metric_stats(rollout_accuracies), per_example_agreement


def compute_omr_from_rollouts(
    model_b_rollouts: List[List[str]],  # [n_examples][n_rollouts]
    model_a_answers: List[str]
) -> MetricStats:
    """
    Compute Overall Match Rate: Pr[B = A] across rollouts.
    """
    if not model_b_rollouts or not model_b_rollouts[0]:
        return MetricStats(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n_samples=0, raw_values=[])
    
    n_rollouts = len(model_b_rollouts[0])
    rollout_omrs = []
    
    for rollout_idx in range(n_rollouts):
        matches = 0
        total = 0
        for ex_idx, a_ans in enumerate(model_a_answers):
            if ex_idx < len(model_b_rollouts) and rollout_idx < len(model_b_rollouts[ex_idx]):
                b_ans = model_b_rollouts[ex_idx][rollout_idx]
                if b_ans == a_ans:
                    matches += 1
                total += 1
        rollout_omrs.append(matches / total if total > 0 else 0.0)
    
    return compute_metric_stats(rollout_omrs)


def compute_mwc_from_rollouts(
    model_b_rollouts: List[List[str]],
    model_a_answers: List[str],
    ground_truths: List[str]
) -> MetricStats:
    """
    Compute Match When Correct: Pr[B = A | A is correct] across rollouts.
    """
    # Find indices where A is correct
    correct_indices = [i for i, (a, gt) in enumerate(zip(model_a_answers, ground_truths)) if a == gt]
    
    if not correct_indices:
        return MetricStats(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n_samples=0, raw_values=[])
    
    if not model_b_rollouts or not model_b_rollouts[0]:
        return MetricStats(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n_samples=0, raw_values=[])
    
    n_rollouts = len(model_b_rollouts[0])
    rollout_mwcs = []
    
    for rollout_idx in range(n_rollouts):
        matches = 0
        for idx in correct_indices:
            if idx < len(model_b_rollouts) and rollout_idx < len(model_b_rollouts[idx]):
                if model_b_rollouts[idx][rollout_idx] == model_a_answers[idx]:
                    matches += 1
        rollout_mwcs.append(matches / len(correct_indices))
    
    return compute_metric_stats(rollout_mwcs)


def compute_mww_from_rollouts(
    model_b_rollouts: List[List[str]],
    model_a_answers: List[str],
    ground_truths: List[str]
) -> MetricStats:
    """
    Compute Match When Wrong: Pr[B = A | A is wrong] across rollouts.
    """
    # Find indices where A is wrong
    wrong_indices = [i for i, (a, gt) in enumerate(zip(model_a_answers, ground_truths)) if a != gt]
    
    if not wrong_indices:
        return MetricStats(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n_samples=0, raw_values=[])
    
    if not model_b_rollouts or not model_b_rollouts[0]:
        return MetricStats(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n_samples=0, raw_values=[])
    
    n_rollouts = len(model_b_rollouts[0])
    rollout_mwws = []
    
    for rollout_idx in range(n_rollouts):
        matches = 0
        for idx in wrong_indices:
            if idx < len(model_b_rollouts) and rollout_idx < len(model_b_rollouts[idx]):
                if model_b_rollouts[idx][rollout_idx] == model_a_answers[idx]:
                    matches += 1
        rollout_mwws.append(matches / len(wrong_indices))
    
    return compute_metric_stats(rollout_mwws)


def compute_flip_rate_from_rollouts(
    baseline_rollouts: List[List[str]],  # [n_examples][n_rollouts]
    intervention_rollouts: List[List[str]]
) -> MetricStats:
    """
    Compute flip rate: how often does the answer change after intervention?
    Compares rollout i of baseline to rollout i of intervention.
    """
    if not baseline_rollouts or not baseline_rollouts[0]:
        return MetricStats(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n_samples=0, raw_values=[])
    
    n_rollouts = len(baseline_rollouts[0])
    rollout_flip_rates = []
    
    for rollout_idx in range(n_rollouts):
        flips = 0
        total = 0
        for ex_idx in range(len(baseline_rollouts)):
            if (ex_idx < len(intervention_rollouts) and 
                rollout_idx < len(baseline_rollouts[ex_idx]) and 
                rollout_idx < len(intervention_rollouts[ex_idx])):
                baseline_ans = baseline_rollouts[ex_idx][rollout_idx]
                intervention_ans = intervention_rollouts[ex_idx][rollout_idx]
                if baseline_ans != intervention_ans:
                    flips += 1
                total += 1
        rollout_flip_rates.append(flips / total if total > 0 else 0.0)
    
    return compute_metric_stats(rollout_flip_rates)


@dataclass
class ExperimentResults:
    """Container for all results from an experiment run."""
    experiment_name: str
    config: Dict[str, Any]
    
    # Model A stats
    model_a_accuracy: Optional[MetricStats] = None
    
    # Model B No-CoT stats
    model_b_no_cot_accuracy: Optional[MetricStats] = None
    
    # Model B Follow-CoT stats (baseline)
    model_b_follow_cot_accuracy: Optional[MetricStats] = None
    omr: Optional[MetricStats] = None
    mwc: Optional[MetricStats] = None
    mww: Optional[MetricStats] = None
    
    # Intervention stats (dict of intervention_name -> metrics)
    intervention_results: Dict[str, Dict[str, MetricStats]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "experiment_name": self.experiment_name,
            "config": self.config,
        }
        
        if self.model_a_accuracy:
            result["model_a_accuracy"] = self.model_a_accuracy.to_dict()
        if self.model_b_no_cot_accuracy:
            result["model_b_no_cot_accuracy"] = self.model_b_no_cot_accuracy.to_dict()
        if self.model_b_follow_cot_accuracy:
            result["model_b_follow_cot_accuracy"] = self.model_b_follow_cot_accuracy.to_dict()
        if self.omr:
            result["omr"] = self.omr.to_dict()
        if self.mwc:
            result["mwc"] = self.mwc.to_dict()
        if self.mww:
            result["mww"] = self.mww.to_dict()
        
        if self.intervention_results:
            result["interventions"] = {
                name: {k: v.to_dict() for k, v in metrics.items()}
                for name, metrics in self.intervention_results.items()
            }
        
        return result
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
