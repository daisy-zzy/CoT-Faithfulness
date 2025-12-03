from typing import List, Dict, Any

from .parsing import normalize_answer


def _get_normalized_triplet(ex: Dict[str, Any]):
    """
    Helper function to extract (A_Q, A_L, A_GT) from one example and normalize them.

    A_Q : answer from Model A (Qwen)  -> ex["answer_a"]
    A_L : answer from Model B (Llama) -> ex["answer_b"]
    A_GT: ground-truth answer         -> ex["ground_truth"]

    Returns:
        (a_q, a_l, a_gt) if all three exist, otherwise (None, None, None).
    """
    a_q = ex.get("answer_a")
    a_l = ex.get("answer_b")
    a_gt = ex.get("ground_truth")

    if a_q is None or a_l is None or a_gt is None:
        return None, None, None

    return (
        normalize_answer(a_q),
        normalize_answer(a_l),
        normalize_answer(a_gt),
    )


def compute_accuracy_vs_gt(
    results: List[Dict[str, Any]],
    use_model: str = "b",
) -> float:
    """
    Compute accuracy of one model against ground truth.

    Args:
        results: list of example dicts, each containing
                 - "ground_truth"
                 - "answer_a" / "answer_b" (depending on use_model)
        use_model: "a" (Model A / Qwen) or "b" (Model B / Llama)

    Returns:
        Accuracy: fraction of examples where normalized prediction equals
        normalized ground truth, ignoring examples with missing values.
    """
    key = f"answer_{use_model}"
    num_valid = 0
    num_correct = 0

    for ex in results:
        pred = ex.get(key)
        gt = ex.get("ground_truth")
        if pred is None or gt is None:
            continue

        pred_n = normalize_answer(pred)
        gt_n = normalize_answer(gt)

        num_valid += 1
        if pred_n == gt_n:
            num_correct += 1

    return num_correct / num_valid if num_valid > 0 else 0.0


def compute_agreement_metrics(
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute agreement-based metrics between Model A and Model B.

    Definitions (following the proposal):

        - Overall Match Rate (OMR):
            Pr[A_L = A_Q]
            How often does Model B's answer match Model A's answer?

        - Match-When-Correct (MWC):
            Pr[A_L = A_Q | A_Q = A_GT]
            Among items where Model A is correct, how often does Model B
            match Model A?

        - Match-When-Wrong (MWW):
            Pr[A_L = A_Q | A_Q != A_GT]
            Among items where Model A is wrong, how often does Model B
            still match Model A?

    Args:
        results: list of example dicts containing keys:
                 - "answer_a"
                 - "answer_b"
                 - "ground_truth"

    Returns:
        A dictionary with:
            {
              "omr": float,
              "mwc": float,
              "mww": float,
              "num_total": int,
              "num_a_correct": int,
              "num_a_wrong": int,
            }
    """
    n_total = 0        # number of valid examples (all three answers present)
    n_omr = 0          # count of A_L == A_Q

    n_a_correct = 0    # count where A_Q == A_GT
    n_mwc = 0          # count where A_Q == A_GT and A_L == A_Q

    n_a_wrong = 0      # count where A_Q != A_GT
    n_mww = 0          # count where A_Q != A_GT and A_L == A_Q

    for ex in results:
        a_q, a_l, a_gt = _get_normalized_triplet(ex)
        if a_q is None:
            # At least one of the three is missing; skip this example.
            continue

        n_total += 1

        # Overall agreement between A and B.
        if a_l == a_q:
            n_omr += 1

        # Condition on whether Model A is correct or wrong.
        if a_q == a_gt:
            n_a_correct += 1
            if a_l == a_q:
                n_mwc += 1
        else:
            n_a_wrong += 1
            if a_l == a_q:
                n_mww += 1

    omr = n_omr / n_total if n_total > 0 else 0.0
    mwc = n_mwc / n_a_correct if n_a_correct > 0 else 0.0
    mww = n_mww / n_a_wrong if n_a_wrong > 0 else 0.0

    return {
        "omr": omr,
        "mwc": mwc,
        "mww": mww,
        "num_total": n_total,
        "num_a_correct": n_a_correct,
        "num_a_wrong": n_a_wrong,
    }


def compute_agreement(results: List[Dict[str, Any]]) -> float:
    """
    Backward-compatible helper that returns only the overall match rate (OMR).

    This keeps the old API used in some scripts:
        compute_agreement(results) -> float

    It simply calls compute_agreement_metrics(results)["omr"].
    """
    metrics = compute_agreement_metrics(results)
    return metrics["omr"]


def compute_flip_rate(
    results: List[Dict[str, Any]],
    intervention_type: str,
) -> Dict[str, float]:
    """
    Compute flip rate for a specific intervention type.
    
    Flip rate is the fraction of items where Model B's answer changes
    after applying the intervention to CoT_A.
    
    Specifically:
    - Compare answer_b (baseline, with original CoT) vs answer_b_perturbed
    - A "flip" occurs when answer_b != answer_b_perturbed
    
    Args:
        results: list of example dicts containing:
                 - "answer_b": Model B's answer with original CoT
                 - "answer_b_perturbed" or intervention-specific field (e.g., "answer_b_truncated", "answer_b_injected")
        intervention_type: One of "truncate_random" or "inject_error"
                          Determines which field to use for perturbed answer
    
    Returns:
        A dictionary with:
            {
              "flip_rate": float,  # Fraction of items that flipped
              "num_total": int,     # Total valid examples
              "num_flipped": int,   # Number of examples that flipped
            }
    """
    num_total = 0
    num_flipped = 0
    
    # Map intervention types to their field names
    field_map = {
        "truncate_random": "answer_b_truncated",
        "inject_error": "answer_b_injected",
    }
    
    # Get the field name for this intervention type, or fall back to generic
    perturbed_field = field_map.get(intervention_type, "answer_b_perturbed")
    
    for ex in results:
        answer_b = ex.get("answer_b")
        answer_b_perturbed = ex.get(perturbed_field)
        
        if answer_b is None or answer_b_perturbed is None:
            continue
        
        num_total += 1
        
        # Normalize and compare
        answer_b_norm = normalize_answer(answer_b)
        answer_b_perturbed_norm = normalize_answer(answer_b_perturbed)
        
        if answer_b_norm != answer_b_perturbed_norm:
            num_flipped += 1
    
    flip_rate = num_flipped / num_total if num_total > 0 else 0.0
    
    return {
        "flip_rate": flip_rate,
        "num_total": num_total,
        "num_flipped": num_flipped,
    }


def compute_truncation_flip_rate(
    results: List[Dict[str, Any]],
    truncation_type: str = "any",
) -> Dict[str, float]:
    """
    Compute truncation flip rate.
    
    This computes flip rates for truncation interventions (random sentence removal).
    
    Args:
        results: list of example dicts with intervention results
        truncation_type: "random" or "any" (both are the same now)
    
    Returns:
        Dictionary with flip_rate, num_total, num_flipped
    """
    # Now we only have "truncate_random", so just compute it directly
    return compute_flip_rate(results, "truncate_random")


def compute_mistake_flip_rate(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute mistake flip rate (for error injection interventions).
    
    Returns:
        Dictionary with flip_rate, num_total, num_flipped
    """
    return compute_flip_rate(results, "inject_error")
