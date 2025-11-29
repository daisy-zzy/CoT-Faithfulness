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
