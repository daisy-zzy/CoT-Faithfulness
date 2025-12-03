import json
from pathlib import Path
from typing import List, Dict, Any
import gc
import torch

from .data import load_math_lighteval_test
from .models import VLLMEngine
from .generation import run_qwen_stage, run_llama_stage, run_intervention_stage
from .eval import (
    compute_agreement,
    compute_accuracy_vs_gt,
    compute_agreement_metrics,
    compute_truncation_flip_rate,
    compute_mistake_flip_rate,
)


def save_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """
    Save a list of Python dicts to a JSONL file, one JSON object per line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.
    """
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def full_pipeline(
    qwen_name: str,
    llama_name: str,
    out_dir: str = "outputs",
    max_examples: int | None = None,
):
    """
    End-to-end pipeline:

        1. Load the MATH-lighteval test split.
        2. Run Model A (Qwen) to generate CoT + final answer.
        3. Run Model B (Llama) conditioned on the CoT_A to generate a new answer.
        4. Save intermediate and final results as JSONL.
        5. Compute agreement-based metrics and accuracy against ground truth.

    Args:
        qwen_name: HuggingFace model ID for Model A (Qwen).
        llama_name: HuggingFace model ID for Model B (Llama).
        out_dir: directory where JSONL outputs will be stored.
        max_examples: if not None, only use the first N examples
                      from the dataset (useful for debugging).
    """
    # 1. Load data
    data = load_math_lighteval_test()
    if max_examples is not None:
        data = data[:max_examples]

    # 2. Stage 1: run Model A (Qwen) to get CoT_A and answer_A
    qwen_engine = VLLMEngine(qwen_name)
    qwen_results = run_qwen_stage(qwen_engine, data)
    save_jsonl(Path(out_dir) / "stage1_qwen.jsonl", qwen_results)

    del qwen_engine      # drop Python reference
    gc.collect()         # run garbage collector
    torch.cuda.empty_cache()  # clear cached CUDA memory

    # 3. Stage 2: run Model B (Llama) with problem + CoT_A (baseline only)
    llama_engine = VLLMEngine(llama_name)
    baseline_results = run_llama_stage(
        llama_engine,
        qwen_results,  # Stage 1 results
    )
    save_jsonl(Path(out_dir) / "stage2_llama.jsonl", baseline_results)

    # 4. Evaluation
    # 4.1 Agreement-based metrics (OMR / MWC / MWW)
    metrics = compute_agreement_metrics(baseline_results)

    # 4.2 Accuracy of each model vs ground truth
    acc_a = compute_accuracy_vs_gt(baseline_results, use_model="a")
    acc_b = compute_accuracy_vs_gt(baseline_results, use_model="b")

    print("=== Agreement-based metrics ===")
    print(f"Overall Match Rate (OMR):         {metrics['omr']:.4f}")
    print(f"Match-When-Correct (MWC):         {metrics['mwc']:.4f}")
    print(f"Match-When-Wrong  (MWW):          {metrics['mww']:.4f}")
    print(
        f"[Counts] total={metrics['num_total']}, "
        f"A_correct={metrics['num_a_correct']}, "
        f"A_wrong={metrics['num_a_wrong']}"
    )

    print("=== Accuracy vs Ground Truth ===")
    print(f"Model A accuracy vs GT:           {acc_a:.4f}")
    print(f"Model B accuracy vs GT:           {acc_b:.4f}")

    overall_agreement = compute_agreement(baseline_results)
    print(f"(Legacy) Model B vs A agreement:  {overall_agreement:.4f}")
    
    # 5. Stage 3: Run interventions
    all_intervention_results = run_intervention_stage(
        llama_engine,
        baseline_results,  # Stage 2 results (already contains baseline)
        run_truncation=True,
        run_error_injection=True,
        truncation_fraction=0.3,
    )
    save_jsonl(Path(out_dir) / "stage3_interventions.jsonl", all_intervention_results)
    
    # 6. Compute intervention metrics
    print("\n=== Causal/Intervention Metrics ===")
    truncation_metrics = compute_truncation_flip_rate(all_intervention_results, truncation_type="any")
    print(f"Truncation Flip Rate:              {truncation_metrics['flip_rate']:.4f}")
    print(
        f"[Counts] total={truncation_metrics['num_total']}, "
        f"flipped={truncation_metrics['num_flipped']}"
    )
    mistake_metrics = compute_mistake_flip_rate(all_intervention_results)
    print(f"Mistake Flip Rate:                 {mistake_metrics['flip_rate']:.4f}")
    print(
        f"[Counts] total={mistake_metrics['num_total']}, "
        f"flipped={mistake_metrics['num_flipped']}"
    )
    