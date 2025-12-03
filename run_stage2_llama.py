from pathlib import Path

from utils.models import VLLMEngine
from utils.generation import run_llama_stage
from utils.pipeline import save_jsonl, load_jsonl
from utils.eval import (
    compute_agreement,
    compute_accuracy_vs_gt,
    compute_agreement_metrics,
)


def run_stage2_llama(
    llama_name: str,
    in_path: str = "outputs/stage1_qwen.jsonl",
    out_dir: str = "outputs",
) -> None:
    """
    Stage 2:
      - Load Stage 1 outputs (problem + CoT_A + answer_A).
      - Run Model B (Llama) conditioned on CoT_A.
      - Save final results and compute evaluation metrics.

    Args:
        llama_name: HuggingFace model ID for Llama (Model B).
        in_path: Path to the JSONL file produced by Stage 1.
        out_dir: Directory where Stage 2 JSONL will be stored.
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Stage 1 file not found: {in_path}. "
            "Please run run_stage1_qwen.py first."
        )

    # 1. Load Stage 1 results
    stage1_results = load_jsonl(in_path)
    print(f"[Stage 2] Loaded {len(stage1_results)} examples from {in_path}")

    # 2. Run Llama as Model B (baseline only)
    llama_engine = VLLMEngine(llama_name)
    final_results = run_llama_stage(
        llama_engine,
        stage1_results,
    )

    # 3. Save Stage 2 results
    out_path = Path(out_dir) / "stage2_llama.jsonl"
    save_jsonl(out_path, final_results)
    print(f"[Stage 2] Saved {len(final_results)} examples to {out_path}")

    # 4. Evaluation
    metrics = compute_agreement_metrics(final_results)
    acc_a = compute_accuracy_vs_gt(final_results, use_model="a")
    acc_b = compute_accuracy_vs_gt(final_results, use_model="b")
    overall_agreement = compute_agreement(final_results)

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

    print(f"(Legacy) Model B vs A agreement:  {overall_agreement:.4f}")


if __name__ == "__main__":
    run_stage2_llama(
        llama_name="meta-llama/Llama-3.2-3B-Instruct",
        in_path="outputs/stage1_qwen.jsonl",
        out_dir="outputs",
    )
