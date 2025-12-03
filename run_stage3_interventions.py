from pathlib import Path

from utils.models import VLLMEngine
from utils.generation import run_intervention_stage
from utils.pipeline import save_jsonl, load_jsonl
from utils.eval import (
    compute_truncation_flip_rate,
    compute_mistake_flip_rate,
)


def run_stage3_interventions(
    llama_name: str,
    in_path: str = "outputs/stage2_llama.jsonl",
    out_dir: str = "outputs",
    truncation_fraction: float = 0.3,
) -> None:
    """
    Stage 3:
      - Load Stage 2 outputs (problem + CoT_A + answer_A + answer_B).
      - Run truncation and error injection interventions.
      - Compute intervention metrics and save results.

    Args:
        llama_name: HuggingFace model ID for Llama (Model B).
        in_path: Path to the JSONL file produced by Stage 2.
        out_dir: Directory where Stage 3 JSONL will be stored.
        truncation_fraction: Fraction of sentences to remove for truncation intervention.
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Stage 2 file not found: {in_path}. "
            "Please run run_stage2_llama.py first."
        )

    # 1. Load Stage 2 results
    stage2_results = load_jsonl(in_path)
    print(f"[Stage 3] Loaded {len(stage2_results)} examples from {in_path}")

    # 2. Run interventions (baseline already exists from Stage 2)
    print("Running both truncation and error injection interventions...")
    llama_engine = VLLMEngine(llama_name)
    all_results = run_intervention_stage(
        llama_engine,
        stage2_results,
        run_truncation=True,
        run_error_injection=True,
        truncation_fraction=truncation_fraction,
    )

    # 3. Save Stage 3 results
    out_path = Path(out_dir) / "stage3_interventions.jsonl"
    save_jsonl(out_path, all_results)
    print(f"[Stage 3] Saved {len(all_results)} examples to {out_path}")

    # 4. Compute intervention metrics
    print("\n=== Causal/Intervention Metrics ===")
    truncation_metrics = compute_truncation_flip_rate(all_results, truncation_type="any")
    print(f"Truncation Flip Rate:              {truncation_metrics['flip_rate']:.4f}")
    print(
        f"[Counts] total={truncation_metrics['num_total']}, "
        f"flipped={truncation_metrics['num_flipped']}"
    )
    mistake_metrics = compute_mistake_flip_rate(all_results)
    print(f"Mistake Flip Rate:                 {mistake_metrics['flip_rate']:.4f}")
    print(
        f"[Counts] total={mistake_metrics['num_total']}, "
        f"flipped={mistake_metrics['num_flipped']}"
    )


if __name__ == "__main__":
    run_stage3_interventions(
        llama_name="meta-llama/Llama-3.2-3B-Instruct",
        in_path="outputs/stage2_llama.jsonl",
        out_dir="outputs",
        truncation_fraction=0.3,
    )
