from pathlib import Path

from utils.data import load_math_lighteval_test
from utils.models import VLLMEngine
from utils.generation import run_qwen_stage
from utils.pipeline import save_jsonl


def run_stage1_qwen(
    qwen_name: str,
    out_dir: str = "outputs",
    max_examples: int | None = None,
) -> None:
    """
    Stage 1:
      - Load the MATH-lighteval test set.
      - Run Model A (Qwen) to generate CoT_A and answer_A.
      - Save results to <out_dir>/stage1_qwen.jsonl.

    Args:
        qwen_name: HuggingFace model ID for Qwen (Model A).
        out_dir: Output directory where JSONL will be written.
        max_examples: If not None, only use the first N examples
                      (useful for debugging).
    """
    # 1. Load data
    data = load_math_lighteval_test()
    if max_examples is not None:
        data = data[:max_examples]

    # 2. Run Qwen as Model A
    qwen_engine = VLLMEngine(qwen_name)
    qwen_results = run_qwen_stage(qwen_engine, data)

    # 3. Save to JSONL
    out_path = Path(out_dir) / "stage1_qwen.jsonl"
    save_jsonl(out_path, qwen_results)
    print(f"[Stage 1] Saved {len(qwen_results)} examples to {out_path}")


if __name__ == "__main__":
    run_stage1_qwen(
        qwen_name="Qwen/Qwen3-4B",
        out_dir="outputs",
        max_examples=800,  # set to None for the full test split
    )
