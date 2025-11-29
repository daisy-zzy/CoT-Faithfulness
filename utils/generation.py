from typing import List, Dict, Any
from tqdm import tqdm

from .models import VLLMEngine
from .prompts import PromptTemplates
from .parsing import extract_cot_and_answer

def run_qwen_stage(
    engine: VLLMEngine,
    examples: List[Dict[str, Any]],
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    For each example:
      - prompt Qwen
      - parse out CoT_A and answer_A
    Return a list of dicts with extra fields.
    """
    results = []
    for i in tqdm(range(0, len(examples), batch_size), desc="Qwen Stage"):
        batch = examples[i : i + batch_size]
        prompts = [
            PromptTemplates.qwen_cot_prompt(ex["problem"])
            for ex in batch
        ]
        outputs = engine.generate(prompts)
        for ex, out in zip(batch, outputs):
            cot, ans = extract_cot_and_answer(out)
            item = {
                "id": ex.get("id", None),
                "problem": ex["problem"],
                "ground_truth": ex.get("ground_truth", None),
                "model_a_raw": out,
                "cot_a": cot,
                "answer_a": ans,
            }
            results.append(item)
    return results

def run_llama_stage(
    engine: VLLMEngine,
    qwen_results: List[Dict[str, Any]],
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    For each Qwen result with (problem, cot_a):
      - build new prompt
      - let Llama generate final answer
    """
    final_results = []
    for i in tqdm(range(0, len(qwen_results), batch_size), desc="Llama Stage"):
        batch = qwen_results[i : i + batch_size]
        prompts = [
            PromptTemplates.llama_from_cot_prompt(ex["problem"], ex["cot_a"] or "")
            for ex in batch
        ]
        outputs = engine.generate(prompts)
        for ex, out in zip(batch, outputs):
            _, ans_b = extract_cot_and_answer(out)
            new_ex = dict(ex)
            new_ex.update(
                {
                    "model_b_raw": out,
                    "answer_b": ans_b,
                }
            )
            final_results.append(new_ex)
    return final_results
