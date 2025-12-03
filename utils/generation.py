from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .models import VLLMEngine
from .prompts import PromptTemplates
from .parsing import extract_cot_and_answer
from .interventions import apply_intervention


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


def _run_baseline(
    engine: VLLMEngine,
    batch: List[Dict[str, Any]],
) -> List[str]:
    """Run baseline generation for a batch. Returns list of outputs (or None if already exists)."""
    if all(ex.get("answer_b") is not None for ex in batch):
        return [None] * len(batch)
    
    prompts = [
        PromptTemplates.llama_from_cot_prompt(ex["problem"], ex["cot_a"] or "")
        for ex in batch
    ]
    return engine.generate(prompts)


def _run_truncation_intervention(
    engine: VLLMEngine,
    batch: List[Dict[str, Any]],
    truncation_fraction: float,
) -> tuple[List[str], List[str], List[str]]:
    """Run truncation intervention. Returns (truncated_cots, prompts, outputs)."""
    cots = [ex.get("cot_a") or "" for ex in batch]
    truncated_cots = apply_intervention(
        cots,
        intervention_type="truncate_random",
        truncation_fraction=truncation_fraction,
    )
    
    prompts = [
        PromptTemplates.llama_from_cot_prompt(ex["problem"], truncated_cot)
        for ex, truncated_cot in zip(batch, truncated_cots)
    ]
    outputs = engine.generate(prompts)
    
    return truncated_cots, prompts, outputs


def _run_error_injection_intervention(
    engine: VLLMEngine,
    batch: List[Dict[str, Any]],
) -> tuple[List[str], List[str], List[str]]:
    """Run error injection intervention. Returns (injected_cots, prompts, outputs)."""
    cots = [ex.get("cot_a") or "" for ex in batch]
    problems = [ex.get("problem", "") for ex in batch]
    injected_cots = apply_intervention(
        cots,
        intervention_type="inject_error",
        problems=problems,
        engine=engine,
    )
    
    prompts = [
        PromptTemplates.llama_from_cot_prompt(ex["problem"], injected_cot)
        for ex, injected_cot in zip(batch, injected_cots)
    ]
    outputs = engine.generate(prompts)
    
    return injected_cots, prompts, outputs


def run_llama_stage(
    engine: VLLMEngine,
    input_results: List[Dict[str, Any]],
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    Run Model B (Llama) baseline - Model B with original CoT_A.
    
    Args:
        engine: VLLMEngine instance for Model B
        input_results: List of dicts from Stage 1, each containing:
                      - "problem"
                      - "cot_a"
                      - "answer_a"
        batch_size: Batch size for generation
        
    Returns:
        List of dicts with baseline results:
        - "model_b_raw": Raw output from Model B
        - "answer_b": Extracted answer from Model B
    """
    final_results = []
    
    for i in tqdm(range(0, len(input_results), batch_size), desc="Llama Stage (baseline)"):
        batch = input_results[i : i + batch_size]
        
        # Run baseline
        baseline_outputs = _run_baseline(engine, batch)
        
        # Combine results
        for ex, baseline_out in zip(batch, baseline_outputs):
            if baseline_out:
                _, ans_b = extract_cot_and_answer(baseline_out)
                result = dict(ex)
                result.update({
                    "model_b_raw": baseline_out,
                    "answer_b": ans_b,
                })
            else:
                result = dict(ex)
            final_results.append(result)
    
    return final_results


def run_intervention_stage(
    engine: VLLMEngine,
    input_results: List[Dict[str, Any]],
    batch_size: int = 8,
    run_truncation: bool = True,
    run_error_injection: bool = True,
    truncation_fraction: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Run intervention stage - applies truncation and/or error injection interventions.
    
    This function runs interventions on the CoT from Model A and evaluates how Model B
    responds to the perturbed reasoning.
    
    Args:
        engine: VLLMEngine instance for Model B (also used for error injection)
        input_results: List of dicts from Stage 2 (baseline), each containing:
                      - "problem"
                      - "cot_a"
                      - "answer_a"
                      - "answer_b" and "model_b_raw" (baseline results)
        batch_size: Batch size for generation
        run_truncation: If True, run truncation intervention (random sentence removal)
        run_error_injection: If True, run error injection intervention (LLM-generated logical errors)
        truncation_fraction: Fraction of sentences to remove (for truncation)
        
    Returns:
        List of dicts with intervention results added. All original fields are preserved.
        Additional fields:
        - Truncation: "cot_a_truncated", "prompt_truncated", "answer_b_truncated", "model_b_truncated_raw"
        - Error injection: "cot_a_injected", "prompt_injected", "answer_b_injected", "model_b_injected_raw"
    """
    # Build progress description
    interventions = []
    if run_truncation:
        interventions.append("truncation")
    if run_error_injection:
        interventions.append("error_injection")
    desc = f"Intervention Stage ({', '.join(interventions)})"
    
    final_results = []
    
    for i in tqdm(range(0, len(input_results), batch_size), desc=desc):
        batch = input_results[i : i + batch_size]
        
        # Run truncation intervention
        truncated_data = (
            _run_truncation_intervention(engine, batch, truncation_fraction)
            if run_truncation
            else None
        )
        
        # Run error injection intervention
        injected_data = (
            _run_error_injection_intervention(engine, batch)
            if run_error_injection
            else None
        )
        
        # Combine results
        for j, ex in enumerate(batch):
            result = dict(ex)
            
            # Add truncation results
            if truncated_data:
                truncated_cot = truncated_data[0][j]
                truncated_prompt = truncated_data[1][j]
                truncated_out = truncated_data[2][j]
                _, ans_b_truncated = extract_cot_and_answer(truncated_out)
                result.update({
                    "cot_a_truncated": truncated_cot,
                    "prompt_truncated": truncated_prompt,
                    "model_b_truncated_raw": truncated_out,
                    "answer_b_truncated": ans_b_truncated,
                })
            
            # Add error injection results
            if injected_data:
                injected_cot = injected_data[0][j]
                injected_prompt = injected_data[1][j]
                injected_out = injected_data[2][j]
                _, ans_b_injected = extract_cot_and_answer(injected_out)
                result.update({
                    "cot_a_injected": injected_cot,
                    "prompt_injected": injected_prompt,
                    "model_b_injected_raw": injected_out,
                    "answer_b_injected": ans_b_injected,
                })
            
            final_results.append(result)
    
    return final_results
