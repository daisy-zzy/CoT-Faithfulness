#!/usr/bin/env python3
"""
Standalone script to run error injection intervention.

This runs WITHOUT Ray to avoid CUDA multiprocessing issues.
It uses the regular VLLMEngine for both error injection (Qwen)
and Model B inference (Llama).
"""

import json
import gc
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch

from experiments.config import ExperimentConfig
from utils.models import VLLMEngine
from utils.parsing import extract_cot_and_answer, normalize_answer
from utils.prompts import PromptTemplates
from utils.interventions_new import inject_error_batch


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def save_jsonl(data: List[Dict], path: Path):
    """Save to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to {path}")


def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    # Load config
    config = ExperimentConfig.from_yaml("configs/experiment_config.yaml")
    rng = random.Random(config.random_seed)

    output_path = config.interventions_dir / "error_injection.jsonl"

    # Check if already done
    if output_path.exists():
        existing = load_jsonl(output_path)
        if len(existing) == 5000:
            print(f"Error injection already complete: {output_path}")
            return
        print(f"Found partial results ({len(existing)}), rerunning...")

    # Load Model A results and fixed indices
    model_a_results = load_jsonl(config.baselines_dir / "model_a_rollouts.jsonl")
    with open(config.baselines_dir / "fixed_cot_indices.json", "r") as f:
        fixed_indices = json.load(f)

    print(f"Loaded {len(model_a_results)} Model A results")

    # Get fixed CoTs and answers
    def get_fixed_cot_and_answer(idx):
        item = model_a_results[idx]
        rollouts = item.get("rollouts", [])
        fixed_idx = fixed_indices[idx]
        if fixed_idx < len(rollouts):
            rollout = rollouts[fixed_idx]
            return rollout.get("cot"), rollout.get("answer_normalized")
        return None, None

    fixed_cots = []
    fixed_answers = []
    for idx in range(len(model_a_results)):
        cot, ans = get_fixed_cot_and_answer(idx)
        fixed_cots.append(cot or "")
        fixed_answers.append(ans)

    problems = [item["problem"] for item in model_a_results]

    # Step 1: Initialize Qwen and inject errors
    print("\n" + "=" * 50)
    print("STEP 1: Injecting errors with Qwen")
    print("=" * 50)

    qwen_engine = VLLMEngine(
        model_name=config.model_a.name,
        temperature=0.7,  # Use some temperature for diverse errors
        max_tokens=config.model_a.max_tokens,
        gpu_memory_utilization=0.85,
    )

    # Inject errors in batches
    modified_cots = []
    batch_size = 16
    for i in tqdm(range(0, len(fixed_cots), batch_size), desc="Error injection"):
        batch_cots = fixed_cots[i : i + batch_size]
        batch_problems = problems[i : i + batch_size]
        modified_batch = inject_error_batch(batch_cots, batch_problems, qwen_engine)
        modified_cots.extend(modified_batch)

    # Clean up Qwen
    del qwen_engine
    cleanup_gpu()

    print(f"Injected errors into {len(modified_cots)} CoTs")

    # Step 2: Initialize Llama and run inference
    print("\n" + "=" * 50)
    print("STEP 2: Running Model B on error-injected CoTs")
    print("=" * 50)

    llama_engine = VLLMEngine(
        model_name=config.model_b.name,
        temperature=config.model_b.temperature,
        max_tokens=config.model_b.max_tokens,
        gpu_memory_utilization=0.85,
    )

    # Build prompts for all rollouts
    n_rollouts = config.n_rollouts
    all_prompts = []
    prompt_to_idx = []  # Track which problem each prompt belongs to

    for idx, (problem, modified_cot) in enumerate(zip(problems, modified_cots)):
        prompt = PromptTemplates.llama_from_cot_prompt(problem, modified_cot)
        for _ in range(n_rollouts):
            all_prompts.append(prompt)
            prompt_to_idx.append(idx)

    # Generate seeds for reproducibility
    seeds = [rng.randint(0, 2**31 - 1) for _ in all_prompts]

    # Run inference in batches
    all_outputs = []
    batch_size = 64
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Model B inference"):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_seeds = seeds[i : i + batch_size]
        outputs = llama_engine.generate_batch(batch_prompts, seeds=batch_seeds)
        all_outputs.extend(outputs)

    # Clean up Llama
    del llama_engine
    cleanup_gpu()

    # Step 3: Process outputs and save results
    print("\n" + "=" * 50)
    print("STEP 3: Processing outputs")
    print("=" * 50)

    # Group outputs by problem
    outputs_by_problem = [[] for _ in range(len(model_a_results))]
    for output, idx in zip(all_outputs, prompt_to_idx):
        outputs_by_problem[idx].append(output)

    # Build results
    results = []
    for idx, item in enumerate(tqdm(model_a_results, desc="Processing outputs")):
        rollout_outputs = []
        for output in outputs_by_problem[idx]:
            _, answer = extract_cot_and_answer(output)
            answer_normalized = normalize_answer(answer) if answer else None
            rollout_outputs.append(
                {
                    "output": output,
                    "answer": answer,
                    "answer_normalized": answer_normalized,
                }
            )

        results.append(
            {
                "id": item.get("id"),
                "problem": item["problem"],
                "ground_truth": item["ground_truth"],
                "ground_truth_normalized": item["ground_truth_normalized"],
                "original_cot": fixed_cots[idx],
                "modified_cot": modified_cots[idx],
                "fixed_answer_a": fixed_answers[idx],
                "rollouts": rollout_outputs,
            }
        )

    # Save results
    save_jsonl(results, output_path)

    # Quick stats
    total = sum(len(r["rollouts"]) for r in results)
    missing = sum(
        1 for r in results for ro in r["rollouts"] if not ro.get("answer_normalized")
    )
    correct = sum(
        1
        for r in results
        for ro in r["rollouts"]
        if ro.get("answer_normalized") == r.get("ground_truth_normalized")
    )
    matches = sum(
        1
        for r in results
        for ro in r["rollouts"]
        if ro.get("answer_normalized")
        and r.get("fixed_answer_a")
        and ro.get("answer_normalized") == r.get("fixed_answer_a")
    )
    valid = sum(
        1
        for r in results
        for ro in r["rollouts"]
        if ro.get("answer_normalized") and r.get("fixed_answer_a")
    )

    print(f"\n" + "=" * 50)
    print("ERROR INJECTION RESULTS")
    print("=" * 50)
    print(f"Total rollouts: {total}")
    print(f"Missing answers: {missing} ({100*missing/total:.1f}%)")
    print(f"Accuracy: {100*correct/total:.1f}%")
    print(f"Match rate: {100*matches/valid:.1f}%")
    print(f"Flip rate: {100*(1-matches/valid):.1f}%")


if __name__ == "__main__":
    main()
