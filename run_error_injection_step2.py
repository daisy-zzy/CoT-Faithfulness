#!/usr/bin/env python3
"""
Step 2: Use Ray + Llama to run inference on error-injected CoTs.
Reads the modified CoTs from Step 1 and generates Model B responses.
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

from utils.ray_inference import RayVLLMInference, init_ray, shutdown_ray
from utils.prompts import PromptTemplates
from utils.parsing import extract_cot_and_answer, normalize_answer


def load_jsonl(path: Path) -> list:
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def save_jsonl(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to {path}")


def main():
    # Paths
    baselines_dir = Path("outputs/baselines")
    interventions_dir = Path("outputs/interventions")

    cots_path = baselines_dir / "error_injected_cots.json"
    output_path = interventions_dir / "error_injection.jsonl"

    # Check if step 1 is done
    if not cots_path.exists():
        print(f"Error: Run step 1 first! Missing: {cots_path}")
        return

    # Check if already done
    if output_path.exists():
        existing = load_jsonl(output_path)
        if len(existing) == 5000:
            print(f"Error injection already complete: {output_path}")
            return

    # Load data
    with open(cots_path, "r") as f:
        cots_data = json.load(f)

    modified_cots = cots_data["modified_cots"]
    fixed_cots = cots_data["fixed_cots"]
    problems = cots_data["problems"]

    model_a_results = load_jsonl(baselines_dir / "model_a_rollouts.jsonl")
    with open(baselines_dir / "fixed_cot_indices.json", "r") as f:
        fixed_indices = json.load(f)

    # Get fixed answers
    fixed_answers = []
    for idx, item in enumerate(model_a_results):
        rollouts = item.get("rollouts", [])
        fixed_idx = fixed_indices[idx]
        ans = (
            rollouts[fixed_idx].get("answer_normalized")
            if fixed_idx < len(rollouts)
            else None
        )
        fixed_answers.append(ans)

    print(f"Loaded {len(modified_cots)} modified CoTs")

    # Build prompts for Model B (with rollouts)
    n_rollouts = 4
    rng = random.Random(42)

    all_prompts = []
    all_seeds = []
    prompt_to_idx = []

    for idx, (problem, modified_cot) in enumerate(zip(problems, modified_cots)):
        prompt = PromptTemplates.llama_from_cot_prompt(problem, modified_cot)
        for _ in range(n_rollouts):
            all_prompts.append(prompt)
            all_seeds.append(rng.randint(0, 2**31 - 1))
            prompt_to_idx.append(idx)

    print(
        f"Total prompts: {len(all_prompts)} ({len(problems)} problems × {n_rollouts} rollouts)"
    )

    # Initialize Ray
    init_ray(num_gpus=8)

    # Create inference engine
    engine = RayVLLMInference(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        num_gpus=8,
        max_model_len=8192,
        max_tokens=1024,
        temperature=0.7,
        batch_size=128,
    )

    # Generate responses
    # IMPORTANT: Pass original indices to handle Ray's out-of-order results
    print("Running Model B inference...")
    extra_data = [
        {"problem_idx": idx, "rollout_num": i % n_rollouts}
        for i, idx in enumerate(prompt_to_idx)
    ]
    results = engine.generate(all_prompts, seeds=all_seeds, extra_data=extra_data)

    # Shutdown Ray
    shutdown_ray()

    # Group outputs by problem using tracked indices
    outputs_by_problem = [[] for _ in range(len(model_a_results))]
    for result in results:
        idx = result["problem_idx"]
        outputs_by_problem[idx].append(result.get("output", ""))

    # Build final results
    final_results = []
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

        final_results.append(
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
    save_jsonl(final_results, output_path)

    # Stats
    total = sum(len(r["rollouts"]) for r in final_results)
    missing = sum(
        1
        for r in final_results
        for ro in r["rollouts"]
        if not ro.get("answer_normalized")
    )
    correct = sum(
        1
        for r in final_results
        for ro in r["rollouts"]
        if ro.get("answer_normalized") == r.get("ground_truth_normalized")
    )
    matches = sum(
        1
        for r in final_results
        for ro in r["rollouts"]
        if ro.get("answer_normalized")
        and r.get("fixed_answer_a")
        and ro.get("answer_normalized") == r.get("fixed_answer_a")
    )
    valid = sum(
        1
        for r in final_results
        for ro in r["rollouts"]
        if ro.get("answer_normalized") and r.get("fixed_answer_a")
    )

    print(f"\n{'='*50}")
    print("ERROR INJECTION RESULTS")
    print(f"{'='*50}")
    print(f"Total rollouts: {total}")
    print(f"Missing answers: {missing} ({100*missing/total:.1f}%)")
    print(f"Accuracy: {100*correct/total:.1f}%")
    print(f"Match rate: {100*matches/valid:.1f}%")
    print(f"Flip rate: {100*(1-matches/valid):.1f}%")


if __name__ == "__main__":
    main()
