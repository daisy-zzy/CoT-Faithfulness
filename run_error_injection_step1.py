#!/usr/bin/env python3
"""
Step 1: Use Ray + Qwen to inject errors into CoTs.
Saves the modified CoTs to a file for Step 2.
"""

import json
import os
import random
from pathlib import Path
from tqdm import tqdm

from utils.ray_inference import RayVLLMInference, init_ray, shutdown_ray
from utils.prompts import PromptTemplates


def load_jsonl(path: Path) -> list:
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def main():
    # Paths
    baselines_dir = Path("outputs/baselines")
    output_path = baselines_dir / "error_injected_cots.json"

    # Check if already done
    if output_path.exists():
        print(f"Error-injected CoTs already exist: {output_path}")
        return

    # Load Model A results and fixed indices
    model_a_results = load_jsonl(baselines_dir / "model_a_rollouts.jsonl")
    with open(baselines_dir / "fixed_cot_indices.json", "r") as f:
        fixed_indices = json.load(f)

    print(f"Loaded {len(model_a_results)} problems")

    # Get fixed CoTs
    fixed_cots = []
    problems = []
    for idx, item in enumerate(model_a_results):
        rollouts = item.get("rollouts", [])
        fixed_idx = fixed_indices[idx]
        cot = rollouts[fixed_idx].get("cot", "") if fixed_idx < len(rollouts) else ""
        fixed_cots.append(cot or "")
        problems.append(item["problem"])

    # Build error injection prompts
    print("Building error injection prompts...")
    prompts = []
    for cot, problem in zip(fixed_cots, problems):
        if not cot.strip():
            # No CoT to inject error into
            prompts.append(None)
        else:
            prompt = PromptTemplates.inject_error_prompt(problem, cot)
            prompts.append(prompt)

    # Filter to only valid prompts
    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [prompts[i] for i in valid_indices]

    print(f"Valid CoTs to inject errors: {len(valid_prompts)} / {len(prompts)}")

    # Initialize Ray
    init_ray(num_gpus=8)

    # Create inference engine
    engine = RayVLLMInference(
        model_name="Qwen/Qwen3-4B",
        num_gpus=8,
        max_model_len=8192,
        max_tokens=4096,
        temperature=0.7,
        batch_size=128,
    )

    # Generate error-injected CoTs
    print("Injecting errors with Qwen...")
    rng = random.Random(42)
    seeds = [rng.randint(0, 2**31 - 1) for _ in valid_prompts]

    # IMPORTANT: Pass original indices as extra_data so we can restore order
    # Ray may return results out of order!
    extra_data = [{"orig_idx": idx} for idx in valid_indices]
    results = engine.generate(valid_prompts, seeds=seeds, extra_data=extra_data)

    # Extract modified CoTs from outputs
    modified_cots = [""] * len(prompts)  # Initialize all as empty

    def extract_output(text: str, original_cot: str) -> str:
        """Extract content from <OUTPUT>...</OUTPUT> tags."""
        import re

        # Try to find <OUTPUT>...</OUTPUT> tags
        match = re.search(r"<OUTPUT>(.*?)</OUTPUT>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: try <output>...</output> (lowercase)
        match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # No tags found - use original CoT
        print(f"Warning: No <OUTPUT> tags found, using original CoT")
        return original_cot

    n_extracted = 0
    n_fallback = 0

    for result in results:
        # Use tracked index instead of assuming order
        orig_idx = result["orig_idx"]
        output = result.get("output", "")
        original_cot = fixed_cots[orig_idx]
        extracted = extract_output(output, original_cot)

        if extracted != original_cot:
            n_extracted += 1
        else:
            n_fallback += 1

        modified_cots[orig_idx] = extracted

    print(f"Successfully extracted: {n_extracted}, Fallback to original: {n_fallback}")

    # For entries without valid CoT, use original
    for i in range(len(prompts)):
        if prompts[i] is None:
            modified_cots[i] = fixed_cots[i]

    # Save results
    output_data = {
        "fixed_cots": fixed_cots,
        "modified_cots": modified_cots,
        "problems": problems,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f)

    print(f"Saved error-injected CoTs to {output_path}")

    # Shutdown Ray
    shutdown_ray()

    # Quick check
    n_modified = sum(1 for orig, mod in zip(fixed_cots, modified_cots) if orig != mod)
    print(f"Modified {n_modified} / {len(fixed_cots)} CoTs")


if __name__ == "__main__":
    main()
