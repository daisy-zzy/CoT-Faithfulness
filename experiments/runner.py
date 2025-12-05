"""Main experiment runner for CoT Faithfulness evaluation."""

import json
import gc
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from experiments.config import ExperimentConfig, InterventionConfig
from utils.parsing import extract_cot_and_answer, normalize_answer
from utils.statistics import (
    MetricStats,
    ExperimentResults,
    compute_accuracy_from_rollouts,
    compute_omr_from_rollouts,
    compute_mwc_from_rollouts,
    compute_mww_from_rollouts,
    compute_flip_rate_from_rollouts,
)


class ExperimentRunner:
    """Runs the full CoT faithfulness experiment pipeline with optimized batching."""

    def __init__(self, config: ExperimentConfig, dry_run: bool = False):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration
            dry_run: If True, skip actual model inference (for testing)
        """
        self.config = config
        self.dry_run = dry_run
        self.rng = random.Random(config.random_seed)

        # Create output directories
        config.baselines_dir.mkdir(parents=True, exist_ok=True)
        config.interventions_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.dataset: Optional[List[Dict]] = None
        self.model_a_results: Optional[List[Dict]] = None
        self.fixed_cot_indices: Optional[List[int]] = None
        self.model_b_no_cot_results: Optional[List[Dict]] = None
        self.model_b_follow_cot_results: Optional[List[Dict]] = None

        # Wikipedia sentences for filler (loaded lazily)
        self._wikipedia_sentences: Optional[List[str]] = None

        # Track if Ray is initialized
        self._ray_initialized: bool = False

    def _cleanup_gpu(self):
        """Clean up GPU memory."""
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _full_cleanup(self):
        """Full cleanup including Ray shutdown to release all GPU memory."""
        import time

        # First do regular cleanup
        self._cleanup_gpu()

        # Shutdown Ray if it's running to fully release GPU memory
        try:
            import ray

            if ray.is_initialized():
                print("Shutting down Ray to release GPU memory...")
                ray.shutdown()
                self._ray_initialized = False
                # Give some time for GPU memory to be released
                time.sleep(5)
                gc.collect()
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Ray shutdown complete.")
        except Exception as e:
            print(f"Warning: Error during Ray shutdown: {e}")

    def _save_jsonl(self, data: List[Dict], path: Path):
        """Save data to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} items to {path}")

    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load data from JSONL file."""
        results = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    def load_data(self):
        """Load the dataset."""
        print("Loading dataset...")
        from utils.data import load_math_lighteval_test

        self.dataset = load_math_lighteval_test()

        if self.config.max_examples is not None:
            self.dataset = self.dataset[: self.config.max_examples]

        print(f"Loaded {len(self.dataset)} examples")

    def _get_wikipedia_sentences(self) -> List[str]:
        """Get Wikipedia sentences for filler replacement (lazy loading)."""
        if self._wikipedia_sentences is None:
            from utils.interventions_new import load_wikipedia_sentences

            cache_path = str(
                self.config.output_base_dir / "wikipedia_sentences_cache.json"
            )
            self._wikipedia_sentences = load_wikipedia_sentences(
                num_articles=self.config.wikipedia_num_articles,
                subset=self.config.wikipedia_subset,
                cache_path=cache_path,
                seed=self.config.random_seed,
            )
        return self._wikipedia_sentences

    def _create_engine(self, model_config):
        """Create a VLLM engine for inference (single-GPU mode)."""
        if self.dry_run:
            return None
        from utils.models import VLLMEngine

        return VLLMEngine(
            model_config.name,
            max_num_seqs=self.config.inference.max_num_seqs,
            max_model_len=self.config.inference.max_model_len,
            enable_chunked_prefill=self.config.inference.enable_chunked_prefill,
        )

    def _generate_with_rollouts(
        self,
        engine,
        prompts: List[str],
        n_rollouts: int,
        temperature: float,
        max_tokens: int,
        model_name: str,
        desc: str = "Generating",
    ) -> List[List[str]]:
        """
        Generate n rollouts for each prompt using efficient batching.

        Supports both single-GPU vLLM and Ray-based multi-GPU inference.

        Args:
            engine: VLLMEngine instance (None if using Ray)
            prompts: List of N prompts
            n_rollouts: Number of rollouts per prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            model_name: Name of the model to use
            desc: Description for progress bar

        Returns:
            List of N lists, each containing n_rollouts outputs
        """
        n_prompts = len(prompts)

        if self.dry_run:
            # Return dummy responses
            return [
                [f"[DRY RUN] Response {j} for prompt {i}" for j in range(n_rollouts)]
                for i in range(n_prompts)
            ]

        # Flatten: repeat each prompt n_rollouts times
        # Create seeds for reproducibility (each rollout gets a different seed)
        flat_prompts = []
        flat_seeds = []
        flat_indices = []  # Track original (prompt_idx, rollout_idx)
        base_seed = self.rng.randint(0, 2**31)

        for prompt_idx, prompt in enumerate(prompts):
            for rollout_idx in range(n_rollouts):
                flat_prompts.append(prompt)
                flat_seeds.append(base_seed + prompt_idx * n_rollouts + rollout_idx)
                flat_indices.append((prompt_idx, rollout_idx))

        print(
            f"{desc}: {n_prompts} prompts × {n_rollouts} rollouts = {len(flat_prompts)} total generations"
        )

        # Choose inference backend
        if self.config.inference.use_ray and self.config.inference.num_gpus > 1:
            flat_outputs = self._generate_with_ray(
                flat_prompts, flat_seeds, temperature, max_tokens, model_name
            )
        else:
            # Single-GPU vLLM
            flat_outputs = engine.generate_batched(
                flat_prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                seeds=flat_seeds,
                batch_size=self.config.inference.batch_size,
                show_progress=True,
            )

        # Unflatten: group outputs back into per-prompt rollouts
        results = []
        for i in range(n_prompts):
            start_idx = i * n_rollouts
            end_idx = start_idx + n_rollouts
            results.append(flat_outputs[start_idx:end_idx])

        return results

    def _generate_with_ray(
        self,
        prompts: List[str],
        seeds: List[int],
        temperature: float,
        max_tokens: int,
        model_name: str,
    ) -> List[str]:
        """
        Generate using Ray for multi-GPU data parallelism.

        Args:
            prompts: Flattened list of all prompts
            seeds: List of seeds for each prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            model_name: Model name to use

        Returns:
            List of generated outputs in same order as prompts
        """
        from utils.ray_inference import RayVLLMInference

        # Don't manually init Ray - let build_llm_processor handle it
        # It will call ray.init() internally with appropriate settings

        engine = RayVLLMInference(
            model_name=model_name,
            num_gpus=self.config.inference.num_gpus,
            max_model_len=self.config.inference.max_model_len,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_size=self.config.inference.batch_size,
            max_num_batched_tokens=self.config.inference.max_num_batched_tokens,
            enable_chunked_prefill=self.config.inference.enable_chunked_prefill,
        )

        # Add indices to track order (Ray may return out of order)
        extra_data = [{"idx": i} for i in range(len(prompts))]

        results = engine.generate(prompts, seeds=seeds, extra_data=extra_data)

        # Sort by original index and extract outputs
        results.sort(key=lambda x: x["idx"])
        outputs = [r["output"] for r in results]

        return outputs

    def run_model_a_baseline(self):
        """
        Stage 1: Run Model A to generate n rollouts of (CoT, answer) per problem.
        Uses efficient batched generation.
        """
        print("\n" + "=" * 60)
        print("STAGE 1: Model A Baseline (n rollouts)")
        print("=" * 60)

        output_path = self.config.baselines_dir / "model_a_rollouts.jsonl"

        # Check if already exists
        if output_path.exists():
            print(f"Loading existing results from {output_path}")
            self.model_a_results = self._load_jsonl(output_path)
            self._load_or_fix_cot_indices()
            return

        from utils.prompts import PromptTemplates

        # Only create engine if not using Ray (Ray manages its own engines)
        engine = None
        if not (self.config.inference.use_ray and self.config.inference.num_gpus > 1):
            engine = self._create_engine(self.config.model_a)

        # Build all prompts
        prompts = [
            PromptTemplates.qwen_cot_prompt(item["problem"]) for item in self.dataset
        ]

        # Generate all rollouts in batched fashion
        all_rollout_outputs = self._generate_with_rollouts(
            engine=engine,
            prompts=prompts,
            n_rollouts=self.config.n_rollouts,
            temperature=self.config.model_a.temperature,
            max_tokens=self.config.model_a.max_tokens,
            model_name=self.config.model_a.name,
            desc="Model A generation",
        )

        # Process outputs
        results = []
        for idx, item in enumerate(
            tqdm(self.dataset, desc="Processing Model A outputs")
        ):
            gt = item.get("ground_truth", "")
            gt_normalized = normalize_answer(gt) if gt else ""

            rollout_outputs = []
            for output in all_rollout_outputs[idx]:
                cot, answer = extract_cot_and_answer(output)
                answer_normalized = normalize_answer(answer) if answer else None

                rollout_outputs.append(
                    {
                        "output": output,
                        "cot": cot,
                        "answer": answer,
                        "answer_normalized": answer_normalized,
                    }
                )

            results.append(
                {
                    "id": item.get("id"),
                    "problem": item["problem"],
                    "ground_truth": gt,
                    "ground_truth_normalized": gt_normalized,
                    "rollouts": rollout_outputs,
                }
            )

        # Save results
        self._save_jsonl(results, output_path)
        self.model_a_results = results

        # Fix which rollout to use for interventions
        self._fix_cot_indices()

        # Cleanup
        if engine is not None:
            del engine
        self._cleanup_gpu()

        # Print stats
        self._print_model_a_stats()

    def _fix_cot_indices(self):
        """Randomly fix which rollout to use for each problem."""
        self.fixed_cot_indices = [
            self.rng.randint(0, self.config.n_rollouts - 1)
            for _ in range(len(self.model_a_results))
        ]

        # Save the fixed indices
        indices_path = self.config.baselines_dir / "fixed_cot_indices.json"
        with open(indices_path, "w") as f:
            json.dump(self.fixed_cot_indices, f)
        print(f"Fixed CoT indices saved to {indices_path}")

    def _load_or_fix_cot_indices(self):
        """Load fixed indices from file or create new ones."""
        indices_path = self.config.baselines_dir / "fixed_cot_indices.json"

        if indices_path.exists():
            with open(indices_path, "r") as f:
                self.fixed_cot_indices = json.load(f)
            print(f"Loaded fixed CoT indices from {indices_path}")
        else:
            self._fix_cot_indices()

    def get_fixed_cot_and_answer(self, example_idx: int) -> Tuple[str, str]:
        """Get the fixed (CoT, answer) for a given example."""
        rollout_idx = self.fixed_cot_indices[example_idx]
        rollout = self.model_a_results[example_idx]["rollouts"][rollout_idx]
        return rollout.get("cot", ""), rollout.get("answer_normalized", "")

    def run_model_b_no_cot_baseline(self):
        """
        Baseline 1: Model B answers without any CoT (n rollouts).
        Uses efficient batched generation.
        """
        print("\n" + "=" * 60)
        print("BASELINE 1: Model B No-CoT (n rollouts)")
        print("=" * 60)

        output_path = self.config.baselines_dir / "model_b_no_cot.jsonl"

        if output_path.exists():
            print(f"Loading existing results from {output_path}")
            self.model_b_no_cot_results = self._load_jsonl(output_path)
            return

        from utils.prompts import PromptTemplates

        # Only create engine if not using Ray
        engine = None
        if not (self.config.inference.use_ray and self.config.inference.num_gpus > 1):
            engine = self._create_engine(self.config.model_b)

        # Build all prompts
        prompts = [
            PromptTemplates.llama_no_cot_prompt(item["problem"])
            for item in self.model_a_results
        ]

        # Generate all rollouts in batched fashion
        all_rollout_outputs = self._generate_with_rollouts(
            engine=engine,
            prompts=prompts,
            n_rollouts=self.config.n_rollouts,
            temperature=self.config.model_b.temperature,
            max_tokens=self.config.model_b.max_tokens,
            model_name=self.config.model_b.name,
            desc="Model B No-CoT generation",
        )

        # Process outputs
        results = []
        for idx, item in enumerate(
            tqdm(self.model_a_results, desc="Processing Model B No-CoT outputs")
        ):
            rollout_outputs = []
            for output in all_rollout_outputs[idx]:
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
                    "rollouts": rollout_outputs,
                }
            )

        self._save_jsonl(results, output_path)
        self.model_b_no_cot_results = results

        if engine is not None:
            del engine
        self._cleanup_gpu()

    def run_model_b_follow_cot_baseline(self):
        """
        Baseline 2: Model B follows fixed CoT from Model A (n rollouts).
        Uses efficient batched generation.
        """
        print("\n" + "=" * 60)
        print("BASELINE 2: Model B Follow-CoT (n rollouts)")
        print("=" * 60)

        output_path = self.config.baselines_dir / "model_b_follow_cot.jsonl"

        if output_path.exists():
            print(f"Loading existing results from {output_path}")
            self.model_b_follow_cot_results = self._load_jsonl(output_path)
            return

        from utils.prompts import PromptTemplates

        # Only create engine if not using Ray
        engine = None
        if not (self.config.inference.use_ray and self.config.inference.num_gpus > 1):
            engine = self._create_engine(self.config.model_b)

        # Build all prompts using fixed CoTs
        prompts = []
        fixed_cots = []
        fixed_answers = []
        for idx, item in enumerate(self.model_a_results):
            fixed_cot, fixed_answer = self.get_fixed_cot_and_answer(idx)
            fixed_cots.append(fixed_cot)
            fixed_answers.append(fixed_answer)
            prompts.append(
                PromptTemplates.llama_from_cot_prompt(item["problem"], fixed_cot or "")
            )

        # Generate all rollouts in batched fashion
        all_rollout_outputs = self._generate_with_rollouts(
            engine=engine,
            prompts=prompts,
            n_rollouts=self.config.n_rollouts,
            temperature=self.config.model_b.temperature,
            max_tokens=self.config.model_b.max_tokens,
            model_name=self.config.model_b.name,
            desc="Model B Follow-CoT generation",
        )

        # Process outputs
        results = []
        for idx, item in enumerate(
            tqdm(self.model_a_results, desc="Processing Model B Follow-CoT outputs")
        ):
            rollout_outputs = []
            for output in all_rollout_outputs[idx]:
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
                    "fixed_cot": fixed_cots[idx],
                    "fixed_answer_a": fixed_answers[idx],
                    "rollouts": rollout_outputs,
                }
            )

        self._save_jsonl(results, output_path)
        self.model_b_follow_cot_results = results

        if engine is not None:
            del engine
        self._cleanup_gpu()

    def run_intervention(self, intervention_config: InterventionConfig) -> List[Dict]:
        """
        Run a single intervention experiment with n rollouts.
        Uses efficient batched generation.
        """
        print("\n" + "-" * 60)
        print(f"INTERVENTION: {intervention_config.run_id}")
        print("-" * 60)

        output_path = (
            self.config.interventions_dir / f"{intervention_config.run_id}.jsonl"
        )

        if output_path.exists():
            print(f"Loading existing results from {output_path}")
            return self._load_jsonl(output_path)

        # Special handling for error injection (needs LLM)
        if intervention_config.name == "error_injection":
            return self._run_error_injection_intervention(output_path)

        from utils.prompts import PromptTemplates
        from utils.interventions_new import get_intervention_fn

        # Get the intervention function
        wiki_sentences = None
        if intervention_config.name == "filler_replacement":
            wiki_sentences = self._get_wikipedia_sentences()

        intervention_fn = get_intervention_fn(
            intervention_config.name,
            intervention_config.params,
            self.rng,
            wiki_sentences,
        )

        # Apply interventions to all CoTs first
        modified_cots = []
        fixed_cots = []
        fixed_answers = []
        for idx in range(len(self.model_a_results)):
            fixed_cot, fixed_answer = self.get_fixed_cot_and_answer(idx)
            fixed_cots.append(fixed_cot)
            fixed_answers.append(fixed_answer)
            modified_cots.append(intervention_fn(fixed_cot or ""))

        # Build all prompts
        prompts = [
            PromptTemplates.llama_from_cot_prompt(item["problem"], modified_cot)
            for item, modified_cot in zip(self.model_a_results, modified_cots)
        ]

        # Only create engine if not using Ray
        engine = None
        if not (self.config.inference.use_ray and self.config.inference.num_gpus > 1):
            engine = self._create_engine(self.config.model_b)

        # Generate all rollouts in batched fashion
        all_rollout_outputs = self._generate_with_rollouts(
            engine=engine,
            prompts=prompts,
            n_rollouts=self.config.n_rollouts,
            temperature=self.config.model_b.temperature,
            max_tokens=self.config.model_b.max_tokens,
            model_name=self.config.model_b.name,
            desc=f"Intervention {intervention_config.run_id}",
        )

        # Process outputs
        results = []
        for idx, item in enumerate(
            tqdm(
                self.model_a_results,
                desc=f"Processing {intervention_config.run_id} outputs",
            )
        ):
            rollout_outputs = []
            for output in all_rollout_outputs[idx]:
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

        self._save_jsonl(results, output_path)

        if engine is not None:
            del engine
        self._cleanup_gpu()

        return results

    def _run_error_injection_intervention(self, output_path: Path) -> List[Dict]:
        """Special handling for error injection which needs an LLM."""
        from utils.prompts import PromptTemplates
        from utils.interventions_new import inject_error_batch

        print("Initializing Model A (Qwen) for error injection...")
        qwen_engine = self._create_engine(self.config.model_a)

        # First, inject errors into all fixed CoTs
        fixed_cots = [
            self.get_fixed_cot_and_answer(i)[0] or ""
            for i in range(len(self.model_a_results))
        ]
        fixed_answers = [
            self.get_fixed_cot_and_answer(i)[1]
            for i in range(len(self.model_a_results))
        ]
        problems = [item["problem"] for item in self.model_a_results]

        print("Injecting errors...")
        if self.dry_run:
            modified_cots = [f"[ERROR INJECTED] {cot}" for cot in fixed_cots]
        else:
            modified_cots = []
            batch_size = min(self.config.batch_size, 16)  # Batch error injection too
            for i in tqdm(
                range(0, len(fixed_cots), batch_size), desc="Error injection"
            ):
                batch_cots = fixed_cots[i : i + batch_size]
                batch_problems = problems[i : i + batch_size]
                modified_batch = inject_error_batch(
                    batch_cots, batch_problems, qwen_engine
                )
                modified_cots.extend(modified_batch)

        if qwen_engine is not None:
            del qwen_engine
        self._cleanup_gpu()

        # Build all prompts for Model B
        prompts = [
            PromptTemplates.llama_from_cot_prompt(problem, modified_cot)
            for problem, modified_cot in zip(problems, modified_cots)
        ]

        # Now run Model B on modified CoTs
        print("Running Model B on error-injected CoTs...")
        # Only create engine if not using Ray
        llama_engine = None
        if not (self.config.inference.use_ray and self.config.inference.num_gpus > 1):
            llama_engine = self._create_engine(self.config.model_b)

        # Generate all rollouts in batched fashion
        all_rollout_outputs = self._generate_with_rollouts(
            engine=llama_engine,
            prompts=prompts,
            n_rollouts=self.config.n_rollouts,
            temperature=self.config.model_b.temperature,
            max_tokens=self.config.model_b.max_tokens,
            model_name=self.config.model_b.name,
            desc="Model B (error injection)",
        )

        # Process outputs
        results = []
        for idx, item in enumerate(
            tqdm(self.model_a_results, desc="Processing error injection outputs")
        ):
            rollout_outputs = []
            for output in all_rollout_outputs[idx]:
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

        self._save_jsonl(results, output_path)

        if llama_engine is not None:
            del llama_engine
        self._cleanup_gpu()

        return results

    def run_all_interventions(self) -> Dict[str, List[Dict]]:
        """Run all intervention experiments."""
        print("\n" + "=" * 60)
        print("RUNNING ALL INTERVENTIONS")
        print("=" * 60)

        intervention_results = {}
        intervention_configs = self.config.get_intervention_configs()

        for i, intervention_config in enumerate(intervention_configs):
            results = self.run_intervention(intervention_config)
            intervention_results[intervention_config.run_id] = results

            # Full cleanup between interventions to prevent GPU memory issues
            # Only do this if there are more interventions to run
            if i < len(intervention_configs) - 1:
                print(f"\nPerforming full cleanup before next intervention...")
                self._full_cleanup()

        return intervention_results

    def compute_all_metrics(self) -> ExperimentResults:
        """Compute all metrics from the collected results."""
        print("\n" + "=" * 60)
        print("COMPUTING METRICS")
        print("=" * 60)

        # Helper to extract rollout predictions
        def extract_rollout_preds(results: List[Dict]) -> List[List[str]]:
            return [
                [r.get("answer_normalized", "") for r in item.get("rollouts", [])]
                for item in results
            ]

        ground_truths = [
            item["ground_truth_normalized"] for item in self.model_a_results
        ]

        # Model A accuracy (use all rollouts)
        model_a_rollouts = [
            [r.get("answer_normalized", "") for r in item.get("rollouts", [])]
            for item in self.model_a_results
        ]
        model_a_acc, _ = compute_accuracy_from_rollouts(model_a_rollouts, ground_truths)
        print(f"Model A Accuracy: {model_a_acc}")

        # Model B No-CoT accuracy
        model_b_no_cot_acc = None
        if self.model_b_no_cot_results:
            model_b_no_cot_rollouts = extract_rollout_preds(self.model_b_no_cot_results)
            model_b_no_cot_acc, _ = compute_accuracy_from_rollouts(
                model_b_no_cot_rollouts, ground_truths
            )
            print(f"Model B No-CoT Accuracy: {model_b_no_cot_acc}")

        # Model B Follow-CoT metrics
        model_b_follow_acc = None
        omr = mwc = mww = None
        model_b_follow_rollouts = None

        if self.model_b_follow_cot_results:
            model_b_follow_rollouts = extract_rollout_preds(
                self.model_b_follow_cot_results
            )
            model_b_follow_acc, _ = compute_accuracy_from_rollouts(
                model_b_follow_rollouts, ground_truths
            )
            print(f"Model B Follow-CoT Accuracy: {model_b_follow_acc}")

            # Fixed Model A answers for OMR/MWC/MWW
            fixed_a_answers = [
                self.get_fixed_cot_and_answer(i)[1]
                for i in range(len(self.model_a_results))
            ]

            omr = compute_omr_from_rollouts(model_b_follow_rollouts, fixed_a_answers)
            mwc = compute_mwc_from_rollouts(
                model_b_follow_rollouts, fixed_a_answers, ground_truths
            )
            mww = compute_mww_from_rollouts(
                model_b_follow_rollouts, fixed_a_answers, ground_truths
            )

            print(f"OMR: {omr}")
            print(f"MWC: {mwc}")
            print(f"MWW: {mww}")

        # Intervention metrics
        intervention_metrics = {}
        for intervention_config in self.config.get_intervention_configs():
            run_id = intervention_config.run_id
            results_path = self.config.interventions_dir / f"{run_id}.jsonl"

            if results_path.exists() and model_b_follow_rollouts is not None:
                int_results = self._load_jsonl(results_path)
                int_rollouts = extract_rollout_preds(int_results)

                flip_rate = compute_flip_rate_from_rollouts(
                    model_b_follow_rollouts, int_rollouts
                )
                int_acc, _ = compute_accuracy_from_rollouts(int_rollouts, ground_truths)

                fixed_a_answers = [
                    self.get_fixed_cot_and_answer(i)[1]
                    for i in range(len(self.model_a_results))
                ]
                int_omr = compute_omr_from_rollouts(int_rollouts, fixed_a_answers)

                intervention_metrics[run_id] = {
                    "flip_rate": flip_rate,
                    "accuracy": int_acc,
                    "omr": int_omr,
                }
                print(f"Intervention {run_id}: Flip Rate = {flip_rate}")

        return ExperimentResults(
            experiment_name="cot_faithfulness",
            config={
                "n_rollouts": self.config.n_rollouts,
                "seed": self.config.random_seed,
            },
            model_a_accuracy=model_a_acc,
            model_b_no_cot_accuracy=model_b_no_cot_acc,
            model_b_follow_cot_accuracy=model_b_follow_acc,
            omr=omr,
            mwc=mwc,
            mww=mww,
            intervention_results=intervention_metrics,
        )

    def run_full_pipeline(self) -> ExperimentResults:
        """Run the complete experiment pipeline."""
        self.load_data()
        self.run_model_a_baseline()
        self.run_model_b_no_cot_baseline()
        self.run_model_b_follow_cot_baseline()
        self.run_all_interventions()

        results = self.compute_all_metrics()

        # Save final results
        results_path = self.config.output_base_dir / "experiment_results.json"
        results.save(str(results_path))
        print(f"\nFinal results saved to {results_path}")

        # Print summary
        self._print_results_summary(results)

        return results

    def _print_model_a_stats(self):
        """Print Model A statistics."""
        if not self.model_a_results:
            return

        correct = 0
        total = 0
        for item in self.model_a_results:
            gt = item.get("ground_truth_normalized", "")
            for rollout in item.get("rollouts", []):
                if rollout.get("answer_normalized") == gt:
                    correct += 1
                total += 1

        if total > 0:
            print(f"Model A accuracy (all rollouts): {correct/total:.3f}")

    def _print_results_summary(self, results: ExperimentResults):
        """Print a summary of all results."""
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 60)

        if results.model_a_accuracy:
            print(f"\nModel A Accuracy: {results.model_a_accuracy}")
        if results.model_b_no_cot_accuracy:
            print(f"Model B No-CoT Accuracy: {results.model_b_no_cot_accuracy}")
        if results.model_b_follow_cot_accuracy:
            print(f"Model B Follow-CoT Accuracy: {results.model_b_follow_cot_accuracy}")

        if results.omr:
            print(f"\nOMR (Pr[B=A]): {results.omr}")
        if results.mwc:
            print(f"MWC (Pr[B=A|A correct]): {results.mwc}")
        if results.mww:
            print(f"MWW (Pr[B=A|A wrong]): {results.mww}")

        if results.intervention_results:
            print("\nIntervention Flip Rates:")
            for name, metrics in sorted(results.intervention_results.items()):
                if "flip_rate" in metrics:
                    print(f"  {name}: {metrics['flip_rate']}")
