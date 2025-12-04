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
    MetricStats, ExperimentResults,
    compute_accuracy_from_rollouts, compute_omr_from_rollouts,
    compute_mwc_from_rollouts, compute_mww_from_rollouts,
    compute_flip_rate_from_rollouts
)


class ExperimentRunner:
    """Runs the full CoT faithfulness experiment pipeline."""
    
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
    
    def _cleanup_gpu(self):
        """Clean up GPU memory."""
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _save_jsonl(self, data: List[Dict], path: Path):
        """Save data to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} items to {path}")
    
    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load data from JSONL file."""
        results = []
        with open(path, 'r') as f:
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
            self.dataset = self.dataset[:self.config.max_examples]
        
        print(f"Loaded {len(self.dataset)} examples")
    
    def _get_wikipedia_sentences(self) -> List[str]:
        """Get Wikipedia sentences for filler replacement (lazy loading)."""
        if self._wikipedia_sentences is None:
            from utils.interventions_new import load_wikipedia_sentences
            cache_path = str(self.config.output_base_dir / "wikipedia_sentences_cache.json")
            self._wikipedia_sentences = load_wikipedia_sentences(
                num_articles=self.config.wikipedia_num_articles,
                subset=self.config.wikipedia_subset,
                cache_path=cache_path,
                seed=self.config.random_seed
            )
        return self._wikipedia_sentences
    
    def _create_engine(self, model_config):
        """Create a VLLM engine for inference."""
        if self.dry_run:
            return None
        from utils.models import VLLMEngine
        return VLLMEngine(model_config.name)
    
    def _generate_batch(
        self, 
        engine, 
        prompts: List[str], 
        temperature: float,
        max_tokens: int
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        if self.dry_run:
            # Return dummy responses for testing
            return [f"[DRY RUN] Response for prompt {i}" for i in range(len(prompts))]
        return engine.generate(prompts, temperature=temperature, max_tokens=max_tokens)
    
    def run_model_a_baseline(self):
        """
        Stage 1: Run Model A to generate n rollouts of (CoT, answer) per problem.
        """
        print("\n" + "="*60)
        print("STAGE 1: Model A Baseline (n rollouts)")
        print("="*60)
        
        output_path = self.config.baselines_dir / "model_a_rollouts.jsonl"
        
        # Check if already exists
        if output_path.exists():
            print(f"Loading existing results from {output_path}")
            self.model_a_results = self._load_jsonl(output_path)
            self._load_or_fix_cot_indices()
            return
        
        from utils.prompts import PromptTemplates
        
        engine = self._create_engine(self.config.model_a)
        
        results = []
        for item in tqdm(self.dataset, desc="Model A generation"):
            problem = item['problem']
            gt = item.get('ground_truth', '')
            gt_normalized = normalize_answer(gt) if gt else ''
            
            # Generate n rollouts
            prompt = PromptTemplates.qwen_cot_prompt(problem)
            rollout_outputs = []
            
            for rollout_idx in range(self.config.n_rollouts):
                if self.dry_run:
                    output = f"<think>Step 1: Consider the problem. Step 2: Apply logic.</think>\nFinal Answer: 42"
                else:
                    outputs = engine.generate(
                        [prompt], 
                        temperature=self.config.model_a.temperature,
                        max_tokens=self.config.model_a.max_tokens
                    )
                    output = outputs[0]
                
                cot, answer = extract_cot_and_answer(output)
                answer_normalized = normalize_answer(answer) if answer else None
                
                rollout_outputs.append({
                    'output': output,
                    'cot': cot,
                    'answer': answer,
                    'answer_normalized': answer_normalized
                })
            
            results.append({
                'id': item.get('id'),
                'problem': problem,
                'ground_truth': gt,
                'ground_truth_normalized': gt_normalized,
                'rollouts': rollout_outputs
            })
        
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
        with open(indices_path, 'w') as f:
            json.dump(self.fixed_cot_indices, f)
        print(f"Fixed CoT indices saved to {indices_path}")
    
    def _load_or_fix_cot_indices(self):
        """Load fixed indices from file or create new ones."""
        indices_path = self.config.baselines_dir / "fixed_cot_indices.json"
        
        if indices_path.exists():
            with open(indices_path, 'r') as f:
                self.fixed_cot_indices = json.load(f)
            print(f"Loaded fixed CoT indices from {indices_path}")
        else:
            self._fix_cot_indices()
    
    def get_fixed_cot_and_answer(self, example_idx: int) -> Tuple[str, str]:
        """Get the fixed (CoT, answer) for a given example."""
        rollout_idx = self.fixed_cot_indices[example_idx]
        rollout = self.model_a_results[example_idx]['rollouts'][rollout_idx]
        return rollout.get('cot', ''), rollout.get('answer_normalized', '')
    
    def run_model_b_no_cot_baseline(self):
        """
        Baseline 1: Model B answers without any CoT (n rollouts).
        """
        print("\n" + "="*60)
        print("BASELINE 1: Model B No-CoT (n rollouts)")
        print("="*60)
        
        output_path = self.config.baselines_dir / "model_b_no_cot.jsonl"
        
        if output_path.exists():
            print(f"Loading existing results from {output_path}")
            self.model_b_no_cot_results = self._load_jsonl(output_path)
            return
        
        from utils.prompts import PromptTemplates
        
        engine = self._create_engine(self.config.model_b)
        
        results = []
        for idx, item in enumerate(tqdm(self.model_a_results, desc="Model B No-CoT")):
            problem = item['problem']
            
            prompt = PromptTemplates.llama_no_cot_prompt(problem)
            rollout_outputs = []
            
            for _ in range(self.config.n_rollouts):
                if self.dry_run:
                    output = "Final Answer: 42"
                else:
                    outputs = engine.generate(
                        [prompt],
                        temperature=self.config.model_b.temperature,
                        max_tokens=self.config.model_b.max_tokens
                    )
                    output = outputs[0]
                
                _, answer = extract_cot_and_answer(output)
                answer_normalized = normalize_answer(answer) if answer else None
                
                rollout_outputs.append({
                    'output': output,
                    'answer': answer,
                    'answer_normalized': answer_normalized
                })
            
            results.append({
                'id': item.get('id'),
                'problem': problem,
                'ground_truth': item['ground_truth'],
                'ground_truth_normalized': item['ground_truth_normalized'],
                'rollouts': rollout_outputs
            })
        
        self._save_jsonl(results, output_path)
        self.model_b_no_cot_results = results
        
        if engine is not None:
            del engine
        self._cleanup_gpu()
    
    def run_model_b_follow_cot_baseline(self):
        """
        Baseline 2: Model B follows fixed CoT from Model A (n rollouts).
        """
        print("\n" + "="*60)
        print("BASELINE 2: Model B Follow-CoT (n rollouts)")
        print("="*60)
        
        output_path = self.config.baselines_dir / "model_b_follow_cot.jsonl"
        
        if output_path.exists():
            print(f"Loading existing results from {output_path}")
            self.model_b_follow_cot_results = self._load_jsonl(output_path)
            return
        
        from utils.prompts import PromptTemplates
        
        engine = self._create_engine(self.config.model_b)
        
        results = []
        for idx, item in enumerate(tqdm(self.model_a_results, desc="Model B Follow-CoT")):
            problem = item['problem']
            fixed_cot, fixed_answer = self.get_fixed_cot_and_answer(idx)
            
            prompt = PromptTemplates.llama_from_cot_prompt(problem, fixed_cot or "")
            rollout_outputs = []
            
            for _ in range(self.config.n_rollouts):
                if self.dry_run:
                    output = "Final Answer: 42"
                else:
                    outputs = engine.generate(
                        [prompt],
                        temperature=self.config.model_b.temperature,
                        max_tokens=self.config.model_b.max_tokens
                    )
                    output = outputs[0]
                
                _, answer = extract_cot_and_answer(output)
                answer_normalized = normalize_answer(answer) if answer else None
                
                rollout_outputs.append({
                    'output': output,
                    'answer': answer,
                    'answer_normalized': answer_normalized
                })
            
            results.append({
                'id': item.get('id'),
                'problem': problem,
                'ground_truth': item['ground_truth'],
                'ground_truth_normalized': item['ground_truth_normalized'],
                'fixed_cot': fixed_cot,
                'fixed_answer_a': fixed_answer,
                'rollouts': rollout_outputs
            })
        
        self._save_jsonl(results, output_path)
        self.model_b_follow_cot_results = results
        
        if engine is not None:
            del engine
        self._cleanup_gpu()
    
    def run_intervention(self, intervention_config: InterventionConfig) -> List[Dict]:
        """
        Run a single intervention experiment with n rollouts.
        """
        print("\n" + "-"*60)
        print(f"INTERVENTION: {intervention_config.run_id}")
        print("-"*60)
        
        output_path = self.config.interventions_dir / f"{intervention_config.run_id}.jsonl"
        
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
            wiki_sentences
        )
        
        engine = self._create_engine(self.config.model_b)
        
        results = []
        for idx, item in enumerate(tqdm(self.model_a_results, desc=f"Intervention {intervention_config.run_id}")):
            problem = item['problem']
            fixed_cot, fixed_answer = self.get_fixed_cot_and_answer(idx)
            
            # Apply intervention to get modified CoT
            modified_cot = intervention_fn(fixed_cot or "")
            
            prompt = PromptTemplates.llama_from_cot_prompt(problem, modified_cot)
            rollout_outputs = []
            
            for _ in range(self.config.n_rollouts):
                if self.dry_run:
                    output = "Final Answer: 99"
                else:
                    outputs = engine.generate(
                        [prompt],
                        temperature=self.config.model_b.temperature,
                        max_tokens=self.config.model_b.max_tokens
                    )
                    output = outputs[0]
                
                _, answer = extract_cot_and_answer(output)
                answer_normalized = normalize_answer(answer) if answer else None
                
                rollout_outputs.append({
                    'output': output,
                    'answer': answer,
                    'answer_normalized': answer_normalized
                })
            
            results.append({
                'id': item.get('id'),
                'problem': problem,
                'ground_truth': item['ground_truth'],
                'ground_truth_normalized': item['ground_truth_normalized'],
                'original_cot': fixed_cot,
                'modified_cot': modified_cot,
                'fixed_answer_a': fixed_answer,
                'rollouts': rollout_outputs
            })
        
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
        fixed_cots = [self.get_fixed_cot_and_answer(i)[0] or "" for i in range(len(self.model_a_results))]
        problems = [item['problem'] for item in self.model_a_results]
        
        print("Injecting errors...")
        if self.dry_run:
            modified_cots = [f"[ERROR INJECTED] {cot}" for cot in fixed_cots]
        else:
            modified_cots = []
            batch_size = min(self.config.batch_size, 8)  # Smaller batches for error injection
            for i in tqdm(range(0, len(fixed_cots), batch_size), desc="Error injection"):
                batch_cots = fixed_cots[i:i+batch_size]
                batch_problems = problems[i:i+batch_size]
                modified_batch = inject_error_batch(batch_cots, batch_problems, qwen_engine)
                modified_cots.extend(modified_batch)
        
        if qwen_engine is not None:
            del qwen_engine
        self._cleanup_gpu()
        
        # Now run Model B on modified CoTs
        print("Running Model B on error-injected CoTs...")
        llama_engine = self._create_engine(self.config.model_b)
        
        results = []
        for idx, item in enumerate(tqdm(self.model_a_results, desc="Model B (error injection)")):
            problem = item['problem']
            fixed_cot, fixed_answer = self.get_fixed_cot_and_answer(idx)
            modified_cot = modified_cots[idx]
            
            prompt = PromptTemplates.llama_from_cot_prompt(problem, modified_cot)
            rollout_outputs = []
            
            for _ in range(self.config.n_rollouts):
                if self.dry_run:
                    output = "Final Answer: 99"
                else:
                    outputs = llama_engine.generate(
                        [prompt],
                        temperature=self.config.model_b.temperature,
                        max_tokens=self.config.model_b.max_tokens
                    )
                    output = outputs[0]
                
                _, answer = extract_cot_and_answer(output)
                answer_normalized = normalize_answer(answer) if answer else None
                
                rollout_outputs.append({
                    'output': output,
                    'answer': answer,
                    'answer_normalized': answer_normalized
                })
            
            results.append({
                'id': item.get('id'),
                'problem': problem,
                'ground_truth': item['ground_truth'],
                'ground_truth_normalized': item['ground_truth_normalized'],
                'original_cot': fixed_cot,
                'modified_cot': modified_cot,
                'fixed_answer_a': fixed_answer,
                'rollouts': rollout_outputs
            })
        
        self._save_jsonl(results, output_path)
        
        if llama_engine is not None:
            del llama_engine
        self._cleanup_gpu()
        
        return results
    
    def run_all_interventions(self) -> Dict[str, List[Dict]]:
        """Run all intervention experiments."""
        print("\n" + "="*60)
        print("RUNNING ALL INTERVENTIONS")
        print("="*60)
        
        intervention_results = {}
        for intervention_config in self.config.get_intervention_configs():
            results = self.run_intervention(intervention_config)
            intervention_results[intervention_config.run_id] = results
        
        return intervention_results
    
    def compute_all_metrics(self) -> ExperimentResults:
        """Compute all metrics from the collected results."""
        print("\n" + "="*60)
        print("COMPUTING METRICS")
        print("="*60)
        
        # Helper to extract rollout predictions
        def extract_rollout_preds(results: List[Dict]) -> List[List[str]]:
            return [
                [r.get('answer_normalized', '') for r in item.get('rollouts', [])]
                for item in results
            ]
        
        ground_truths = [item['ground_truth_normalized'] for item in self.model_a_results]
        
        # Model A accuracy (use all rollouts)
        model_a_rollouts = [
            [r.get('answer_normalized', '') for r in item.get('rollouts', [])]
            for item in self.model_a_results
        ]
        model_a_acc, _ = compute_accuracy_from_rollouts(model_a_rollouts, ground_truths)
        print(f"Model A Accuracy: {model_a_acc}")
        
        # Model B No-CoT accuracy
        model_b_no_cot_acc = None
        if self.model_b_no_cot_results:
            model_b_no_cot_rollouts = extract_rollout_preds(self.model_b_no_cot_results)
            model_b_no_cot_acc, _ = compute_accuracy_from_rollouts(model_b_no_cot_rollouts, ground_truths)
            print(f"Model B No-CoT Accuracy: {model_b_no_cot_acc}")
        
        # Model B Follow-CoT metrics
        model_b_follow_acc = None
        omr = mwc = mww = None
        model_b_follow_rollouts = None
        
        if self.model_b_follow_cot_results:
            model_b_follow_rollouts = extract_rollout_preds(self.model_b_follow_cot_results)
            model_b_follow_acc, _ = compute_accuracy_from_rollouts(model_b_follow_rollouts, ground_truths)
            print(f"Model B Follow-CoT Accuracy: {model_b_follow_acc}")
            
            # Fixed Model A answers for OMR/MWC/MWW
            fixed_a_answers = [
                self.get_fixed_cot_and_answer(i)[1] 
                for i in range(len(self.model_a_results))
            ]
            
            omr = compute_omr_from_rollouts(model_b_follow_rollouts, fixed_a_answers)
            mwc = compute_mwc_from_rollouts(model_b_follow_rollouts, fixed_a_answers, ground_truths)
            mww = compute_mww_from_rollouts(model_b_follow_rollouts, fixed_a_answers, ground_truths)
            
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
                
                flip_rate = compute_flip_rate_from_rollouts(model_b_follow_rollouts, int_rollouts)
                int_acc, _ = compute_accuracy_from_rollouts(int_rollouts, ground_truths)
                
                fixed_a_answers = [
                    self.get_fixed_cot_and_answer(i)[1] 
                    for i in range(len(self.model_a_results))
                ]
                int_omr = compute_omr_from_rollouts(int_rollouts, fixed_a_answers)
                
                intervention_metrics[run_id] = {
                    'flip_rate': flip_rate,
                    'accuracy': int_acc,
                    'omr': int_omr
                }
                print(f"Intervention {run_id}: Flip Rate = {flip_rate}")
        
        return ExperimentResults(
            experiment_name="cot_faithfulness",
            config={"n_rollouts": self.config.n_rollouts, "seed": self.config.random_seed},
            model_a_accuracy=model_a_acc,
            model_b_no_cot_accuracy=model_b_no_cot_acc,
            model_b_follow_cot_accuracy=model_b_follow_acc,
            omr=omr,
            mwc=mwc,
            mww=mww,
            intervention_results=intervention_metrics
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
            gt = item.get('ground_truth_normalized', '')
            for rollout in item.get('rollouts', []):
                if rollout.get('answer_normalized') == gt:
                    correct += 1
                total += 1
        
        if total > 0:
            print(f"Model A accuracy (all rollouts): {correct/total:.3f}")
    
    def _print_results_summary(self, results: ExperimentResults):
        """Print a summary of all results."""
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        
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
                if 'flip_rate' in metrics:
                    print(f"  {name}: {metrics['flip_rate']}")
