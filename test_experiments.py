#!/usr/bin/env python3
"""
Test script to verify the experiment infrastructure works correctly.
Runs with --dry-run to avoid needing GPU.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config():
    """Test configuration loading."""
    print("=" * 60)
    print("TEST: Configuration")
    print("=" * 60)
    
    from experiments.config import ExperimentConfig, InterventionConfig
    
    # Test default config
    config = ExperimentConfig()
    assert config.n_rollouts == 4
    assert config.batch_size == 32
    print(f"✓ Default config: n_rollouts={config.n_rollouts}, batch_size={config.batch_size}")
    
    # Test intervention configs
    interventions = config.get_intervention_configs()
    print(f"✓ Generated {len(interventions)} intervention configs:")
    for ic in interventions[:5]:
        print(f"    - {ic.run_id}")
    print(f"    ... and {len(interventions) - 5} more")
    
    # Test config from YAML if exists
    if os.path.exists("configs/experiment_config.yaml"):
        config_yaml = ExperimentConfig.from_yaml("configs/experiment_config.yaml")
        print(f"✓ Loaded config from YAML: n_rollouts={config_yaml.n_rollouts}")
    
    print()


def test_interventions():
    """Test intervention functions."""
    print("=" * 60)
    print("TEST: Interventions")
    print("=" * 60)
    
    from utils.interventions_new import (
        split_sentences,
        truncate_first_k,
        truncate_last_k,
        truncate_contiguous_k,
        truncate_percent,
        filler_replacement,
        get_intervention_fn
    )
    import random
    
    # Test text
    test_cot = (
        "First, we identify the problem. "
        "Next, we apply the formula x = 2y + 3. "
        "Then we substitute y = 5 into the equation. "
        "This gives us x = 2(5) + 3 = 13. "
        "Finally, the answer is 13."
    )
    
    # Test sentence splitting
    sentences = split_sentences(test_cot)
    assert len(sentences) == 5, f"Expected 5 sentences, got {len(sentences)}"
    print(f"✓ split_sentences: {len(sentences)} sentences")
    
    # Test truncate_first_k
    result = truncate_first_k(test_cot, 2)
    assert "First" not in result
    assert "Finally" in result
    print(f"✓ truncate_first_k(k=2): removed first 2 sentences")
    
    # Test truncate_last_k
    result = truncate_last_k(test_cot, 2)
    assert "First" in result
    assert "Finally" not in result
    print(f"✓ truncate_last_k(k=2): removed last 2 sentences")
    
    # Test truncate_contiguous_k
    rng = random.Random(42)
    result = truncate_contiguous_k(test_cot, 2, rng)
    result_sentences = split_sentences(result)
    assert len(result_sentences) == 3
    print(f"✓ truncate_contiguous_k(k=2): {len(result_sentences)} sentences remaining")
    
    # Test truncate_percent
    result = truncate_percent(test_cot, 0.4, rng)
    result_sentences = split_sentences(result)
    assert len(result_sentences) < 5
    print(f"✓ truncate_percent(p=0.4): {len(result_sentences)} sentences remaining")
    
    # Test filler_replacement (with fallback sentences since we're not loading Wikipedia)
    rng = random.Random(42)
    result = filler_replacement(test_cot, 0.2, rng)
    # Just check it returns something
    assert len(result) > 0
    print(f"✓ filler_replacement(p=0.2): {len(split_sentences(result))} sentences")
    
    # Test intervention factory
    fn = get_intervention_fn("truncate_first", {"k": 1}, rng)
    result = fn(test_cot)
    assert "First" not in result
    print(f"✓ get_intervention_fn('truncate_first', k=1): works")
    
    print()


def test_statistics():
    """Test statistics utilities."""
    print("=" * 60)
    print("TEST: Statistics")
    print("=" * 60)
    
    from utils.statistics import (
        compute_metric_stats,
        compute_accuracy_from_rollouts,
        compute_omr_from_rollouts,
        compute_flip_rate_from_rollouts,
        MetricStats
    )
    
    # Test compute_metric_stats
    values = [0.8, 0.85, 0.82, 0.78]
    stats = compute_metric_stats(values)
    assert 0.7 < stats.mean < 0.9
    assert stats.std > 0
    print(f"✓ compute_metric_stats: mean={stats.mean:.3f}, std={stats.std:.3f}")
    
    # Test accuracy computation
    rollout_preds = [
        ["42", "42", "43", "42"],  # Example 1: 3/4 correct
        ["10", "10", "10", "10"],  # Example 2: 4/4 correct if gt=10
        ["5", "6", "5", "7"],      # Example 3: 2/4 correct if gt=5
    ]
    ground_truths = ["42", "10", "5"]
    
    acc_stats, per_example = compute_accuracy_from_rollouts(rollout_preds, ground_truths)
    assert 0.6 < acc_stats.mean < 0.9
    print(f"✓ compute_accuracy_from_rollouts: mean={acc_stats.mean:.3f}")
    
    # Test OMR computation
    model_a_answers = ["42", "10", "5"]
    omr_stats = compute_omr_from_rollouts(rollout_preds, model_a_answers)
    print(f"✓ compute_omr_from_rollouts: mean={omr_stats.mean:.3f}")
    
    # Test flip rate
    baseline_rollouts = [
        ["42", "42", "42", "42"],
        ["10", "10", "10", "10"],
    ]
    intervention_rollouts = [
        ["42", "43", "44", "42"],  # 2 flips
        ["10", "10", "11", "11"],  # 2 flips
    ]
    flip_stats = compute_flip_rate_from_rollouts(baseline_rollouts, intervention_rollouts)
    assert flip_stats.mean == 0.5  # 4 flips out of 8
    print(f"✓ compute_flip_rate_from_rollouts: mean={flip_stats.mean:.3f}")
    
    print()


def test_runner_dry_run():
    """Test experiment runner in dry-run mode."""
    print("=" * 60)
    print("TEST: Experiment Runner (dry-run)")
    print("=" * 60)
    
    from experiments.config import ExperimentConfig
    from experiments.runner import ExperimentRunner
    from pathlib import Path
    import tempfile
    import shutil
    
    # Create temp output directory
    temp_dir = tempfile.mkdtemp(prefix="cot_test_")
    
    try:
        # Create minimal config
        config = ExperimentConfig(
            n_rollouts=2,
            max_examples=3,
            output_base_dir=Path(temp_dir),
            truncate_k_values=[1, 2],
            truncate_p_values=[0.2],
            filler_p_values=[0.2],
        )
        
        print(f"✓ Created test config with {len(config.get_intervention_configs())} interventions")
        
        # Create runner in dry-run mode
        runner = ExperimentRunner(config, dry_run=True)
        print("✓ Created ExperimentRunner in dry-run mode")
        
        # Load data
        runner.load_data()
        print(f"✓ Loaded {len(runner.dataset)} examples")
        
        # Run Model A baseline
        runner.run_model_a_baseline()
        assert runner.model_a_results is not None
        assert len(runner.model_a_results) == 3
        print(f"✓ Model A baseline: {len(runner.model_a_results)} results")
        
        # Check fixed indices
        assert runner.fixed_cot_indices is not None
        assert len(runner.fixed_cot_indices) == 3
        print(f"✓ Fixed CoT indices: {runner.fixed_cot_indices}")
        
        # Run Model B No-CoT baseline
        runner.run_model_b_no_cot_baseline()
        assert runner.model_b_no_cot_results is not None
        print(f"✓ Model B No-CoT baseline: {len(runner.model_b_no_cot_results)} results")
        
        # Run Model B Follow-CoT baseline
        runner.run_model_b_follow_cot_baseline()
        assert runner.model_b_follow_cot_results is not None
        print(f"✓ Model B Follow-CoT baseline: {len(runner.model_b_follow_cot_results)} results")
        
        # Run one intervention
        intervention_config = config.get_intervention_configs()[0]
        int_results = runner.run_intervention(intervention_config)
        assert len(int_results) == 3
        print(f"✓ Intervention {intervention_config.run_id}: {len(int_results)} results")
        
        # Compute metrics
        results = runner.compute_all_metrics()
        print(f"✓ Computed metrics")
        print(f"    Model A accuracy: {results.model_a_accuracy}")
        print(f"    OMR: {results.omr}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print()


def test_prompts():
    """Test prompt templates."""
    print("=" * 60)
    print("TEST: Prompt Templates")
    print("=" * 60)
    
    from utils.prompts import PromptTemplates
    
    problem = "What is 2 + 2?"
    cot = "First, we add 2 and 2. This gives us 4."
    
    # Test Qwen CoT prompt
    prompt = PromptTemplates.qwen_cot_prompt(problem)
    assert "2 + 2" in prompt
    assert "<think>" in prompt
    print("✓ qwen_cot_prompt")
    
    # Test Llama from CoT prompt
    prompt = PromptTemplates.llama_from_cot_prompt(problem, cot)
    assert "2 + 2" in prompt
    assert "add 2 and 2" in prompt
    print("✓ llama_from_cot_prompt")
    
    # Test Llama no-CoT prompt
    prompt = PromptTemplates.llama_no_cot_prompt(problem)
    assert "2 + 2" in prompt
    assert "Final Answer" in prompt
    print("✓ llama_no_cot_prompt")
    
    # Test error injection prompt
    prompt = PromptTemplates.inject_error_prompt(problem, cot)
    assert "2 + 2" in prompt
    assert "error" in prompt.lower()
    print("✓ inject_error_prompt")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CoT Faithfulness Experiment Infrastructure Tests")
    print("=" * 60 + "\n")
    
    try:
        test_config()
        test_prompts()
        test_interventions()
        test_statistics()
        test_runner_dry_run()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
