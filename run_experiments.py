#!/usr/bin/env python3
"""
Main entry point for running CoT Faithfulness experiments.

Usage:
    python run_experiments.py                    # Run with default config
    python run_experiments.py --config path.yaml # Run with custom config
    python run_experiments.py --dry-run          # Test without GPU inference
    python run_experiments.py --max-examples 10  # Quick test with few examples
"""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run CoT Faithfulness experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to test pipeline without GPU
    python run_experiments.py --dry-run --max-examples 5

    # Run specific stage only
    python run_experiments.py --stage model_a

    # Run with custom config
    python run_experiments.py --config configs/experiment_config.yaml
    
    # Multi-GPU with Ray (8 GPUs)
    python run_experiments.py --num-gpus 8 --use-ray
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test pipeline without actual model inference",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Override max_examples in config (for quick testing)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "all",
            "model_a",
            "model_b_no_cot",
            "model_b_follow",
            "interventions",
            "metrics",
        ],
        default="all",
        help="Which stage to run",
    )
    parser.add_argument(
        "--n-rollouts", type=int, default=None, help="Override n_rollouts in config"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for data parallelism (default: from config)",
    )
    parser.add_argument(
        "--use-ray", action="store_true", help="Use Ray for multi-GPU data parallelism"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: from config)",
    )
    args = parser.parse_args()

    # Import here to avoid slow imports for --help
    from experiments.config import ExperimentConfig
    from experiments.runner import ExperimentRunner

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from {config_path}")
        config = ExperimentConfig.from_yaml(str(config_path))
    else:
        print("Config file not found, using default configuration")
        config = ExperimentConfig()

    # Override config with command line args
    if args.max_examples is not None:
        config.max_examples = args.max_examples
    if args.n_rollouts is not None:
        config.n_rollouts = args.n_rollouts
    if args.num_gpus is not None:
        config.inference.num_gpus = args.num_gpus
    if args.use_ray:
        config.inference.use_ray = True
    if args.batch_size is not None:
        config.inference.batch_size = args.batch_size

    # Print configuration
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"n_rollouts: {config.n_rollouts}")
    print(f"batch_size: {config.inference.batch_size}")
    print(f"num_gpus: {config.inference.num_gpus}")
    print(f"use_ray: {config.inference.use_ray}")
    print(f"random_seed: {config.random_seed}")
    print(f"max_examples: {config.max_examples or 'all'}")
    print(f"Model A: {config.model_a.name}")
    print(f"Model B: {config.model_b.name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Output dir: {config.output_base_dir}")
    print(f"Dry run: {args.dry_run}")

    print(f"\nInterventions to run ({len(config.get_intervention_configs())} total):")
    for ic in config.get_intervention_configs():
        print(f"  - {ic.run_id}")

    # Initialize runner
    runner = ExperimentRunner(config, dry_run=args.dry_run)

    # Run requested stage(s)
    if args.stage == "all":
        results = runner.run_full_pipeline()
    else:
        runner.load_data()

        if args.stage == "model_a":
            runner.run_model_a_baseline()
        elif args.stage == "model_b_no_cot":
            runner.run_model_a_baseline()  # Need Model A results first
            runner.run_model_b_no_cot_baseline()
        elif args.stage == "model_b_follow":
            runner.run_model_a_baseline()
            runner.run_model_b_follow_cot_baseline()
        elif args.stage == "interventions":
            runner.run_model_a_baseline()
            runner.run_model_b_follow_cot_baseline()
            runner.run_all_interventions()
        elif args.stage == "metrics":
            # Load all existing results and compute metrics
            runner.run_model_a_baseline()
            runner.run_model_b_no_cot_baseline()
            runner.run_model_b_follow_cot_baseline()
            results = runner.compute_all_metrics()
            results.save(str(config.output_base_dir / "experiment_results.json"))
            runner._print_results_summary(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
