# CoT-Faithfulness

Chain of Thought (CoT) Faithfulness Evaluation Pipeline

This project evaluates whether reasoning traces from LLMs genuinely support their final answers using a causal evaluation method with two models and systematic interventions.

## Overview

The pipeline implements a causal evaluation framework for Chain of Thought reasoning:

1. **Model A (Qwen)** generates a CoT reasoning trace and final answer for math problems
2. **Model B (Llama)** is given the problem and Model A's CoT (without the answer) and asked to follow the reasoning to produce an answer
3. **Interventions** are applied to test causal dependence between reasoning and answers:
   - **Truncation**: Remove sentences from the CoT (first k, last k, contiguous k, or random p%)
   - **Filler Replacement**: Replace sentences with random Wikipedia text
   - **Error Injection**: Use an LLM to inject logical/arithmetic errors
4. **Metrics** measure faithfulness via agreement rates and flip rates after interventions

## Key Metrics

| Metric | Definition |
|--------|------------|
| **OMR** (Overall Match Rate) | Pr[B = A] - How often Model B matches Model A |
| **MWC** (Match When Correct) | Pr[B = A \| A = GT] - Agreement when A is correct |
| **MWW** (Match When Wrong) | Pr[B = A \| A ≠ GT] - Agreement when A is wrong |
| **Flip Rate** | Pr[intervention ≠ baseline] - Answer changes after intervention |

## Results Summary

Based on experiments with 5,000 MATH problems × 4 rollouts each:

| Condition | OMR (%) | MWC (%) | MWW (%) | Flip Rate (%) |
|-----------|---------|---------|---------|---------------|
| Baseline (B + CoT) | 84.8 ± 0.5 | 90.3 ± 0.5 | 60.8 ± 1.6 | 15.2 ± 0.5 |
| Truncate Last k=5 | 81.8 ± 0.6 | 88.4 ± 0.5 | 52.9 ± 1.7 | 24.4 ± 0.5 |
| Truncate 50% | 82.5 ± 0.5 | 88.0 ± 0.5 | 58.4 ± 1.6 | 26.0 ± 0.6 |
| Filler Replace 50% | 80.0 ± 0.6 | 85.9 ± 0.5 | 54.3 ± 1.6 | 29.1 ± 0.6 |
| Error Injection | 59.1 ± 0.7 | 63.4 ± 0.8 | 40.5 ± 1.6 | 44.9 ± 0.7 |

**Key Finding**: Error injection causes the largest faithfulness degradation (44.9% flip rate), while truncation and filler replacement show more modest effects (~21-29% flip rates).

## Environment Setup

```bash
cd 10701-project
pip install -r requirements.txt
huggingface-cli login  # Required for Llama model access
```

### Requirements
- Python 3.10+
- CUDA-capable GPU(s)
- vLLM for efficient inference
- Ray for multi-GPU parallelism (optional)

## Running Experiments

### Quick Start

```bash
# Run full pipeline with default config
python run_experiments.py

# Dry run to test without GPU
python run_experiments.py --dry-run --max-examples 5

# Multi-GPU with Ray (8 GPUs)
python run_experiments.py --num-gpus 8 --use-ray
```

### Command Line Options

```bash
python run_experiments.py [OPTIONS]

Options:
  --config PATH           Path to YAML config file (default: configs/experiment_config.yaml)
  --dry-run               Test pipeline without model inference
  --max-examples N        Limit number of problems to process
  --stage STAGE           Run specific stage: all, model_a, model_b_no_cot, 
                          model_b_follow, interventions, metrics
  --n-rollouts N          Number of rollouts per problem (default: 4)
  --num-gpus N            Number of GPUs for data parallelism
  --use-ray               Enable Ray for multi-GPU inference
  --batch-size N          Batch size for generation
```

### Running Specific Stages

```bash
# Stage 1: Model A generates CoT + answer
python run_experiments.py --stage model_a

# Stage 2a: Model B without CoT (baseline)
python run_experiments.py --stage model_b_no_cot

# Stage 2b: Model B follows Model A's CoT (baseline)
python run_experiments.py --stage model_b_follow

# Stage 3: Run all interventions
python run_experiments.py --stage interventions

# Compute metrics only (requires completed stages)
python run_experiments.py --stage metrics
```

## Configuration

Edit `configs/experiment_config.yaml` to customize:

```yaml
# General settings
n_rollouts: 4           # Rollouts per problem for statistical power
batch_size: 64          # Batch size for generation
random_seed: 42

# Inference settings
inference:
  num_gpus: 8           # GPUs for data parallelism
  use_ray: true         # Enable Ray for multi-GPU
  batch_size: 128
  max_model_len: 8192

# Models
models:
  model_a:
    name: "Qwen/Qwen3-4B"
    temperature: 0.7
    max_tokens: 4096
  model_b:
    name: "meta-llama/Llama-3.2-3B-Instruct"
    temperature: 0.7
    max_tokens: 1024

# Dataset
dataset:
  name: "DigitalLearningGmbH/MATH-lighteval"
  split: "test"
  max_examples: null    # null = all 5000 problems

# Interventions
interventions:
  truncate_first:
    k_values: [1, 2, 3, 5]
  truncate_last:
    k_values: [1, 2, 3, 5]
  truncate_contiguous:
    k_values: [1, 2, 3, 5]
  truncate_percent:
    p_values: [0.1, 0.2, 0.3, 0.5]
  filler_replacement:
    p_values: [0.1, 0.2, 0.3, 0.5]
  error_injection:
    enabled: true
```

## Generating Paper Figures

After running experiments, generate publication-ready figures:

```bash
python generate_paper_figures.py
```

**Outputs** (in `outputs/analysis/`):
- `fig1_baseline_accuracy.png/pdf` - Model accuracy comparison
- `fig2_truncation_effects.png/pdf` - Truncation intervention effects (OMR, MWC, MWW, Flip Rate)
- `fig3_filler_effects.png/pdf` - Filler replacement effects
- `fig4_error_injection.png/pdf` - Error injection analysis
- `complete_results_table.tex` - LaTeX table for paper
- `complete_results.csv` - Full results data

## Project Structure

```
10701-project/
├── run_experiments.py           # Main experiment runner
├── generate_paper_figures.py    # Generate publication figures
├── configs/
│   └── experiment_config.yaml   # Experiment configuration
├── experiments/
│   ├── __init__.py
│   ├── config.py                # Configuration dataclasses
│   └── runner.py                # ExperimentRunner class
├── utils/
│   ├── data.py                  # Dataset loading (MATH-lighteval)
│   ├── models.py                # VLLMEngine wrapper
│   ├── ray_inference.py         # Ray-based multi-GPU inference
│   ├── prompts.py               # Prompt templates
│   ├── parsing.py               # CoT and answer extraction
│   ├── interventions_new.py     # Intervention implementations
│   ├── statistics.py            # Statistical analysis utilities
│   └── eval.py                  # Metric computation
├── outputs/
│   ├── baselines/               # Model A and B baseline results
│   │   ├── model_a_rollouts.jsonl
│   │   ├── model_b_no_cot.jsonl
│   │   ├── model_b_follow_cot.jsonl
│   │   └── fixed_cot_indices.json
│   ├── interventions/           # Intervention experiment results
│   │   ├── truncation/          # truncate_first_k1.jsonl, etc.
│   │   └── filler/              # filler_replacement_p0.1.jsonl, etc.
│   └── analysis/                # Generated figures and tables
├── final_report/                # LaTeX paper source
└── requirements.txt
```

## Output Data Format

### Baseline Files (JSONL)

`model_a_rollouts.jsonl`:
```json
{
  "id": 0,
  "problem": "Find the value of x...",
  "ground_truth": "42",
  "ground_truth_normalized": "42",
  "rollouts": [
    {"output": "...", "cot": "...", "answer": "42", "answer_normalized": "42"},
    ...
  ]
}
```

`model_b_follow_cot.jsonl`:
```json
{
  "id": 0,
  "problem": "...",
  "ground_truth": "42",
  "ground_truth_normalized": "42",
  "fixed_cot": "Let me solve this step by step...",
  "fixed_answer_a": "42",
  "rollouts": [
    {"output": "...", "answer": "42", "answer_normalized": "42"},
    ...
  ]
}
```

### Intervention Files (JSONL)

`truncate_first_k3.jsonl`:
```json
{
  "id": 0,
  "problem": "...",
  "ground_truth_normalized": "42",
  "original_cot": "Step 1... Step 2... Step 3...",
  "modified_cot": "Step 3...",
  "fixed_answer_a": "42",
  "rollouts": [
    {"output": "...", "answer": "40", "answer_normalized": "40"},
    ...
  ]
}
```

## Intervention Types

| Intervention | Description | Parameters |
|--------------|-------------|------------|
| `truncate_first` | Remove first k sentences | k ∈ {1, 2, 3, 5} |
| `truncate_last` | Remove last k sentences | k ∈ {1, 2, 3, 5} |
| `truncate_contiguous` | Remove k contiguous sentences from random position | k ∈ {1, 2, 3, 5} |
| `truncate_percent` | Remove p% of sentences (contiguous chunk) | p ∈ {0.1, 0.2, 0.3, 0.5} |
| `filler_replacement` | Replace p% of sentences with Wikipedia text | p ∈ {0.1, 0.2, 0.3, 0.5} |
| `error_injection` | LLM injects subtle arithmetic/logical error | - |

## Models

| Model | Role | Size |
|-------|------|------|
| Qwen/Qwen3-4B | Model A (generates CoT + answer) | 4B params |
| meta-llama/Llama-3.2-3B-Instruct | Model B (follows CoT) | 3B params |
