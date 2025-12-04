# 10701-project

Chain of Thought (CoT) Faithfulness Evaluation Pipeline

This project evaluates whether the reasoning trace from Model A genuinely supports its final answer by using Model B to reconstruct answers from the reasoning, and measuring how interventions on the reasoning affect Model B's responses.

## Overview

The pipeline implements a causal evaluation method for Chain of Thought reasoning:

1. **Model A (Qwen)** generates a CoT and final answer for a math problem
2. **Model B (Llama)** is given the problem and Model A's CoT (without the answer) and asked to follow the reasoning
3. **Interventions** are applied to the CoT to test causal dependence:
   - **Truncation**: Randomly remove sentences from the CoT
   - **Error Injection**: Use an LLM to inject logical/arithmetic errors into the CoT
4. **Metrics** measure how often Model B's answer changes (flips) after interventions

## Environment Setup

```bash
cd 10701-project
pip install -r requirements.txt
huggingface-cli login
```

## Pipeline Stages

### Stage 1: Model A (Qwen) Generation
Generates Chain of Thought reasoning and answers for math problems.

```bash
python run_stage1_qwen.py
```

**Output**: `outputs/stage1_qwen.jsonl`
- Contains: `problem`, `cot_a`, `answer_a`, `model_a_raw`, `ground_truth`

### Stage 2: Model B (Llama) Baseline
Model B attempts to reconstruct Model A's answer from the CoT.

```bash
python run_stage2_llama.py
```

**Output**: `outputs/stage2_llama.jsonl`
- Contains: All Stage 1 fields + `answer_b`, `model_b_raw`

**Metrics Computed**:
- **OMR** (Overall Match Rate): Pr[A_L = A_Q]
- **MWC** (Match-When-Correct): Pr[A_L = A_Q | A_Q = A_GT]
- **MWW** (Match-When-Wrong): Pr[A_L = A_Q | A_Q ≠ A_GT]

**Results from `outputs/stage2_llama.jsonl`**:
```
=== Agreement-based metrics ===
Overall Match Rate (OMR):         0.4772
Match-When-Correct (MWC):         0.8121
Match-When-Wrong  (MWW):          0.0902
[Counts] total=526, A_correct=282, A_wrong=244
=== Accuracy vs Ground Truth ===
Model A accuracy vs GT:           0.5202
Model B accuracy vs GT:           0.5696
(Legacy) Model B vs A agreement:  0.4772
```

### Stage 3: Interventions
Applies truncation and error injection interventions to test causal dependence.

```bash
python run_stage3_interventions.py
```

**Output**: `outputs/stage3_interventions.jsonl`
- Contains: All Stage 2 fields + intervention results:
  - Truncation: `cot_a_truncated`, `prompt_truncated`, `answer_b_truncated`, `model_b_truncated_raw`
  - Error Injection: `cot_a_injected`, `prompt_injected`, `answer_b_injected`, `model_b_injected_raw`

**Metrics Computed**:
- **Truncation Flip Rate**: Fraction of items where B's answer changes after truncation
- **Mistake Flip Rate**: Fraction of items where B's answer changes after error injection

**Results from `outputs/stage3_interventions.jsonl`**:
```
=== Causal/Intervention Metrics ===
Truncation Flip Rate:              0.5182
[Counts] total=631, flipped=327
Mistake Flip Rate:                 0.7087
[Counts] total=666, flipped=472
```

## Running the Full Pipeline

Run all stages sequentially:

```bash
python run_full_pipeline.py
```

Or use the pipeline function directly:

```python
from utils.pipeline import full_pipeline

full_pipeline(
    qwen_name="Qwen/Qwen3-4B",
    llama_name="meta-llama/Llama-3.2-3B-Instruct",
    out_dir="outputs",
    max_examples=800,  # Optional: limit number of examples
)
```

## Utility Scripts

### View JSONL Entries

Print a specific entry from a JSONL file:

```bash
python print_jsonl_entry.py --index 0 --file outputs/stage3_interventions.jsonl
```

Options:
- `--index, -i`: Entry index (0-based, default: 0)
- `--file, -f`: Path to JSONL file (default: `outputs/stage3_interventions.jsonl`)

## Project Structure

```
10701-project/
├── run_stage1_qwen.py          # Stage 1: Qwen generation
├── run_stage2_llama.py          # Stage 2: Llama baseline
├── run_stage3_interventions.py  # Stage 3: Interventions
├── run_full_pipeline.py         # Full pipeline runner
├── print_jsonl_entry.py         # Utility to view JSONL entries
├── utils/
│   ├── data.py                  # Data loading (MATH-lighteval)
│   ├── models.py                 # VLLMEngine wrapper
│   ├── prompts.py                # Prompt templates
│   ├── generation.py             # Stage execution functions
│   ├── interventions.py          # Intervention implementations
│   ├── parsing.py                # CoT and answer extraction
│   ├── eval.py                  # Metric computation
│   └── pipeline.py              # Full pipeline orchestration
└── outputs/                     # Output JSONL files
```

## Configuration

Default batch size: 8 (can be adjusted in function calls)

Default truncation fraction: 0.3 (30% of sentences removed)

Models:
- Model A: `Qwen/Qwen3-4B` (configurable)
- Model B: `meta-llama/Llama-3.2-3B-Instruct` (configurable)

## Output Format

All JSONL files contain dictionaries with the following structure:

```python
{
    "id": int,
    "problem": str,
    "ground_truth": str,
    "model_a_raw": str,
    "cot_a": str,
    "answer_a": str,
    "model_b_raw": str,           # Stage 2+
    "answer_b": str,              # Stage 2+
    "cot_a_truncated": str,       # Stage 3+
    "prompt_truncated": str,      # Stage 3+
    "answer_b_truncated": str,    # Stage 3+
    "model_b_truncated_raw": str, # Stage 3+
    "cot_a_injected": str,        # Stage 3+
    "prompt_injected": str,       # Stage 3+
    "answer_b_injected": str,     # Stage 3+
    "model_b_injected_raw": str,  # Stage 3+
}
```