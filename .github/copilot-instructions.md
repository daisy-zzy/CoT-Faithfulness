# Copilot Instructions for CoT-Faithfulness

## Project Overview

This is a research pipeline evaluating **Chain of Thought (CoT) faithfulness** in LLMs. It tests whether reasoning traces genuinely support final answers using a causal evaluation method with two models and interventions.

## Architecture

### Three-Stage Pipeline
1. **Stage 1** (`run_stage1_qwen.py`): Model A (Qwen) generates CoT + answer for math problems
2. **Stage 2** (`run_stage2_llama.py`): Model B (Llama) reconstructs answers from Model A's CoT only
3. **Stage 3** (`run_stage3_interventions.py`): Apply interventions (truncation, error injection) to test causal dependence

### Key Data Flow
```
MATH-lighteval dataset → Qwen (CoT+answer) → Llama (follow CoT) → Interventions → Metrics
```

Output files are JSONL in `outputs/`: `stage1_qwen.jsonl` → `stage2_llama.jsonl` → `stage3_interventions.jsonl`

## Core Modules (`utils/`)

| Module | Purpose |
|--------|---------|
| `models.py` | `VLLMEngine` wrapper for vLLM inference |
| `prompts.py` | All prompt templates (Qwen CoT, Llama follow, error injection) |
| `parsing.py` | Extract CoT from `<think>` tags, parse `\boxed{}`, normalize answers |
| `generation.py` | Batch generation logic for each stage |
| `interventions.py` | `truncate_sentences()` and `inject_error_batch()` |
| `eval.py` | Metrics: OMR, MWC, MWW, flip rates, accuracy |
| `data.py` | Loads `DigitalLearningGmbH/MATH-lighteval` dataset |

## Critical Patterns

### Answer Parsing
- Model outputs use `<think>...</think>` tags for CoT reasoning
- Final answers follow `Final Answer:` pattern
- Ground truth extracted from `\boxed{...}` in dataset solutions
- **Always use `normalize_answer()` from `parsing.py`** for consistent comparison

### Prompt Format (defined in `prompts.py`)
```python
# Qwen: generates reasoning in <think> tags then Final Answer
PromptTemplates.qwen_cot_prompt(problem)

# Llama: follows provided reasoning to produce answer
PromptTemplates.llama_from_cot_prompt(problem, cot)
```

### Intervention Types
- **Truncation**: Removes ~30% of sentences randomly (`truncate_sentences()`)
- **Error Injection**: LLM introduces subtle arithmetic/logical errors

### Key Metrics
- **OMR**: Pr[Model B = Model A] (overall agreement)
- **MWC**: Pr[B = A | A correct] (faithfulness when reasoning is valid)
- **MWW**: Pr[B = A | A wrong] (blind following of flawed reasoning)
- **Flip Rate**: Answer changes after intervention (causal dependence)

## Running the Pipeline

```bash
# Full pipeline (requires GPU with vLLM)
python run_full_pipeline.py

# Or run stages individually (each depends on prior stage output)
python run_stage1_qwen.py
python run_stage2_llama.py
python run_stage3_interventions.py
```

Default models: `Qwen/Qwen3-4B` and `meta-llama/Llama-3.2-3B-Instruct`

## Development Notes

- **GPU Required**: vLLM needs CUDA. Adjust `gpu_memory_utilization` in `VLLMEngine` if needed
- **HuggingFace Auth**: Run `huggingface-cli login` before first use (Llama requires access)
- **Batch Processing**: All generation uses batched inference (default `batch_size=8`)
- **Memory Management**: Pipeline explicitly calls `gc.collect()` and `torch.cuda.empty_cache()` between stages
- Inspect JSONL outputs with: `python print_jsonl_entry.py outputs/<file>.jsonl <index>`
