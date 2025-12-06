import json
from pathlib import Path
import gc
import torch
from .data import load_math_lighteval_test
from .models import VLLMEngine
from .generation import run_qwen_stage, run_llama_stage, run_intervention_stage
from .eval import compute_agreement, compute_accuracy_vs_gt, compute_agreement_metrics, compute_truncation_flip_rate, compute_mistake_flip_rate

def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def load_jsonl(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows

def full_pipeline(qwen_name, llama_name, out_dir='outputs', max_examples=None):
    data = load_math_lighteval_test()
    if max_examples is not None:
        data = data[:max_examples]
    qwen_engine = VLLMEngine(qwen_name)
    qwen_results = run_qwen_stage(qwen_engine, data)
    save_jsonl(Path(out_dir) / 'stage1_qwen.jsonl', qwen_results)
    del qwen_engine
    gc.collect()
    torch.cuda.empty_cache()
    llama_engine = VLLMEngine(llama_name)
    baseline_results = run_llama_stage(llama_engine, qwen_results)
    save_jsonl(Path(out_dir) / 'stage2_llama.jsonl', baseline_results)
    metrics = compute_agreement_metrics(baseline_results)
    acc_a = compute_accuracy_vs_gt(baseline_results, use_model='a')
    acc_b = compute_accuracy_vs_gt(baseline_results, use_model='b')
    print('=== Agreement-based metrics ===')
    print(f"Overall Match Rate (OMR):         {metrics['omr']:.4f}")
    print(f"Match-When-Correct (MWC):         {metrics['mwc']:.4f}")
    print(f"Match-When-Wrong  (MWW):          {metrics['mww']:.4f}")
    print(f"[Counts] total={metrics['num_total']}, A_correct={metrics['num_a_correct']}, A_wrong={metrics['num_a_wrong']}")
    print('=== Accuracy vs Ground Truth ===')
    print(f'Model A accuracy vs GT:           {acc_a:.4f}')
    print(f'Model B accuracy vs GT:           {acc_b:.4f}')
    overall_agreement = compute_agreement(baseline_results)
    print(f'(Legacy) Model B vs A agreement:  {overall_agreement:.4f}')
    all_intervention_results = run_intervention_stage(llama_engine, baseline_results, run_truncation=True, run_error_injection=True, truncation_fraction=0.3)
    save_jsonl(Path(out_dir) / 'stage3_interventions.jsonl', all_intervention_results)
    print('\n=== Causal/Intervention Metrics ===')
    truncation_metrics = compute_truncation_flip_rate(all_intervention_results, truncation_type='any')
    print(f"Truncation Flip Rate:              {truncation_metrics['flip_rate']:.4f}")
    print(f"[Counts] total={truncation_metrics['num_total']}, flipped={truncation_metrics['num_flipped']}")
    mistake_metrics = compute_mistake_flip_rate(all_intervention_results)
    print(f"Mistake Flip Rate:                 {mistake_metrics['flip_rate']:.4f}")
    print(f"[Counts] total={mistake_metrics['num_total']}, flipped={mistake_metrics['num_flipped']}")