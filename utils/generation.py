from tqdm import tqdm
from .models import VLLMEngine
from .prompts import PromptTemplates
from .parsing import extract_cot_and_answer
from .interventions import apply_intervention

def run_qwen_stage(engine, examples, batch_size=8):
    results = []
    for i in tqdm(range(0, len(examples), batch_size), desc='Qwen Stage'):
        batch = examples[i:i + batch_size]
        prompts = [PromptTemplates.qwen_cot_prompt(ex['problem']) for ex in batch]
        outputs = engine.generate(prompts)
        for ex, out in zip(batch, outputs):
            cot, ans = extract_cot_and_answer(out)
            item = {'id': ex.get('id', None), 'problem': ex['problem'], 'ground_truth': ex.get('ground_truth', None), 'model_a_raw': out, 'cot_a': cot, 'answer_a': ans}
            results.append(item)
    return results

def _run_baseline(engine, batch):
    if all((ex.get('answer_b') is not None for ex in batch)):
        return [None] * len(batch)
    prompts = [PromptTemplates.llama_from_cot_prompt(ex['problem'], ex['cot_a'] or '') for ex in batch]
    return engine.generate(prompts)

def _run_truncation_intervention(engine, batch, truncation_fraction):
    cots = [ex.get('cot_a') or '' for ex in batch]
    truncated_cots = apply_intervention(cots, intervention_type='truncate_random', truncation_fraction=truncation_fraction)
    prompts = [PromptTemplates.llama_from_cot_prompt(ex['problem'], truncated_cot) for ex, truncated_cot in zip(batch, truncated_cots)]
    outputs = engine.generate(prompts)
    return (truncated_cots, prompts, outputs)

def _run_error_injection_intervention(engine, batch):
    cots = [ex.get('cot_a') or '' for ex in batch]
    problems = [ex.get('problem', '') for ex in batch]
    injected_cots = apply_intervention(cots, intervention_type='inject_error', problems=problems, engine=engine)
    prompts = [PromptTemplates.llama_from_cot_prompt(ex['problem'], injected_cot) for ex, injected_cot in zip(batch, injected_cots)]
    outputs = engine.generate(prompts)
    return (injected_cots, prompts, outputs)

def run_llama_stage(engine, input_results, batch_size=8):
    final_results = []
    for i in tqdm(range(0, len(input_results), batch_size), desc='Llama Stage (baseline)'):
        batch = input_results[i:i + batch_size]
        baseline_outputs = _run_baseline(engine, batch)
        for ex, baseline_out in zip(batch, baseline_outputs):
            if baseline_out:
                _, ans_b = extract_cot_and_answer(baseline_out)
                result = dict(ex)
                result.update({'model_b_raw': baseline_out, 'answer_b': ans_b})
            else:
                result = dict(ex)
            final_results.append(result)
    return final_results

def run_intervention_stage(engine, input_results, batch_size=8, run_truncation=True, run_error_injection=True, truncation_fraction=0.3):
    interventions = []
    if run_truncation:
        interventions.append('truncation')
    if run_error_injection:
        interventions.append('error_injection')
    desc = f"Intervention Stage ({', '.join(interventions)})"
    final_results = []
    for i in tqdm(range(0, len(input_results), batch_size), desc=desc):
        batch = input_results[i:i + batch_size]
        truncated_data = _run_truncation_intervention(engine, batch, truncation_fraction) if run_truncation else None
        injected_data = _run_error_injection_intervention(engine, batch) if run_error_injection else None
        for j, ex in enumerate(batch):
            result = dict(ex)
            if truncated_data:
                truncated_cot = truncated_data[0][j]
                truncated_prompt = truncated_data[1][j]
                truncated_out = truncated_data[2][j]
                _, ans_b_truncated = extract_cot_and_answer(truncated_out)
                result.update({'cot_a_truncated': truncated_cot, 'prompt_truncated': truncated_prompt, 'model_b_truncated_raw': truncated_out, 'answer_b_truncated': ans_b_truncated})
            if injected_data:
                injected_cot = injected_data[0][j]
                injected_prompt = injected_data[1][j]
                injected_out = injected_data[2][j]
                _, ans_b_injected = extract_cot_and_answer(injected_out)
                result.update({'cot_a_injected': injected_cot, 'prompt_injected': injected_prompt, 'model_b_injected_raw': injected_out, 'answer_b_injected': ans_b_injected})
            final_results.append(result)
    return final_results