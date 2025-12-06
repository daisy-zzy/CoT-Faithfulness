import re
import random
from .prompts import PromptTemplates

def split_into_sentences(cot):
    if not cot or not cot.strip():
        return []
    parts = re.split('([.!?]\\s+)', cot)
    sentences = []
    current = ''
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:
            sentences.append(current.strip())
            current = ''
    if current.strip():
        sentences.append(current.strip())
    return sentences if sentences else [cot]

def truncate_sentences(cot, fraction=0.3):
    if not cot or not cot.strip():
        return cot
    sentences = split_into_sentences(cot)
    if len(sentences) < 2:
        return cot
    num_to_remove = max(1, int(len(sentences) * fraction))
    num_to_remove = min(num_to_remove, len(sentences) - 1)
    if num_to_remove <= 0:
        return cot
    indices_to_remove = set(random.sample(range(len(sentences)), num_to_remove))
    remaining_sentences = [sent for i, sent in enumerate(sentences) if i not in indices_to_remove]
    return '\n'.join(remaining_sentences)

def inject_error_batch(cots, problems, engine):
    if len(cots) != len(problems):
        raise ValueError('cots and problems must have the same length')
    if not cots:
        return []
    prompts = [PromptTemplates.inject_error_prompt(problem, cot) for cot, problem in zip(cots, problems)]
    outputs = engine.generate(prompts)
    results = []
    for i, (cot, output) in enumerate(zip(cots, outputs)):
        if output and output.strip():
            results.append(output.strip())
        else:
            results.append(cot)
    return results

def apply_intervention(cots, intervention_type, problems=None, engine=None, truncation_fraction=0.3):
    if not cots:
        return []
    if intervention_type == 'truncate_random':
        return [truncate_sentences(cot, fraction=truncation_fraction) for cot in cots]
    elif intervention_type == 'inject_error':
        if problems is None or engine is None:
            raise ValueError('problems and engine are required for inject_error intervention')
        if len(cots) != len(problems):
            raise ValueError('cots and problems must have the same length')
        return inject_error_batch(cots, problems, engine)
    else:
        raise ValueError(f'Unknown intervention type: {intervention_type}')