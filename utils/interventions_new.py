import re
import random
from .prompts import PromptTemplates

def split_sentences(text):
    if not text or not text.strip():
        return []
    parts = re.split('([.!?]\\s+)', text.strip())
    sentences = []
    current = ''
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:
            if current.strip():
                sentences.append(current.strip())
            current = ''
    if current.strip():
        sentences.append(current.strip())
    return sentences if sentences else [text.strip()]

def truncate_first_k(text, k):
    sentences = split_sentences(text)
    if not sentences:
        return text
    if k >= len(sentences):
        return sentences[-1]
    return ' '.join(sentences[k:])

def truncate_last_k(text, k):
    sentences = split_sentences(text)
    if not sentences:
        return text
    if k >= len(sentences):
        return sentences[0]
    return ' '.join(sentences[:-k])

def truncate_contiguous_k(text, k, rng=None):
    if rng is None:
        rng = random.Random()
    sentences = split_sentences(text)
    if not sentences:
        return text
    if k >= len(sentences):
        return sentences[0]
    max_start = len(sentences) - k
    start = rng.randint(0, max_start)
    remaining = sentences[:start] + sentences[start + k:]
    return ' '.join(remaining) if remaining else sentences[0]

def truncate_percent(text, p, rng=None):
    if rng is None:
        rng = random.Random()
    sentences = split_sentences(text)
    if not sentences or len(sentences) <= 1:
        return text
    k = max(1, int(len(sentences) * p))
    k = min(k, len(sentences) - 1)
    return truncate_contiguous_k(text, k, rng)

def truncate_sentences(text, fraction=0.3, rng=None):
    if rng is None:
        rng = random.Random()
    if not text or not text.strip():
        return text
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text
    keep_count = max(1, int(len(sentences) * (1 - fraction)))
    indices = sorted(rng.sample(range(len(sentences)), keep_count))
    return ' '.join((sentences[i] for i in indices))

def inject_error_batch(cots, problems, engine, temperature=0.7, max_tokens=2048):
    if len(cots) != len(problems):
        raise ValueError('cots and problems must have the same length')
    if not cots:
        return []
    prompts = [PromptTemplates.inject_error_prompt(problem, cot) for cot, problem in zip(cots, problems)]
    outputs = engine.generate(prompts, temperature=temperature, max_tokens=max_tokens)
    results = []
    for cot, output in zip(cots, outputs):
        if output and output.strip():
            results.append(output.strip())
        else:
            results.append(cot)
    return results
_WIKIPEDIA_SENTENCES = None

def load_wikipedia_sentences(num_articles=1000, subset='20231101.en', cache_path=None, seed=42):
    global _WIKIPEDIA_SENTENCES
    if _WIKIPEDIA_SENTENCES is not None:
        return _WIKIPEDIA_SENTENCES
    if cache_path:
        try:
            import json
            with open(cache_path, 'r') as f:
                _WIKIPEDIA_SENTENCES = json.load(f)
            print(f'Loaded {len(_WIKIPEDIA_SENTENCES)} Wikipedia sentences from cache')
            return _WIKIPEDIA_SENTENCES
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    try:
        from datasets import load_dataset
        print(f'Loading Wikipedia dataset (subset={subset})...')
        wiki = load_dataset('wikimedia/wikipedia', subset, split='train', streaming=True, trust_remote_code=True)
        rng = random.Random(seed)
        sentences = []
        print(f'Extracting sentences from {num_articles} articles...')
        for i, article in enumerate(wiki):
            if i >= num_articles:
                break
            text = article.get('text', '')
            if text:
                article_sentences = split_sentences(text)
                good_sentences = [s for s in article_sentences if 20 < len(s) < 200 and (not s.startswith('='))]
                sentences.extend(good_sentences)
            if (i + 1) % 100 == 0:
                print(f'  Processed {i + 1} articles, {len(sentences)} sentences collected')
        rng.shuffle(sentences)
        _WIKIPEDIA_SENTENCES = sentences
        if cache_path and sentences:
            import json
            import os
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(sentences, f)
            print(f'Cached {len(sentences)} sentences to {cache_path}')
        print(f'Loaded {len(sentences)} Wikipedia sentences')
        return _WIKIPEDIA_SENTENCES
    except Exception as e:
        print(f'Warning: Could not load Wikipedia dataset: {e}')
        print('Falling back to default filler sentences')
        _WIKIPEDIA_SENTENCES = ['The quick brown fox jumps over the lazy dog.', 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.', 'This is placeholder text that serves no mathematical purpose.', 'The weather today is particularly pleasant for outdoor activities.', 'Historical records indicate significant changes over time.', 'Consider the implications of this statement carefully.', 'Many factors contribute to the final outcome of any process.', 'The relationship between these elements is quite complex.', 'Scientists have discovered new evidence supporting this theory.', 'The development of technology has transformed modern society.', 'Education plays a crucial role in personal development.', 'Environmental concerns have become increasingly important.', 'The economic impact of these changes is still being studied.', 'Cultural traditions vary significantly across different regions.', 'Research in this field continues to yield interesting results.']
        return _WIKIPEDIA_SENTENCES

def filler_replacement(text, p, rng=None, wikipedia_sentences=None):
    if rng is None:
        rng = random.Random()
    sentences = split_sentences(text)
    if not sentences or len(sentences) <= 1:
        return text
    if wikipedia_sentences is None:
        wikipedia_sentences = load_wikipedia_sentences()
    if not wikipedia_sentences:
        return text
    num_replace = max(1, int(len(sentences) * p))
    num_replace = min(num_replace, len(sentences) - 1)
    indices = rng.sample(range(len(sentences)), num_replace)
    for idx in indices:
        sentences[idx] = rng.choice(wikipedia_sentences)
    return ' '.join(sentences)

def get_intervention_fn(name, params, rng=None, wikipedia_sentences=None):
    if rng is None:
        rng = random.Random()
    if name == 'truncate_first':
        k = params.get('k', 3)
        return lambda text: truncate_first_k(text, k)
    elif name == 'truncate_last':
        k = params.get('k', 3)
        return lambda text: truncate_last_k(text, k)
    elif name == 'truncate_contiguous':
        k = params.get('k', 3)
        return lambda text: truncate_contiguous_k(text, k, rng)
    elif name == 'truncate_percent':
        p = params.get('p', 0.3)
        return lambda text: truncate_percent(text, p, rng)
    elif name == 'filler_replacement':
        p = params.get('p', 0.3)
        wiki_sents = wikipedia_sentences or load_wikipedia_sentences()
        return lambda text: filler_replacement(text, p, rng, wiki_sents)
    elif name == 'error_injection':
        raise ValueError('Error injection must be handled separately with an LLM engine')
    else:
        raise ValueError(f'Unknown intervention: {name}')

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