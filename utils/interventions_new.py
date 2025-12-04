"""Intervention utilities for CoT manipulation."""

import re
import random
from typing import Optional, List, Callable
from .prompts import PromptTemplates


# =============================================================================
# Sentence Splitting Utilities
# =============================================================================

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling common edge cases.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences (non-empty)
    """
    if not text or not text.strip():
        return []
    
    # Split by sentence boundaries (., !, ?) followed by space or newline
    parts = re.split(r'([.!?]\s+)', text.strip())
    sentences = []
    current = ""
    
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:  # At a sentence boundary
            if current.strip():
                sentences.append(current.strip())
            current = ""
    
    # Add the last part if it exists
    if current.strip():
        sentences.append(current.strip())
    
    return sentences if sentences else [text.strip()]


# =============================================================================
# Intervention 1a: Truncate First K Sentences
# =============================================================================

def truncate_first_k(text: str, k: int) -> str:
    """
    Remove first k sentences from reasoning.
    
    Args:
        text: Original CoT reasoning
        k: Number of sentences to remove from the beginning
        
    Returns:
        Modified CoT with first k sentences removed
    """
    sentences = split_sentences(text)
    if not sentences:
        return text
    if k >= len(sentences):
        return sentences[-1]  # Keep at least the last sentence
    return ' '.join(sentences[k:])


# =============================================================================
# Intervention 1b: Truncate Last K Sentences
# =============================================================================

def truncate_last_k(text: str, k: int) -> str:
    """
    Remove last k sentences from reasoning.
    
    Args:
        text: Original CoT reasoning
        k: Number of sentences to remove from the end
        
    Returns:
        Modified CoT with last k sentences removed
    """
    sentences = split_sentences(text)
    if not sentences:
        return text
    if k >= len(sentences):
        return sentences[0]  # Keep at least the first sentence
    return ' '.join(sentences[:-k])


# =============================================================================
# Intervention 1c: Truncate Random Contiguous K Sentences
# =============================================================================

def truncate_contiguous_k(text: str, k: int, rng: Optional[random.Random] = None) -> str:
    """
    Remove k contiguous sentences from a random position.
    
    Args:
        text: Original CoT reasoning
        k: Number of contiguous sentences to remove
        rng: Random number generator for reproducibility
        
    Returns:
        Modified CoT with k contiguous sentences removed
    """
    if rng is None:
        rng = random.Random()
    
    sentences = split_sentences(text)
    if not sentences:
        return text
    if k >= len(sentences):
        return sentences[0]  # Keep at least one sentence
    
    # Pick random start position
    max_start = len(sentences) - k
    start = rng.randint(0, max_start)
    remaining = sentences[:start] + sentences[start + k:]
    return ' '.join(remaining) if remaining else sentences[0]


# =============================================================================
# Intervention 1d: Truncate Random Contiguous P% of Sentences
# =============================================================================

def truncate_percent(text: str, p: float, rng: Optional[random.Random] = None) -> str:
    """
    Remove a contiguous p% chunk of sentences from a random position.
    
    Args:
        text: Original CoT reasoning
        p: Fraction of sentences to remove (0.0 to 1.0)
        rng: Random number generator for reproducibility
        
    Returns:
        Modified CoT with ~p% of sentences removed
    """
    if rng is None:
        rng = random.Random()
    
    sentences = split_sentences(text)
    if not sentences or len(sentences) <= 1:
        return text
    
    # Calculate number of sentences to remove
    k = max(1, int(len(sentences) * p))
    k = min(k, len(sentences) - 1)  # Keep at least one sentence
    
    return truncate_contiguous_k(text, k, rng)


# =============================================================================
# Legacy: Truncate Random (Non-Contiguous) Sentences
# =============================================================================

def truncate_sentences(text: str, fraction: float = 0.3, rng: Optional[random.Random] = None) -> str:
    """
    Randomly truncate some non-contiguous sentences from the CoT.
    Kept for backward compatibility with existing results.
    
    Args:
        text: Original CoT reasoning
        fraction: Fraction of sentences to remove (0.0 to 1.0)
        rng: Random number generator for reproducibility
        
    Returns:
        Modified CoT with random sentences removed
    """
    if rng is None:
        rng = random.Random()
    
    if not text or not text.strip():
        return text
    
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text
    
    keep_count = max(1, int(len(sentences) * (1 - fraction)))
    indices = sorted(rng.sample(range(len(sentences)), keep_count))
    return ' '.join(sentences[i] for i in indices)


# =============================================================================
# Intervention 2: Error Injection (LLM-based)
# =============================================================================

def inject_error_batch(
    cots: List[str],
    problems: List[str],
    engine,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> List[str]:
    """
    Use an LLM to inject logical or arithmetic errors into a batch of CoTs.
    
    Args:
        cots: List of original Chain of Thought reasoning strings
        problems: List of original problem statements (for context)
        engine: VLLMEngine instance to use for generating the errors
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        List of CoTs with injected logical/arithmetic errors
    """
    if len(cots) != len(problems):
        raise ValueError("cots and problems must have the same length")
    
    if not cots:
        return []
    
    # Build prompts for all items in the batch
    prompts = [
        PromptTemplates.inject_error_prompt(problem, cot)
        for cot, problem in zip(cots, problems)
    ]
    
    # Generate all perturbed CoTs in one batch
    outputs = engine.generate(prompts, temperature=temperature, max_tokens=max_tokens)
    
    # Process outputs, falling back to original if generation fails
    results = []
    for cot, output in zip(cots, outputs):
        if output and output.strip():
            results.append(output.strip())
        else:
            results.append(cot)
    
    return results


# =============================================================================
# Intervention 3: Filler Text Replacement (Wikipedia-based)
# =============================================================================

# Global cache for Wikipedia sentences
_WIKIPEDIA_SENTENCES: Optional[List[str]] = None


def load_wikipedia_sentences(
    num_articles: int = 1000,
    subset: str = "20231101.en",
    cache_path: Optional[str] = None,
    seed: int = 42
) -> List[str]:
    """
    Load random sentences from Wikipedia articles.
    
    Args:
        num_articles: Number of articles to sample
        subset: Wikipedia dump subset to use
        cache_path: Path to cache the sentences (optional)
        seed: Random seed for reproducible sampling
        
    Returns:
        List of sentences from Wikipedia
    """
    global _WIKIPEDIA_SENTENCES
    
    # Return cached if available
    if _WIKIPEDIA_SENTENCES is not None:
        return _WIKIPEDIA_SENTENCES
    
    # Try to load from cache file
    if cache_path:
        try:
            import json
            with open(cache_path, 'r') as f:
                _WIKIPEDIA_SENTENCES = json.load(f)
            print(f"Loaded {len(_WIKIPEDIA_SENTENCES)} Wikipedia sentences from cache")
            return _WIKIPEDIA_SENTENCES
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    # Load from HuggingFace
    try:
        from datasets import load_dataset
        print(f"Loading Wikipedia dataset (subset={subset})...")
        
        # Load a streaming dataset and take a sample
        wiki = load_dataset(
            "wikimedia/wikipedia",
            subset,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        # Sample articles
        rng = random.Random(seed)
        sentences = []
        
        print(f"Extracting sentences from {num_articles} articles...")
        for i, article in enumerate(wiki):
            if i >= num_articles:
                break
            
            text = article.get('text', '')
            if text:
                article_sentences = split_sentences(text)
                # Filter out very short or very long sentences
                good_sentences = [
                    s for s in article_sentences 
                    if 20 < len(s) < 200 and not s.startswith('=')
                ]
                sentences.extend(good_sentences)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} articles, {len(sentences)} sentences collected")
        
        # Shuffle sentences
        rng.shuffle(sentences)
        _WIKIPEDIA_SENTENCES = sentences
        
        # Save to cache if path provided
        if cache_path and sentences:
            import json
            import os
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(sentences, f)
            print(f"Cached {len(sentences)} sentences to {cache_path}")
        
        print(f"Loaded {len(sentences)} Wikipedia sentences")
        return _WIKIPEDIA_SENTENCES
        
    except Exception as e:
        print(f"Warning: Could not load Wikipedia dataset: {e}")
        print("Falling back to default filler sentences")
        
        _WIKIPEDIA_SENTENCES = [
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "This is placeholder text that serves no mathematical purpose.",
            "The weather today is particularly pleasant for outdoor activities.",
            "Historical records indicate significant changes over time.",
            "Consider the implications of this statement carefully.",
            "Many factors contribute to the final outcome of any process.",
            "The relationship between these elements is quite complex.",
            "Scientists have discovered new evidence supporting this theory.",
            "The development of technology has transformed modern society.",
            "Education plays a crucial role in personal development.",
            "Environmental concerns have become increasingly important.",
            "The economic impact of these changes is still being studied.",
            "Cultural traditions vary significantly across different regions.",
            "Research in this field continues to yield interesting results.",
        ]
        return _WIKIPEDIA_SENTENCES


def filler_replacement(
    text: str,
    p: float,
    rng: Optional[random.Random] = None,
    wikipedia_sentences: Optional[List[str]] = None,
) -> str:
    """
    Replace p% of sentences with random Wikipedia filler text.
    
    Args:
        text: Original CoT reasoning
        p: Fraction of sentences to replace (0.0 to 1.0)
        rng: Random number generator for reproducibility
        wikipedia_sentences: Pre-loaded Wikipedia sentences (optional)
        
    Returns:
        Modified CoT with some sentences replaced by filler
    """
    if rng is None:
        rng = random.Random()
    
    sentences = split_sentences(text)
    if not sentences or len(sentences) <= 1:
        return text
    
    # Load Wikipedia sentences if not provided
    if wikipedia_sentences is None:
        wikipedia_sentences = load_wikipedia_sentences()
    
    if not wikipedia_sentences:
        return text
    
    # Calculate number of sentences to replace
    num_replace = max(1, int(len(sentences) * p))
    num_replace = min(num_replace, len(sentences) - 1)  # Keep at least one original
    
    # Pick random indices to replace
    indices = rng.sample(range(len(sentences)), num_replace)
    
    # Replace with random Wikipedia sentences
    for idx in indices:
        sentences[idx] = rng.choice(wikipedia_sentences)
    
    return ' '.join(sentences)


# =============================================================================
# Intervention Factory
# =============================================================================

def get_intervention_fn(
    name: str, 
    params: dict, 
    rng: Optional[random.Random] = None,
    wikipedia_sentences: Optional[List[str]] = None,
) -> Callable[[str], str]:
    """
    Factory function to get an intervention function by name.
    
    Args:
        name: Intervention name
        params: Intervention parameters
        rng: Random number generator for reproducibility
        wikipedia_sentences: Pre-loaded Wikipedia sentences for filler replacement
        
    Returns:
        Function that takes a CoT string and returns modified CoT
    """
    if rng is None:
        rng = random.Random()
    
    if name == "truncate_first":
        k = params.get("k", 3)
        return lambda text: truncate_first_k(text, k)
    
    elif name == "truncate_last":
        k = params.get("k", 3)
        return lambda text: truncate_last_k(text, k)
    
    elif name == "truncate_contiguous":
        k = params.get("k", 3)
        return lambda text: truncate_contiguous_k(text, k, rng)
    
    elif name == "truncate_percent":
        p = params.get("p", 0.3)
        return lambda text: truncate_percent(text, p, rng)
    
    elif name == "filler_replacement":
        p = params.get("p", 0.3)
        wiki_sents = wikipedia_sentences or load_wikipedia_sentences()
        return lambda text: filler_replacement(text, p, rng, wiki_sents)
    
    elif name == "error_injection":
        raise ValueError("Error injection must be handled separately with an LLM engine")
    
    else:
        raise ValueError(f"Unknown intervention: {name}")


# =============================================================================
# Legacy: apply_intervention (for backward compatibility)
# =============================================================================

def apply_intervention(
    cots: List[str],
    intervention_type: str,
    problems: Optional[List[str]] = None,
    engine=None,
    truncation_fraction: float = 0.3,
) -> List[str]:
    """
    Apply a specific intervention to a batch of CoTs.
    Kept for backward compatibility.
    
    Args:
        cots: List of original Chain of Thought reasoning strings
        intervention_type: One of "truncate_random", "inject_error"
        problems: List of original problems (required for inject_error)
        engine: VLLMEngine instance (required for inject_error)
        truncation_fraction: Fraction of sentences to remove (for truncation)
        
    Returns:
        List of perturbed CoT strings
    """
    if not cots:
        return []
    
    if intervention_type == "truncate_random":
        return [truncate_sentences(cot, fraction=truncation_fraction) for cot in cots]
    elif intervention_type == "inject_error":
        if problems is None or engine is None:
            raise ValueError("problems and engine are required for inject_error intervention")
        if len(cots) != len(problems):
            raise ValueError("cots and problems must have the same length")
        return inject_error_batch(cots, problems, engine)
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
