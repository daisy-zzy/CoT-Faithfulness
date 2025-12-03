import re
import random
from typing import Optional, List
from .prompts import PromptTemplates


def split_into_sentences(cot: str) -> List[str]:
    """
    Split a CoT string into sentences.
    
    Uses sentence boundaries (periods, exclamation marks, question marks)
    followed by space or newline.
    
    Args:
        cot: The Chain of Thought reasoning string
        
    Returns:
        List of sentences
    """
    if not cot or not cot.strip():
        return []
    
    # Split by sentence boundaries (., !, ?) followed by space or newline
    # This regex captures the sentence boundary as well
    parts = re.split(r'([.!?]\s+)', cot)
    sentences = []
    current = ""
    
    for i, part in enumerate(parts):
        current += part
        # Every other part (starting from index 1) is a sentence boundary
        if i % 2 == 1:  # We're at a sentence boundary
            sentences.append(current.strip())
            current = ""
    
    # Add the last part if it exists
    if current.strip():
        sentences.append(current.strip())
    
    return sentences if sentences else [cot]


def truncate_sentences(cot: str, fraction: float = 0.3) -> str:
    """
    Randomly truncate some sentences from the CoT.
    
    Args:
        cot: The original Chain of Thought reasoning
        fraction: Fraction of sentences to remove (0.0 to 1.0)
        
    Returns:
        Truncated CoT string with some sentences randomly removed
    """
    if not cot or not cot.strip():
        return cot
    
    sentences = split_into_sentences(cot)
    
    if len(sentences) < 2:
        # If we can't split into sentences, return original
        return cot
    
    # Calculate how many sentences to remove
    num_to_remove = max(1, int(len(sentences) * fraction))
    # Make sure we don't remove all sentences
    num_to_remove = min(num_to_remove, len(sentences) - 1)
    
    if num_to_remove <= 0:
        return cot
    
    # Randomly select sentences to remove
    indices_to_remove = set(random.sample(range(len(sentences)), num_to_remove))
    
    # Keep sentences that are not selected for removal
    remaining_sentences = [
        sent for i, sent in enumerate(sentences) if i not in indices_to_remove
    ]
    
    # Join sentences back together
    return '\n'.join(remaining_sentences)


def inject_error_batch(
    cots: List[str],
    problems: List[str],
    engine,
) -> List[str]:
    """
    Use an LLM to inject logical or arithmetic errors into a batch of CoTs.
    
    Args:
        cots: List of original Chain of Thought reasoning strings
        problems: List of original problem statements (for context)
        engine: VLLMEngine instance to use for generating the errors
        
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
    outputs = engine.generate(prompts)
    
    # Process outputs, falling back to original if generation fails
    results = []
    for i, (cot, output) in enumerate(zip(cots, outputs)):
        if output and output.strip():
            results.append(output.strip())
        else:
            # Fallback to original if generation fails
            results.append(cot)
    
    return results


def apply_intervention(
    cots: List[str],
    intervention_type: str,
    problems: Optional[List[str]] = None,
    engine=None,
    truncation_fraction: float = 0.3,
) -> List[str]:
    """
    Apply a specific intervention to a batch of CoTs.
    
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

