"""
Print the i-th entry from a JSONL file in a readable format.
Usage:
    python print_jsonl_entry.py --index <index> [--file <file_path>]
    python print_jsonl_entry.py --index 0  # Print first entry (index 0)
    python print_jsonl_entry.py --index 5 --file outputs/stage2_llama.jsonl
"""

import json
import argparse
from pathlib import Path
import pprint


def print_entry(entry: dict, index: int):
    """Print a JSONL entry in a readable format with all fields."""
    print("=" * 100)
    print(f"ENTRY {index} (ID: {entry.get('id', 'N/A')})")
    print("=" * 100)
    print()
    
    # Basic info
    print("=" * 100)
    print("BASIC INFO")
    print("=" * 100)
    print(f"ID (id): {entry.get('id', 'N/A')}")
    print(f"Problem (problem): {entry.get('problem', 'N/A')}")
    print(f"Ground Truth (ground_truth): {entry.get('ground_truth', 'N/A')}")
    print()
    
    # Model A
    print("=" * 100)
    print("MODEL A (QWEN)")
    print("=" * 100)
    print(f"Answer A (answer_a): {entry.get('answer_a', 'N/A')}")
    print()
    print("--- Model A Raw Output (model_a_raw) ---")
    print(entry.get('model_a_raw', 'N/A'))
    print()
    print("--- Chain of Thought (cot_a) ---")
    print(entry.get('cot_a', 'N/A'))
    print()
    
    # Model B Baseline
    print("=" * 100)
    print("MODEL B (LLAMA) - BASELINE")
    print("=" * 100)
    print(f"Answer B (answer_b): {entry.get('answer_b', 'N/A')}")
    print()
    print("--- Model B Raw Output (model_b_raw) ---")
    print(entry.get('model_b_raw', 'N/A'))
    print()
    
    # Truncated Intervention
    if 'answer_b_truncated' in entry:
        print("=" * 100)
        print("MODEL B (LLAMA) - TRUNCATED INTERVENTION")
        print("=" * 100)
        print(f"Answer B Truncated (answer_b_truncated): {entry.get('answer_b_truncated', 'N/A')}")
        flip = entry.get('answer_b', '') != entry.get('answer_b_truncated', '')
        print(f"Flip? {flip}")
        print()
        print("--- Truncated Chain of Thought (cot_a_truncated) ---")
        print(entry.get('cot_a_truncated', 'N/A'))
        print()
        print("--- Truncated Prompt (prompt_truncated) ---")
        print(entry.get('prompt_truncated', 'N/A'))
        print()
        print("--- Model B Truncated Raw Output (model_b_truncated_raw) ---")
        print(entry.get('model_b_truncated_raw', 'N/A'))
        print()
    
    # Error Injection Intervention
    if 'answer_b_injected' in entry:
        print("=" * 100)
        print("MODEL B (LLAMA) - ERROR INJECTION INTERVENTION")
        print("=" * 100)
        print(f"Answer B Injected (answer_b_injected): {entry.get('answer_b_injected', 'N/A')}")
        flip = entry.get('answer_b', '') != entry.get('answer_b_injected', '')
        print(f"Flip? {flip}")
        print()
        print("--- Injected Chain of Thought (cot_a_injected) ---")
        print(entry.get('cot_a_injected', 'N/A'))
        print()
        print("--- Injected Prompt (prompt_injected) ---")
        print(entry.get('prompt_injected', 'N/A'))
        print()
        print("--- Model B Injected Raw Output (model_b_injected_raw) ---")
        print(entry.get('model_b_injected_raw', 'N/A'))
        print()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Print the i-th entry from a JSONL file in a readable format"
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=0,
        help="Index of the entry to print (0-based)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default="outputs/stage3_interventions.jsonl",
        help="Path to the JSONL file (default: outputs/stage3_interventions.jsonl)"
    )
    
    args = parser.parse_args()
    
    file_path = args.file
    index = args.index
    
    # Check if file exists
    if not Path(file_path).exists():
        raise ValueError(f"Error: File '{file_path}' not found")
    
    # Read and print the entry
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if index < 0 or index >= len(lines):
        raise ValueError(
            f"Error: Index {index} is out of range. "
            f"File has {len(lines)} entries (indices 0-{len(lines)-1})"
        )
    
    entry = json.loads(lines[index])
    print_entry(entry, index)
    

if __name__ == "__main__":
    main()
