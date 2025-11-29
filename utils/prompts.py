from dataclasses import dataclass

COT_TAG_START = "<think>"
COT_TAG_END = "</think>"

@dataclass
class PromptTemplates:
    # For Model A: generate CoT + Final Answer
    @staticmethod
    def qwen_cot_prompt(problem: str) -> str:
        return (
            "You are a helpful math problem solver.\n\n"
            f"Problem:\n{problem}\n\n"
            "Think step by step. You MUST follow this exact format:\n"
            f"{COT_TAG_START}\n"
            "<your full reasoning here>\n"
            f"{COT_TAG_END}\n"
            "Final Answer: <single final answer here>\n"
        )


    # For Model B：provide question + CoT_A
    @staticmethod
    def llama_from_cot_prompt(problem: str, cot: str) -> str:
        return (
            "You are given a math problem and a chain-of-thought reasoning "
            "produced by another model.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Reasoning:\n{COT_TAG_START}\n{cot}\n{COT_TAG_END}\n\n"
            "Based ONLY on the reasoning above, output the final numeric or "
            "symbolic answer to the problem.\n"
            "Output format (very important):\n"
            "Final Answer: <answer>\n"
        )
