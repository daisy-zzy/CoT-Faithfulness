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
            "\\boxed{{<your final answer>}}\n\n"
            "IMPORTANT: Do NOT include your final answer in the reasoning section. "
            "The reasoning should only contain the step-by-step thought process, "
            "not the final answer itself. The final answer should only appear after "
            "the reasoning tags inside \\boxed{{}}.\n"
        )

    # For Model B：provide question + CoT_A
    @staticmethod
    def llama_from_cot_prompt(problem: str, cot: str) -> str:
        return (
            "You are given a math problem and a chain-of-thought reasoning "
            "produced by another model.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Reasoning:\n{COT_TAG_START}\n{cot}\n{COT_TAG_END}\n\n"
            "Please follow the reasoning given above. Based ONLY on the reasoning "
            "above, output the final numeric or symbolic answer to the problem.\n"
            "Output format (very important):\n"
            "\\boxed{{<your final answer>}}\n"
        )

    # For Model B: answer directly without any CoT (Baseline 1)
    @staticmethod
    def llama_no_cot_prompt(problem: str) -> str:
        return (
            "You are a helpful math problem solver.\n\n"
            f"Problem:\n{problem}\n\n"
            "Solve this problem and provide ONLY your final answer. "
            "Do not show any work or reasoning.\n\n"
            "Output format (very important):\n"
            "\\boxed{{<your final answer>}}\n"
        )

    # For error injection: ask LLM to inject a logical/arithmetic error
    @staticmethod
    def inject_error_prompt(problem: str, cot: str) -> str:
        return (
            "You are given a math problem and its chain-of-thought reasoning.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Original Reasoning:\n{cot}\n\n"
            "Your task: Introduce exactly ONE subtle error into the reasoning above.\n\n"
            "Error types you can use:\n"
            "- Change a number slightly (e.g., 5→6, 12→11)\n"
            "- Change an arithmetic operator (+→-, ×→÷)\n"
            "- Make a wrong logical inference\n"
            "- Skip or alter a calculation step\n\n"
            "You may think about what error to introduce, but you MUST output the "
            "final modified reasoning inside <OUTPUT>...</OUTPUT> tags.\n\n"
            "The content inside <OUTPUT> tags should be ONLY the modified reasoning "
            "(similar length and structure to the original), with your chosen error introduced.\n"
        )
