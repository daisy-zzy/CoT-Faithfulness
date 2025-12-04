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
            "Final Answer: <single final answer here>\n\n"
            "IMPORTANT: Do NOT include your final answer in the reasoning section. "
            "The reasoning should only contain the step-by-step thought process, "
            "not the final answer itself. The final answer should only appear after "
            "the reasoning tags.\n"
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
            "Final Answer: <answer>\n"
        )

    # For error injection: ask LLM to inject a logical/arithmetic error
    @staticmethod
    def inject_error_prompt(problem: str, cot: str) -> str:
        return (
            "You are given a math problem and a chain-of-thought reasoning solution.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Original Reasoning:\n{cot}\n\n"
            "Your task is to introduce a SINGLE, SUBTLE logical or arithmetic error "
            "into the reasoning above. The error should be realistic and not immediately obvious.\n\n"
            "Examples of errors you could introduce:\n"
            "- Change a number slightly (e.g., 5 to 6, or 10 to 9)\n"
            "- Change an arithmetic operator (e.g., + to -, * to /)\n"
            "- Make a logical mistake in a step (e.g., wrong assumption, incorrect inference)\n"
            "- Skip or modify a calculation step\n\n"
            "IMPORTANT: Keep the overall structure and flow of the reasoning similar. "
            "Only introduce ONE error. Do not change the format or add explanations.\n\n"
            "Output ONLY the modified reasoning with the error. Do not add any "
            "introductory text, explanations, or comments. Just output the reasoning itself:\n"
        )
