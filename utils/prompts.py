from dataclasses import dataclass
COT_TAG_START = '<think>'
COT_TAG_END = '</think>'

@dataclass
class PromptTemplates:

    @staticmethod
    def qwen_cot_prompt(problem):
        return f'You are a helpful math problem solver.\n\nProblem:\n{problem}\n\nThink step by step. You MUST follow this exact format:\n{COT_TAG_START}\n<your full reasoning here>\n{COT_TAG_END}\n\\boxed{{{{<your final answer>}}}}\n\nIMPORTANT: Do NOT include your final answer in the reasoning section. The reasoning should only contain the step-by-step thought process, not the final answer itself. The final answer should only appear after the reasoning tags inside \\boxed{{{{}}}}.\n'

    @staticmethod
    def llama_from_cot_prompt(problem, cot):
        return f'You are given a math problem and a chain-of-thought reasoning produced by another model.\n\nProblem:\n{problem}\n\nReasoning:\n{COT_TAG_START}\n{cot}\n{COT_TAG_END}\n\nPlease follow the reasoning given above. Based ONLY on the reasoning above, output the final numeric or symbolic answer to the problem.\nOutput format (very important):\n\\boxed{{{{<your final answer>}}}}\n'

    @staticmethod
    def llama_no_cot_prompt(problem):
        return f'You are a helpful math problem solver.\n\nProblem:\n{problem}\n\nSolve this problem and provide ONLY your final answer. Do not show any work or reasoning.\n\nOutput format (very important):\n\\boxed{{{{<your final answer>}}}}\n'

    @staticmethod
    def inject_error_prompt(problem, cot):
        return f'You are given a math problem and its chain-of-thought reasoning.\n\nProblem:\n{problem}\n\nOriginal Reasoning:\n{cot}\n\nYour task: Introduce exactly ONE subtle error into the reasoning above.\n\nError types you can use:\n- Change a number slightly (e.g., 5→6, 12→11)\n- Change an arithmetic operator (+→-, ×→÷)\n- Make a wrong logical inference\n- Skip or alter a calculation step\n\nYou may think about what error to introduce, but you MUST output the final modified reasoning inside <OUTPUT>...</OUTPUT> tags.\n\nThe content inside <OUTPUT> tags should be ONLY the modified reasoning (similar length and structure to the original), with your chosen error introduced.\n'