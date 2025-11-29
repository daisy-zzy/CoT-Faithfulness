import re
from typing import Optional, Tuple

from .prompts import COT_TAG_START, COT_TAG_END

COT_PATTERN = re.compile(
    re.escape(COT_TAG_START) + r"(.*?)" + re.escape(COT_TAG_END),
    re.DOTALL | re.IGNORECASE,
)

# "Final Answer: ..." pattern.
# - case-insensitive
# - optional "is"
# - optional colon
FINAL_PATTERN = re.compile(
    r"""(?ix)          # ignore case, allow comments
        final \s* answer   # "Final Answer"
        (?: \s* is )?      # optional "is"
        \s*[:：]?          # optional colon
        \s* (.+)           # capture the rest of the line
    """
)

# Any \boxed{...} occurrence inside a string
BOXED_ANYWHERE = re.compile(r"\\boxed\{(.*?)}")


# =====================
# Ground-truth parsing
# =====================

def extract_boxed_answer(solution: str) -> Optional[str]:
    """
    Extract the content of the *last* \\boxed{...} in a LaTeX solution.

    This is mainly used for parsing ground-truth from the MATH(-lighteval)
    dataset, where the official answer is enclosed in a \\boxed{...}.

    Returns:
        A normalized answer string, or None if no \\boxed is found.
    """
    matches = list(BOXED_ANYWHERE.finditer(solution))
    if not matches:
        return None

    last = matches[-1].group(1)
    return normalize_answer(last)


# ==========================
# Model output CoT + answer
# ==========================

def extract_cot_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract CoT and final answer from a model's raw output.

    Priority:
      1) If there is an explicit <think>...</think> block, use that as CoT.
      2) Otherwise, if we can find a 'Final Answer', treat the prefix
         (everything before the last 'Final Answer') as CoT.
    """
    if text is None:
        return None, None

    text = text.strip()

    # -------- 1. Try explicit <think>...</think> --------
    cot: Optional[str] = None
    m_cot = COT_PATTERN.search(text)
    if m_cot:
        cot = m_cot.group(1).strip()

    # -------- 2. Find the LAST 'Final Answer' line --------
    ans: Optional[str] = None
    matches = list(FINAL_PATTERN.finditer(text))
    if matches:
        last = matches[-1]
        raw_ans = last.group(1)
        ans = normalize_answer(raw_ans)

        # If we still don't have a CoT, fall back to "everything before Final Answer"
        if cot is None:
            cot_candidate = text[: last.start()].strip()
            # Sometimes the model spuriously emits '</think>' at the very start.
            cot_candidate = re.sub(r"^</?think>\s*", "", cot_candidate,
                                   flags=re.IGNORECASE)
            if cot_candidate:
                cot = cot_candidate

    return cot, ans



# ====================
# Answer normalization
# ====================

def normalize_answer(ans: str) -> str:
    """
    Heuristic normalization shared by:
      - model predictions (answer_a, answer_b)
      - ground-truth extracted from LaTeX solutions.
    """
    if ans is None:
        return ""

    # Basic strip
    ans = ans.strip()
    if not ans:
        return ""

    # Cut anything after an HTML/XML-like tag (</think>, <stop>, etc.)
    parts = re.split(r"<[^>]+>", ans)
    ans = parts[0].strip() if parts else ""
    if not ans:
        return ""

    # Often the answer is on a single line; drop later lines just in case.
    lines = ans.splitlines()
    if lines:
        ans = lines[0].strip()
    else:
        ans = ""
    if not ans:
        return ""

    # Prefer boxed content if present anywhere
    m_box = BOXED_ANYWHERE.search(ans)
    if m_box:
        ans = m_box.group(1).strip()
        if not ans:
            return ""

    # Remove outer \( ... \), \[ ... \], $$ ... $$ if present
    if ans.startswith(r"\(") and ans.endswith(r"\)"):
        ans = ans[2:-2].strip()
    if ans.startswith(r"\[") and ans.endswith(r"\]"):
        ans = ans[2:-2].strip()
    if ans.startswith("$$") and ans.endswith("$$"):
        ans = ans[2:-2].strip()

    # Remove any remaining $ that just denote math mode
    ans = ans.replace("$", "").strip()

    # Collapse multiple spaces
    ans = re.sub(r"\s+", " ", ans)

    return ans
