import re
from .prompts import COT_TAG_START, COT_TAG_END
COT_PATTERN = re.compile(re.escape(COT_TAG_START) + '(.*?)' + re.escape(COT_TAG_END), re.DOTALL | re.IGNORECASE)
FINAL_PATTERN = re.compile('(?ix)          # ignore case, allow comments\n        final \\s* answer\n        (?: \\s* is )?\n        \\s*[:：]?\n        \\s* (.+)\n    ')
BOXED_ANYWHERE = re.compile('\\\\boxed\\{(.*?)}')

def extract_boxed_answer(solution):
    matches = list(BOXED_ANYWHERE.finditer(solution))
    if not matches:
        return None
    last = matches[-1].group(1)
    return normalize_answer(last)

def extract_cot_and_answer(text):
    if text is None:
        return (None, None)
    text = text.strip()
    cot = None
    m_cot = COT_PATTERN.search(text)
    if m_cot:
        cot = m_cot.group(1).strip()
    ans = None
    answer_start_pos = None
    boxed_matches = list(BOXED_ANYWHERE.finditer(text))
    if boxed_matches:
        last_boxed = boxed_matches[-1]
        raw_ans = last_boxed.group(1)
        ans = normalize_answer(raw_ans)
        answer_start_pos = last_boxed.start()
    if ans is None:
        final_matches = list(FINAL_PATTERN.finditer(text))
        if final_matches:
            last = final_matches[-1]
            raw_ans = last.group(1)
            ans = normalize_answer(raw_ans)
            answer_start_pos = last.start()
    if cot is None and answer_start_pos is not None:
        cot_candidate = text[:answer_start_pos].strip()
        cot_candidate = re.sub('^</?think>\\s*', '', cot_candidate, flags=re.IGNORECASE)
        if cot_candidate:
            cot = cot_candidate
    return (cot, ans)

def normalize_answer(ans):
    if ans is None:
        return ''
    ans = ans.strip()
    if not ans:
        return ''
    parts = re.split('<[^>]+>', ans)
    ans = parts[0].strip() if parts else ''
    if not ans:
        return ''
    lines = ans.splitlines()
    if lines:
        ans = lines[0].strip()
    else:
        ans = ''
    if not ans:
        return ''
    m_box = BOXED_ANYWHERE.search(ans)
    if m_box:
        ans = m_box.group(1).strip()
        if not ans:
            return ''
    if ans.startswith('\\(') and ans.endswith('\\)'):
        ans = ans[2:-2].strip()
    if ans.startswith('\\[') and ans.endswith('\\]'):
        ans = ans[2:-2].strip()
    if ans.startswith('$$') and ans.endswith('$$'):
        ans = ans[2:-2].strip()
    ans = ans.replace('$', '').strip()
    ans = re.sub('\\s+', ' ', ans)
    return ans