from datasets import load_dataset
from .parsing import extract_boxed_answer

def load_math_lighteval_test():
    ds = load_dataset('DigitalLearningGmbH/MATH-lighteval', 'default', split='test')
    data = []
    for i, ex in enumerate(ds):
        problem = ex['problem']
        solution = ex['solution']
        gt = extract_boxed_answer(solution)
        data.append({'id': i, 'problem': problem, 'solution': solution, 'ground_truth': gt, 'level': ex.get('level'), 'type': ex.get('type')})
    return data