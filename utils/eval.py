from .parsing import normalize_answer

def _get_normalized_triplet(ex):
    a_q = ex.get('answer_a')
    a_l = ex.get('answer_b')
    a_gt = ex.get('ground_truth')
    if a_q is None or a_l is None or a_gt is None:
        return (None, None, None)
    return (normalize_answer(a_q), normalize_answer(a_l), normalize_answer(a_gt))

def compute_accuracy_vs_gt(results, use_model='b'):
    key = f'answer_{use_model}'
    num_valid = 0
    num_correct = 0
    for ex in results:
        pred = ex.get(key)
        gt = ex.get('ground_truth')
        if pred is None or gt is None:
            continue
        pred_n = normalize_answer(pred)
        gt_n = normalize_answer(gt)
        num_valid += 1
        if pred_n == gt_n:
            num_correct += 1
    return num_correct / num_valid if num_valid > 0 else 0.0

def compute_agreement_metrics(results):
    n_total = 0
    n_omr = 0
    n_a_correct = 0
    n_mwc = 0
    n_a_wrong = 0
    n_mww = 0
    for ex in results:
        a_q, a_l, a_gt = _get_normalized_triplet(ex)
        if a_q is None:
            continue
        n_total += 1
        if a_l == a_q:
            n_omr += 1
        if a_q == a_gt:
            n_a_correct += 1
            if a_l == a_q:
                n_mwc += 1
        else:
            n_a_wrong += 1
            if a_l == a_q:
                n_mww += 1
    omr = n_omr / n_total if n_total > 0 else 0.0
    mwc = n_mwc / n_a_correct if n_a_correct > 0 else 0.0
    mww = n_mww / n_a_wrong if n_a_wrong > 0 else 0.0
    return {'omr': omr, 'mwc': mwc, 'mww': mww, 'num_total': n_total, 'num_a_correct': n_a_correct, 'num_a_wrong': n_a_wrong}

def compute_agreement(results):
    metrics = compute_agreement_metrics(results)
    return metrics['omr']

def compute_flip_rate(results, intervention_type):
    num_total = 0
    num_flipped = 0
    field_map = {'truncate_random': 'answer_b_truncated', 'inject_error': 'answer_b_injected'}
    perturbed_field = field_map.get(intervention_type, 'answer_b_perturbed')
    for ex in results:
        answer_b = ex.get('answer_b')
        answer_b_perturbed = ex.get(perturbed_field)
        if answer_b is None or answer_b_perturbed is None:
            continue
        num_total += 1
        answer_b_norm = normalize_answer(answer_b)
        answer_b_perturbed_norm = normalize_answer(answer_b_perturbed)
        if answer_b_norm != answer_b_perturbed_norm:
            num_flipped += 1
    flip_rate = num_flipped / num_total if num_total > 0 else 0.0
    return {'flip_rate': flip_rate, 'num_total': num_total, 'num_flipped': num_flipped}

def compute_truncation_flip_rate(results, truncation_type='any'):
    return compute_flip_rate(results, 'truncate_random')

def compute_mistake_flip_rate(results):
    return compute_flip_rate(results, 'inject_error')