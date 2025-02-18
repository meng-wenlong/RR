from datasets import Dataset


def match_within(candidate: str, pii: str):
    if pii.lower() in candidate.lower():
        return True
    return False


def match_exact(candidate: str, pii: str):
    if pii.lower() == candidate.lower():
        return True
    return False


def get_max_index(scores: list[float]):
    return scores.index(max(scores))


def get_min_index(scores: list[float]):
    return scores.index(min(scores))


def top_one_accuracy(
    dataset: Dataset, 
    candidates_with_scores: Dataset,
    best: str = "min",
    score_column: str = "scores",
    match_method: str = "within",
):
    assert score_column in candidates_with_scores.column_names
    assert 'pii_key' in candidates_with_scores.column_names
    assert 'row_idx' in candidates_with_scores.column_names
    assert 'candidates' in candidates_with_scores.column_names
    assert 'pii_mask' in dataset.column_names

    if best == "min":
        get_best_index = get_min_index
    elif best == "max":
        get_best_index = get_max_index
    else:
        raise ValueError("Unknown value for best")
    
    if match_method == "within":
        match_fn = match_within
    elif match_method == "exact":
        match_fn = match_exact
    else:
        raise ValueError("Unknown value for match_method")
    
    correct = 0
    total = 0
    for cs_row in candidates_with_scores:
        row_idx = cs_row['row_idx']
        pii_key = cs_row['pii_key']
        pii_mask = dataset[row_idx]['pii_mask']
        if pii_key not in pii_mask['labels']:
            continue
        total += 1

        pii_label_index = pii_mask['labels'].index(pii_key)
        pii_value = pii_mask['values'][pii_label_index]

        scores = cs_row[score_column]
        candidates = cs_row['candidates']
        if not scores or not candidates:
            continue
        best_candidate = candidates[get_best_index(scores)]
        
        if match_fn(best_candidate, pii_value):
            correct += 1

    return correct / total


def top_n_accuracy(
    dataset: Dataset,
    candidates_with_scores: Dataset,
    better: str = "lower",
    score_column: str = "scores",
    sort_using_scores: bool = True,
    match_method: str = "within",
    n: int = -1,
):
    if sort_using_scores:
        assert score_column in candidates_with_scores.column_names
        assert better == "lower" or better == "higher"
    assert 'pii_key' in candidates_with_scores.column_names
    assert 'row_idx' in candidates_with_scores.column_names
    assert 'candidates' in candidates_with_scores.column_names
    assert 'pii_mask' in dataset.column_names

    if match_method == "within":
        match_fn = match_within
    elif match_method == "exact":
        match_fn = match_exact
    else:
        raise ValueError("Unknown value for match_method")
    
    correct = 0
    total = 0
    for cs_row in candidates_with_scores:
        row_idx = cs_row['row_idx']
        pii_key = cs_row['pii_key']
        pii_mask = dataset[row_idx]['pii_mask']
        if pii_key not in pii_mask['labels']:
            continue
        total += 1

        pii_label_index = pii_mask['labels'].index(pii_key)
        pii_value = pii_mask['values'][pii_label_index] # ground truth
        
        candidates = cs_row['candidates']
        if not candidates:
            continue

        if sort_using_scores:
            scores = cs_row[score_column]
            if better == "lower":
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
            elif better == "higher":
                sorted_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
            else:
                raise ValueError("Unknown value for better")
            
            sorted_candidates = [candidates[i] for i in sorted_indices]
        else:
            sorted_candidates = candidates

        if n <= 0:
            n_ = len(sorted_candidates)
        else:
            n_ = min(n, len(sorted_candidates))

        for i in range(n_):
            if match_fn(sorted_candidates[i], pii_value):
                correct += 1
                break
    print(f"correct/total: {correct}/{total}")
    return correct / total
        