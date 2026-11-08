"""Observed-only local context and target-covered validation for scientific panels."""

import numpy as np

import evaluate_realistic_science as original
from evaluate_science_delays import delayed_matrix


def cover_validation_targets(context, original_mask, targets, seed=61291, minimum=12):
    observed = np.isfinite(context)
    mask = np.asarray(original_mask, dtype=bool).copy() & observed
    random = np.random.default_rng(seed)
    for column in targets:
        available = int(observed[:, column].sum())
        required = min(minimum, available // 4)
        if required < 4:
            raise ValueError('Too few observed measurements for supported validation')
        missing_positions = ~observed[:, column]
        transitions = np.diff(np.r_[False, missing_positions, False].astype(int))
        lengths = np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)
        block_length = min(4, max(1, int(np.median(lengths)))) if len(lengths) else 1
        for start in random.permutation(len(context)):
            if mask[:, column].sum() >= required:
                break
            candidates = np.arange(start, min(start + block_length, len(context)))
            candidates = candidates[observed[candidates, column] & ~mask[candidates, column]]
            remaining = required - int(mask[:, column].sum())
            mask[candidates[:remaining], column] = True
        if mask[:, column].sum() < required:
            raise ValueError('Could not cover every target column')
    assert not (mask & ~observed).any()
    return mask


def temporal_context_predict(matrix, targets, model, distribution, forward, rows, delays):
    augmented = delayed_matrix(matrix, targets, delays)
    augmented, inverse_global = original.transform(augmented, 'column')
    result = np.empty_like(augmented, dtype=float)
    block_size = min(16, rows)
    for start in range(0, len(matrix), block_size):
        stop = min(start + block_size, len(matrix))
        center = (start + stop) // 2
        left = max(0, min(center - rows // 2, len(matrix) - rows))
        right = min(left + rows, len(matrix))
        local, inverse = original.transform(augmented[left:right], 'column')
        prediction, _ = forward(local, model, distribution)
        restored = inverse(prediction)
        result[start:stop] = restored[start-left:stop-left]
    result = inverse_global(result)[:, :matrix.shape[1]]
    result[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite local-context prediction')
    return result
