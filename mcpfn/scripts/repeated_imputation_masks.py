"""Versioned, rate-controlled MissBench-family masks for repeated evaluation."""

import hashlib
import math

import numpy as np
from scipy.special import expit, logit


VERSION = 'missbench_families_rate_controlled_v1'
PATTERNS = {
    'MCAR': ('MCAR', 'MCAR'),
    'MNARLatentFactorPattern': ('Latent-MCAR', 'MCAR'),
    'MNARClusterLevelPattern': ('Clust-MCAR', 'MCAR'),
    'MAR': ('Col-MAR', 'MAR'),
    'MNARTwoPhaseSubsetPattern': ('Phase-MAR', 'MAR'),
    'MNAR': ('Self-MNAR', 'MNAR'),
    'MAR_Neural': ('NN-MNAR', 'MNAR'),
    'MAR_BlockNeural': ('Block-MNAR', 'MNAR'),
    'MNARPolarizationPattern': ('Pol-MNAR', 'MNAR'),
    'MNARSoftPolarizationPattern': ('Soft-Pol-MNAR', 'MNAR'),
    'MNARCensoringPattern': ('Censor-MNAR', 'MNAR'),
}
DETERMINISTIC = {'MNARPolarizationPattern'}
NEURAL_TILE = (32, 16)


def seed_for(*parts):
    return int(hashlib.sha256('/'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    return (values - center) / np.where(scale > 1e-12, scale, 1.)


def calibrated_probabilities(scores, eligible, requested):
    scores = np.asarray(scores, dtype=np.float64)
    eligible = np.broadcast_to(eligible, scores.shape)
    maximum = float(eligible.mean())
    if not 0 < requested < 1 or not np.isfinite(scores).all():
        raise ValueError('Require finite logits and a rate strictly between zero and one')
    if requested >= maximum:
        raise ValueError(f'Rate {requested} is infeasible with eligible fraction {maximum}')
    low, high = -float(np.max(scores)) - 50., -float(np.min(scores)) + 50.
    for unused in range(60):
        middle = (low + high) / 2
        if np.mean(expit(scores + middle) * eligible) < requested:
            low = middle
        else:
            high = middle
    return expit(scores + (low + high) / 2) * eligible


def neural_scores(values, generator):
    import torch
    from tabimpute.prior.mar_missing import MAR_missingness

    result = np.empty(values.shape, dtype=np.float64)
    row_groups = np.array_split(np.arange(values.shape[0]), math.ceil(values.shape[0] / NEURAL_TILE[0]))
    column_groups = np.array_split(np.arange(values.shape[1]), math.ceil(values.shape[1] / NEURAL_TILE[1]))
    for rows in row_groups:
        for columns in column_groups:
            if len(rows) < 2 or len(columns) < 2:
                raise ValueError('Neural masks need at least two rows and columns per tile')
            config = dict(num_layers_upper=3, hidden_lower=1, hidden_upper=100,
                          activation='relu', N=len(rows), T=len(columns),
                          row_neighbor_upper=max(1, len(rows) // 2),
                          col_neighbor_upper=max(1, len(columns) // 2),
                          seed=int(generator.integers(0, 2**31)), neighbor_type='random')
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                network = MAR_missingness(config)
                probabilities = network(torch.tensor(values[np.ix_(rows, columns)], dtype=torch.float32)).numpy()
            result[np.ix_(rows, columns)] = logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
    return result


def family_state(truth, pattern, seed):
    if pattern not in PATTERNS:
        raise ValueError(f'Unknown pattern: {pattern}')
    truth = np.asarray(truth, dtype=np.float64)
    if truth.ndim != 2 or min(truth.shape) < 2 or not np.isfinite(truth).all():
        raise ValueError('Masks require a finite two-dimensional complete matrix')
    generator = np.random.default_rng(seed)
    rows, columns = truth.shape
    values = standardize(truth)
    state = dict(pattern=pattern, seed=seed, truth=truth,
                 eligible=np.ones(truth.shape, dtype=bool), scores=np.zeros(truth.shape),
                 uniforms=generator.random(truth.shape), protected_columns=[])
    if pattern == 'MNARLatentFactorPattern':
        rank = int(generator.integers(1, min(5, rows, columns) + 1))
        state['scores'] = (generator.normal(size=(rows, rank)) @ generator.normal(size=(rank, columns))
                           / np.sqrt(rank) + generator.normal(size=(rows, 1)) + generator.normal(size=(1, columns)))
    elif pattern == 'MNARClusterLevelPattern':
        row_clusters = generator.integers(0, 5, rows)
        column_clusters = generator.integers(0, 4, columns)
        state['scores'] = (generator.normal(size=5)[row_clusters, None]
                           + generator.normal(size=4)[None, column_clusters]
                           + generator.normal(size=truth.shape))
    elif pattern in ('MAR', 'MNARTwoPhaseSubsetPattern'):
        protected = generator.permutation(columns)[:max(1, int(.4 * columns))]
        state['protected_columns'] = protected.tolist()
        state['eligible'][:, protected] = False
        row_score = values[:, protected] @ generator.normal(size=len(protected))
        row_score = (row_score - row_score.mean()) / max(float(row_score.std()), 1e-12)
        state['scores'] = np.broadcast_to(2 * row_score[:, None], truth.shape)
        if pattern == 'MNARTwoPhaseSubsetPattern':
            state['uniforms'] = np.broadcast_to(generator.random((rows, 1)), truth.shape)
    elif pattern == 'MNAR':
        state['scores'] = 2 * values * generator.choice([-1., 1.], size=columns)
    elif pattern == 'MAR_Neural':
        state['scores'] = neural_scores(values, generator)
    elif pattern == 'MAR_BlockNeural':
        row_groups = np.array_split(np.arange(rows), max(2, rows // 3))
        column_groups = np.array_split(np.arange(columns), max(2, columns // 3))
        pooled = np.array([[values[np.ix_(row_group, column_group)].mean()
                            for column_group in column_groups] for row_group in row_groups])
        scores = neural_scores(pooled, generator)
        uniforms = generator.random(pooled.shape)
        for row_index, row_group in enumerate(row_groups):
            for column_index, column_group in enumerate(column_groups):
                state['scores'][np.ix_(row_group, column_group)] = scores[row_index, column_index]
                state['uniforms'][np.ix_(row_group, column_group)] = uniforms[row_index, column_index]
    elif pattern == 'MNARSoftPolarizationPattern':
        deviations = np.abs(values - np.median(values, axis=0)) ** 2.5
        normalized = deviations / np.maximum(deviations.max(axis=0), 1e-12)
        state['scores'] = logit(1 - (.05 + .9 * normalized))
    elif pattern == 'MNARCensoringPattern':
        state['left_censoring'] = generator.random(columns) < .5
    return state


def mask_at_rate(state, rate):
    truth = state['truth']
    pattern = state['pattern']
    if pattern == 'MNARPolarizationPattern':
        lower, upper = np.quantile(truth, [(1 - rate) / 2, (1 + rate) / 2], axis=0)
        mask = (truth > lower) & (truth < upper)
        expected = None
    elif pattern == 'MNARCensoringPattern':
        lower, upper = np.quantile(truth, [rate, 1 - rate], axis=0)
        mask = np.where(state['left_censoring'][None, :], truth < lower, truth > upper)
        expected = None
    else:
        probabilities = calibrated_probabilities(state['scores'], state['eligible'], rate)
        mask = state['uniforms'] < probabilities
        expected = float(probabilities.mean())
    diagnostics = dict(requested_rate=rate, expected_rate=expected, realized_rate=float(mask.mean()),
                       masked_cells=int(mask.sum()), empty_rows=int(mask.all(axis=1).sum()),
                       empty_columns=int(mask.all(axis=0).sum()),
                       protected_columns=state['protected_columns'], deterministic=pattern in DETERMINISTIC)
    return mask, diagnostics
