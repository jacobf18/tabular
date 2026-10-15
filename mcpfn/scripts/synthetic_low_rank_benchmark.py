"""Deterministic synthetic evaluation data, observed-only baselines, and scores."""

from contextlib import contextmanager
import hashlib
import itertools
import json
import time
import warnings

import numpy as np

from tabimpute.routing import TwoExpertRouter, observed_holdout


BASELINES = ('column_mean', 'knn', 'softimpute', 'iterative_svd')
NEURAL_METHODS = ('matrixformer', 'recsys_specialist', 'matrixformer_plus')
METHODS = (*BASELINES, *NEURAL_METHODS)
LABELS = {'column_mean': 'Column mean', 'knn': 'K-NN', 'softimpute': 'SoftImpute',
          'iterative_svd': 'IterativeSVD', 'matrixformer': 'MatrixFormer',
          'recsys_specialist': 'RecSys specialist', 'matrixformer_plus': 'MatrixFormer+'}
CASE_AXES = ('rows', 'cols', 'rank', 'noise_std', 'missing_rate', 'seed')


def cases_for_profile(profile):
    cases = []
    for shape, rank, noise, missing, seed in itertools.product(profile['shapes'], profile['ranks'],
            profile['noise_stds'], profile['missing_rates'], profile['seeds']):
        rows, cols = shape
        if not 0 < rank <= min(rows, cols) or min(rows, cols) < 2:
            raise ValueError('Every registered rank must fit every registered matrix shape')
        if not 0 < missing < 1 or not np.isfinite(noise) or noise < 0:
            raise ValueError('Invalid noise or missingness')
        case = dict(zip(CASE_AXES, (rows, cols, rank, noise, missing, seed)))
        case['case_id'] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:20]
        cases.append(case)
    if not cases or len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('Require nonempty, unique registered cases')
    return cases


def generate_case(case, master_seed):
    rows, cols, rank = (case[name] for name in ('rows', 'cols', 'rank'))
    sequence = np.random.SeedSequence([master_seed, rows, cols, rank, case['seed']])
    factor_seed, noise_seed, mask_seed = sequence.spawn(3)
    generator = np.random.default_rng(factor_seed)
    left = generator.standard_normal((rows, rank))
    right = generator.standard_normal((cols, rank))
    signal = (left @ right.T / np.sqrt(rank)).astype(np.float32)
    noise = np.random.default_rng(noise_seed).standard_normal((rows, cols))
    truth = (signal.astype(np.float64) + case['noise_std'] * noise).astype(np.float32)
    count = int(np.floor(rows * cols * case['missing_rate'] + .5))
    if not 0 < count < rows * cols:
        raise ValueError('Require observed and missing cells; masks are not repaired')
    ordering = np.random.default_rng(mask_seed).permutation(rows * cols)
    missing = np.zeros(rows * cols, dtype=bool)
    missing[ordering[:count]] = True
    missing = missing.reshape(rows, cols)
    observed = truth.copy()
    observed[missing] = np.nan
    return signal, truth, observed, missing


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def support_diagnostics(observed, rank):
    present = np.isfinite(observed)
    rows, cols = present.shape
    row_counts, col_counts = present.sum(1), present.sum(0)
    degrees = rank * (rows + cols - rank)
    return {'observed_cells': int(present.sum()), 'missing_cells': int((~present).sum()),
        'actual_missing_rate': float((~present).mean()), 'rank_degrees_of_freedom': degrees,
        'observations_per_degree_of_freedom': float(present.sum() / degrees),
        'empty_rows': int((row_counts == 0).sum()), 'empty_columns': int((col_counts == 0).sum()),
        'rows_with_fewer_than_rank_observations': int((row_counts < rank).sum()),
        'columns_with_fewer_than_rank_observations': int((col_counts < rank).sum()),
        'minimum_row_support': int(row_counts.min()), 'minimum_column_support': int(col_counts.min())}


def score_prediction(truth, signal, observed, prediction):
    prediction = np.asarray(prediction)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError('Require finite, full-shape predictions; never drop difficult cells')
    present = np.isfinite(observed)
    if not np.array_equal(prediction[present], observed[present]):
        raise ValueError('Observed cells changed')
    missing = ~present
    if not missing.any():
        raise ValueError('No test targets')
    prediction = prediction.astype(np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError('Predictions must remain finite at the registered storage precision')
    errors = prediction[missing].astype(np.float64) - truth[missing].astype(np.float64)
    clean_errors = prediction[missing].astype(np.float64) - signal[missing].astype(np.float64)
    supported = present.any(1)[:, None] & present.any(0)[None, :]
    results = {'rmse': float(np.sqrt(np.mean(errors ** 2))), 'mae': float(np.mean(np.abs(errors))),
               'signal_rmse': float(np.sqrt(np.mean(clean_errors ** 2))), 'test_cells': int(missing.sum())}
    for name, subset in (('supported', missing & supported), ('cold_axis', missing & ~supported)):
        results[f'{name}_cells'] = int(subset.sum())
        results[f'{name}_rmse'] = (float(np.sqrt(np.mean((prediction[subset].astype(np.float64)
            - truth[subset].astype(np.float64)) ** 2))) if subset.any() else None)
    return results, prediction[missing]


def observed_normalization(matrix):
    present = np.isfinite(matrix)
    count = present.sum(0)
    global_mean = float(np.sum(np.where(present, matrix, 0.)) / max(int(present.sum()), 1))
    mean = np.divide(np.sum(np.where(present, matrix, 0.), axis=0), count,
                     out=np.full(matrix.shape[1], global_mean), where=count > 0)
    variance = np.sum(np.where(present, (matrix - mean) ** 2, 0.), axis=0) / np.maximum(count, 1)
    scale = np.where(np.sqrt(variance) > 1e-8, np.sqrt(variance), 1.)
    return (matrix - mean) / scale, mean, scale


@contextmanager
def numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def baseline_prediction(matrix, method, parameters, seed, max_iters, tolerance):
    normalized, mean, scale = observed_normalization(np.asarray(matrix, dtype=np.float64))
    present = np.isfinite(normalized)
    rows, cols = present.any(1), present.any(0)
    completed = np.zeros_like(normalized)
    if method != 'column_mean' and rows.any() and cols.any():
        block = normalized[np.ix_(rows, cols)].copy()
        if np.isnan(block).any():
            with numpy_seed(seed):
                if method == 'knn':
                    from sklearn.impute import KNNImputer
                    result = KNNImputer(n_neighbors=parameters['neighbors'], weights='distance').fit_transform(block)
                elif method == 'softimpute':
                    from fancyimpute import SoftImpute
                    singular = np.linalg.svd(np.nan_to_num(block), compute_uv=False)[0]
                    shrinkage = max(float(singular) * parameters['shrinkage_fraction'], 1e-8)
                    result = SoftImpute(shrinkage_value=shrinkage, max_iters=max_iters,
                        convergence_threshold=tolerance, verbose=False).fit_transform(block)
                elif method == 'iterative_svd':
                    from fancyimpute import IterativeSVD
                    if min(block.shape) < 2:
                        result = np.nan_to_num(block)
                    else:
                        result = IterativeSVD(rank=min(parameters['rank'], min(block.shape) - 1),
                            max_iters=max_iters, convergence_threshold=tolerance, verbose=False).fit_transform(block)
                else:
                    raise ValueError(f'Unknown baseline: {method}')
            completed[np.ix_(rows, cols)] = result
        else:
            completed[np.ix_(rows, cols)] = block
    elif method not in BASELINES:
        raise ValueError(f'Unknown baseline: {method}')
    prediction = completed * scale + mean
    prediction[present] = matrix[present]
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite baseline output')
    return prediction.astype(np.float32)


def baseline_grid(method, settings):
    if method == 'column_mean':
        return [{}]
    key, grid = {'knn': ('neighbors', 'knn_neighbors'), 'softimpute': ('shrinkage_fraction', 'softimpute_shrinkage_fractions'),
                 'iterative_svd': ('rank', 'iterative_svd_ranks')}[method]
    return [{key: value} for value in settings[grid]]


def effective_svd_rank(matrix, parameters):
    present = np.isfinite(matrix)
    minimum = min(int(present.any(0).sum()), int(present.any(1).sum()))
    return min(parameters['rank'], minimum - 1) if minimum >= 2 else None


def evaluate_baseline(matrix, method, settings, seed):
    heldout = observed_holdout(matrix, settings['validation_fraction'], seed)
    reduced = matrix.copy()
    reduced[heldout] = np.nan
    trials = []
    started = time.perf_counter()
    grid = baseline_grid(method, settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for parameters in grid:
            if not heldout.any() or len(grid) == 1:
                trials.append({'parameters': parameters, 'validation_mse': None})
                break
            try:
                prediction = baseline_prediction(reduced, method, parameters, seed,
                    settings['max_iters'], settings['tolerance'])
                mse = float(np.mean((prediction[heldout].astype(np.float64) - matrix[heldout]) ** 2))
                if not np.isfinite(mse):
                    raise ValueError('Nonfinite observed-only validation score')
                trials.append({'parameters': parameters, 'validation_mse': mse,
                    'effective_svd_rank': effective_svd_rank(reduced, parameters) if method == 'iterative_svd' else None})
            except Exception as error:
                trials.append({'parameters': parameters, 'validation_mse': None,
                               'error': f'{type(error).__name__}: {error}'})
        valid = [entry for entry in trials if 'error' not in entry]
        if not valid:
            raise ValueError(f'All observed-only baseline candidates failed: {trials}')
        selected = min(valid, key=lambda entry: entry['validation_mse'] if entry['validation_mse'] is not None else 0.)
        prediction = baseline_prediction(matrix, method, selected['parameters'], seed,
            settings['max_iters'], settings['tolerance'])
    return prediction, {'seconds': time.perf_counter() - started, 'selected_parameters': selected['parameters'],
        'effective_svd_rank': effective_svd_rank(matrix, selected['parameters']) if method == 'iterative_svd' else None,
        'validation_cells': int(heldout.sum()), 'validation_mask_sha256': array_digest(heldout),
        'selection_trials': trials, 'warnings': sorted({str(warning.message) for warning in caught}),
        'empty_axis_fallback': 'Observed-column mean; observed-global mean for entirely missing columns; zero without observations',
        'parameter_selection': 'Observed-only holdout; no hidden targets or generating rank supplied'}


def predict_experts(matrix, general, specialist):
    router = TwoExpertRouter(general, specialist, normalization='column', policy='observed_blend')
    result = router.predict_all(matrix, include_holdout=False)
    mapping = {'matrixformer': 'general', 'recsys_specialist': 'specialist', 'matrixformer_plus': 'observed_blend'}
    return {name: result.predictions[key].astype(np.float32) for name, key in mapping.items()}, {
        'decision': result.decisions['observed'], 'expert_seconds': result.seconds,
        'routing_policy': 'observed_blend', 'neural_updates': 0}
