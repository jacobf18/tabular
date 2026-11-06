"""Deterministic synthetic capability and neighborhood evaluation primitives."""

import hashlib
import itertools
import json

import numpy as np
from scipy.spatial.distance import cdist

from synthetic_low_rank_benchmark import baseline_prediction, observed_normalization
from tabimpute.routing import observed_holdout


METHOD_NAMES = {
    'matrixformer': 'MatrixFormer', 'column_mean': 'Column mean',
    'row_knn': 'Row K-NN', 'column_knn': 'Column K-NN', 'two_way_knn': 'Two-way K-NN',
    'iterative_svd': 'IterativeSVD', 'softimpute': 'SoftImpute',
    'oracle_neighbors': 'Oracle neighbors', 'oracle_block': 'Oracle block mean',
}
KNOBS = {'knn': [{'neighbors': count} for count in (3, 5, 10, 20)],
         'iterative_svd': [{'rank': rank} for rank in (2, 4, 8, 16, 32)],
         'softimpute': [{'shrinkage_fraction': value} for value in (.01, .05, .2)]}
MASTER_SEED = 20261103


def protocol(profile='main'):
    conditions = [('low_rank', rank, 0.) for rank in (2, 4, 8, 16, 32)]
    conditions += [('clusters', 4, noise) for noise in (0., .25, 1.)]
    conditions += [('noise', 0, 1.)]
    cases = []
    for (family, rank, noise), observed_fraction, seed in itertools.product(
            conditions, (.2, .4, .6, .8), range(10)):
        case = dict(family=family, rank=rank, noise=noise, observed_fraction=observed_fraction,
                    seed=seed, rows=64, columns=64)
        case['case_id'] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]
        cases.append(case)
    if profile == 'smoke':
        cases = [case for case in cases if case['observed_fraction'] == .4 and case['seed'] == 0
                 and ((case['family'] == 'low_rank' and case['rank'] == 2)
                      or (case['family'] == 'clusters' and case['noise'] == .25)
                      or case['family'] == 'noise')]
    elif profile != 'main':
        raise ValueError(profile)
    return dict(profile=profile, master_seed=MASTER_SEED, cases=cases, knobs=KNOBS,
                solver_iterations=100, solver_tolerance=1e-4, validation_fraction=.1,
                attention_queries=8, intervention_queries=4, removed_donors=3, random_controls=3,
                precision='FP32 weights, BF16 autocast; standard orientation; observed-column scaling',
                attention='Read-only pre-forward hooks; average all heads/layers for donor removal; '
                          'known groups are shuffled before inference; no best-head selection',
                oracle='Only latent distances or cluster memberships are supplied; donor values are observed only',
                uncertainty='95% t intervals across ten matrices within each condition; no cell-level CI',
                selection='No checkpoint training or test-score selection; all registered cases are retained')


def generate(case):
    family_code = {'low_rank': 0, 'clusters': 1, 'noise': 2}[case['family']]
    streams = np.random.SeedSequence([MASTER_SEED, family_code, case['rank'], case['seed']]).spawn(3)
    latent_rng, noise_rng, mask_rng = [np.random.default_rng(stream) for stream in streams]
    rows, columns = case['rows'], case['columns']
    row_labels = np.full(rows, -1)
    column_labels = np.full(columns, -1)
    row_latent, column_latent = np.zeros((rows, 1)), np.zeros((columns, 1))
    if case['family'] == 'low_rank':
        row_latent = latent_rng.normal(size=(rows, case['rank']))
        column_latent = latent_rng.normal(size=(columns, case['rank']))
        signal = row_latent @ column_latent.T / np.sqrt(case['rank'])
    elif case['family'] == 'clusters':
        row_labels = latent_rng.permutation(np.arange(rows) % case['rank'])
        column_labels = latent_rng.permutation(np.arange(columns) % case['rank'])
        block_values = latent_rng.normal(size=(case['rank'], case['rank']))
        signal = block_values[row_labels[:, None], column_labels[None, :]]
    else:
        signal = np.zeros((rows, columns))
    truth = signal + case['noise'] * noise_rng.normal(size=signal.shape)
    observed = np.zeros(signal.size, dtype=bool)
    observed[mask_rng.permutation(signal.size)[:round(signal.size * case['observed_fraction'])]] = True
    observed = observed.reshape(signal.shape)
    return dict(signal=signal.astype(np.float32), truth=truth.astype(np.float32), observed=observed,
                context=np.where(observed, truth, np.nan).astype(np.float32),
                row_labels=row_labels, column_labels=column_labels,
                row_latent=row_latent, column_latent=column_latent)


def selected_attention_case(case):
    return case['observed_fraction'] == .4 and (case['family'] != 'low_rank' or case['rank'] in (2, 16))


def donor_predict(matrix, distances=None, labels=None, neighbors=10):
    matrix = np.asarray(matrix, dtype=np.float64)
    _, means, _ = observed_normalization(matrix)
    prediction = np.where(np.isfinite(matrix), matrix, means)
    for row, column in np.argwhere(~np.isfinite(matrix)):
        eligible = np.flatnonzero(np.isfinite(matrix[:, column]))
        eligible = eligible[eligible != row]
        if labels is not None:
            eligible = eligible[labels[eligible] == labels[row]]
            if eligible.size:
                prediction[row, column] = matrix[eligible, column].mean()
        elif eligible.size:
            order = np.argsort(distances[row, eligible], kind='stable')[:neighbors]
            selected = eligible[order]
            distance = distances[row, selected]
            weights = (distance == 0).astype(float) if np.any(distance == 0) else 1 / np.maximum(distance, 1e-12)
            prediction[row, column] = np.average(matrix[selected, column], weights=weights)
    return prediction.astype(np.float32)


def oracle_prediction(matrix, data, family, neighbors=10, block=False):
    if family == 'noise':
        raise ValueError('Independent noise has no oracle neighborhood')
    if block:
        if family != 'clusters':
            raise ValueError('Block oracle requires clustered data')
        _, means, _ = observed_normalization(matrix)
        prediction = np.where(np.isfinite(matrix), matrix, means)
        for row_group, column_group in itertools.product(np.unique(data['row_labels']), np.unique(data['column_labels'])):
            membership = ((data['row_labels'][:, None] == row_group)
                          & (data['column_labels'][None, :] == column_group))
            donors = matrix[membership & np.isfinite(matrix)]
            if donors.size:
                prediction[membership & ~np.isfinite(matrix)] = donors.mean()
        return prediction.astype(np.float32)
    predictions = []
    for axis, context in (('row', matrix), ('column', matrix.T)):
        if family == 'clusters':
            prediction = donor_predict(context, labels=data[f'{axis}_labels'])
        else:
            distances = cdist(data[f'{axis}_latent'], data[f'{axis}_latent'])
            prediction = donor_predict(context, distances=distances, neighbors=neighbors)
        predictions.append(prediction if axis == 'row' else prediction.T)
    return .5 * (predictions[0] + predictions[1])


def fit_baselines(context, data, case):
    validation = observed_holdout(context, .1, MASTER_SEED + case['seed'])
    reduced = context.copy()
    reduced[validation] = np.nan
    predictions = {'column_mean': baseline_prediction(context, 'column_mean', {}, 0, 100, 1e-4)}
    tuning = {'validation_cells': int(validation.sum()), 'validation_indices': np.flatnonzero(validation).tolist()}
    for method in ('row_knn', 'column_knn', 'iterative_svd', 'softimpute', 'oracle_neighbors'):
        if method == 'oracle_neighbors' and case['family'] == 'noise':
            continue
        primitive = 'knn' if method in ('row_knn', 'column_knn', 'oracle_neighbors') else method
        grid = [{}] if method == 'oracle_neighbors' and case['family'] == 'clusters' else KNOBS[primitive]

        def predict(visible, parameters):
            if method == 'oracle_neighbors':
                return oracle_prediction(visible, data, case['family'], parameters.get('neighbors', 10))
            oriented = visible.T if method == 'column_knn' else visible
            prediction = baseline_prediction(oriented, primitive, parameters, MASTER_SEED + case['seed'], 100, 1e-4)
            return prediction.T if method == 'column_knn' else prediction

        trials = []
        for parameters in grid:
            if not validation.any() or len(grid) == 1:
                trials.append(dict(parameters=parameters, validation_mse=None))
                break
            prediction = predict(reduced, parameters)
            loss = float(np.mean((prediction[validation].astype(float) - context[validation]) ** 2))
            if not np.isfinite(loss):
                raise ValueError(f'Nonfinite validation error: {method}')
            trials.append(dict(parameters=parameters, validation_mse=loss))
        chosen = min(trials, key=lambda trial: trial['validation_mse'] if trial['validation_mse'] is not None else 0)
        predictions[method] = predict(context, chosen['parameters'])
        tuning[method] = dict(chosen=chosen, trials=trials)
    predictions['two_way_knn'] = .5 * (predictions['row_knn'] + predictions['column_knn'])
    if case['family'] == 'clusters':
        predictions['oracle_block'] = oracle_prediction(context, data, case['family'], block=True)
    return predictions, tuning


def scores(prediction, data):
    missing = ~data['observed']
    if prediction.shape != data['truth'].shape or not np.isfinite(prediction).all():
        raise ValueError('Predictions must be complete, finite and shape-matched')
    if not np.array_equal(prediction[data['observed']], data['context'][data['observed']]):
        raise ValueError('Observed entries changed')
    error = prediction[missing].astype(float) - data['truth'][missing]
    signal_error = prediction[missing].astype(float) - data['signal'][missing]
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))),
                signal_rmse=float(np.sqrt(np.mean(signal_error ** 2))), test_cells=int(missing.sum()))


def attention_statistics(weights, targets, data, axis):
    observed = data['observed']
    labels = data[f'{axis}_labels']
    coordinates = targets[:, 0 if axis == 'row' else 1]
    contexts = targets[:, 1 if axis == 'row' else 0]
    visible = observed if axis == 'row' else observed.T
    size = visible.shape[0]
    special = weights.shape[-1] - size
    records = []
    for query, (coordinate, context) in enumerate(zip(coordinates, contexts)):
        eligible = visible[:, context].copy()
        eligible[coordinate] = False
        same = eligible & (labels == labels[coordinate])
        for head in range(weights.shape[1]):
            probabilities = weights[query, head]
            data_weights = probabilities[special:]
            donor_mass = float(data_weights[eligible].sum())
            record = dict(query=query, head=head, donor_mass=donor_mass,
                          special_mass=float(probabilities[:special].sum()),
                          self_mass=float(data_weights[coordinate]),
                          missing_key_mass=float(data_weights[~visible[:, context]].sum()),
                          group_mass=None, group_chance=None, group_excess=None, top5_group_fraction=None)
            if labels[coordinate] >= 0 and eligible.any() and donor_mass > 0:
                group_mass = float(data_weights[same].sum() / donor_mass)
                group_chance = float(same.sum() / eligible.sum())
                top = np.flatnonzero(eligible)[np.argsort(-data_weights[eligible], kind='stable')[:5]]
                record.update(group_mass=group_mass, group_chance=group_chance,
                              group_excess=group_mass - group_chance,
                              top5_group_fraction=float(np.mean(labels[top] == labels[coordinate])))
            records.append(record)
    return records


def matched_random_donors(eligible, selected, coordinate, rng):
    remaining = list(map(int, eligible))
    result = []
    for donor in selected:
        target_band = int(np.log2(1 + abs(int(donor) - coordinate)))
        bands = np.array([int(np.log2(1 + abs(candidate - coordinate))) for candidate in remaining])
        difference = np.abs(bands - target_band)
        candidates = np.flatnonzero(difference == difference.min())
        index = int(rng.choice(candidates))
        result.append(remaining.pop(index))
    return np.asarray(result)
