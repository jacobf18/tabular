"""Unpooled cell-embedding probes within fixed columns and fixed rows."""

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score


CLUSTER_SEED = 20261104
LAYERS = (0, 8)
AXES = ('row', 'column')
EXAMPLE_CASE = '7f7a39e928ff6b43'


def data_grid(output, special_rows, special_columns):
    if output.ndim != 4 or output.shape[0] != 1:
        raise ValueError('Expected a single cell-embedding grid')
    grid = output.detach()[0, special_rows:, special_columns:, :].float().cpu().numpy().copy()
    if not all(grid.shape) or not np.isfinite(grid).all():
        raise ValueError('Expected finite data-cell embeddings after special-token removal')
    return grid


def capture(predictor, context, means, scales):
    model = predictor.imputer.model
    grids, handles = {}, []

    def hook(layer):
        def collect(module, arguments, output):
            if layer in grids:
                raise ValueError('Expected one forward pass, without chunking')
            encoder = model.feature_encoder
            grids[layer] = data_grid(output, encoder.last_num_cls_rows, encoder.last_num_cls_cols)
        return collect

    handles.append(model.feature_encoder.register_forward_hook(hook(0)))
    handles.append(model.transformer_blocks[7].register_forward_hook(hook(8)))
    try:
        prediction = predictor.predict(context, means, scales)
    finally:
        for handle in handles:
            handle.remove()
    if set(grids) != set(LAYERS) or any(grid.shape[:2] != context.shape for grid in grids.values()):
        raise ValueError('Incomplete cell-embedding capture')
    return prediction, grids


def slice_features(grid, axis, context_index):
    if axis == 'row':
        return grid[:, context_index, :]
    if axis == 'column':
        return grid[context_index, :, :]
    raise ValueError('Unknown axis')


def predetermined_indices(labels):
    return [int(np.flatnonzero(labels == group)[0]) for group in range(4)]


def fit_slice(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 4 or not np.isfinite(features).all():
        raise ValueError('Clustering requires at least four finite cell vectors')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        fitted = KMeans(n_clusters=4, n_init=20, max_iter=300, algorithm='lloyd',
                        random_state=CLUSTER_SEED).fit(features)
    return fitted.labels_


def score_slice(assignments, labels, observed):
    records = []
    for scope, selected in (('all', np.ones(len(labels), dtype=bool)), ('masked', ~observed)):
        eligible = selected.sum() >= 2 and np.unique(labels[selected]).size >= 2
        records.append(dict(scope=scope, cells=int(selected.sum()), eligible=bool(eligible),
                            fitted_clusters=int(np.unique(assignments).size),
                            ari=float(adjusted_rand_score(labels[selected], assignments[selected]))
                            if eligible else np.nan))
    return records


def evaluate_slice(grid, observed, labels, axis, context_index):
    features = slice_features(grid, axis, context_index)
    assignments = fit_slice(features)
    selected_observed = observed[:, context_index] if axis == 'row' else observed[context_index, :]
    return assignments, score_slice(assignments, labels, selected_observed)
