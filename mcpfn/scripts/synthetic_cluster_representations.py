"""Label-blind representation extraction and clustering for frozen synthetic cases."""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch


CLUSTER_SEED = 20261104
POOLINGS = ('mean', 'profile')
BASELINES = ('observed', 'mask', 'completed', 'oracle_complete')


def compact_profile(grid, axis):
    oriented = grid if axis == 'row' else grid.transpose(0, 1)
    flattened = oriented.reshape(oriented.shape[0], -1).double()
    centered = flattened - flattened.mean(0, keepdim=True)
    gram = (centered @ centered.T).cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    coordinates = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[None, :]
    return coordinates.astype(np.float32)


def pool_grid(output, special_rows, special_columns):
    if output.ndim != 4 or output.shape[0] != 1:
        raise ValueError('Expected a single cell-embedding grid')
    grid = output.detach()[0, special_rows:, special_columns:, :].float()
    if not grid.shape[0] or not grid.shape[1]:
        raise ValueError('No data cells after removing special tokens')
    return {'mean_row': grid.mean(1).cpu().numpy(),
            'mean_column': grid.mean(0).cpu().numpy(),
            'profile_row': compact_profile(grid, 'row'),
            'profile_column': compact_profile(grid, 'column')}


def baseline_features(context, prediction, truth, means, scales):
    observed = np.isfinite(context)
    normalized = (context.astype(float) - means) / (scales + 1e-16)
    completed = (prediction.astype(float) - means) / (scales + 1e-16)
    oracle = (truth.astype(float) - means) / (scales + 1e-16)
    result = {}
    for axis in ('row', 'column'):
        orient = lambda array: array if axis == 'row' else array.T
        mask = orient(observed).astype(np.float32)
        result[f'observed_{axis}_-1'] = np.concatenate(
            [orient(np.nan_to_num(normalized)), mask], axis=1).astype(np.float32)
        result[f'mask_{axis}_-1'] = mask
        result[f'completed_{axis}_-1'] = orient(completed).astype(np.float32)
        result[f'oracle_complete_{axis}_-1'] = orient(oracle).astype(np.float32)
    return result


def fit_clusters(features, clusters=4):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Clustering requires finite features')
    fitted = KMeans(n_clusters=clusters, n_init=20, random_state=CLUSTER_SEED,
                    algorithm='lloyd', max_iter=300).fit(features)
    projection = PCA(n_components=2, svd_solver='full').fit(features)
    return fitted.labels_, projection.transform(features), projection.explained_variance_ratio_


def cluster_metrics(features, assignments, labels):
    labels = np.asarray(labels)
    distance = cdist(np.asarray(features, dtype=float), np.asarray(features, dtype=float))
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind='stable')[:, :5]
    purity = float((labels[neighbors] == labels[:, None]).mean())
    chance = float(((labels[:, None] == labels[None, :]).sum(1) - 1).mean() / (len(labels) - 1))
    generator = np.random.default_rng(CLUSTER_SEED)
    null = [adjusted_rand_score(generator.permutation(labels), assignments) for _ in range(20)]
    return dict(ari=float(adjusted_rand_score(labels, assignments)),
                nmi=float(normalized_mutual_info_score(labels, assignments)),
                knn5_purity=purity, knn5_excess=purity - chance,
                shuffled_ari=float(np.mean(null)))


def attention_probabilities(module, projected_queries, projected_keys, start):
    from tabimpute.model.model_new_stable import _apply_rope_to_subset
    queries = module._reshape_heads(projected_queries)
    keys = module._reshape_heads(projected_keys)
    if module.rope is not None and start < queries.size(2):
        cosine, sine = module.rope.get_cos_sin(queries.size(2) - start, queries.device, queries.dtype)
        queries = _apply_rope_to_subset(queries, cosine, sine, start, module.rotary_dim)
        keys = _apply_rope_to_subset(keys, cosine, sine, start, module.rotary_dim)
    with torch.autocast(device_type=queries.device.type, enabled=False):
        return torch.softmax(queries.float() @ keys.float().transpose(-2, -1)
                             * module.head_dim ** -.5, dim=-1)


def capture(predictor, context, means, scales, target=None):
    model = predictor.imputer.model
    features, attention, handles = {}, {}, []
    projected = {}

    def embeddings(layer):
        def collect(module, arguments, output):
            encoder = model.feature_encoder
            pooled = pool_grid(output, encoder.last_num_cls_rows, encoder.last_num_cls_cols)
            for name, array in pooled.items():
                key = f'{name}_{layer}'
                if key in features:
                    raise ValueError('Unexpected repeated model pass')
                features[key] = array
        return collect

    def projection(layer, axis, name):
        def collect(module, arguments, output):
            projected[layer, axis, name] = output.detach()
        return collect

    def probabilities(layer, axis):
        def collect(module, arguments, kwargs, output):
            source = arguments[0]
            encoder = model.feature_encoder
            special_rows, special_columns = encoder.last_num_cls_rows, encoder.last_num_cls_cols
            special_context = special_columns if axis == 'row' else special_rows
            context_count = context.shape[1] if axis == 'row' else context.shape[0]
            if source.shape[0] != context_count + special_context:
                raise ValueError('Attention examples require an unchunked single matrix')
            start = kwargs.get('rope_start_index', 0)
            weights = attention_probabilities(module, projected.pop((layer, axis, 'query')),
                                              projected.pop((layer, axis, 'key')), start)[special_context:]
            coordinate = int(target[1] if axis == 'row' else target[0])
            attention[f'context_mean_{axis}_{layer}'] = weights.mean(0).cpu().numpy()
            attention[f'fixed_context_{axis}_{layer}'] = weights[coordinate].cpu().numpy()
        return collect

    handles.append(model.feature_encoder.register_forward_hook(embeddings(0)))
    for layer, block in enumerate(model.transformer_blocks, start=1):
        handles.append(block.register_forward_hook(embeddings(layer)))
        if target is not None:
            for axis, module in (('row', block.self_attention_between_datapoints),
                                 ('column', block.self_attention_between_features)):
                handles.append(module.q_proj.register_forward_hook(projection(layer, axis, 'query')))
                handles.append(module.k_proj.register_forward_hook(projection(layer, axis, 'key')))
                handles.append(module.register_forward_hook(probabilities(layer, axis), with_kwargs=True))
    try:
        prediction = predictor.predict(context, means, scales)
    finally:
        for handle in handles:
            handle.remove()
    return prediction, features, attention
