"""Completion of real, quality-masked SDSS galaxy flux spectra distributed by astroML."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.impute import KNNImputer
from sklearn.utils.extmath import randomized_svd
import torch

import evaluate_cylinder_completion as shared
import evaluate_measured_physics as measured
from evaluate_physics_impute import forward_native
from tabimpute.routing import observed_holdout


DATA = measured.DATA_ROOT / 'sdss/spec4000.npz'
TRANSFORMS = ('global', 'column', 'row', 'row_flux', 'asinh')
METHODS = ('Spectral interpolation', 'Spectral smoothing', 'KNN', 'SoftImpute', 'Iterative PCA',
           'MatrixFormer', 'MatrixFormer-Impute')


def transform_observed(matrix, name):
    if name in ('global', 'column', 'row'):
        return measured.transform_observed(matrix, name)
    if name == 'row_flux':
        flux = np.nanmean(matrix, axis=1, keepdims=True)
        fallback = max(float(np.nanmedian(np.abs(matrix))), 1e-8)
        flux = np.where(np.isfinite(flux) & (np.abs(flux) > 1e-8), flux, fallback)
        return matrix / flux, lambda values: values * flux
    scale = max(float(np.nanmedian(np.abs(matrix))), 1e-8)
    return np.arcsinh(matrix / scale), lambda values: np.sinh(values) * scale


def candidates(method):
    if method.startswith('MatrixFormer'):
        return [dict(transform=name) for name in TRANSFORMS]
    if method == 'SoftImpute':
        return [dict(transform=name, shrinkage=value) for name in TRANSFORMS for value in (.001, .005, .02, .1)]
    if method == 'Iterative PCA':
        return [dict(transform=name, rank=rank) for name in TRANSFORMS for rank in (2, 4, 8, 16, 32)]
    if method == 'KNN':
        return [dict(transform=name, neighbors=count) for name in TRANSFORMS for count in (3, 10, 30)]
    if method == 'Spectral smoothing':
        return [dict(transform=name, sigma=sigma) for name in ('global', 'asinh') for sigma in (1., 2., 4., 8.)]
    return [dict(transform=name) for name in ('global', 'asinh')]


def iterative_pca(matrix, rank, iterations=200):
    observed = np.isfinite(matrix)
    filled = np.nan_to_num(matrix)
    for iteration in range(iterations):
        previous = filled.copy()
        left, spectrum, right = randomized_svd(filled, n_components=rank, n_iter=4, random_state=11)
        filled = np.where(observed, matrix, (left * spectrum) @ right)
        relative_step = np.linalg.norm(filled - previous) / max(np.linalg.norm(previous), 1)
        if relative_step < 1e-5:
            break
    return filled, dict(iterations=iteration + 1, relative_step=float(relative_step))


def predict(matrix, method, setting, models, distribution):
    values, inverse = transform_observed(matrix, setting['transform'])
    details = {}
    if method.startswith('MatrixFormer'):
        prediction, features = forward_native(values, models[method], distribution)
        del features
    elif method == 'SoftImpute':
        prediction, details = measured.soft_complete(values, setting['shrinkage'])
    elif method == 'Iterative PCA':
        prediction, details = iterative_pca(values, setting['rank'])
    elif method == 'KNN':
        prediction = KNNImputer(n_neighbors=setting['neighbors'], weights='distance', keep_empty_features=True).fit_transform(values)
    elif method == 'Spectral interpolation':
        prediction = measured.spectral_interpolation(values)
    else:
        observed = np.isfinite(values)
        numerator = gaussian_filter1d(np.nan_to_num(values), setting['sigma'], axis=1, mode='nearest')
        denominator = gaussian_filter1d(observed.astype(float), setting['sigma'], axis=1, mode='nearest')
        prediction = numerator / np.maximum(denominator, 1e-12)
    result = inverse(prediction)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite predicted flux')
    result[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    return result, details


def choose(matrix, validation, method, models, distribution):
    reduced = matrix.copy()
    reduced[validation] = np.nan
    trials = []
    for setting in candidates(method):
        prediction, details = predict(reduced, method, setting, models, distribution)
        score = float(np.sqrt(np.mean((prediction[validation] - matrix[validation]) ** 2)))
        trials.append(dict(setting=setting, validation_rmse=score, solver=details))
    selected = min(trials, key=lambda trial: trial['validation_rmse'])
    prediction, details = predict(matrix, method, selected['setting'], models, distribution)
    return prediction, dict(selected=selected, trials=trials, final_solver=details)


def make_split(data, split, rows=256):
    band_indices = np.arange(100, 900, 2)
    flux = data['spectra'][:, band_indices].astype(float)
    valid = ~data['mask'][:, band_indices] & np.isfinite(flux)
    eligible = np.flatnonzero(valid.mean(axis=1) >= .9)
    shuffled = np.random.default_rng(860).permutation(eligible)
    offset = 0 if split == 'development' else rows
    selected = shuffled[offset:offset + rows]
    if len(selected) != rows:
        raise ValueError('Insufficient measured spectra')
    return flux[selected], valid[selected], selected, band_indices


def register(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    data = np.load(DATA)
    truth, valid, rows, bands = make_split(data, args.split)
    for rate in args.rates:
        for seed in args.seeds:
            missing = (np.random.default_rng(seed).random(truth.shape) < rate) & valid
            context = np.where(valid & ~missing, truth, np.nan)
            validation = observed_holdout(context, fraction=.1, seed=seed + 104729)
            np.savez_compressed(args.output / f'sdss_r{round(rate*100):02}_s{seed}_input.npz',
                                truth=truth, valid=valid, missing=missing, validation=validation,
                                row_indices=rows, band_indices=bands)
    sources = [Path(__file__), Path(measured.__file__), Path(shared.__file__), Path(__file__).with_name('evaluate_physics_impute.py')]
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), data_sha256=shared.sha256(DATA),
                    split=args.split, shape=list(truth.shape), object_indices=rows.tolist(), band_indices=bands.tolist(),
                    measurement='Actual rest-frame rebinned galaxy spectra; source-quality-masked entries are never targets or observations',
                    forbidden='No source PCA coefficients, eigenvectors, norms, imputed flux, class labels or synthetic noise',
                    methods=list(METHODS), settings={method: candidates(method) for method in METHODS},
                    source_sha256={str(path): shared.sha256(path) for path in sources},
                    inputs={path.name: shared.sha256(path) for path in args.output.glob('*_input.npz')},
                    checkpoints={name: dict(path=str(path), sha256=shared.sha256(path),
                                           config_sha256=shared.sha256(path.with_name('config.json')))
                                 for name, path in [('MatrixFormer', shared.SINGLE), ('MatrixFormer-Impute', shared.GENERAL)]},
                    tuning='Only 10% of actual observed entries; transforms fit after inner holdout',
                    primary_metric='Outer-held-out flux RMSE / observed global population SD',
                    secondary='Mean per-object RMSE / observed per-object RMS flux; negative measured flux retained',
                    confirmation='Disjoint objects from the same survey, not an independent instrument', training=False)
    shared.write_json(args.output / 'protocol.json', protocol)
    shutil.copyfile(__file__, args.output / 'source.py')
    print(f'Registered {len(protocol["inputs"])} SDSS {args.split} cases at {args.output}', flush=True)


def execute(args):
    protocol = json.loads((args.output / 'protocol.json').read_text())
    for filename, expected in protocol['source_sha256'].items():
        assert shared.sha256(filename) == expected, filename
    assert shared.sha256(DATA) == protocol['data_sha256']
    methods = args.methods or protocol['methods']
    checkpoints = {name: Path(record['path']) for name, record in protocol['checkpoints'].items() if name in methods}
    for name, path in checkpoints.items():
        assert shared.sha256(path) == protocol['checkpoints'][name]['sha256']
        assert shared.sha256(path.with_name('config.json')) == protocol['checkpoints'][name]['config_sha256']
    models, distribution = shared.load_models(checkpoints) if checkpoints else ({}, None)
    for filename, expected in protocol['inputs'].items():
        assert shared.sha256(args.output / filename) == expected
        arrays = np.load(args.output / filename)
        truth, valid, missing, validation = [arrays[key] for key in ('truth', 'valid', 'missing', 'validation')]
        matrix = np.where(valid & ~missing, truth, np.nan)
        identifier = filename.removesuffix('_input.npz')
        _, rate, seed = identifier.rsplit('_', 2)
        for method in methods:
            destination = args.output / 'records' / f'{identifier}_{method.replace(" ", "_")}.json'
            if destination.exists():
                continue
            print(f'Start {identifier} {method}', flush=True)
            started = time.perf_counter()
            prediction, tuning = choose(matrix, validation, method, models, distribution)
            scores = shared.point_metrics(truth, prediction, missing, float(np.nanstd(matrix)))
            row_rmse = np.sqrt(np.sum(np.where(missing, (prediction - truth) ** 2, 0), axis=1) / missing.sum(axis=1))
            row_scale = np.sqrt(np.nanmean(matrix ** 2, axis=1))
            scores['mean_per_object_normalized_rmse'] = float(np.mean(row_rmse / np.maximum(row_scale, 1e-8)))
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction.astype(np.float32))
            shared.write_json(destination, dict(dataset='sdss', method=method, rate=int(rate[1:])/100, seed=int(seed[1:]),
                                               scores=scores, tuning=tuning, seconds=time.perf_counter()-started))
            print(f'Done {identifier} {method}: NRMSE={scores["nrmse"]:.6f}; {time.perf_counter()-started:.1f}s', flush=True)
    records = [json.loads(path.read_text()) for path in (args.output / 'records').glob('*.json')]
    frame = pd.DataFrame([{**{key: record[key] for key in ('dataset', 'rate', 'seed', 'method', 'seconds')}, **record['scores']}
                          for record in records])
    frame.to_csv(args.output / 'scores.csv', index=False)
    print(frame.pivot_table(index='rate', columns='method', values='nrmse').round(5).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--rates', nargs='+', type=float, default=[.5, .8, .95])
    parser.add_argument('--seeds', nargs='+', type=int, default=[710])
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    register(arguments) if arguments.action == 'register' else execute(arguments)
