"""Observed-validation completion of measured hyperspectral and ARPES data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter
from sklearn.impute import KNNImputer
import torch

import evaluate_cylinder_completion as shared
from evaluate_physics_impute import forward_native
from tabimpute.routing import observed_holdout


TRANSFORMS = ('global', 'column', 'row', 'sqrt')
METHODS = ('Spectral interpolation', 'Gaussian interpolation', 'KNN', 'SoftImpute',
           'HaLRTC', 'TNN', 'MatrixFormer', 'MatrixFormer-Impute')
DATA_ROOT = Path('/mnt/matrixformer_scratch/measured_physics_20261106/data')


def load_measurement(name, data_root=DATA_ROOT, patch_size=32):
    sources = {'indian': ('Indian_pines_corrected.mat', 'indian_pines_corrected'),
               'salinas': ('Salinas_corrected.mat', 'salinas_corrected'),
               'pavia': ('PaviaU.mat', 'paviaU')}
    if name in sources:
        filename, key = sources[name]
        path = data_root / 'HybridSN/data' / filename
        scene = loadmat(path)[key]
        starts = [(dimension - patch_size) // 2 for dimension in scene.shape[:2]]
        cube = scene[starts[0]:starts[0] + patch_size, starts[1]:starts[1] + patch_size].astype(float)
        description = dict(source=str(path), scene_shape=list(scene.shape), spatial_starts=starts,
                           crop='geometrical center, fixed before scoring', axes=['spatial row', 'spatial column', 'band index'])
    else:
        filename = {'arpes_photon': 'photon_energy', 'arpes_temperature': 'temperature_dependence',
                    'arpes_fermi': 'fermi_surface', 'xps': 'nano_xps'}[name]
        path = data_root / 'pyarpes-datasets/data' / f'{filename}.nc'
        with h5py.File(path) as source:
            values = source['__xarray_dataarray_variable__']
            axes = [source[axis[0]].name.lstrip('/') for axis in values.attrs['DIMENSION_LIST']]
            cube = values[:].astype(float)
            if name == 'xps':
                description = dict(source=str(path), axes=axes, selection='entire distributed cube')
            else:
                index = cube.shape[0] // 2
                cube = cube[index].T
                description = dict(source=str(path), axes=[axes[2], axes[1]],
                                   selection=f'central {axes[0]} slice', slice_index=index,
                                   coordinate=float(source[axes[0]][index]))
    if not np.isfinite(cube).all() or cube.min() < 0:
        raise ValueError('Expected finite nonnegative measured intensities')
    description.update(shape=list(cube.shape), sha256=shared.sha256(path))
    return cube, description


def transform_observed(matrix, name):
    values = np.array(matrix, dtype=float, copy=True)
    if name == 'sqrt':
        values = np.sqrt(values)
    axis = {'global': None, 'sqrt': None, 'column': 0, 'row': 1}[name]
    center = np.nanmean(values, axis=axis, keepdims=True)
    scale = np.nanstd(values, axis=axis, keepdims=True)
    fallback_center = np.nanmean(values)
    fallback_scale = max(float(np.nanstd(values)), 1e-8)
    center = np.where(np.isfinite(center), center, fallback_center)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback_scale)

    def inverse(prediction):
        restored = prediction * scale + center
        return np.maximum(restored, 0) ** 2 if name == 'sqrt' else restored

    return (values - center) / scale, inverse


def soft_complete(matrix, shrinkage, iterations=500):
    observed = np.isfinite(matrix)
    filled = np.nan_to_num(matrix)
    threshold = shrinkage * svd(filled, compute_uv=False, check_finite=False)[0]
    extrapolated = filled.copy()
    momentum = 1.
    for iteration in range(iterations):
        previous = filled.copy()
        argument = np.where(observed, matrix, extrapolated)
        left, spectrum, right = svd(argument, full_matrices=False, check_finite=False)
        low_rank = (left * np.maximum(spectrum - threshold, 0)) @ right
        filled = np.where(observed, matrix, low_rank)
        relative_step = np.linalg.norm(filled - previous) / max(np.linalg.norm(previous), 1)
        if relative_step <= 1e-6:
            break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = filled + (momentum - 1) / next_momentum * (filled - previous)
        momentum = next_momentum
    return filled, dict(iterations=iteration + 1, threshold=float(threshold), relative_step=float(relative_step),
                        algorithm='accelerated proximal gradient for observed squared error + nuclear norm')


def nuclear_prox(matrix, threshold):
    left, spectrum, right = np.linalg.svd(matrix, full_matrices=False)
    return (left * np.maximum(spectrum - threshold, 0)[..., None, :]) @ right


def tensor_complete(cube, method, initial_rho, iterations=200):
    observed = np.isfinite(cube)
    filled = np.nan_to_num(cube)
    rho = initial_rho
    if method == 'HaLRTC':
        duals = [np.zeros_like(filled) for axis in range(cube.ndim)]
    else:
        dual = np.zeros_like(filled)
    residual = float('inf')
    for iteration in range(iterations):
        previous = filled.copy()
        if method == 'HaLRTC':
            auxiliaries = []
            for axis in range(cube.ndim):
                moved = np.moveaxis(filled + duals[axis] / rho, axis, 0)
                unfolded = moved.reshape(moved.shape[0], -1)
                projected = nuclear_prox(unfolded, 1 / (cube.ndim * rho))
                auxiliaries.append(np.moveaxis(projected.reshape(moved.shape), 0, axis))
            filled = np.mean([auxiliary - dual / rho for auxiliary, dual in zip(auxiliaries, duals)], axis=0)
            filled[observed] = cube[observed]
            residual = max(float(np.linalg.norm(filled - auxiliary)) for auxiliary in auxiliaries)
            duals = [dual + rho * (filled - auxiliary) for dual, auxiliary in zip(duals, auxiliaries)]
        else:
            fourier = np.fft.rfft(filled + dual / rho, axis=-1)
            projected = nuclear_prox(np.moveaxis(fourier, -1, 0), 1 / rho)
            auxiliary = np.fft.irfft(np.moveaxis(projected, 0, -1), n=cube.shape[-1], axis=-1)
            filled = auxiliary - dual / rho
            filled[observed] = cube[observed]
            residual = float(np.linalg.norm(filled - auxiliary))
            dual += rho * (filled - auxiliary)
        denominator = max(float(np.linalg.norm(previous)), 1)
        if residual / denominator < 1e-5 and np.linalg.norm(filled - previous) / denominator < 1e-5:
            break
        rho = min(rho * 1.05, 1e5)
    return filled, dict(iterations=iteration + 1, relative_residual=residual / denominator)


def spectral_interpolation(matrix):
    result = matrix.copy()
    coordinates = np.arange(matrix.shape[1])
    fallback = float(np.nanmean(matrix))
    for row_index, row in enumerate(matrix):
        observed = np.isfinite(row)
        result[row_index] = np.interp(coordinates, coordinates[observed], row[observed]) if observed.any() else fallback
    return result


def candidates(method, ndim):
    if method.startswith('MatrixFormer'):
        return [dict(transform=name) for name in TRANSFORMS]
    if method == 'SoftImpute':
        return [dict(transform=name, shrinkage=value) for name in TRANSFORMS for value in (.001, .005, .02, .1)]
    if method == 'KNN':
        return [dict(transform=name, neighbors=count) for name in ('global', 'column', 'row') for count in (3, 10, 30)]
    if method == 'Spectral interpolation':
        return [dict(transform=name) for name in ('global', 'sqrt')]
    if method == 'Gaussian interpolation':
        return [dict(transform=name, sigma=[spatial] * (ndim - 1) + [spectral])
                for name in ('global', 'sqrt') for spatial in (1., 2., 4.) for spectral in (1., 3., 7.)]
    return [dict(transform=name, rho=rho) for name in ('global', 'sqrt') for rho in (.01, .1, 1.)]


def predict(matrix, cube_shape, method, setting, models, distribution):
    values, inverse = transform_observed(matrix, setting['transform'])
    details = {}
    if method.startswith('MatrixFormer'):
        predicted, features = forward_native(values, models[method], distribution)
        del features
    elif method == 'SoftImpute':
        predicted, details = soft_complete(values, setting['shrinkage'])
    elif method == 'KNN':
        predicted = KNNImputer(n_neighbors=setting['neighbors'], weights='distance', keep_empty_features=True).fit_transform(values)
    elif method == 'Spectral interpolation':
        predicted = spectral_interpolation(values)
    elif method == 'Gaussian interpolation':
        cube = values.reshape(cube_shape)
        observed = np.isfinite(cube)
        numerator = gaussian_filter(np.nan_to_num(cube), setting['sigma'], mode='nearest')
        denominator = gaussian_filter(observed.astype(float), setting['sigma'], mode='nearest')
        predicted = (numerator / np.maximum(denominator, 1e-12)).reshape(matrix.shape)
    else:
        predicted, details = tensor_complete(values.reshape(cube_shape), method, setting['rho'])
        predicted = predicted.reshape(matrix.shape)
    prediction = np.maximum(inverse(predicted), 0)
    if not np.isfinite(prediction).all():
        raise ValueError(f'Nonfinite {method} prediction')
    prediction[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    return prediction, details


def select(matrix, validation, cube_shape, method, models, distribution):
    reduced = matrix.copy()
    reduced[validation] = np.nan
    trials = []
    for setting in candidates(method, len(cube_shape)):
        prediction, details = predict(reduced, cube_shape, method, setting, models, distribution)
        error = prediction[validation] - matrix[validation]
        trials.append(dict(setting=setting, validation_rmse=float(np.sqrt(np.mean(error ** 2))), solver=details))
    chosen = min(trials, key=lambda trial: trial['validation_rmse'])
    prediction, details = predict(matrix, cube_shape, method, chosen['setting'], models, distribution)
    return prediction, dict(selected=chosen, trials=trials, final_solver=details)


def register(args):
    if args.output.exists():
        raise ValueError('Registration requires a fresh output directory')
    args.output.mkdir(parents=True)
    measurements = {}
    for name in args.datasets:
        cube, description = load_measurement(name, args.data, args.patch_size)
        matrix = cube.reshape(-1, cube.shape[-1])
        measurements[name] = description
        for rate in args.rates:
            for seed in args.seeds:
                random = np.random.default_rng(seed)
                missing = random.random(matrix.shape) < rate
                context = np.where(missing, np.nan, matrix)
                validation = observed_holdout(context, fraction=.1, seed=seed + 104729)
                if not validation.any():
                    raise ValueError('Empty validation mask')
                identifier = f'{name}_r{round(100 * rate):02}_s{seed}'
                np.savez_compressed(args.output / f'{identifier}_input.npz', truth=matrix, missing=missing,
                                    validation=validation, cube_shape=cube.shape)
    checkpoints = {'MatrixFormer': shared.SINGLE, 'MatrixFormer-Impute': shared.GENERAL}
    source_paths = [Path(__file__), Path(shared.__file__), Path(__file__).with_name('evaluate_physics_impute.py')]
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), datasets=args.datasets,
                    rates=args.rates, seeds=args.seeds, measurements=measurements, methods=list(METHODS),
                    settings={method: {str(ndim): candidates(method, ndim) for ndim in (2, 3)} for method in METHODS},
                    inputs={path.name: shared.sha256(path) for path in args.output.glob('*_input.npz')},
                    source_sha256={str(path): shared.sha256(path) for path in source_paths},
                    checkpoints={name: dict(path=str(path), sha256=shared.sha256(path),
                                           config_sha256=shared.sha256(path.with_name('config.json')))
                                 for name, path in checkpoints.items()},
                    outer_mask='independent Bernoulli missingness; unsupported rows/columns are retained',
                    selection='10% of outer observations, support-preserving; all transforms fit after inner masking',
                    metric='RMSE / observed global population SD; equal weight per case',
                    spectral_coordinates='retained-band index, not physical wavelength; corrected HSI has removed bands',
                    training=False, transpose=False, reconstruction='nonnegative and exact observed-value restoration',
                    scope='development unless explicitly designated independent confirmation')
    shared.write_json(args.output / 'protocol.json', protocol)
    shutil.copyfile(__file__, args.output / 'source.py')
    print(f'Registered {len(protocol["inputs"])} cases at {args.output}', flush=True)


def execute(args):
    protocol = json.loads((args.output / 'protocol.json').read_text())
    for name, expected in protocol['source_sha256'].items():
        if shared.sha256(name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    for name, expected in protocol['inputs'].items():
        if shared.sha256(args.output / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    methods = args.methods or protocol['methods']
    if not set(methods) <= set(protocol['methods']):
        raise ValueError('Unregistered method')
    neural = {name: Path(record['path']) for name, record in protocol['checkpoints'].items() if name in methods}
    for name, path in neural.items():
        if shared.sha256(path) != protocol['checkpoints'][name]['sha256']:
            raise ValueError('Checkpoint changed')
        if shared.sha256(path.with_name('config.json')) != protocol['checkpoints'][name]['config_sha256']:
            raise ValueError('Checkpoint configuration changed')
    models, distribution = shared.load_models(neural) if neural else ({}, None)
    for name in protocol['inputs']:
        arrays = np.load(args.output / name)
        truth, missing, validation = [arrays[key] for key in ('truth', 'missing', 'validation')]
        cube_shape = tuple(arrays['cube_shape'])
        matrix = np.where(missing, np.nan, truth)
        scale = float(np.nanstd(matrix))
        identifier = name.removesuffix('_input.npz')
        dataset, rate, seed = identifier.rsplit('_', 2)
        for method in methods:
            if method == 'TNN' and len(cube_shape) != 3:
                continue
            destination = args.output / 'records' / f'{identifier}_{method.replace(" ", "_")}.json'
            if destination.exists():
                continue
            started = time.perf_counter()
            print(f'Start {identifier} {method}', flush=True)
            prediction, tuning = select(matrix, validation, cube_shape, method, models, distribution)
            scores = shared.point_metrics(truth, prediction, missing, scale)
            completed = prediction.copy()
            cosine = np.sum(completed * truth, axis=1) / np.maximum(np.linalg.norm(completed, axis=1) * np.linalg.norm(truth, axis=1), 1e-12)
            scores['spectral_angle_radians'] = float(np.arccos(np.clip(cosine, -1, 1)).mean())
            scores['normalized_mae'] = scores['mae'] / scale
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction.astype(np.float32))
            shared.write_json(destination, dict(dataset=dataset, rate=int(rate[1:]) / 100, seed=int(seed[1:]),
                                               method=method, scores=scores, tuning=tuning,
                                               seconds=time.perf_counter() - started,
                                               missing_cells=int(missing.sum()), validation_cells=int(validation.sum()),
                                               unsupported_rows=int(missing.all(1).sum()), unsupported_columns=int(missing.all(0).sum())))
            print(f'Done {identifier} {method}: NRMSE={scores["nrmse"]:.6f}; {time.perf_counter()-started:.1f}s', flush=True)
    records = [json.loads(path.read_text()) for path in (args.output / 'records').glob('*.json')]
    frame = pd.DataFrame([{**{key: record[key] for key in ('dataset', 'rate', 'seed', 'method', 'seconds')},
                           **record['scores']} for record in records])
    frame.to_csv(args.output / 'scores.csv', index=False)
    print(frame.pivot_table(index=['dataset', 'rate'], columns='method', values='nrmse').round(5).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=DATA_ROOT)
    parser.add_argument('--datasets', nargs='+', default=['indian', 'arpes_photon'])
    parser.add_argument('--rates', nargs='+', type=float, default=[.5, .8, .95])
    parser.add_argument('--seeds', nargs='+', type=int, default=[710])
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    register(arguments) if arguments.action == 'register' else execute(arguments)
