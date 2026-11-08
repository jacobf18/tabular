"""Stricter-convergence SoftImpute controls, selected without outer test targets."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.linalg import eigh

import evaluate_cylinder_completion as shared
import evaluate_measured_physics as measured
import evaluate_sdss_completion as sdss


def gram_prox(matrix, threshold):
    if matrix.shape[0] < matrix.shape[1]:
        return gram_prox(matrix.T, threshold).T
    eigenvalues, vectors = eigh(matrix.T @ matrix, check_finite=False)
    spectrum = np.sqrt(np.maximum(eigenvalues, 0))
    keep = spectrum > threshold
    vectors = vectors[:, keep]
    return ((matrix @ vectors) * (1 - threshold / spectrum[keep])) @ vectors.T


def complete(matrix, shrinkage, iterations=3000):
    observed = np.isfinite(matrix)
    filled = np.nan_to_num(matrix)
    initial_spectrum = np.linalg.svd(filled, compute_uv=False)
    threshold = shrinkage * initial_spectrum[0]
    extrapolated = filled.copy()
    momentum = 1.
    fixed_point_residual = 1.
    for iteration in range(iterations):
        previous = filled
        argument = np.where(observed, matrix, extrapolated)
        low_rank = gram_prox(argument, threshold)
        filled = np.where(observed, matrix, low_rank)
        relative_step = np.linalg.norm(filled - previous) / max(np.linalg.norm(previous), 1)
        if relative_step < 1e-7:
            fixed_point = np.where(observed, matrix, gram_prox(filled, threshold))
            fixed_point_residual = np.linalg.norm(fixed_point - filled) / max(np.linalg.norm(filled), 1)
            if fixed_point_residual < 1e-7:
                break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = filled + (momentum - 1) / next_momentum * (filled - previous)
        momentum = next_momentum
    fixed_point = np.where(observed, matrix, gram_prox(filled, threshold))
    fixed_point_residual = np.linalg.norm(fixed_point - filled) / max(np.linalg.norm(filled), 1)
    return filled, dict(iterations=iteration + 1, relative_step=float(relative_step),
                        fixed_point_residual=float(fixed_point_residual), threshold=float(threshold))


def prediction(matrix, setting, is_sdss):
    transform = sdss.transform_observed if is_sdss else measured.transform_observed
    normalized, inverse = transform(matrix, setting['transform'])
    result, details = complete(normalized, setting['shrinkage'])
    result = inverse(result)
    if not is_sdss:
        result = np.maximum(result, 0)
    result[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    return result, details


def run(parent, output, datasets):
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    if not output.exists():
        output.mkdir(parents=True)
        inputs = {name: expected for name, expected in parent_protocol['inputs'].items()
                  if not datasets or any(name.startswith(dataset + '_r') for dataset in datasets)}
        for name, expected in inputs.items():
            assert shared.sha256(parent / name) == expected
            shutil.copyfile(parent / name, output / name)
        protocol = dict(created=datetime.now(timezone.utc).isoformat(), parent=str(parent), inputs=inputs,
                        methods=['SoftImpute (converged)'], selection='Original grids; stricter solver, observed validation only',
                        tolerance=1e-7, maximum_iterations=3000,
                        source_sha256={str(Path(__file__)): shared.sha256(__file__),
                                       str(Path(measured.__file__)): shared.sha256(measured.__file__),
                                       str(Path(sdss.__file__)): shared.sha256(sdss.__file__)})
        shared.write_json(output / 'protocol.json', protocol)
        shutil.copyfile(__file__, output / 'source.py')
    protocol = json.loads((output / 'protocol.json').read_text())
    for filename, expected in protocol['source_sha256'].items():
        assert shared.sha256(filename) == expected
    for filename, expected in protocol['inputs'].items():
        assert shared.sha256(output / filename) == expected
        identifier = filename.removesuffix('_input.npz')
        destination = output / 'records' / f'{identifier}_SoftImpute_converged.json'
        if destination.exists():
            continue
        arrays = np.load(output / filename)
        truth, missing, validation = [arrays[key] for key in ('truth', 'missing', 'validation')]
        valid = arrays['valid'] if 'valid' in arrays else np.ones(truth.shape, dtype=bool)
        matrix = np.where(valid & ~missing, truth, np.nan)
        reduced = matrix.copy()
        reduced[validation] = np.nan
        is_sdss = identifier.startswith('sdss_')
        settings = sdss.candidates('SoftImpute') if is_sdss else measured.candidates('SoftImpute', len(arrays['cube_shape']))
        print(f'Start {identifier}', flush=True)
        started = time.perf_counter()
        trials = []
        for setting in settings:
            result, details = prediction(reduced, setting, is_sdss)
            score = float(np.sqrt(np.mean((result[validation] - matrix[validation]) ** 2)))
            trials.append(dict(setting=setting, validation_rmse=score, solver=details))
        chosen = min(trials, key=lambda trial: trial['validation_rmse'])
        result, details = prediction(matrix, chosen['setting'], is_sdss)
        scores = shared.point_metrics(truth, result, missing, float(np.nanstd(matrix)))
        dataset, rate, seed = identifier.rsplit('_', 2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination.with_suffix('.npz'), prediction=result.astype(np.float32))
        shared.write_json(destination, dict(dataset=dataset, rate=int(rate[1:])/100, seed=int(seed[1:]),
                                           method='SoftImpute (converged)', scores=scores,
                                           tuning=dict(selected=chosen, trials=trials, final_solver=details),
                                           seconds=time.perf_counter() - started))
        print(f'Done {identifier}: NRMSE={scores["nrmse"]:.6f}; {time.perf_counter()-started:.1f}s; {details}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+')
    args = parser.parse_args()
    run(args.parent, args.output, args.datasets)
