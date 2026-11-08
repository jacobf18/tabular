"""Observed-validation SDSS PCA control with exact SVD projections and longer fits."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.linalg import eigh

import evaluate_cylinder_completion as shared
import evaluate_sdss_completion as sdss


def complete(matrix, rank, iterations=2000):
    transpose = matrix.shape[0] < matrix.shape[1]
    matrix = matrix.T if transpose else matrix
    observed = np.isfinite(matrix)
    filled = np.nan_to_num(matrix)
    for iteration in range(iterations):
        previous = filled
        active_rank = min(rank, 2 ** min(iteration // 30, 5))
        _, vectors = eigh(filled.T @ filled, subset_by_index=[filled.shape[1] - active_rank, filled.shape[1] - 1], check_finite=False)
        projected = (filled @ vectors) @ vectors.T
        filled = np.where(observed, matrix, projected)
        relative_step = np.linalg.norm(filled - previous) / max(np.linalg.norm(previous), 1)
        if active_rank == rank and relative_step < 1e-6:
            break
    return filled.T if transpose else filled, dict(iterations=iteration + 1, relative_step=float(relative_step),
                                                   exact_projection=True, gradual_rank=True)


def predict(matrix, setting):
    normalized, inverse = sdss.transform_observed(matrix, setting['transform'])
    result, details = complete(normalized, setting['rank'])
    result = inverse(result)
    result[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    return result, details


def run(parent, output):
    if not output.exists():
        original = json.loads((parent / 'protocol.json').read_text())
        output.mkdir(parents=True)
        for name, digest in original['inputs'].items():
            assert shared.sha256(parent / name) == digest
            shutil.copyfile(parent / name, output / name)
        protocol = dict(created=datetime.now(timezone.utc).isoformat(), inputs=original['inputs'], parent=str(parent),
                        methods=['Iterative PCA (extended)'], maximum_iterations=2000,
                        selection='Original rank and transform grid; observed validation only; exact projections with gradual rank increase',
                        source_sha256={str(Path(__file__)): shared.sha256(__file__), str(Path(sdss.__file__)): shared.sha256(sdss.__file__)})
        shared.write_json(output / 'protocol.json', protocol)
        shutil.copyfile(__file__, output / 'source.py')
    protocol = json.loads((output / 'protocol.json').read_text())
    for filename, digest in protocol['source_sha256'].items():
        assert shared.sha256(filename) == digest
    for filename, digest in protocol['inputs'].items():
        assert shared.sha256(output / filename) == digest
        identifier = filename.removesuffix('_input.npz')
        destination = output / 'records' / f'{identifier}_PCA_extended.json'
        if destination.exists():
            continue
        arrays = np.load(output / filename)
        truth, valid, missing, validation = [arrays[key] for key in ('truth', 'valid', 'missing', 'validation')]
        matrix = np.where(valid & ~missing, truth, np.nan)
        reduced = matrix.copy()
        reduced[validation] = np.nan
        started = time.perf_counter()
        print(f'Start {identifier}', flush=True)
        trials = []
        for setting in sdss.candidates('Iterative PCA'):
            result, details = predict(reduced, setting)
            error = float(np.sqrt(np.mean((result[validation] - matrix[validation]) ** 2)))
            trials.append(dict(setting=setting, validation_rmse=error, solver=details))
        chosen = min(trials, key=lambda trial: trial['validation_rmse'])
        result, details = predict(matrix, chosen['setting'])
        scores = shared.point_metrics(truth, result, missing, float(np.nanstd(matrix)))
        dataset, rate, seed = identifier.rsplit('_', 2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination.with_suffix('.npz'), prediction=result.astype(np.float32))
        shared.write_json(destination, dict(dataset=dataset, rate=int(rate[1:])/100, seed=int(seed[1:]),
                                           method='Iterative PCA (extended)', scores=scores,
                                           tuning=dict(selected=chosen, trials=trials, final_solver=details),
                                           seconds=time.perf_counter()-started))
        print(f'Done {identifier}: NRMSE={scores["nrmse"]:.6f}; {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.output)
