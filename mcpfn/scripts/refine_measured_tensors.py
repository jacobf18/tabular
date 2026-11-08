"""Extend tensor-completion ADMM initialization toward the reference defaults."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

import evaluate_cylinder_completion as shared
import evaluate_measured_physics as measured


def predict(matrix, shape, method, setting):
    normalized, inverse = measured.transform_observed(matrix, setting['transform'])
    result, details = measured.tensor_complete(normalized.reshape(shape), method, setting['rho'], iterations=500)
    result = np.maximum(inverse(result.reshape(matrix.shape)), 0)
    result[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    return result, details


def run(parent, output, datasets):
    if not output.exists():
        original = json.loads((parent / 'protocol.json').read_text())
        output.mkdir(parents=True)
        inputs = {name: digest for name, digest in original['inputs'].items()
                  if not datasets or any(name.startswith(dataset + '_r') for dataset in datasets)}
        for name, digest in inputs.items():
            assert shared.sha256(parent / name) == digest
            shutil.copyfile(parent / name, output / name)
        protocol = dict(created=datetime.now(timezone.utc).isoformat(), inputs=inputs, parent=str(parent),
                        methods=['HaLRTC (extended)', 'TNN (extended)'],
                        rationale='Selected original ADMM initialization was on the grid boundary; add reference-scale 1e-4 and 1e-3, retaining 1e-2 and 500 iterations',
                        source_sha256={str(Path(__file__)): shared.sha256(__file__),
                                       str(Path(measured.__file__)): shared.sha256(measured.__file__)})
        shared.write_json(output / 'protocol.json', protocol)
        shutil.copyfile(__file__, output / 'source.py')
    protocol = json.loads((output / 'protocol.json').read_text())
    for filename, digest in protocol['source_sha256'].items():
        assert shared.sha256(filename) == digest
    for filename, digest in protocol['inputs'].items():
        assert shared.sha256(output / filename) == digest
        arrays = np.load(output / filename)
        truth, missing, validation = [arrays[key] for key in ('truth', 'missing', 'validation')]
        shape = tuple(arrays['cube_shape'])
        matrix = np.where(missing, np.nan, truth)
        reduced = matrix.copy()
        reduced[validation] = np.nan
        identifier = filename.removesuffix('_input.npz')
        dataset, rate, seed = identifier.rsplit('_', 2)
        for method in ('HaLRTC', 'TNN'):
            destination = output / 'records' / f'{identifier}_{method}.json'
            if destination.exists():
                continue
            started = time.perf_counter()
            trials = []
            print(f'Start {identifier} {method}', flush=True)
            for transform in ('global', 'sqrt'):
                for rho in (.0001, .001, .01):
                    setting = dict(transform=transform, rho=rho)
                    prediction, details = predict(reduced, shape, method, setting)
                    error = float(np.sqrt(np.mean((prediction[validation] - matrix[validation]) ** 2)))
                    trials.append(dict(setting=setting, validation_rmse=error, solver=details))
            chosen = min(trials, key=lambda trial: trial['validation_rmse'])
            prediction, details = predict(matrix, shape, method, chosen['setting'])
            scores = shared.point_metrics(truth, prediction, missing, float(np.nanstd(matrix)))
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction.astype(np.float32))
            shared.write_json(destination, dict(dataset=dataset, rate=int(rate[1:])/100, seed=int(seed[1:]),
                                               method=method + ' (extended)', scores=scores,
                                               tuning=dict(selected=chosen, trials=trials, final_solver=details),
                                               seconds=time.perf_counter()-started))
            print(f'Done {identifier} {method}: NRMSE={scores["nrmse"]:.6f}; {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+')
    args = parser.parse_args()
    run(args.parent, args.output, args.datasets)
