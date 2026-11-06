"""Rank-preserving exact-SVD sensitivity control, tuned only on observed cells."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from recsys_evaluation_common import atomic_npz, sha256, write_json
from synthetic_matrixformer_study import MASTER_SEED, generate, scores
from tabimpute.routing import observed_holdout


def complete(context, rank, max_iterations=500, tolerance=1e-6):
    observed = np.isfinite(context)
    scale = max(float(np.nanstd(context)), 1e-8)
    original = np.where(observed, context / scale, 0.).astype(float)
    filled = original.copy()
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        left, singular, right = np.linalg.svd(filled, full_matrices=False)
        reconstructed = (left[:, :rank] * singular[:rank]) @ right[:rank]
        change = np.linalg.norm((reconstructed - filled)[~observed])
        previous = max(np.linalg.norm(filled[~observed]), 1e-12)
        filled[~observed] = reconstructed[~observed]
        if change / previous < tolerance:
            break
    prediction = (filled * scale).astype(np.float32)
    prediction[observed] = context[observed]
    return prediction, iterations


def run(output):
    destination = output / 'uncentered_svd'
    destination.mkdir(exist_ok=True)
    manifest = dict(protocol_sha256=sha256(output / 'protocol.json'), source_sha256=sha256(Path(__file__)),
                    ranks=[1, 2, 4, 8, 16, 32], max_iterations=500, relative_change_tolerance=1e-6,
                    normalization='Observed global standard deviation, no centering; original rank preserved',
                    tuning='Same support-preserving 10% observed holdout as primary study; refit on all observations',
                    rationale='Exploratory implementation sensitivity after smoke inspection: per-column centering '
                              'can increase generating rank; retain rather than replace original baselines')
    manifest_file = destination / 'manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise ValueError('SVD control configuration changed')
    write_json(manifest_file, manifest)
    specification = json.loads((output / 'protocol.json').read_text())
    for index, case in enumerate(specification['cases']):
        path = destination / case['case_id']
        if path.with_suffix('.json').exists():
            continue
        started = time.perf_counter()
        data = generate(case)
        context = data['context']
        validation = observed_holdout(context, .1, MASTER_SEED + case['seed'])
        reduced = context.copy()
        reduced[validation] = np.nan
        trials = []
        for rank in manifest['ranks']:
            prediction, iterations = complete(reduced, rank)
            loss = float(np.mean((prediction[validation].astype(float) - context[validation]) ** 2))
            trials.append(dict(rank=rank, validation_mse=loss, iterations=iterations))
        chosen = min(trials, key=lambda trial: trial['validation_mse'])
        prediction, iterations = complete(context, chosen['rank'])
        atomic_npz(path.with_suffix('.npz'), prediction=prediction)
        write_json(path.with_suffix('.json'), dict(**case, method='uncentered_svd', **scores(prediction, data),
                   trials=trials, chosen_rank=chosen['rank'], refit_iterations=iterations,
                   prediction_sha256=sha256(path.with_suffix('.npz'))))
        print(json.dumps(dict(index=index + 1, case=case['case_id'], seconds=time.perf_counter() - started,
                              rank=chosen['rank'], rmse=scores(prediction, data)['rmse'])), flush=True)


def audit(output):
    specification = json.loads((output / 'protocol.json').read_text())
    rows = []
    for case in specification['cases']:
        path = output / 'uncentered_svd' / case['case_id']
        row = json.loads(path.with_suffix('.json').read_text())
        if sha256(path.with_suffix('.npz')) != row['prediction_sha256']:
            raise ValueError('SVD predictions changed')
        with np.load(path.with_suffix('.npz')) as stored:
            for name, value in scores(stored['prediction'], generate(case)).items():
                if row[name] != value:
                    raise ValueError(f'SVD metric mismatch: {name}')
        rows.append({name: value for name, value in row.items() if name != 'trials'})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output)
    print(f'Independently rescored {len(audit(arguments.output))} SVD controls.')
