"""Validate saved predictions against frozen inputs and independently recompute NRMSE."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_dwip_missbench_uci import load_original
from grzesiak_benchmark import REPO, WORK, sha256, write_json
from grzesiak_evaluate import cases_for, load_matrix


def audit(work):
    frozen = json.loads((work / 'fresh/input_manifest.json').read_text())
    for relative, expected in frozen.items():
        if sha256(work / relative) != expected:
            raise ValueError(f'Frozen fresh input changed: {relative}')
    score_paths = {'missbench': REPO / 'benchmark/plot_options.py',
                   'uci': REPO / 'benchmark/uci/plot_uci_negative_rmse.py'}
    scorers = {suite: load_original(path, 'compute_normalized_rmse_columnwise') for suite, path in score_paths.items()}
    cases = {case['case_id']: case for scope in ('fresh', 'broad') for case in cases_for(work, scope)}
    counts, maximum = {}, 0.0
    for path in sorted((work / 'evaluations').rglob('*.json')):
        record = json.loads(path.read_text())
        case = cases[record['case_id']]
        for source, checksum in [('truth_path', 'truth_sha256'), ('missing_path', 'missing_sha256')]:
            if sha256(case[source]) != record[checksum]:
                raise ValueError(f'Input checksum mismatch: {path}')
        if record['status'] != 'ok':
            continue
        predicted = work / record['prediction_path']
        if sha256(predicted) != record['prediction_sha256']:
            raise ValueError(f'Prediction checksum mismatch: {path}')
        truth, missing = load_matrix(case['truth_path']), load_matrix(case['missing_path'])
        with np.load(predicted, allow_pickle=False) as saved:
            if tuple(saved['shape']) != truth.shape or saved['missing_values'].size != np.isnan(missing).sum():
                raise ValueError(f'Prediction shape mismatch: {path}')
            prediction = missing.astype(np.float64)
            prediction[np.isnan(missing)] = saved['missing_values']
        expected = float(scorers['missbench' if case['suite'] == 'missbench' else 'uci'](truth, prediction, np.isnan(missing)))
        difference = abs(expected - record['nrmse'])
        maximum = max(maximum, difference)
        if not np.isclose(expected, record['nrmse'], atol=1e-9, rtol=1e-7):
            raise ValueError(f'NRMSE mismatch: {path}: {difference}')
        counts[case['suite']] = counts.get(case['suite'], 0) + 1
    natural = []
    for path in sorted((work / 'natural').glob('*/*.json')):
        record = json.loads(path.read_text())
        if record['status'] != 'ok':
            continue
        if sha256(path.with_suffix('.npz')) != record['prediction_sha256']:
            raise ValueError(f'Natural prediction checksum mismatch: {path}')
        columns = pd.read_csv(path.with_suffix('.columns.csv'))
        recovered = float((columns.score * columns.weight / columns.weight.sum()).sum())
        if abs(recovered - record['energy_iscore']) > 5.1e-5:
            raise ValueError(f'Natural score/column mismatch: {path}')
        natural.append({'dataset': record['dataset'], 'method': record['method'], 'energy_iscore_exact': recovered,
                        'energy_iscore_json': record['energy_iscore'], 'scored_columns': int(columns.score.notna().sum()),
                        'requested_columns': len(columns)})
    pd.DataFrame(natural).to_csv(work / 'natural/audited_scores.csv', index=False)
    result = {'status': 'passed', 'prediction_counts': counts, 'maximum_nrmse_difference': maximum,
              'natural_scores_checked': len(natural), 'energy_distance_validation': 'Python/R numerical parity regression test',
              'natural_precision': 'Full precision recovered from archived column contributions; initial R JSON serialization rounded to four decimals.',
              'nrmse_original_source_sha256': {suite: sha256(path) for suite, path in score_paths.items()}}
    write_json(work / 'audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    audit(parser.parse_args().work)
