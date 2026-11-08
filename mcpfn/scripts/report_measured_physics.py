"""Audit saved measured-data predictions and summarize case- and seed-level errors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def audit_run(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    inputs = {}
    for name in protocol['inputs']:
        arrays = np.load(root / name)
        inputs[name.removesuffix('_input.npz')] = {key: arrays[key] for key in arrays.files}
    records = []
    for path in sorted((root / 'records').glob('*.json')):
        record = json.loads(path.read_text())
        identifier = f'{record["dataset"]}_r{round(record["rate"]*100):02}_s{record["seed"]}'
        arrays = inputs[identifier]
        truth, missing, validation = [arrays[key] for key in ('truth', 'missing', 'validation')]
        valid = arrays.get('valid', np.ones(truth.shape, dtype=bool))
        observed = valid & ~missing
        assert not (missing & ~valid).any()
        assert not (validation & ~observed).any()
        prediction = np.load(path.with_suffix('.npz'))['prediction'].astype(float)
        assert np.isfinite(prediction).all()
        np.testing.assert_allclose(prediction[observed], truth[observed], rtol=1e-6, atol=1e-6)
        rmse = float(np.sqrt(np.mean((prediction[missing] - truth[missing]) ** 2)))
        score = rmse / float(np.std(truth[observed]))
        np.testing.assert_allclose(score, record['scores']['nrmse'], rtol=2e-5, atol=1e-7)
        selected = record['tuning']['selected']
        assert selected['validation_rmse'] == min(trial['validation_rmse'] for trial in record['tuning']['trials'])
        records.append(dict(run=str(root), **{key: record[key] for key in ('dataset', 'method', 'rate', 'seed', 'seconds')},
                            **record['scores']))
    expected = 0
    for arrays in inputs.values():
        expected += len(protocol['methods']) - int('TNN' in protocol['methods'] and len(arrays['cube_shape']) != 3)
    return pd.DataFrame(records), dict(path=str(root), expected_records=expected, completed_records=len(records),
                                      complete=expected == len(records), predictions_audited=True)


def seed_summary(frame, groups, metric='nrmse'):
    seed_means = frame.groupby(groups + ['seed'], as_index=False)[metric].mean()
    summary = seed_means.groupby(groups)[metric].agg(mean='mean', sd='std', seeds='count').reset_index()
    summary['ci95_half_width'] = student_t.ppf(.975, summary['seeds'] - 1) * summary['sd'] / np.sqrt(summary['seeds'])
    return summary


def summarize(roots, output):
    output.mkdir(parents=True, exist_ok=True)
    frames, audits = zip(*(audit_run(root) for root in roots))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(['dataset', 'method', 'rate', 'seed']).any():
        raise ValueError('Do not combine runs with duplicate cases')
    frame.to_csv(output / 'scores.csv', index=False)
    by_rate = seed_summary(frame, ['dataset', 'rate', 'method'])
    by_rate.to_csv(output / 'by_rate.csv', index=False)
    seed_summary(frame, ['dataset', 'method']).to_csv(output / 'overall.csv', index=False)
    comparisons = []
    for dataset, subset in frame.groupby('dataset'):
        if not all(audit['complete'] for audit in audits):
            continue
        for scope, selected in [('all_rates', subset), ('sparse_80_95', subset[subset.rate >= .8])]:
            pivot = selected.pivot(index=['rate', 'seed'], columns='method', values='nrmse')
            baselines = [name for name in pivot if not name.startswith('MatrixFormer')]
            if not baselines:
                continue
            oracle = pivot[baselines].min(axis=1)
            strongest = pivot[baselines].mean().idxmin()
            for neural in [name for name in pivot if name.startswith('MatrixFormer')]:
                comparisons.append(dict(dataset=dataset, scope=scope, model=neural,
                                        model_nrmse=float(pivot[neural].mean()), strongest_baseline=strongest,
                                        baseline_nrmse=float(pivot[strongest].mean()),
                                        reduction_percent=float(100*(1-pivot[neural].mean()/pivot[strongest].mean())),
                                        per_case_oracle_baseline_nrmse=float(oracle.mean()),
                                        reduction_vs_oracle_percent=float(100*(1-pivot[neural].mean()/oracle.mean())),
                                        case_wins=int((pivot[neural] < oracle).sum()), cases=len(pivot),
                                        interpretation='Exploratory unless confirmation; no population CI from individual pixels'))
    pd.DataFrame(comparisons).to_csv(output / 'comparisons.csv', index=False)
    (output / 'audit.json').write_text(json.dumps(list(audits), indent=2) + '\n')
    print(by_rate.pivot(index=['dataset', 'rate'], columns='method', values='mean').round(5).to_string())
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    summarize(arguments.runs, arguments.output)
