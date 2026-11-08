"""Audit saved science-panel predictions and summarize window-level uncertainty."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import seaborn as sns

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
import evaluate_science_panels as panels
from fetch_science_panels import ROOT


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None
    return mean, float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def audit(run, methods):
    protocol = json.loads((run / 'protocol.json').read_text())
    rows, analytes, audits = [], [], []
    for case in protocol['cases']:
        arrays_path = run / f'{case["identifier"]}.npz'
        assert shared.sha256(arrays_path) == case['input_sha256']
        arrays = np.load(arrays_path)
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        assert not (missing & validation).any()
        assert np.isfinite(truth[missing]).all() and np.isfinite(truth[validation]).all()
        context = np.where(missing, np.nan, truth)
        observed = np.isfinite(context)
        completed = []
        for method in methods:
            path = run / 'records' / f'{case["identifier"]}_{method.replace(" ", "_")}.json'
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            assert record['protocol_sha256'] == shared.sha256(run / 'protocol.json')
            assert record['input_sha256'] == case['input_sha256']
            assert record['prediction_sha256'] == shared.sha256(path.with_suffix('.npz'))
            prediction = np.load(path.with_suffix('.npz'))['prediction']
            assert prediction.shape == truth.shape and np.isfinite(prediction).all()
            np.testing.assert_array_equal(prediction[observed], context[observed])
            recomputed = original.metrics(truth, prediction, missing, context, case['targets'])
            np.testing.assert_allclose(recomputed['nrmse'], record['scores']['nrmse'], rtol=1e-12)
            trials = record['tuning']['trials']
            assert [trial['setting'] for trial in trials] == protocol['grids'][method]
            assert record['tuning']['selected'] == min(trials, key=lambda trial: trial['validation_nrmse'])
            rows.append(dict(dataset=case['dataset'], split=case['split'], identifier=case['identifier'], method=method,
                             nrmse=recomputed['nrmse'], target_cells=recomputed['target_cells'],
                             target_columns=recomputed['target_columns'], seconds=record['seconds']))
            columns = protocol['datasets'][case['dataset']]['columns']
            for entry in recomputed['per_column']:
                analytes.append(dict(dataset=case['dataset'], split=case['split'], identifier=case['identifier'], method=method,
                                    analyte=columns[entry['column']], **entry))
            completed.append(method)
        audits.append(dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'],
                           complete=set(methods) == set(completed), missing_methods=sorted(set(methods) - set(completed))))
    for dataset in protocol['datasets']:
        windows = [case for case in protocol['cases'] if case['dataset'] == dataset]
        for position, case in enumerate(windows):
            for other in windows[position + 1:]:
                assert abs(case['target_start'] - other['target_start']) >= protocol['datasets'][dataset]['window']
    return pd.DataFrame(rows), pd.DataFrame(analytes), audits


def summary(frame, group, metric):
    result = []
    for key, values in frame.groupby(group):
        mean, half = interval(values[metric])
        result.append(dict(zip(group, key if isinstance(key, tuple) else [key]), metric=metric,
                           mean=mean, ci95_half_width=half, windows=len(values)))
    return pd.DataFrame(result)


def compare(frame, audits, split='confirmation'):
    result = []
    for dataset, current in frame[frame.split == split].groupby('dataset'):
        relevant = [record for record in audits if record['dataset'] == dataset and record['split'] == split]
        if not relevant or not all(record['complete'] for record in relevant):
            continue
        pivot = current.pivot(index='identifier', columns='method', values='nrmse')
        baseline_names = [method for method in pivot if not method.startswith('MatrixFormer')]
        if not baseline_names:
            continue
        baseline = min(baseline_names, key=lambda name: (round(pivot[name].mean(), 12), name != 'Temporal interpolation', name))
        for method in ('MatrixFormer', 'MatrixFormer-Impute'):
            if method not in pivot:
                continue
            differences = pivot[baseline] - pivot[method]
            mean, half = interval(differences)
            result.append(dict(dataset=dataset, model=method, baseline=baseline, windows=len(pivot),
                               model_nrmse=float(pivot[method].mean()), baseline_nrmse=float(pivot[baseline].mean()),
                               relative_reduction_percent=100 * float(1 - pivot[method].mean() / pivot[baseline].mean()),
                               paired_improvement=mean, paired_ci95_half_width=half,
                               wins=int((differences > 0).sum())))
    return pd.DataFrame(result)


def plot(table, output, split='confirmation'):
    current = table[table.split == split]
    sns.set_theme(style='whitegrid', font='STIXGeneral', font_scale=.95)
    figure, axes = plt.subplots(1, current.dataset.nunique(), figsize=(5.2 * current.dataset.nunique(), 4.5), squeeze=False)
    titles = dict(wastewater='Wastewater chemistry', intel_temperature='Indoor sensor temperatures',
                  potomac='Potomac nitrate', susquehanna='Susquehanna nitrate', ocean_alkalinity='Ocean alkalinity')
    for axis, (dataset, values) in zip(axes[0], current.groupby('dataset', sort=False)):
        values = values.sort_values('mean')
        colors = ['#2463A6' if name == 'MatrixFormer' else '#75AADD' if name == 'MatrixFormer-Impute' else '#B5B5B5'
                  for name in values.method]
        axis.barh(values.method, values['mean'], color=colors, edgecolor='none',
                  xerr=values.ci95_half_width.fillna(0), error_kw=dict(ecolor='#333333', capsize=2, lw=.8))
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        unit = 'cruises' if dataset == 'ocean_alkalinity' else 'windows'
        axis.set_title(f'{titles[dataset]}\n{int(values.windows.iloc[0])} held-out {unit}')
        axis.set_xlabel('Normalized RMSE (lower is better)')
        axis.grid(axis='y', visible=False)
        sns.despine(ax=axis)
    figure.tight_layout()
    figure.savefig(output / 'science_panels.pdf', bbox_inches='tight')
    figure.savefig(output / 'science_panels.png', bbox_inches='tight', dpi=180)
    plt.close(figure)


def report(run, methods, output, split='confirmation', datasets=None):
    output.mkdir(parents=True, exist_ok=True)
    frame, analytes, audits = audit(run, methods)
    if datasets:
        frame = frame[frame.dataset.isin(datasets)]
        analytes = analytes[analytes.dataset.isin(datasets)]
        audits = [record for record in audits if record['dataset'] in datasets]
    if frame.empty:
        raise ValueError('No completed records')
    table = summary(frame, ['dataset', 'split', 'method'], 'nrmse')
    comparisons = compare(frame, audits, split)
    frame.to_csv(output / 'per_case.csv', index=False)
    analytes.to_csv(output / 'per_analyte.csv', index=False)
    table.to_csv(output / 'summary.csv', index=False)
    comparisons.to_csv(output / 'comparisons.csv', index=False)
    summary(analytes, ['dataset', 'split', 'method', 'analyte'], 'rmse').to_csv(output / 'native_rmse.csv', index=False)
    shared.write_json(output / 'audit.json', dict(records=len(frame), cases=audits, methods=methods,
                                                complete=all(record['complete'] for record in audits)))
    shared.write_json(output / 'scope.json', dict(run=str(run), methods=methods,
                                                uncertainty='Student-t 95% intervals across nonoverlapping target windows, not cells; within-site temporal dependence remains possible',
                                                comparison='Strongest completed registered baseline by aggregate mean; paired intervals descriptive, not multiplicity-adjusted',
                                                limitation='No native-error pooling across different physical quantities'))
    relevant = [record for record in audits if record['split'] == split]
    if relevant and all(record['complete'] for record in relevant):
        plot(table, output, split)
    print(comparisons.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'replay_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='confirmation')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--methods', nargs='+', choices=panels.METHODS, default=list(panels.SCREEN))
    arguments = parser.parse_args()
    report(arguments.run, arguments.methods, arguments.output, arguments.split, arguments.datasets)
