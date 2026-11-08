"""Audit saved realistic-science predictions and report complete comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as evaluation


def summarize(frame, metric):
    if metric.startswith('pooled_'):
        frame = frame[frame.dataset == 'beijing']
    if frame.empty or metric not in frame:
        return pd.DataFrame(columns=['dataset', 'split', 'method', 'mean', 'sd', 'windows', 'ci95_half_width', 'metric'])
    result = frame.groupby(['dataset', 'split', 'method'])[metric].agg(mean='mean', sd='std', windows='count').reset_index()
    result['ci95_half_width'] = student_t.ppf(.975, result.windows - 1) * result.sd / np.sqrt(result.windows)
    result['metric'] = metric
    return result


def audit(root, controls=()):
    protocol = json.loads((root / 'protocol.json').read_text())
    for filename, expected in protocol['source_sha256'].items():
        assert shared.sha256(filename) == expected
    method_sources = {method: root for method in protocol['methods']}
    for control in controls:
        control_protocol = json.loads((control / 'protocol.json').read_text())
        assert Path(control_protocol['parent']) == root
        assert shared.sha256(root / 'protocol.json') == control_protocol['parent_protocol_sha256']
        for filename, expected in control_protocol['source_sha256'].items():
            assert shared.sha256(filename) == expected
        for method in control_protocol['methods']:
            assert method not in method_sources
            method_sources[method] = control
    rows, physical, audits = [], [], []
    for case in protocol['cases']:
        source = root / f'{case["identifier"]}.npz'
        assert shared.sha256(source) == case['input_sha256']
        arrays = np.load(source)
        truth, missing, validation = [arrays[key] for key in ('truth', 'missing', 'validation')]
        assert np.isfinite(truth[missing]).all()
        assert np.isfinite(truth[validation]).all()
        assert not (missing & validation).any()
        context = np.where(missing, np.nan, truth)
        observed = np.isfinite(context)
        metadata = protocol['datasets'][case['dataset']]
        methods = [method for method in method_sources
                   if method != 'Spatial residual interpolation' or case['dataset'] == 'beijing']
        completed = []
        for method in methods:
            path = method_sources[method] / 'records' / f'{case["identifier"]}_{method.replace(" ", "_")}.json'
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            prediction = np.load(path.with_suffix('.npz'))['prediction'].astype(float)
            assert np.isfinite(prediction).all()
            np.testing.assert_allclose(prediction[observed], context[observed], rtol=1e-6, atol=1e-6)
            scores = evaluation.metrics(truth, prediction, missing, context, metadata['targets'])
            for metric in ('nrmse', 'pooled_rmse', 'pooled_mae', 'mean_rmse', 'mean_mae'):
                np.testing.assert_allclose(scores[metric], record['scores'][metric], rtol=2e-5, atol=1e-6)
            tuning = record['tuning']
            assert tuning['selected']['validation_nrmse'] == min(trial['validation_nrmse'] for trial in tuning['trials'])
            identity = dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'], method=method)
            excluded = {'per_column'}
            if case['dataset'] == 'air_quality':
                excluded.update(('pooled_rmse', 'pooled_mae', 'mean_rmse', 'mean_mae'))
            rows.append(dict(**identity, **{key: value for key, value in scores.items() if key not in excluded},
                             seconds=record['seconds'], target_date=case['target_date'],
                             donor_missing_fraction=case['donor_missing_fraction'],
                             artificial_hidden_fraction=case['artificial_hidden_fraction'],
                             chosen_transform=tuning['selected']['setting']['transform']))
            for measurement in scores['per_column']:
                physical.append(dict(**identity, **measurement, name=metadata['columns'][measurement['column']]))
            completed.append(method)
        audits.append(dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'],
                           expected=len(methods), completed=len(completed), complete=set(methods) == set(completed),
                           predictions_audited=True))
    return pd.DataFrame(rows), pd.DataFrame(physical), pd.DataFrame(audits)


def compare(frame, audits):
    rows = []
    for (dataset, split), group in frame.groupby(['dataset', 'split']):
        coverage = audits[(audits.dataset == dataset) & (audits.split == split)]
        if not coverage.complete.all():
            continue
        for metric in ('nrmse', 'pooled_rmse'):
            if metric == 'pooled_rmse' and dataset != 'beijing':
                continue
            pivot = group.pivot(index='identifier', columns='method', values=metric)
            baselines = [name for name in pivot.columns if not name.startswith('MatrixFormer')]
            strongest = pivot[baselines].mean().idxmin()
            oracle = pivot[baselines].min(axis=1)
            for model in ('MatrixFormer', 'MatrixFormer-Impute'):
                delta = pivot[strongest] - pivot[model]
                half_width = student_t.ppf(.975, len(delta) - 1) * delta.std() / np.sqrt(len(delta))
                rows.append(dict(dataset=dataset, split=split, metric=metric, model=model,
                                 model_mean=float(pivot[model].mean()), baseline=strongest,
                                 baseline_mean=float(pivot[strongest].mean()),
                                 reduction_percent=float(100 * (1 - pivot[model].mean() / pivot[strongest].mean())),
                                 paired_difference=float(delta.mean()), paired_ci95_half_width=float(half_width),
                                 wins=int((delta > 0).sum()), windows=len(delta),
                                 oracle_mean=float(oracle.mean()),
                                 reduction_vs_oracle_percent=float(100 * (1 - pivot[model].mean() / oracle.mean()))))
    return pd.DataFrame(rows)


def plot(summary, audits, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='whitegrid', font='STIXGeneral', font_scale=.9)
    plt.rcParams.update({'pdf.fonttype': 42, 'mathtext.fontset': 'stix'})
    for (dataset, split), group in summary.groupby(['dataset', 'split']):
        coverage = audits[(audits.dataset == dataset) & (audits.split == split)]
        if not coverage.complete.all():
            continue
        data = group[group.metric == 'nrmse'].sort_values('mean')
        figure, axis = plt.subplots(figsize=(6.6, .27 * len(data) + 1.3))
        colors = ['#2A6FBB' if method.startswith('MatrixFormer') else '#b9bec8' for method in data.method]
        axis.barh(data.method, data['mean'], xerr=data.ci95_half_width, color=colors, capsize=2)
        axis.invert_yaxis()
        axis.set_xlabel('Normalized RMSE ↓')
        title = 'Italian chemical sensors' if dataset == 'air_quality' else 'Beijing monitoring network'
        axis.set_title(f'{title} · {split}\nRecorded gap masks')
        axis.grid(axis='y', visible=False)
        axis.set_xlim(left=0)
        sns.despine(ax=axis)
        figure.tight_layout()
        for extension in ('pdf', 'png'):
            figure.savefig(output / f'{dataset}_{split}.{extension}', dpi=180)
        plt.close(figure)


def main(root, output, controls=(), datasets=None):
    output.mkdir(parents=True, exist_ok=True)
    frame, physical, audits = audit(root, controls)
    if datasets:
        frame = frame[frame.dataset.isin(datasets)]
        physical = physical[physical.dataset.isin(datasets)]
        audits = audits[audits.dataset.isin(datasets)]
    audits.to_json(output / 'audit.json', orient='records', indent=2)
    frame.to_csv(output / 'per_case.csv', index=False)
    physical.to_csv(output / 'per_analyte.csv', index=False)
    if frame.empty:
        return
    summaries = [summarize(frame, metric) for metric in ('nrmse', 'pooled_rmse', 'pooled_mae')]
    summary = pd.concat([table for table in summaries if not table.empty], ignore_index=True)
    summary.to_csv(output / 'summary.csv', index=False)
    comparisons = compare(frame, audits)
    comparisons.to_csv(output / 'comparisons.csv', index=False)
    plot(summary, audits, output)
    print(frame.pivot_table(index=['dataset', 'split', 'identifier'], columns='method', values='nrmse').round(4).to_string())
    print('\nComplete comparisons only:\n', comparisons.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls', nargs='+', type=Path, default=[])
    parser.add_argument('--datasets', nargs='+', choices=list(evaluation.PLANS))
    arguments = parser.parse_args()
    main(arguments.run, arguments.output, arguments.controls, arguments.datasets)
