"""Independently rescore the saved observed-only physics adaptation experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_physics_transforms as experiment


def audit(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    for section in ('source_sha256', 'input_sha256'):
        for name, expected in protocol[section].items():
            assert experiment.shared.sha256(name) == expected, name
    assert experiment.shared.sha256(protocol['checkpoint']) == protocol['checkpoint_sha256']
    paths = sorted((output / 'cases').glob('*.json'))
    expected_cases = {f'{benchmark}_{path.stem}' for benchmark, parent in
                      (('cylinder', experiment.CYLINDER), ('quantum', experiment.QUANTUM))
                      for path in (parent / 'cases').glob('*.json')}
    assert {path.stem for path in paths} == expected_cases
    _, operators = experiment.quantum.pauli_basis(4)
    checked, settings = 0, []
    for path in paths:
        record = json.loads(path.read_text())
        assert experiment.shared.sha256(path.with_suffix('.npz')) == record['arrays_sha256']
        benchmark, case_id = record['benchmark'], record['metadata']['case_id']
        with np.load(path.with_suffix('.npz')) as arrays:
            matrix, validation = arrays['matrix'], arrays['validation']
            assert not (validation & ~np.isfinite(matrix)).any()
            if benchmark == 'cylinder':
                with np.load(experiment.CYLINDER / 'dataset.npz') as original:
                    truth = original['truth']
                with np.load(experiment.CYLINDER / 'predictions' / f'{case_id}.npz') as original:
                    np.testing.assert_array_equal(matrix, np.where(original['missing'], np.nan, truth))
                    np.testing.assert_array_equal(validation, original['validation'])
                _, _, scale = experiment.shared.normalize_observed(matrix)
            else:
                with np.load(experiment.QUANTUM / 'cases' / f'{case_id}.npz') as original:
                    np.testing.assert_array_equal(matrix, original['matrix'])
                observed = np.flatnonzero(np.isfinite(matrix.ravel()))
                observed = observed[observed != 0]
                selected = np.random.default_rng(20261106 + record['metadata']['seed']).choice(
                    observed, size=max(1, len(observed) // 10), replace=False)
                expected_validation = np.zeros_like(validation)
                expected_validation.ravel()[selected] = True
                np.testing.assert_array_equal(validation, expected_validation)
                with np.load(experiment.QUANTUM / 'states' / f"{record['metadata']['state_id']}.npz") as original:
                    truth, density = original['coefficients'], original['density']
            for method, key in record['prediction_keys'].items():
                prediction = arrays[key]
                assert np.isfinite(prediction).all()
                if benchmark == 'cylinder':
                    metrics = experiment.shared.point_metrics(truth, prediction, ~np.isfinite(matrix), scale)
                else:
                    metrics = experiment.quantum.reconstruction_metrics(density, truth, prediction, matrix, operators)
                assert set(metrics) == set(record['metrics'][method])
                for metric, value in metrics.items():
                    np.testing.assert_allclose(value, record['metrics'][method][metric], rtol=1e-10, atol=1e-12)
                    checked += 1
            trials = record['decisions']['trials']
            for method, selected in record['decisions']['selected'].items():
                if benchmark == 'cylinder':
                    eligible = [trial for trial in trials if (trial['transform'] == 'row_mean_control') == (method == 'Low-rank refinement control')]
                    if method == 'MF-Impute input transform':
                        eligible = [trial for trial in eligible if trial['rank'] is None]
                elif method == 'MF-Impute input transform':
                    eligible = record['decisions']['input_trials']
                else:
                    eligible = [trial for trial in trials if (trial['transform'] == 'zero_control') == (method == 'Zero-prior + PSD refinement')]
                best = min(eligible, key=lambda trial: trial['validation_rmse'])
                for key in ('transform', 'rank', 'strength', 'validation_rmse'):
                    assert best.get(key) == selected.get(key)
                settings.append(dict(benchmark=benchmark, case_id=case_id, method=method,
                                     **{key: selected[key] for key in ('transform', 'rank', 'strength', 'validation_rmse') if key in selected}))
    pd.DataFrame(settings).to_csv(output / 'selected_settings.csv', index=False)
    return dict(status='passed', cases=len(paths), metric_values_checked=checked,
                observed_only_validation_masks_verified=True, validation_argmins_verified=True,
                source_and_input_hashes_verified=True, saved_prediction_hashes_verified=True)


def markdown_table(frame, columns, formats=None):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    formats = formats or {}
    for _, row in frame.iterrows():
        values = [formats[name](row[name]) if name in formats else str(row[name]) for name in columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def report(output):
    audited = audit(output)
    experiment.summarize(output)
    comparison = pd.read_csv(output / 'comparison_per_case.csv')
    overall = pd.read_csv(output / 'overall.csv')
    settings = pd.read_csv(output / 'selected_settings.csv')
    text = [
        '# Physics adaptation: MatrixFormer-Impute versus the baselines', '',
        'Exploratory results on the same 18 cylinder and 18 simulated quantum cases. No MRI. '
        'The checkpoint is frozen; these changes are inference-time preprocessing and physical refinement, not training.', '',
        '## Selection protocol', '',
        'Every candidate grid was frozen before this run. Choose settings using RMSE on 10% of the '
        'original observed cells, with all preprocessing refit after masking those cells. Cylinder '
        'uses the archived baseline validation mask; quantum holds out nonidentity observations. '
        'Refit on all original observations, then score the unchanged original hidden targets. '
        'Neither ranks, transforms, nor regularization strengths are selected using test RMSE or fidelity.', '',
        'This is not an independent confirmation set: the original benchmark outcomes motivated '
        'the experiment. Quantum validation sets are small (5 or 12 noisy coefficients), and cylinder '
        'random-cell validation may not predict performance on long sensor outages. '
        'Do not interpret the displayed sample SDs as confidence intervals over independent physical systems.', '',
        '## Cylinder flow', '',
        'Metric: hidden-cell RMSE divided by the global population SD of observed values; lower is better. '
        'Each case receives equal weight. Input transforms subtract each spatial row\'s observed temporal '
        'mean, optionally divide by its observed SD (floor 5% of global SD), and optionally concatenate '
        'one-step shifted copies of time. All copies of a held-out value are masked before lag construction. '
        'Inverse-transformed predictions from the three lag copies are averaged.', '',
        'The hybrid initializes iterative rank-constrained completion with the neural prediction, '
        'then alternates rank projection and exact restoration of observations (up to 40 steps). '
        'Ranks are 2, 4, 8, 16, 32; unrefined neural outputs are also candidates. '
        'The no-neural control uses the same rank grid and iterations initialized with row means. '
        'This is a hybrid algorithm, not a claim that the unmodified network attains its scores.', '',
    ]
    for benchmark in ('cylinder', 'quantum'):
        if benchmark == 'quantum':
            text.extend([
                '## Quantum tomography', '',
                'Metric: squared state fidelity after PSD trace-one reconstruction; higher is better. '
                'The same nine states are evaluated at two missingness rates. '
                'The alternative input removes the fixed identity coefficient from neural normalization '
                'and averages predictions across the three two-qubit/two-qubit tensor unfoldings. '
                'Identity remains known in every physical reconstruction.', '',
                'The hybrid solves observed-coefficient mean squared error plus lambda times the '
                'mean squared deviation of hidden coefficients from the clipped neural prior, '
                'subject to a positive-semidefinite, trace-one density matrix. '
                'Lambda is selected from 0, 0.0001, 0.001, 0.01, 0.1, 1. '
                'The zero-prior control uses the same constraint, objective, solver, and lambda grid. '
                'At lambda=0 the neural prior contributes nothing; improvement from that case cannot '
                'be attributed to MatrixFormer. SCS uses the mean-scaled objective, whereas the archived '
                'PSD baseline used a summed objective; their exact minimizers coincide but numerical '
                'solutions can differ in underdetermined cases.', '',
            ])
        current = overall[overall.benchmark == benchmark].sort_values('mean', ascending=benchmark == 'cylinder').copy()
        assert (current['count'] == 18).all()
        current['Mean ± sample SD'] = current.apply(lambda row: f"{row['mean']:.5f} ± {row['std']:.5f}", axis=1)
        text.extend([markdown_table(current, ['method', 'Mean ± sample SD', 'count']), ''])
        frame = comparison[comparison.benchmark == benchmark]
        metric = 'nrmse' if benchmark == 'cylinder' else 'fidelity'
        hybrid = 'MF-Impute + low-rank refinement' if benchmark == 'cylinder' else 'MF-Impute + PSD refinement'
        comparator = 'SoftImpute' if benchmark == 'cylinder' else 'PSD least squares'
        paired = frame[frame.method.isin([hybrid, comparator])].pivot(index='case_id', columns='method', values=metric)
        assert len(paired) == 18 and not paired.isna().any().any()
        wins = paired[hybrid] < paired[comparator] if benchmark == 'cylinder' else paired[hybrid] > paired[comparator]
        text.extend([f'Observed-validation-selected hybrid beats {comparator} on {int(wins.sum())}/18 cases.', ''])
        groups = ['pattern', 'rate'] if benchmark == 'cylinder' else ['rate']
        methods = ['MatrixFormer-Impute', hybrid, comparator,
                   'Low-rank refinement control' if benchmark == 'cylinder' else 'Zero-prior + PSD refinement']
        summary = frame[frame.method.isin(methods)].groupby(groups + ['method'])[metric].mean().unstack('method').reset_index()
        text.extend([markdown_table(summary, groups + methods, {method: lambda value: f'{value:.5f}' for method in methods}), ''])
    selections = settings.groupby(['benchmark', 'method', 'transform'], dropna=False).size().reset_index(name='Cases')
    text.extend(['## Chosen input transforms', '', markdown_table(selections, list(selections.columns)), '',
                 'Full choices and validation scores: `selected_settings.csv`; all candidate validation '
                 'losses, predictions, masks, physical solver diagnostics and timings: `cases/`. '
                 'Detailed metrics: `comparison_per_case.csv`, `overall.csv`, and the by-condition CSVs.', '',
                 '## Limitations', '',
                 'No uncertainty intervals are propagated through lag averaging, rank refinement, or '
                 'physical constrained optimization. Native model intervals cannot be presented as '
                 'calibrated uncertainty for the refined predictions without a separate analysis. '
                 'The cylinder data are a single noiseless simulated trajectory; quantum data are simulated '
                 'Pauli measurements, not hardware. Further independent states/flows are needed before '
                 'treating these exploratory gains as established generalization.', ''])
    (output / 'RESULTS.md').write_text('\n'.join(text))
    audited['report_sha256'] = experiment.shared.sha256(Path(__file__))
    audited['artifacts_sha256'] = {path.name: experiment.shared.sha256(path) for path in
                                 [output / 'RESULTS.md', *sorted(output.glob('*.csv'))]}
    experiment.shared.write_json(output / 'audit.json', audited)
    experiment.shared.write_json(output / 'status.json', dict(status='complete', cases=36, expected=36, audit='passed'))
    print(overall[['benchmark', 'method', 'mean']].to_string(index=False), flush=True)
    print(f"Audit passed: {audited['metric_values_checked']} metric values rescored", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    report(parser.parse_args().output)
