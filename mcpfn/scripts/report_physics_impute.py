"""Audit cylinder and quantum specialist predictions against archived baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_physics_impute as evaluation


PRIMARY = {'cylinder': 'nrmse', 'quantum': 'fidelity', 'synthetic_mri': 'complex_image_nrmse',
           'real_mri': 'rss_image_nrmse'}
REPORT_BENCHMARKS = ('cylinder', 'quantum')


def check_metrics(actual, expected):
    if set(actual) != set(expected):
        raise ValueError(f'Metric fields differ: {set(actual) ^ set(expected)}')
    for name, value in actual.items():
        if isinstance(value, str):
            assert value == expected[name]
        else:
            np.testing.assert_allclose(value, expected[name], rtol=1e-10, atol=1e-12, err_msg=name)
    return len(actual)


def audit_case(case, record, arrays, reference):
    benchmark = case['benchmark']
    parent_record = json.loads(Path(case['parent_record']).read_text())
    checked = 0
    with np.load(case['parent_artifact']) as parent:
        if benchmark in ('cylinder', 'quantum'):
            if benchmark == 'cylinder':
                truth = reference['truth']
                np.testing.assert_array_equal(arrays['missing'], parent['missing'])
                np.testing.assert_array_equal(arrays['uncertainty_indices'], parent['uncertainty_indices'])
                matrix = np.where(parent['missing'], np.nan, truth)
                _, _, scale = evaluation.shared.normalize_observed(matrix)
                scores = evaluation.shared.point_metrics(truth, arrays['prediction'], parent['missing'], scale)
            else:
                np.testing.assert_array_equal(arrays['matrix'], parent['matrix'])
                np.testing.assert_array_equal(arrays['counts'], parent['counts'])
                np.testing.assert_array_equal(arrays['uncertainty_indices'], parent['hidden_indices'])
                matrix = parent['matrix']
                with np.load(evaluation.PARENTS[benchmark] / 'states' / f"{parent_record['state_id']}.npz") as state:
                    truth, density = state['coefficients'], state['density']
                scores, reconstructed = evaluation.quantum_scores(density, truth, matrix, arrays['prediction'], reference['operators'])
                for key, values in reconstructed.items():
                    np.testing.assert_allclose(arrays[key], values, rtol=0, atol=1e-12)
                _, _, scale = evaluation.shared.normalize_observed(matrix)
            native = {key: arrays[f'uncertainty_{key}'] for key in ('crps', 'nll', 'quantiles', 'std')}
            indices = arrays['uncertainty_indices']
            assert np.isfinite(native['quantiles']).all()
            assert np.all(np.diff(native['quantiles'], axis=1) >= 0)
            assert (native['std'] >= 0).all()
            scores.update(evaluation.uncertainty_scores(native, truth.ravel()[indices], scale))
            if benchmark == 'quantum':
                bounds = native['quantiles']
                scores['interval_90_outside_physical_range'] = float(((bounds[:, 1] < -1) | (bounds[:, 4] > 1)).mean())
            assert set(record['metrics']) == {'MatrixFormer-Impute'}
            checked += check_metrics(scores, record['metrics']['MatrixFormer-Impute'])
        else:
            np.testing.assert_array_equal(arrays['acquired'], parent['acquired'])
            acquired = arrays['acquired']
            assert set(record['metrics']) == {'MatrixFormer-Impute raw', 'MatrixFormer-Impute S'}
            for key, method in (('raw', 'MatrixFormer-Impute raw'), ('lifted', 'MatrixFormer-Impute S')):
                prediction = arrays[f'{key}_kspace']
                truth = reference['truth']
                np.testing.assert_array_equal(prediction[acquired], truth[acquired])
                if benchmark == 'synthetic_mri':
                    scores = evaluation.synthetic.score_reconstruction(reference['image'], truth, prediction, acquired)
                else:
                    scores = evaluation.real.score_reconstruction(truth, prediction, acquired)
                checked += check_metrics(scores, record['metrics'][method])
    return checked


def paired_comparison(comparison, benchmark):
    metric = PRIMARY[benchmark]
    records = []
    pairs = [('MatrixFormer-Impute', 'MatrixFormer')] if benchmark in ('cylinder', 'quantum') else [
        ('MatrixFormer-Impute raw', 'MatrixFormer raw'), ('MatrixFormer-Impute S', 'MatrixFormer S')]
    for specialist, combined in pairs:
        selected = comparison[comparison.method.isin([specialist, combined])]
        pivot = selected.pivot(index='case_id', columns='method', values=metric)
        if pivot.isna().any().any() or len(pivot) != evaluation.COUNTS[benchmark]:
            raise ValueError('Incomplete paired comparison')
        difference = pivot[specialist] - pivot[combined]
        wins = difference > 0 if benchmark == 'quantum' else difference < 0
        records.append(dict(benchmark=benchmark, method=specialist, comparator=combined, metric=metric,
                            cases=len(pivot), combined_mean=float(pivot[combined].mean()),
                            impute_mean=float(pivot[specialist].mean()),
                            impute_minus_combined=float(difference.mean()),
                            difference_sd=float(difference.std(ddof=1)),
                            impute_wins=int(wins.sum()), ties=int((difference == 0).sum())))
    return records


def mean_sd_table(frame, metric, groups, methods):
    keys = groups + ['method']
    summary = frame.groupby(keys)[metric].agg(['mean', 'std', 'count']).reset_index()
    header = '| ' + ' | '.join(groups + methods) + ' |'
    lines = [header, '| ' + ' | '.join(['---'] * (len(groups) + len(methods))) + ' |']
    for values, group in summary.groupby(groups):
        if not isinstance(values, tuple):
            values = (values,)
        labels = [f'{value:g}' if isinstance(value, (float, np.floating)) else str(value) for value in values]
        lookup = group.set_index('method')
        cells = [f"{lookup.loc[method, 'mean']:.4f} ± {lookup.loc[method, 'std']:.4f}" for method in methods]
        lines.append('| ' + ' | '.join(labels + cells) + ' |')
    return '\n'.join(lines)


def report(output):
    protocol = evaluation.verify(output)
    expected = sum(evaluation.COUNTS[benchmark] for benchmark in REPORT_BENCHMARKS)
    excluded_benchmarks = sorted(set(evaluation.PARENTS) - set(REPORT_BENCHMARKS))
    checked, cases, replays, paired, all_rows = 0, 0, {}, [], []
    new_method_case_scores = 0
    comparisons = {}
    for benchmark in REPORT_BENCHMARKS:
        parent = evaluation.PARENTS[benchmark]
        directory = output / benchmark
        paths = sorted((directory / 'cases').glob('*.json'))
        if len(paths) != evaluation.COUNTS[benchmark]:
            raise ValueError(f'{benchmark}: expected {evaluation.COUNTS[benchmark]} cases, found {len(paths)}')
        replay = json.loads((directory / 'combined_replay.json').read_text())
        assert replay['passed']
        replays[benchmark] = replay
        reference = evaluation.load_reference(benchmark)
        selected_cases = {case['case_id']: case for case in protocol['cases'] if case['benchmark'] == benchmark}
        new_rows = []
        for path in paths:
            record = json.loads(path.read_text())
            artifact = path.with_suffix('.npz')
            assert evaluation.shared.sha256(artifact) == record['arrays_sha256']
            case = selected_cases[record['case_id']]
            assert evaluation.shared.sha256(case['parent_record']) == record['parent_record_sha256']
            with np.load(artifact) as arrays:
                checked += audit_case(case, record, arrays, reference)
            metadata = {key: record[key] for key in ('case_id', 'pattern', 'family', 'state_id', 'seed', 'rate') if key in record}
            new_rows.extend(dict(**metadata, method=method, **scores) for method, scores in record['metrics'].items())
            cases += 1
        new_frame = pd.DataFrame(new_rows)
        new_method_case_scores += len(new_frame)
        saved_frame = pd.read_csv(directory / 'per_case.csv')
        order = ['case_id', 'method']
        pd.testing.assert_frame_equal(new_frame.sort_values(order).reset_index(drop=True),
                                      saved_frame.sort_values(order).reset_index(drop=True),
                                      check_dtype=False, rtol=1e-10, atol=1e-12)
        combined = pd.concat([pd.read_csv(parent / 'per_case.csv'), new_frame], ignore_index=True)
        assert not combined.duplicated(['case_id', 'method']).any()
        comparisons[benchmark] = combined
        combined.to_csv(directory / 'comparison_per_case.csv', index=False)
        group_keys = ['family', 'rate', 'method'] if benchmark == 'quantum' else ['pattern', 'rate', 'method']
        excluded = {'seed', 'rate', 'actual_missing_fraction', 'observed_std', 'uncertainty_cells'}
        metrics = [name for name in new_frame.select_dtypes(include='number').columns if name not in excluded]
        summary = combined.groupby(group_keys)[metrics].agg(['mean', 'std', 'count'])
        summary.columns = ['_'.join(parts) for parts in summary.columns]
        summary.reset_index().to_csv(directory / 'summary.csv', index=False)
        if benchmark == 'quantum':
            overall = combined.groupby(['rate', 'method'])[metrics].agg(['mean', 'std', 'count'])
            overall.columns = ['_'.join(parts) for parts in overall.columns]
            overall.reset_index().to_csv(directory / 'overall_by_missingness.csv', index=False)
        paired.extend(paired_comparison(combined, benchmark))
        all_rows.append(combined.assign(benchmark=benchmark))
    pd.concat(all_rows, ignore_index=True).to_csv(output / 'comparison_per_case.csv', index=False)
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output / 'paired_summary.csv', index=False)
    cylinder_methods = ['Column mean', 'Temporal interpolation', 'Iterative gappy POD', 'SoftImpute',
                        'MatrixFormer', 'MatrixFormer-Impute', 'MatrixFormer*']
    quantum_methods = ['Zero-fill + PSD', 'PSD least squares', 'MatrixFormer', 'MatrixFormer-Impute', 'MatrixFormer*']
    text = [
        '# MatrixFormer-Impute on the original physics pilots', '',
        f'All {expected} retained cylinder and quantum cases completed. The imputation specialist is frozen; no physics training, '
        'checkpoint selection, new masks, new measurement noise, transpose averaging, or routing. '
        'Classical baselines and the original combined/ensemble results are reused unchanged.', '',
        'MRI is excluded at the user\'s request. The original 60-case protocol and any already saved '
        'MRI artifacts are preserved as history, not included in these results. '
        'See `scope_change.json` for the cancellation record.', '',
        f"Checkpoint: `{protocol['checkpoint']}`", f"SHA256: `{protocol['checkpoint_sha256']}`", '',
        'The combined checkpoint was replayed on the first lexicographic case of each benchmark; '
        'its predictions match the archived run numerically. '
        'Every retained new saved prediction is audited.', '',
        '## Cylinder-wake NRMSE (lower is better)', '',
        'Hidden-entry RMSE divided by the global observed-value population SD. Mean ± sample SD over '
        'three mask seeds, not a confidence interval over independent flows. Full 2,048 × 151 context.', '',
        mean_sd_table(comparisons['cylinder'], 'nrmse', ['pattern', 'rate'], cylinder_methods), '',
        '## Quantum state fidelity (higher is better)', '',
        'Mean ± sample SD over the same nine simulated four-qubit states per rate. This is a descriptive '
        'mixture of three state families, not quantum hardware. Pauli measurements and shot counts are '
        'unchanged; all reconstructions use the original PSD, trace-one projection. Full 16 × 16 context.', '',
        mean_sd_table(comparisons['quantum'], 'fidelity', ['rate'], quantum_methods), '',
        '## Predictive distributions and artifacts', '',
        'Cylinder and quantum native CRPS, NLL, 50/90/95% interval coverage and widths are included in '
        'the per-case and summary CSVs. Cylinder uses exactly the original sampled hidden cells; quantum '
        'uses all hidden Pauli coefficients. Intervals refer to raw coefficient predictions, not projected '
        'density matrices.', '',
        '`comparison_per_case.csv` retains every original and new cylinder/quantum method score; `paired_summary.csv` '
        'compares the specialist to the combined model on identical cases. The benchmark subdirectories '
        'contain masks, predictions, distributions, timing, per-case scores, and complete summaries. '
        'No cross-domain average is taken: the metrics and normalizations differ.', '',
    ]
    (output / 'RESULTS.md').write_text('\n'.join(text))
    artifact_paths = [output / 'RESULTS.md', output / 'paired_summary.csv', output / 'comparison_per_case.csv']
    if (output / 'scope_change.json').exists():
        artifact_paths.append(output / 'scope_change.json')
    for benchmark in REPORT_BENCHMARKS:
        artifact_paths.extend(sorted((output / benchmark).glob('*.csv')))
    assert cases == expected
    audit = dict(status='passed', cases=cases, new_method_case_scores=new_method_case_scores, metric_values_checked=checked,
                 benchmarks=REPORT_BENCHMARKS, excluded_benchmarks=excluded_benchmarks,
                 combined_replays=replays, original_inputs_and_sources_unchanged=True,
                 original_baselines_reused=True,
                 script_sha256=evaluation.shared.sha256(Path(__file__)),
                 artifacts_sha256={str(path.relative_to(output)): evaluation.shared.sha256(path) for path in artifact_paths})
    evaluation.shared.write_json(output / 'audit.json', audit)
    evaluation.shared.write_json(output / 'status.json', dict(status='complete', cases=cases, expected=expected, audit='passed',
                                 benchmarks=REPORT_BENCHMARKS, excluded_benchmarks=excluded_benchmarks))
    print(paired_frame.to_string(index=False), flush=True)
    print(f'Audit passed: {checked} saved metric values checked; results at {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report(args.output)
