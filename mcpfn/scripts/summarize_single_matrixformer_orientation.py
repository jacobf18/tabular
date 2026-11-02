"""Audit and summarize the combined-model orientation campaign, without paper edits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_single_matrixformer_orientation as campaign
from recsys_evaluation_common import sha256, write_json


LABELS = {'standard': 'Standard', 'transpose': 'Transpose', 'average': 'Average'}
PATTERNS = {'MCAR': 'MCAR', 'MAR': 'Col-MAR', 'MNAR': 'Self-MNAR', 'MAR_Neural': 'NN-MNAR',
            'MAR_BlockNeural': 'Block-MNAR', 'MNARPolarizationPattern': 'Pol-MNAR',
            'MNARSoftPolarizationPattern': 'Soft-Pol-MNAR', 'MNARLatentFactorPattern': 'Latent-MCAR',
            'MNARClusterLevelPattern': 'Clust-MCAR', 'MNARTwoPhaseSubsetPattern': 'Phase-MAR',
            'MNARCensoringPattern': 'Censor-MNAR', 'Overall': 'Overall'}


def reporting_scope(output, manifest):
    cases = manifest['protocols']['recsys']['cases']
    path = output / 'scope_override.json'
    if not path.exists():
        return cases, None
    scope = campaign.read(path)
    if scope.get('manifest_sha256') != sha256(output / 'manifest.json'):
        raise ValueError('Scope override does not match the frozen manifest')
    if scope.get('version') != 1 or scope.get('excluded_datasets') != {'recsys': ['10M']}:
        raise ValueError('Only the explicitly requested MovieLens 10M exclusion is supported')
    if not scope.get('user_request') or not scope.get('reason'):
        raise ValueError('Scope override must document the user request and reason')
    if sorted(case['dataset'] for case in cases) != ['100K', '10M', '1M', 'netflix']:
        raise ValueError('Expected all four original rating datasets in the frozen manifest')
    return [case for case in cases if case['dataset'] != '10M'], scope


def verify_completion(output, scope):
    statuses, parity_hashes = {}, {}
    for stage, expected_checks in (('tabular', 16), ('llm', 12), ('recsys', 1)):
        status = campaign.read(output / stage / 'status.json')
        statuses[stage] = status
        cancelled_exclusion = (stage == 'recsys' and scope is not None
                               and status.get('state') == 'evaluating'
                               and status.get('dataset') == '10M'
                               and status.get('completed') == 3 and status.get('expected') == 4)
        if status.get('state') != 'complete' and not cancelled_exclusion:
            raise ValueError(f'{stage} is not complete within the reporting scope')
        path = output / stage / 'parity.json'
        checks = campaign.read(path)['checks']
        if (len(checks) != expected_checks
                or len({check['case_id'] for check in checks}) != expected_checks
                or any(check['bit_identical'] is not True or check['max_abs_difference'] != 0 for check in checks)):
            raise ValueError(f'{stage} standard-orientation parity checks failed')
        parity_hashes[stage] = sha256(path)
    return statuses, parity_hashes


def bootstrap(values, repetitions=1000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Require finite, paired, independent-unit scores for all three orientations')
    generator = np.random.default_rng(0)
    replicates = []
    for start in range(0, repetitions, 25):
        indices = generator.integers(0, len(values), size=(min(25, repetitions - start), len(values)))
        replicates.append(values[indices].mean(axis=1))
    replicates = np.concatenate(replicates)
    lower, upper = np.percentile(replicates, [2.5, 97.5], axis=0)
    differences = replicates - replicates[:, :1]
    delta_lower, delta_upper = np.percentile(differences, [2.5, 97.5], axis=0)
    mean = values.mean(axis=0)
    return [{'variant': variant, 'mean': float(mean[index]), 'ci_low': float(lower[index]),
             'ci_high': float(upper[index]), 'delta_standard': float(mean[index] - mean[0]),
             'delta_ci_low': float(delta_lower[index]), 'delta_ci_high': float(delta_upper[index]),
             'units': len(values), 'bootstrap_repetitions': repetitions}
            for index, variant in enumerate(campaign.VARIANTS)]


def user_errors(users, targets, predictions):
    identities, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    error = predictions - targets
    return identities, {'RMSE': np.sqrt(np.bincount(inverse, weights=error ** 2) / counts),
                        'MAE': np.bincount(inverse, weights=np.abs(error)) / counts}


def tabular_summaries(frame):
    if frame.duplicated(['case_id', 'variant']).any():
        raise ValueError('Duplicate tabular cases')
    patterns = frame.groupby(['suite', 'pattern', 'dataset', 'variant'], as_index=False).nrmse.mean()
    overall = frame.groupby(['suite', 'dataset', 'variant'], as_index=False).nrmse.mean().assign(pattern='Overall')
    datasets = pd.concat([patterns, overall], ignore_index=True)
    rows = []
    for (suite, pattern), group in datasets.groupby(['suite', 'pattern'], sort=True):
        wide = group.pivot(index='dataset', columns='variant', values='nrmse').loc[:, list(campaign.VARIANTS)]
        for interval in bootstrap(wide.to_numpy(), 10000):
            rows.append({'suite': suite, 'pattern': pattern, 'pattern_label': PATTERNS[pattern],
                         'metric': 'NRMSE', 'resampling_unit': 'dataset', **interval})
    return pd.DataFrame(rows), datasets


def audit_case(output, manifest, stage, case):
    key = campaign.case_key(stage, case)
    path, record = campaign.cache(output, stage, key, sha256(output / 'manifest.json'))
    if record is None:
        raise ValueError(f'Missing result: {stage}/{key}')
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    original_record, original_arrays = campaign.original(stage, case, manifest)
    standard_key = 'student_z' if stage == 'llm' else 'prediction'
    np.testing.assert_array_equal(arrays['standard'], original_arrays[standard_key])
    expected = campaign.combine(arrays['standard'], arrays['transpose'])
    np.testing.assert_array_equal(arrays['average'], expected['average'])
    if stage == 'tabular':
        observed, truth = campaign.tabular.load_case(case)
        mask = np.isnan(observed)
        if tuple(arrays['shape']) != observed.shape:
            raise ValueError('Tabular geometry mismatch')
        for variant in campaign.VARIANTS:
            prediction = observed.astype(arrays[variant].dtype)
            prediction[mask] = arrays[variant]
            score = campaign.tabular.score_predictions(truth, prediction, observed, case['suite'])['nrmse_original']
            np.testing.assert_allclose(score, record['metrics'][variant]['nrmse'], rtol=1e-12, atol=1e-12)
    elif stage == 'llm':
        for name in ('target_z', 'query_rows', 'query_columns'):
            np.testing.assert_array_equal(arrays[name], original_arrays[name])
        for variant in campaign.VARIANTS:
            actual = campaign.llm.checked_metrics(arrays['target_z'], arrays[variant])
            for metric in ('r2', 'rmse'):
                np.testing.assert_allclose(actual[metric], record['metrics'][variant][metric], rtol=1e-12, atol=1e-12)
    else:
        campaign.recsys.verify_hashes(case['reference_sha256'])
        split = campaign.recsys.load_split(case['split'])
        for destination, source in (('test_users', 'test_users'), ('test_items', 'test_items'), ('target', 'test_ratings')):
            np.testing.assert_array_equal(arrays[destination], split[source])
        for variant in campaign.VARIANTS:
            campaign.recsys.audit_scores(split, arrays[variant], record['metrics'][variant])
    return record, arrays, sha256(path)


def summarize_tabular(output, manifest, audit_hashes):
    rows = []
    for index, case in enumerate(manifest['protocols']['tabular']['cases']):
        record, _, digest = audit_case(output, manifest, 'tabular', case)
        audit_hashes['tabular/' + case['case_id']] = digest
        for variant in campaign.VARIANTS:
            rows.append({key: record[key] for key in ('case_id', 'suite', 'dataset', 'pattern', 'missingness', 'repeat')}
                        | {'variant': variant, **record['metrics'][variant]})
        if (index + 1) % 300 == 0:
            print(f'Audited tabular {index + 1}/2262', flush=True)
    frame = pd.DataFrame(rows)
    summary, datasets = tabular_summaries(frame)
    frame.to_csv(output / 'tabular/per_case.csv', index=False)
    datasets.to_csv(output / 'tabular/per_dataset.csv', index=False)
    summary.to_csv(output / 'tabular/summary.csv', index=False)
    return summary


def summarize_llm(output, manifest, audit_hashes):
    rows = []
    for case in manifest['protocols']['llm']['cases']:
        record, _, digest = audit_case(output, manifest, 'llm', case)
        audit_hashes['llm/' + case['case_id']] = digest
        for variant in campaign.VARIANTS:
            rows.append({key: case[key] for key in ('case_id', 'dataset', 'method', 'budget', 'fold')}
                        | {'variant': variant, **record['metrics'][variant]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'llm/per_fold.csv', index=False)
    panels = []
    for (dataset, method, budget), group in frame.groupby(['dataset', 'method', 'budget'], sort=True):
        for metric in ('r2', 'rmse'):
            wide = group.pivot(index='fold', columns='variant', values=metric).loc[:, list(campaign.VARIANTS)]
            if list(wide.index) != list(range(10)):
                raise ValueError('Incomplete LLM panel')
            panels += [{'dataset': dataset, 'method': method, 'budget': budget, 'metric': metric,
                        'resampling_unit': 'fold', **interval} for interval in bootstrap(wide.to_numpy())]
    pd.DataFrame(panels).to_csv(output / 'llm/panels.csv', index=False)
    rows = []
    for scope in ('all_settings', 'mi_5_10_15'):
        selected = frame if scope == 'all_settings' else frame[(frame.method == 'mi') & frame.budget.isin([5, 10, 15])]
        folded = selected.groupby(['dataset', 'fold', 'variant'], as_index=False)[['r2', 'rmse']].mean()
        for dataset, group in folded.groupby('dataset', sort=True):
            for metric in ('r2', 'rmse'):
                wide = group.pivot(index='fold', columns='variant', values=metric).loc[:, list(campaign.VARIANTS)]
                rows += [{'dataset': dataset, 'scope': scope, 'metric': metric, 'resampling_unit': 'fold', **interval}
                         for interval in bootstrap(wide.to_numpy())]
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'llm/summary.csv', index=False)
    return summary


def summarize_recsys(output, manifest, audit_hashes, cases):
    rows = []
    for case in cases:
        record, arrays, digest = audit_case(output, manifest, 'recsys', case)
        audit_hashes['recsys/' + case['dataset']] = digest
        columns = {}
        for variant in campaign.VARIANTS:
            users, metrics = user_errors(arrays['test_users'], arrays['target'], arrays[variant])
            for metric, values in metrics.items():
                columns[variant + '_' + metric] = values
                np.testing.assert_allclose(values.mean(), record['metrics'][variant][metric], rtol=1e-12, atol=1e-12)
        pd.DataFrame({'user': users, **columns}).to_csv(output / 'recsys' / (case['dataset'] + '_per_user.csv.gz'), index=False)
        for metric in ('RMSE', 'MAE'):
            values = np.column_stack([columns[variant + '_' + metric] for variant in campaign.VARIANTS])
            rows += [{'dataset': case['label'], 'metric': metric, 'resampling_unit': 'test user', **interval}
                     for interval in bootstrap(values)]
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'recsys/summary.csv', index=False)
    return summary


def table(frame, keys):
    rows = ['| ' + ' | '.join([*keys, 'Standard', 'Transpose', 'Average', 'Average − standard']) + ' |',
            '|' + '---|' * len(keys) + '---:|---:|---:|---:|']
    for identity, group in frame.groupby(keys, sort=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        values = group.set_index('variant')['mean']
        rows.append('| ' + ' | '.join([*map(str, identity), *[f'{values[variant]:.6f}' for variant in campaign.VARIANTS],
                                       f"{values['average'] - values['standard']:+.6f}"]) + ' |')
    return rows + ['']


def report(output, tables, scope):
    lines = ['# Combined MatrixFormer: scale-preserving orientation evaluation', '',
             'All requested benchmarks completed. Same combined checkpoint; no training, tuning,',
             'orientation selection, baseline refits or paper changes. Standard predictions are',
             'hash-verified October 20 predictions, with representative bit-level parity checks.', '',
             *(['MovieLens 10M was explicitly excluded by the user because it is not used in the paper.',
                'Its unfinished inference was stopped; partial window accumulators are preserved but',
                'never scored or included here. The original manifest and suite status are unchanged;',
                '`scope_override.json` records the revised reporting scope.', ''] if scope else []),
             '## Protocol', '',
             '- Standardize on the original axes using observed entries only; freeze those statistics',
             '  before transposing. This differs from the older orientation campaign and from the',
             '  causal run, which refitted column statistics separately for each orientation.',
             '- MissBench and LLM use whole matrices (LLM retains the original context/test batches).',
             '  UCI and ratings retain 2,000-row × 1,000-column limits in each orientation;',
             '  transposition therefore changes window membership and available row/column context.',
             '- Ratings retain original user scaling and biases, not newly fitted item scaling.',
             '  Each orientation is clipped to the frozen training rating range before averaging.',
             '- LLM predictions are averaged before the unchanged standardized scorer clips to [-10,10].',
             '- No per-case best-orientation selection. All three outcomes are retained.', '',
             '## Uncertainty', '',
             'The CSV summaries include 95% paired percentile-bootstrap intervals for each score',
             'and for its change from standard orientation. Resampling units are datasets for',
             'tabular results (10,000 replicates), test users for ratings (1,000), and folds for',
             'LLM results (1,000), all with seed 0. Methods share bootstrap draws. Within a dataset,',
             'tabular rates/repeats/patterns are averaged before resampling; LLM settings are',
             'averaged within each fold for dataset summaries. These are conditional uncertainty',
             'estimates, not new-dataset or training-seed guarantees; folds share training context.', '',
             'The tables below show point estimates. Read `ci_low`, `ci_high`, `delta_ci_low` and',
             '`delta_ci_high` in the corresponding summary CSVs for exact intervals.', '',
             '## Tabular NRMSE (lower is better)', '']
    tabular_table = tables['tabular']
    lines += table(tabular_table[tabular_table.pattern == 'Overall'], ['suite'])
    lines += ['### All missingness patterns', '']
    lines += table(tabular_table[tabular_table.pattern != 'Overall'], ['suite', 'pattern_label'])
    lines += ['## Rating errors (user-macro; lower is better)', '']
    lines += table(tables['recsys'], ['dataset', 'metric'])
    lines += ['## LLM: all selection methods and budgets', '',
              'R² is higher-is-better; standardized RMSE is lower-is-better. All 1,800 fold settings',
              '(180 ten-fold panels) are included. Per-panel results are in `llm/panels.csv`.', '']
    lines += table(tables['llm'][tables['llm'].scope == 'all_settings'], ['dataset', 'metric'])
    lines += ['### Greedy MI, budgets 5/10/15', '']
    lines += table(tables['llm'][tables['llm'].scope == 'mi_5_10_15'], ['dataset', 'metric'])
    lines += ['## Reproduce', '', 'Re-audit and summarize saved predictions from `/root/tabular`:', '', '```bash',
              'PYTHONPATH=mcpfn/scripts:mcpfn/src python mcpfn/scripts/summarize_single_matrixformer_orientation.py',
              '```', '', 'Prediction caches, original inputs, source snapshots, checkpoint hashes,',
              'parity controls and final audits are retained alongside this report.', '']
    if scope:
        lines += ['Do not resume the original full inference command: its immutable manifest still',
                  'contains MovieLens 10M. All results in the revised scope are already complete.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=campaign.OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output
    manifest = campaign.load_manifest(output)
    recsys_cases, scope = reporting_scope(output, manifest)
    for stage in campaign.STAGES:
        campaign.tabular.verify_hashes(manifest['protocols'][stage]['reference_sha256'])
    campaign.llm.verify_frozen(campaign.llm.REFERENCE)
    statuses, parity_hashes = verify_completion(output, scope)
    audit_hashes, tables = {}, {}
    for stage, function in (('tabular', summarize_tabular), ('llm', summarize_llm)):
        tables[stage] = function(output, manifest, audit_hashes)
    tables['recsys'] = summarize_recsys(output, manifest, audit_hashes, recsys_cases)
    expected_artifacts = 2262 + 1800 + len(recsys_cases)
    if len(audit_hashes) != expected_artifacts:
        raise ValueError('Incomplete final audit')
    report(output, tables, scope)
    write_json(output / 'audit.json', {'passed': True, 'prediction_artifacts': len(audit_hashes),
               'expected_artifacts': expected_artifacts, 'all_average_predictions_exact': True,
               'all_standard_predictions_unchanged': True, 'all_metrics_rescored': True,
               'scope_override': scope,
               'scope_override_sha256': sha256(output / 'scope_override.json') if scope else None,
               'included_rating_datasets': [case['dataset'] for case in recsys_cases],
               'original_suite_statuses': statuses, 'parity_sha256': parity_hashes,
               'artifact_sha256': audit_hashes, 'summarizer_sha256': sha256(Path(__file__)),
               'manifest_sha256': sha256(output / 'manifest.json'),
               'output_sha256': {str(path.relative_to(output)): sha256(path) for path in output.rglob('*.csv')},
               'report_sha256': sha256(output / 'RESULTS.md')})
    write_json(output / 'status.json', {'state': 'complete_revised_scope' if scope else 'complete',
               'excluded_datasets': scope['excluded_datasets'] if scope else {},
               'audit_sha256': sha256(output / 'audit.json')})
    print((output / 'RESULTS.md').read_text())


if __name__ == '__main__':
    main()
