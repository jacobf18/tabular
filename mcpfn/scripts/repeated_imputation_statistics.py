"""Dataset-level uncertainty and paired tests, never seed-level pseudoreplication."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    return float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def dataset_scores(cases, results, methods):
    expected = pd.DataFrame(cases)
    if expected.case_id.duplicated().any():
        raise ValueError('Duplicate registered cases')
    actual = pd.DataFrame(results)
    if actual.empty:
        actual = pd.DataFrame(columns=['case_id', 'method', 'status', 'nrmse'])
    if actual.duplicated(['case_id', 'method']).any():
        raise ValueError('Duplicate method/case records')
    if not set(actual.case_id).issubset(set(expected.case_id)) or not set(actual.method).issubset(set(methods)):
        raise ValueError('Unregistered method or case')
    outputs, coverage = [], []
    for method in methods:
        requested = expected if method != 'tabimpute' else expected[expected.suite == 'missbench']
        matched = requested.merge(actual[actual.method == method][['case_id', 'status', 'nrmse']], on='case_id', how='left')
        matched['valid'] = matched.status.eq('ok') & np.isfinite(pd.to_numeric(matched.nrmse, errors='coerce'))
        for suite, suite_cases in matched.groupby('suite'):
            coverage.append(dict(method=method, suite=suite, expected=len(suite_cases),
                                 successful=int(suite_cases.valid.sum()), failed=int(suite_cases.status.notna().sum() - suite_cases.valid.sum()),
                                 pending=int(suite_cases.status.isna().sum()), complete=bool(suite_cases.valid.all())))
        for (suite, dataset), group in matched.groupby(['suite', 'dataset']):
            per_pattern = {}
            for pattern, pattern_cases in group.groupby('pattern'):
                if not pattern_cases.valid.all():
                    continue
                by_rate = pattern_cases.groupby('missingness').nrmse.mean()
                score = float(by_rate.mean())
                per_pattern[pattern] = score
                outputs.append(dict(method=method, suite=suite, dataset=dataset, panel=pattern,
                                    nrmse=score, cases=len(pattern_cases)))
            if len(per_pattern) == group.pattern.nunique():
                outputs.append(dict(method=method, suite=suite, dataset=dataset, panel='Overall',
                                    nrmse=float(np.mean(list(per_pattern.values()))), cases=len(group)))
            for family, family_cases in group.groupby('family'):
                patterns = list(family_cases.pattern.unique())
                if all(pattern in per_pattern for pattern in patterns):
                    outputs.append(dict(method=method, suite=suite, dataset=dataset, panel=f'family/{family}',
                                        nrmse=float(np.mean([per_pattern[pattern] for pattern in patterns])), cases=len(family_cases)))
    return outputs, coverage


def summarize(cases, results, methods):
    scores, coverage = dataset_scores(cases, results, methods)
    if not scores:
        return scores, coverage, [], []
    frame = pd.DataFrame(scores)
    expected = pd.DataFrame(cases).groupby('suite').dataset.nunique().to_dict()
    summaries = []
    for (suite, method, panel), group in frame.groupby(['suite', 'method', 'panel']):
        summaries.append(dict(suite=suite, method=method, panel=panel, mean=float(group.nrmse.mean()),
                              ci95_half_width=interval(group.nrmse), datasets=len(group),
                              expected_datasets=expected[suite], complete=len(group) == expected[suite],
                              ci_unit='dataset', aggregation='seed -> rate -> pattern -> dataset'))
    tests = []
    for suite, group in frame[frame.panel == 'Overall'].groupby('suite'):
        pivot = group.pivot(index='dataset', columns='method', values='nrmse')
        if 'matrixformer' not in pivot:
            continue
        for method in pivot.columns:
            if method == 'matrixformer':
                continue
            pair = pivot[['matrixformer', method]].dropna()
            if len(pair) != expected[suite]:
                continue
            differences = pair.matrixformer.to_numpy() - pair[method].to_numpy()
            probability = 1. if np.all(differences == 0) else float(wilcoxon(differences, alternative='two-sided').pvalue)
            tests.append(dict(suite=suite, reference='matrixformer', comparison=method, datasets=len(pair),
                              alternative='two-sided', p_value=probability, mean_difference=float(differences.mean()),
                              difference_ci95_half_width=interval(differences)))
    for suite in expected:
        family = sorted([record for record in tests if record['suite'] == suite], key=lambda record: record['p_value'])
        previous = 0.
        for index, record in enumerate(family):
            previous = max(previous, min(1., record['p_value'] * (len(family) - index)))
            record['p_holm'] = previous
    return scores, coverage, summaries, tests


def export(output, manifest):
    output = Path(output)
    results = [json.loads(path.read_text()) for path in (output / 'results').glob('*/cases/**/*.json')]
    scores, coverage, summaries, tests = summarize(manifest['cases'], results, manifest['methods'])
    destination = output / 'summary'
    destination.mkdir(exist_ok=True)
    for name, rows in [('per_case', results), ('dataset_scores', scores), ('coverage', coverage),
                       ('summary', summaries), ('paired_tests', tests)]:
        temporary = destination / f'{name}.csv.tmp'
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(destination / f'{name}.csv')
    lines = ['# Repeated imputation development benchmark', '',
             'Five mask seeds for stochastic mechanisms; one for deterministic Pol-MNAR.',
             'Historical single-mask numbers are not pooled with this new protocol.',
             'Primary NRMSE uses held-out-truth column SD, matching the historical scorer.',
             'Observed-only SD NRMSE is also saved per case as a separate diagnostic.',
             '95% t intervals and two-sided Wilcoxon tests use dataset means, not mask seeds.',
             'Missing/failed cases are explicit; incomplete dataset-panels are excluded, not silently averaged.',
             'Overall is the equal-pattern mean after averaging seeds and rates within datasets.',
             'Partial summaries are provisional and must not be used for method rankings.', '',
             '| Suite | Method | Overall NRMSE ± 95% CI | Complete datasets |',
             '|---|---|---:|---:|']
    for row in summaries:
        if row['panel'] == 'Overall':
            half = row['ci95_half_width']
            formatted = f"{row['mean']:.4f} ± {half:.4f}" if half is not None else f"{row['mean']:.4f} (CI unavailable)"
            lines.append(f"| {row['suite']} | {row['method']} | {formatted} | {row['datasets']}/{row['expected_datasets']} |")
    temporary = destination / 'SUMMARY.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(destination / 'SUMMARY.md')
