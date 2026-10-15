"""Summarize complete saved development boundaries without selecting a model."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


METRICS = ('rmse', 'mae', 'missbench_nrmse', 'uci_nrmse', 'common_nrmse')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def cohort(case):
    return 'recsys' if case['domain'] == 'recsys' else 'structured_overall' if case.get('structured', False) else 'imputation'


def stored(report, group):
    if group in ('imputation', 'recsys'):
        return report['student'][group]
    if group == 'structured_overall':
        return report['student']['structured']['overall']
    return report['student']['structured']['families'][group]


def prefix(run, boundary):
    raw = []
    with (run / 'training.jsonl').open('rb') as stream:
        for _index in range(boundary):
            line = stream.readline()
            assert line.endswith(b'\n')
            raw.append(line)
    records = [json.loads(line) for line in raw]
    assert [record['step'] for record in records] == list(range(1, boundary + 1))
    return records, hashlib.sha256(b''.join(raw)).hexdigest()


def review(run, boundary, output, config_sha256):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require an explicitly CUDA-hidden saved-metric review")
    if output.exists() or boundary not in (10, 25, 50, 64):
        raise ValueError("Use a new output and a completed original trained boundary")
    if shutil.disk_usage(output.parent).free < 50 * 1024**3 + 8 * 1024**2:
        raise OSError("Keep50 GiB plus8 MiB free for the standalone scratch report")
    config = read(run / 'config.json')
    assert digest(run / 'config.json') == config_sha256
    assert digest(run / 'config.json') == read(run / 'registration.json')['config_sha256']
    steps = tuple(step for step in config['boundaries'] if step <= boundary)
    assert steps == tuple(step for step in (0, 10, 25, 50, 64) if step <= boundary)
    reports = {step: read(run / f'validation_{step}.json') for step in steps}
    paths = [run / 'config.json', run / 'registration.json', Path(__file__),
        Path(__file__).resolve().parents[1] / 'tests/test_single_model_training_boundary_review.py']
    paths.extend(run / f'validation_{step}.json' for step in steps)
    inputs = {str(path): digest(path) for path in paths}
    training, prefix_sha = prefix(run, boundary)
    assert reports[0]['best_step'] == 0 and reports[0]['selected_checkpoint_sha256'] is None
    for step, report in reports.items():
        assert report['step'] == step and len(report['cases']) == 152
        assert report['accepted_updates'] == (training[step - 1]['accepted_updates'] if step else 0)
        assert Counter(cohort(case) for case in report['cases']) == {'imputation': 40, 'structured_overall': 96, 'recsys': 16}
        assert len(report['decision']['comparisons']) == 22
    case_rows, coverage_rows = [], []
    for index, initial in enumerate(reports[0]['cases']):
        assert initial['index'] == index and initial['gradient_data'] == 'synthetic'
        for step, report in reports.items():
            case = report['cases'][index]
            assert all(initial.get(key) == case.get(key) for key in ('index', 'seed', 'domain', 'rows', 'columns', 'prior', 'component', 'family', 'gradient_data', 'targets'))
            assert cohort(case) == cohort(initial)
            if cohort(case) != 'recsys':
                coverage = case['student']['column_coverage']
                assert coverage['queried_columns'] == initial['student']['column_coverage']['queried_columns']
                assert coverage['common_nonconstant_columns'] == initial['student']['column_coverage']['common_nonconstant_columns']
                coverage_rows.append({'step': step, 'index': index, 'cohort': cohort(case), **coverage})
        for metric in (('rmse', 'mae') if cohort(initial) == 'recsys' else METRICS):
            values = {step: report['cases'][index]['student'][metric] for step, report in reports.items()}
            assert all(math.isfinite(value) for value in values.values())
            case_rows.append({'index': index, 'seed': initial['seed'], 'cohort': cohort(initial),
                'family': initial.get('family', ''), 'prior': initial.get('prior', initial.get('component', '')),
                'rows': initial['rows'], 'columns': initial['columns'], 'metric': metric,
                **{f'step_{step}': value for step, value in values.items()},
                'latest_minus_initialization': values[boundary] - values[0],
                'latest_minus_previous': values[boundary] - values[steps[-2]]})
    assert len(case_rows) == 712 and len(coverage_rows) == 136 * len(steps)
    families = tuple(reports[0]['student']['structured']['families'])
    assert len(families) == 12
    summary_rows = []
    for group in ('imputation', 'recsys', 'structured_overall', *families):
        indices = [index for index, case in enumerate(reports[0]['cases']) if cohort(case) == group or (case.get('structured', False) and case['family'] == group)]
        assert len(indices) == {'imputation': 40, 'recsys': 16, 'structured_overall': 96}.get(group, 8)
        for metric in (('rmse', 'mae') if group == 'recsys' else METRICS):
            values = {}
            for step, report in reports.items():
                values[step] = math.fsum(report['cases'][index]['student'][metric] for index in indices) / len(indices)
                close(values[step], stored(report, group)[metric])
            summary_rows.append({'cohort': group, 'cases': len(indices), 'metric': metric,
                **{f'step_{step}': value for step, value in values.items()},
                'latest_minus_initialization': values[boundary] - values[0],
                'latest_minus_previous': values[boundary] - values[steps[-2]]})
    assert len(summary_rows) == 72
    guards = []
    for position, initial in enumerate(reports[0]['decision']['comparisons']):
        row = {'group': initial['group'], 'metric': initial['metric'], 'original_limit': initial['initialization']}
        for step, report in reports.items():
            guard = report['decision']['comparisons'][position]
            assert all(guard[key] == initial[key] for key in ('group', 'metric', 'initialization'))
            close(guard['candidate'], stored(report, guard['group'])[guard['metric']])
            assert guard['retained'] == (guard['candidate'] <= guard['initialization'])
            row[f'step_{step}'] = guard['candidate']
            row[f'step_{step}_retained'] = guard['retained']
        guards.append(row)
    for report in reports.values():
        decision = report['decision']
        close(decision['score'], max(decision['ratios'].values()))
        assert decision['retained'] == all(guard['retained'] for guard in decision['comparisons'])
        assert decision['research_selection_tolerance'] == 0
    failed = [row for row in guards if not row[f'step_{boundary}_retained']]
    if failed:
        assert reports[boundary]['decision']['passed'] is False
    for path, expected in inputs.items():
        assert digest(Path(path)) == expected
    assert prefix(run, boundary)[1] == prefix_sha
    output.mkdir()
    write_csv(output / 'case_metrics.csv', case_rows)
    write_csv(output / 'summary_metrics.csv', summary_rows)
    write_csv(output / 'original_guards.csv', guards)
    write_csv(output / 'role_specific_coverage.csv', coverage_rows)
    header = ' | '.join('Actual initializer' if step == 0 else f'Step {step}' for step in steps)
    separator = '|---|---|' + '---:|' * len(steps)
    lines = [f'# {run.name} development boundary {boundary}', '',
        '**Consumed synthetic development results only; not real-benchmark scores.**', '',
        f"Attempts: {boundary}; accepted updates: {reports[boundary]['accepted_updates']}; failed original guards: {len(failed)}/22.",
        f"Original best step: {reports[boundary]['best_step']}; selected checkpoint SHA256: `{reports[boundary]['selected_checkpoint_sha256']}`.",
        'Initializer step 0 is ineligible. This review does not select a checkpoint.', '',
        '## Summary', '', f'| Cohort | Metric | {header} |', separator]
    for row in summary_rows:
        if (row['cohort'] in ('imputation', 'structured_overall') and row['metric'] in ('rmse', 'common_nrmse')) or row['cohort'] == 'recsys':
            values = ' | '.join(f"{row[f'step_{step}']:.12f}" for step in steps)
            lines.append(f"| {row['cohort']} | {row['metric']} | {values} |")
    lines += ['', 'NRMSE uses original per-case eligible-column means, then equal case weights.',
        'The imputation RMSE entries are equal-case means of per-case pooled RMSE, not globally pooled errors.',
        'Canonical role-specific coverage is preserved in `role_specific_coverage.csv`, not replaced by a shared intersection.', '',
        '## All original guards', '', 'Original guard limits are distinct from actual V29 initializer scores.', '',
        '| Group | Metric | Original limit | Latest value | Retained |', '|---|---|---:|---:|---|']
    for row in guards:
        lines.append(f"| {row['group']} | {row['metric']} | {row['original_limit']:.12f} | {row[f'step_{boundary}']:.12f} | {'yes' if row[f'step_{boundary}_retained'] else 'NO'} |")
    lines += ['', '## Minimax and complete evidence', '', '| Boundary | Accepted updates | Minimax | Bottleneck |', '|---|---:|---:|---|']
    for step, report in reports.items():
        ratios = report['decision']['ratios']
        lines.append(f"| {step} | {report['accepted_updates']} | {report['decision']['score']:.12f} | {max(ratios, key=ratios.get)} |")
    lines += ['', f"The latest boundary worsens {sum(row['latest_minus_initialization'] > 0 for row in case_rows)}/712 case/metric rows and {sum(row['latest_minus_initialization'] > 0 for row in summary_rows)}/72 summary rows versus the actual initializer.",
        'These include correlated metric formulas, not independent significance tests.',
        '- `case_metrics.csv`: all 712 rows over all 152 original cases at every included boundary.',
        '- `summary_metrics.csv`: all 72 cohort/family/metric rows, including all 12 structured families.',
        '- `original_guards.csv`: all 22 original limits and every included boundary outcome.',
        '- `role_specific_coverage.csv`: all imputation case/boundary coverage, including undefined columns.',
        '- `recheck.json`: immutable source/report/CSV hashes and the bound original training-log prefix.', '',
        f'Independent stored-case arithmetic checks {72 * len(steps)} means and {22 * len(steps)} boundary/guard comparisons.',
        'No grids or models are rerun; this is not full historical optimizer/autograd replay or the original mandatory full audit.',
        'No new seeds, training modification, checkpoint selection, gate relaxation, confirmation or real-benchmark evaluation occurs.',
        'Keep the original trial and queued audits unchanged. The single-checkpoint goal remains active and unmet.', '']
    (output / 'OUTCOME.md').write_text('\n'.join(lines))
    receipt = {'complete': True, 'utc': datetime.now(timezone.utc).isoformat(), 'boundary': boundary, 'included_boundaries': steps,
        'input_sha256': inputs, 'training_log_prefix_sha256': prefix_sha, 'training_log_prefix_attempts': boundary,
        'nonaccepted_attempts': [{'step': record['step'], 'reason': record['reason']} for record in training if record['reason'] != 'accepted'],
        'accepted_updates': reports[boundary]['accepted_updates'], 'case_metric_rows': 712, 'summary_metric_rows': 72,
        'case_aggregation_checks': 72 * len(steps), 'guard_checks': 22 * len(steps), 'original_guards': 22,
        'failed_guards': [{'group': row['group'], 'metric': row['metric']} for row in failed],
        'best_step': reports[boundary]['best_step'], 'selected_checkpoint_sha256': reports[boundary]['selected_checkpoint_sha256'],
        'output_sha256': {str(path): digest(path) for path in output.iterdir()},
        'prediction_grids_recomputed': False, 'model_inference_performed': False, 'training_modified': False,
        'gate_changed': False, 'real_data_used': False}
    (output / 'recheck.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'complete': True, 'output': str(output), 'failed_guards': receipt['failed_guards'],
        'nonaccepted_attempts': receipt['nonaccepted_attempts'], 'report_sha256': digest(output / 'OUTCOME.md'),
        'recheck_sha256': digest(output / 'recheck.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--boundary', type=int, choices=(10, 25, 50, 64), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    arguments = parser.parse_args()
    review(arguments.run.resolve(), arguments.boundary, arguments.output.absolute(), arguments.config_sha256)
