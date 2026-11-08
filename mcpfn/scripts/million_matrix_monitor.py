"""UCI-first development monitoring without changing historical benchmark scorers."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import science_retention_benchmarks as retention
import science_continuation_monitor as science
from evaluate_scientific_uci import aggregate, select_uci_cases
from recsys_evaluation_common import sha256, write_csv, write_json


WARM_RUN = retention.ROOT / 'retained_v2'
WARM_CHECKPOINT = WARM_RUN / 'snapshots/step_3000/checkpoint.pth'
WARM_UCI = WARM_RUN / 'full_uci_step3000_v1'
LIMITS = dict(missbench_overall_relative=.10, llm_rmse_relative=.02, llm_r2_drop=.01, recsys_error_relative=.02)


def read(path):
    return json.loads(Path(path).read_text())


def register_protocol(output):
    screen = read(retention.ROOT / 'retention_screen_v1/protocol.json')
    full = read(retention.ROOT / 'retention_full_v1/protocol.json')
    protocol = deepcopy(screen)
    protocol['stages']['tabular'] = [case for case in screen['stages']['tabular'] if case['suite'] == 'missbench'] + select_uci_cases(full)
    protocol['input_sha256'].update(full['input_sha256'])
    protocol['scope'] = 'Full UCI 1800, full MissBench 462, LLM screen 72, full 100K/Netflix: 2336 records'
    protocol['historical_gate_is_diagnostic_only'] = True
    protocol['million_matrix_limits'] = LIMITS
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)


def warm_rows():
    rows = [json.loads(line) for line in (WARM_RUN / 'snapshots/step_3000/retention/records.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['suite'] != 'uci']
    for row in pd.read_csv(WARM_UCI / 'per_case.csv').to_dict('records'):
        rows.append(dict(stage='tabular', suite='uci', dataset=row['dataset'], pattern=row['pattern'], case_id=row['case_id'],
                         error=row['candidate_nrmse'], parent_error=row['original_nrmse']))
    if len(rows) != 2336 or len({(row['stage'], row['case_id']) for row in rows}) != 2336:
        raise ValueError('Incomplete warm-start monitoring reference')
    return rows


def summarize(rows, science_summary, warm=None):
    frame = pd.DataFrame(rows)
    values = {}
    for suite in ('uci', 'missbench'):
        group = frame[(frame.stage == 'tabular') & (frame.suite == suite)]
        means = group.groupby('dataset')[['error', 'parent_error']].mean().mean()
        values[suite] = float(means.error)
        values[f'{suite}_original'] = float(means.parent_error)
        values[f'{suite}_ratio'] = float(means.error / means.parent_error)
    values['science_ratio'] = float(np.exp(np.log([row['candidate'] / row['parent'] for row in science_summary]).mean()))
    values['science_datasets_improved'] = sum(row['candidate'] < row['parent'] for row in science_summary)
    values['science_worst_ratio'] = max(row['candidate'] / row['parent'] for row in science_summary)
    values['joint_score'] = float(np.sqrt(values['uci_ratio'] * values['science_ratio']))
    checks = []
    checks.append(dict(name='missbench/overall', change=values['missbench_ratio'] - 1, limit=LIMITS['missbench_overall_relative']))
    for dataset, group in frame[frame.stage == 'llm'].groupby('dataset'):
        checks += [dict(name=f'llm/{dataset}/rmse', change=float(group.error.mean() / group.parent_error.mean() - 1), limit=LIMITS['llm_rmse_relative']),
                   dict(name=f'llm/{dataset}/r2', change=float(group.parent_r2.mean() - group.r2.mean()), limit=LIMITS['llm_r2_drop'])]
    for row in frame[frame.stage == 'recsys'].itertuples():
        checks += [dict(name=f'recsys/{row.dataset}/rmse', change=row.error / row.parent_error - 1, limit=LIMITS['recsys_error_relative']),
                   dict(name=f'recsys/{row.dataset}/mae', change=row.mae / row.parent_mae - 1, limit=LIMITS['recsys_error_relative'])]
    for check in checks:
        if not np.isfinite(check['change']):
            raise ValueError('Nonfinite retention comparison')
        check['passed'] = bool(check['change'] <= check['limit'])
    baseline = warm or values
    values['joint_eligible'] = bool(values['uci'] <= baseline['uci'] and values['science_ratio'] <= baseline['science_ratio']
                                    and checks[0]['passed'])
    values['balanced_eligible'] = bool(values['joint_eligible'] and all(check['passed'] for check in checks))
    values['checks'] = checks
    values['failures'] = [check for check in checks if not check['passed']]
    return values


def warm_reference():
    fingerprint = sha256(WARM_CHECKPOINT)
    for path in (WARM_UCI / 'manifest.json', WARM_RUN / 'snapshots/step_3000/science/manifest.json',
                 WARM_RUN / 'snapshots/step_3000/retention/manifest.json'):
        if read(path)['checkpoint_sha256'] != fingerprint:
            raise ValueError('Warm-start results refer to another checkpoint')
    if read(WARM_UCI / 'status.json')['completed'] != 1800:
        raise ValueError('Require completed full UCI reference')
    summary = pd.read_csv(WARM_RUN / 'snapshots/step_3000/science/summary.csv').to_dict('records')
    return summarize(warm_rows(), summary)


def evaluate(checkpoint, registry, science_registry, output, warm):
    output.mkdir(parents=True, exist_ok=False)
    science.evaluate(checkpoint, science_registry, output / 'science')
    retention.evaluate(checkpoint, registry, output / 'legacy')
    rows = [json.loads(line) for line in (output / 'legacy/records.jsonl').read_text().splitlines()]
    expected = sum(len(cases) for cases in read(registry / 'protocol.json')['stages'].values())
    if len(rows) != expected:
        raise ValueError('Incomplete benchmark coverage')
    scientific = pd.read_csv(output / 'science/summary.csv').to_dict('records')
    result = summarize(rows, scientific, warm)
    by_id = {case['case_id']: case for case in read(registry / 'protocol.json')['stages']['tabular']}
    uci = [dict(case_id=row['case_id'], dataset=row['dataset'], pattern=row['pattern'],
                missingness=by_id[row['case_id']]['missingness'], repeat=by_id[row['case_id']]['repeat'],
                original_nrmse=row['parent_error'], candidate_nrmse=row['error']) for row in rows if row['suite'] == 'uci']
    if len(uci) != 1800:
        raise ValueError('Full UCI is required for selection')
    for name, dimensions in [('overall', ()), ('per_pattern', ('pattern',)), ('per_dataset', ('dataset',)), ('per_missingness', ('missingness',))]:
        write_csv(output / f'uci_{name}.csv', aggregate(uci, dimensions))
    write_csv(output / 'uci_per_case.csv', uci)
    write_json(output / 'selection.json', result)
    return result
