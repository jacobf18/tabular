"""Recheck the completed probe's identities, coverage and complete saved tables."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

import preflight_single_model_all_imputation_column_mean as probe


ROOT = probe.OUTPUT
MEAN = ROOT.with_name(ROOT.name + '_mean_audit')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13):
        raise ValueError(f'Saved scalar/regrouping mismatch: {actual} != {expected}')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_new(name, text):
    path = ROOT / name
    if path.exists():
        raise FileExistsError(path)
    patch = f'*** Begin Patch\n*** Add File: {path}\n' + ''.join('+' + line + '\n' for line in text.splitlines()) + '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, check=True)


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    names = ('config.json', 'registration.json', 'status.json', 'result.json', 'audit.json', 'retention_audit.json', 'gate.json')
    config, registration, status, result, audit, retention, gate = (read(ROOT / name) for name in names)
    mean_config, mean = read(MEAN / 'config.json'), read(MEAN / 'audit.json')
    assert digest(ROOT / 'config.json') == registration['config_sha256'] == '669d4b633a362e5e577771ed48d7838253a3ced0ea7e496d76087bc7c93f6d11'
    assert status['state'] == 'complete' and status['retention_audited']
    assert not Path(f'/proc/{status["pid"]}').exists() and not Path(f'/proc/{result["training_pid"]}').exists()
    assert gate['passed'] and gate['preflight_only'] and not gate['real_benchmark_success']
    assert gate['audit_sha256'] == retention['original_audit_sha256'] == digest(ROOT / 'audit.json')
    assert audit['complete'] and audit['training_tasks'] == 28 and audit['prediction_grids'] == 56
    assert audit['teacher_replay']['full_grids'] == 28 and audit['teacher_replay']['component_grids'] == 52
    assert audit['teacher_replay']['bitwise_equal'] and audit['teacher_replay']['observed_weights_refitted']
    assert audit['data_admission']['eligible'] and audit['data_admission']['structured_draws'] == 24
    assert audit['data_admission']['rejected_task_indices'] == []
    assert retention['complete'] and retention['reference_grids'] == 26 and retention['component_grids'] == 50
    assert retention['component_role_counts'] == {'v8': 24, 'v29': 26}
    assert retention['reference_role_counts'] == {'v29': 2, 'structured_mean': 24}
    assert result['frozen']['frozen_tensors'] == 16 and len(result['frozen']['changed_trainable_tensors']) == 238
    assert result['frozen']['unselected_parameters_and_all_buffers_bitwise_unchanged']
    assert not result['data_admission_only'] and not audit['cuda_initialized'] and not audit['autograd_replayed']
    assert mean['complete'] and mean['tasks'] == 28 and mean['columns'] > 0
    assert mean['native_reweighted_tasks'] == 2 and mean['structured_reweighted_tasks'] == 24 and mean['imputation_reweighted_tasks'] == 26 and mean['unchanged_rating_tasks'] == 2
    assert not mean['new_gate'] and not mean['model_inference_performed'] and not mean['cuda_initialized']
    assert mean['config_sha256'] == digest(MEAN / 'config.json')
    inventory = {}

    def merge_reports(value):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            if key in {'source_sha256', 'input_sha256', 'artifact_sha256', 'seed_inventory'}:
                for path, expected in child.items():
                    assert path not in inventory or inventory[path] == expected, path
                    inventory[path] = expected
            elif isinstance(child, dict):
                merge_reports(child)

    for report in (config, audit, retention, mean_config, mean):
        merge_reports(report)
    verified_bytes = 0
    for path, expected in inventory.items():
        assert digest(path) == expected, path
        verified_bytes += Path(path).stat().st_size
    used = set().union(*(probe.training.alignment.fixed.seeds_in(read(Path(path))) for path in config['seed_inventory']))
    fresh = probe.training.alignment.fixed.seeds_in(config['batch'])
    assert len(fresh) == 56 and not fresh.intersection(used)
    record = result['record']
    assert record['accepted'] and record['accepted_updates'] == 1 and record['step'] == 1
    assert record['displacement']['admissible'] and record['displacement']['paired_directions'] == 30
    individual = [{'comparison': name, 'before': before, 'candidate': record['point_after'][name],
        'change': record['point_after'][name] - before} for name, before in record['point_before'].items()]
    assert len(individual) == 56
    for row in individual:
        close(row['change'], record['finite']['individual_draw_diagnostics']['changes'][row['comparison']])
    grouped = []
    for name, stored_change in record['finite']['changes'].items():
        if name in record['point_before']:
            members = [name]
        else:
            cohort, metric = name.split('/')
            members = [f'{cohort}_draw_{replicate}/{metric}' for replicate in range(2)]
        before = math.fsum(record['point_before'][member] for member in members) / len(members)
        candidate = math.fsum(record['point_after'][member] for member in members) / len(members)
        close(candidate - before, stored_change)
        grouped.append({'comparison': name, 'before': before, 'candidate': candidate, 'change': stored_change})
    assert len(grouped) == 30 and all(row['change'] < 0 for row in grouped)
    tasks, columns = read_csv(MEAN / 'tasks.csv'), read_csv(MEAN / 'columns.csv')
    assert len(tasks) == 28 and len(columns) == mean['columns']
    assert Counter(row['imputation_reweighting'] for row in tasks) == {'True': 26, 'False': 2}
    by_task = defaultdict(list)
    for row in columns:
        by_task[int(row['task_index'])].append(row)
    mean_checks = 0
    for index, row in enumerate(tasks):
        assert int(row['task_index']) == index
        registered = config['batch']['tasks'][index]
        assert row['task'] == registered['name'] and row['cohort'] == registered['cohort']
        assert (row['imputation_reweighting'] == 'True') == (registered['cohort'] != 'recsys')
        close(float(row['task_weight']), registered['weight'])
        selected = by_task[index]
        assert len(selected) == int(row['columns'])
        assert sum(int(column['queries']) for column in selected) == int(row['queries'])
        for endpoint in ('before', 'candidate'):
            for weighting in ('uniform', 'applied'):
                field = f'{endpoint}_{weighting}_mean20'
                close(math.fsum(float(column[field + '_contribution']) for column in selected), float(row[field]))
                mean_checks += 1
        close(float(row['candidate_applied_mean20']) - float(row['before_applied_mean20']), float(row['applied_change']))
        close(registered['weight'] * float(row['applied_change']), float(row['mixture_weighted_change']))
    penalties = retention['endpoint_penalties']
    assert len(penalties) == 26
    for row in penalties:
        for endpoint in ('before', 'candidate'):
            close(row[endpoint] * row['weighting']['task_weight'], row[endpoint + '_mixture_weighted'])
    adverse = [row for row in individual if row['change'] > 0]
    adverse_mean = [row for row in tasks if float(row['applied_change']) > 0]
    assert len(adverse_mean) == mean['adverse_teacher_mean_tasks']
    assert all(math.isfinite(row['change']) for row in individual)
    report_paths = [ROOT / name for name in names] + [MEAN / name for name in ('config.json', 'audit.json', 'tasks.csv', 'columns.csv')]
    summary = {'complete': True, 'checked_unix': time.time(), 'config_sha256': registration['config_sha256'],
        'scope': 'Independent byte checks, seed inventory and complete saved-value regrouping; not another model or autograd replay',
        'verified_files': len(inventory), 'verified_bytes_path_counted': verified_bytes,
        'verified_hash_inventory_sha256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        'input_report_sha256': {str(path): digest(path) for path in report_paths},
        'recheck_source_sha256': digest(__file__), 'historical_configs': len(config['seed_inventory']),
        'historical_seed_values': len(used), 'scheduled_seed_values': len(fresh), 'seed_overlap': 0,
        'grouped_point_improvements': 30, 'individual_point_improvements': sum(row['change'] < 0 for row in individual), 'individual_adverse_changes': adverse,
        'mean_tasks': 28, 'mean_columns': len(columns), 'independently_summed_mean_terms': mean_checks,
        'mean_adverse_tasks': len(adverse_mean),
        'mean_mixture_weighted_change': math.fsum(float(row['mixture_weighted_change']) for row in tasks),
        'retention_adverse_tasks': sum(row['candidate'] > row['before'] for row in penalties),
        'retention_mixture_weighted_before': math.fsum(row['before_mixture_weighted'] for row in penalties),
        'retention_mixture_weighted_candidate': math.fsum(row['candidate_mixture_weighted'] for row in penalties),
        'training_repeated': False, 'new_model_inference_performed': False, 'confirmation_evaluated': False,
        'real_benchmarks_evaluated': False, 'goal_achieved': False}
    write_new('point_diagnostics.csv', csv_text(individual))
    write_new('grouped_point_guards.csv', csv_text(grouped))
    write_new('external_recheck.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    text = ['# All-imputation column-mean probe: audited feasibility pass', '',
        '**One synthetic feasibility update only. No full trial, qualification or benchmark success. Never warm-start a trial from this probe.**', '',
        'The actual V29 initializer receives one fresh update. Extend native count-shrunk equal-column mean20 Huber weighting to all structured imputation tasks, uniformly across every family and component. Native and rating objectives, mixed priors, teachers, task masses and protected optimizer stay unchanged. Native retention remains actual V29, structured retention stays the exact FP32 V8/V29 mean, and ratings have no retention term.', '',
        f'All30 original grouped finite guards improve. Of56 individual draw metrics,{summary["individual_point_improvements"]} improve and{len(adverse)} worsen. Every adverse value is retained in the complete tables below.', '',
        'The original independent audit verifies28 complete tasks,56 endpoint grids,28 original teacher grids and52 imputation-teacher components. All50 retention components (24 V8/26 V29) and26 targets reproduce, with zero general retention inference. All16 frozen buffers remain unchanged;238 trainable tensors change.', '',
        f'The fixed-half candidate displacement norm is{record["candidate_norm"]:.15g}, compared with Adam norm{record["adam_norm"]:.15g}; ratio{record["displacement"]["norm_ratio"]:.15g} satisfies the unchanged2x cap. The raw audit checks84 gradients and77,998,496 coordinates; maximum analytic AdamW error is{audit["raw_endpoint"]["adam"]["maximum_analytic_adamw_error"]:.15g}. Both displacement comparisons have zero non-bitwise elements. No full historical autograd/optimizer trajectory is replayed.', '',
        f'Peak reserved/allocated GPU memory: {result["peak_reserved_gib"]:.8f}/{result["peak_gib"]:.8f} GiB.', '',
        f'Retention penalties worsen on{summary["retention_adverse_tasks"]}/26 tasks, and their mixture-weighted total changes from{summary["retention_mixture_weighted_before"]:.15g} to{summary["retention_mixture_weighted_candidate"]:.15g}. Mean20 terms worsen on{len(adverse_mean)}/28 tasks; the complete mixture-weighted change is{summary["mean_mixture_weighted_change"]:+.15g}. These are training/reference distances, not benchmark accuracy.', '',
        '## All individual point diagnostics', '', '| Comparison | Before | Candidate | Change |', '|---|---:|---:|---:|']
    for row in individual:
        text.append(f'|{row["comparison"]}|{row["before"]:.12g}|{row["candidate"]:.12g}|{row["change"]:+.12g}|')
    text.extend(['', '## All grouped finite guards', '', '| Comparison | Before | Candidate | Change |', '|---|---:|---:|---:|'])
    for row in grouped:
        text.append(f'|{row["comparison"]}|{row["before"]:.12g}|{row["candidate"]:.12g}|{row["change"]:+.12g}|')
    text.extend(['', '## All retention penalties', '', '| Task | Before | Candidate | Mixture-weighted change |', '|---|---:|---:|---:|'])
    for row in penalties:
        text.append(f'|{row["task"]}|{row["before"]:.12g}|{row["candidate"]:.12g}|{row["candidate_mixture_weighted"] - row["before_mixture_weighted"]:+.12g}|')
    text.extend(['', '## All teacher-mean20 terms', '', '| Task | Imputation column weighting | Change | Mixture-weighted change |', '|---|---|---:|---:|'])
    for row in tasks:
        text.append(f'|{row["task"]}|{row["imputation_reweighting"]}|{float(row["applied_change"]):+.12g}|{float(row["mixture_weighted_change"]):+.12g}|')
    text.extend(['', '## Verification and next steps', '',
        f'Independent external recheck verifies{len(inventory)} artifact/source/input/historical-config paths, all56 fresh seeds against{len(used)} prior seed values,30 independently regrouped guards,56 individual deltas,112 column-to-task mean sums and all26 penalty weights. The mean-only audit includes all{len(columns)} columns, including query-empty, singleton and constant columns.', '',
        'The probe passes feasibility only. A new full trial must initialize afresh from actual V29, preserve all22 original development guards, all32 fresh confirmation comparisons and every frozen real-suite/full-review requirement. First preserve sufficient closed artifacts to satisfy the unchanged checkpoint reserve plus full-stage allowance; no artifact is deleted or relocated by this report.', '',
        'All scratch storage is machine-local, not external/provider-persistent backup. The full single-model goal remains active and unmet.'])
    write_new('OUTCOME.md', '\n'.join(text) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key not in ('input_report_sha256',)}, indent=2))


if __name__ == '__main__':
    main()
