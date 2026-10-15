"""Review the original standardized-Huber probe without repeating scientific stages."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/mnt/checkpoints_200/single_model_goal_20261005/standardized_teacher_huber_preflight_v1')
MEAN = ROOT.with_name(ROOT.name + '_mean_audit')
CONFIG_SHA256 = '7bf3acc2e92274eda8e5254546913f49353f1700091292d78b4c0ed93a0641d8'
PARENT_SHA256 = '9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def close(actual, expected):
    require(math.isfinite(actual) and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13),
        f'Saved scalar mismatch: {actual} != {expected}')


def delta(before, candidate):
    if before is None or candidate is None:
        require(before is None and candidate is None, 'Preserve truth-defined nulls at both endpoints')
        return None
    require(math.isfinite(before) and math.isfinite(candidate), 'Never hide nonfinite point diagnostics')
    return candidate - before


def point_tables(record, tasks):
    expected = {f'{task["name"]}/{metric}' for task in tasks
        for metric in ('rmse', 'mae' if task['cohort'] == 'recsys' else 'common_nrmse')}
    before, candidate = record['point_before'], record['point_after']
    individual_changes = record['finite']['individual_draw_diagnostics']['changes']
    require(set(before) == set(candidate) == set(individual_changes) == expected,
        'Require both point objectives for every registered task')
    individual = []
    grouped_members = defaultdict(list)
    for name, initial in before.items():
        change = delta(initial, candidate[name])
        if change is None:
            require(name.endswith('/common_nrmse') and individual_changes[name] is None,
                'Only truth-defined common NRMSE may remain undefined')
        else:
            close(change, individual_changes[name])
        individual.append({'comparison': name, 'before': initial, 'candidate': candidate[name], 'change': change})
        task_name, metric = name.rsplit('/', 1)
        cohort, draw = task_name.rsplit('_draw_', 1)
        require(draw in {'0', '1'}, 'Preserve the two original draw identities')
        grouped_members[name if cohort == 'legacy' else f'{cohort}/{metric}'].append(name)
    require(set(grouped_members) == set(record['finite']['changes']), 'Preserve all original grouped guard slots')
    grouped = []
    for name, members in grouped_members.items():
        require(len(members) == (1 if name.startswith('legacy_draw_') else 2), 'No missing or duplicate draw')
        selected = [member for member in members if before[member] is not None]
        initial = math.fsum(before[member] for member in selected) / len(selected) if selected else None
        final = math.fsum(candidate[member] for member in selected) / len(selected) if selected else None
        change = delta(initial, final)
        stored = record['finite']['changes'][name]
        if change is None:
            require(stored is None, 'Never fabricate undefined grouped guards')
        else:
            close(change, stored)
        grouped.append({'comparison': name, 'before': initial, 'candidate': final, 'change': change})
    return individual, grouped


def merge_hashes(value, inventory):
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        if key in {'source_sha256', 'input_sha256', 'artifact_sha256', 'seed_inventory'}:
            for path, expected in child.items():
                require(path not in inventory or inventory[path] == expected, f'Conflicting original hash: {path}')
                inventory[path] = expected
        elif isinstance(child, dict):
            merge_hashes(child, inventory)


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_new(name, text):
    path = ROOT / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    patch = f'*** Begin Patch\n*** Add File: {path}\n' + ''.join('+' + line + '\n' for line in text.splitlines()) + '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, check=True)


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Require CUDA-hidden read-only evidence review')
    import preflight_single_model_standardized_teacher_huber as probe

    require(not probe.torch.cuda.is_initialized(), 'No CUDA initialization during review')
    names = ('config.json', 'registration.json', 'status.json', 'result.json', 'audit.json', 'retention_audit.json', 'gate.json')
    config, registration, status, result, audit, retention, gate = (read(ROOT / name) for name in names)
    mean_config, mean = read(MEAN / 'config.json'), read(MEAN / 'audit.json')
    require(digest(ROOT / 'config.json') == registration['config_sha256'] == CONFIG_SHA256, 'Original registration must match')
    require(config['batch'] == probe.batch() and config['structured_component_balance'] == probe.components.SPECIFICATION
        and config['native_teacher_mean'] == probe.retention.MEAN_SPECIFICATION, 'Keep the complete registered recipe')
    require(config['parent'] == str(probe.ROOT / 'legacy_strata_v29/selected_checkpoint.pth')
        and digest(config['parent']) == PARENT_SHA256, 'Use actual V29, not inherited initialization labels')
    require(Counter(task['specification']['structured']['component'] for task in config['batch']['tasks']
        if 'structured' in task['specification']) == {component: 6 for component in probe.components.COMPONENTS},
        'Preserve six draws per structured component')
    require(status['state'] == 'complete' and status['retention_audited'], 'Require terminal original audits')
    for owner in (status['pid'], result['training_pid']):
        require(not Path(f'/proc/{owner}').exists(), 'Original workers must be stopped')
    worker_scripts = {'preflight_single_model_standardized_teacher_huber.py',
        'audit_single_model_standardized_teacher_huber_probe.py'}
    for command_path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = command_path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(not any(Path(argument).name in worker_scripts for argument in arguments),
            'Original probe, replay and supplementary audit must all have stopped')
    require(gate['preflight_only'] and not gate['real_benchmark_success'] and not result['data_admission_only'],
        'This review covers a full feasibility proposal, not qualification or data-only rejection')
    require(gate['audit_sha256'] == retention['original_audit_sha256'] == digest(ROOT / 'audit.json'), 'Bind original audit')
    require(audit['complete'] and audit['training_tasks'] == 28 and audit['prediction_grids'] == 56
        and not audit['cuda_initialized'] and not audit['autograd_replayed'], 'Retain exact replay scope')
    teacher = audit['teacher_replay']
    require(teacher['full_grids'] == 28 and teacher['component_grids'] == 52
        and teacher['bitwise_equal'] and teacher['observed_weights_refitted'], 'Replay all teacher components')
    admission = audit['data_admission']
    require(admission['eligible'] and admission['structured_draws'] == 24 and not admission['rejected_task_indices'],
        'Never omit a rejected task')
    require(retention['complete'] and retention['reference_grids'] == 26 and retention['component_grids'] == 50
        and retention['component_role_counts'] == {'v8': 24, 'v29': 26}
        and retention['reference_role_counts'] == {'v29': 2, 'structured_mean': 24}, 'Retain all original retention targets')
    require(result['frozen']['frozen_tensors'] == 16
        and result['frozen']['unselected_parameters_and_all_buffers_bitwise_unchanged'], 'Preserve frozen buffers')
    require(mean['complete'] and mean['tasks'] == 28 and mean['columns'] > 0
        and mean['native_reweighted_tasks'] == 2 and mean['structured_reweighted_tasks'] == 24
        and mean['imputation_reweighted_tasks'] == 26 and mean['unchanged_rating_tasks'] == 2
        and not mean['new_gate'] and not mean['model_inference_performed'] and not mean['cuda_initialized']
        and mean['config_sha256'] == digest(MEAN / 'config.json'), 'Require independent standardized-mean audit')
    record = result['record']
    require(record['step'] == 1 and record['accepted_updates'] == int(record['accepted'])
        and gate['passed'] == record['accepted'] and audit['raw_endpoint']['accepted'] == record['accepted'],
        'Preserve the original acceptance decision, including rejection')
    require(audit['raw_endpoint']['gradient_vectors'] == 84
        and audit['raw_endpoint']['fixed_half_step']['parameters'] == 77998496
        and audit['raw_endpoint']['actual_final_parameters_checked']
        and record['displacement']['paired_directions'] == 30, 'Preserve full-network gradient and endpoint coverage')
    individual, grouped = point_tables(record, config['batch']['tasks'])
    require(len(individual) == 56 and len(grouped) == 30, 'Preserve all point slots including nulls')
    if record['accepted']:
        require(record['displacement']['admissible'] and record['finite']['accepted']
            and len(result['frozen']['changed_trainable_tensors']) == 238, 'Check accepted full-network update')
    tasks, columns = read_csv(MEAN / 'tasks.csv'), read_csv(MEAN / 'columns.csv')
    require(len(tasks) == 28 and len(columns) == mean['columns'], 'Preserve every audited column')
    by_task = defaultdict(list)
    for row in columns:
        by_task[int(row['task_index'])].append(row)
    require(set(by_task) == set(range(28)), 'No omitted or extra task columns')
    for index, row in enumerate(tasks):
        registered = config['batch']['tasks'][index]
        require(int(row['task_index']) == index and row['task'] == registered['name']
            and row['cohort'] == registered['cohort']
            and (row['imputation_reweighting'] == 'True') == (registered['cohort'] != 'recsys'), 'Bind every task identity')
        selected = by_task[index]
        require(len(selected) == int(row['columns']) and sum(int(column['queries']) for column in selected) == int(row['queries']),
            'Retain every query and empty column')
        close(float(row['task_weight']), registered['weight'])
        for endpoint in ('before', 'candidate'):
            for weighting in ('uniform', 'applied'):
                field = f'{endpoint}_{weighting}_mean20'
                close(math.fsum(float(column[field + '_contribution']) for column in selected), float(row[field]))
        close(float(row['candidate_applied_mean20']) - float(row['before_applied_mean20']), float(row['applied_change']))
        close(registered['weight'] * float(row['applied_change']), float(row['mixture_weighted_change']))
    penalties = retention['endpoint_penalties']
    require(len(penalties) == 26, 'Retain all penalty changes, including adverse changes')
    for row in penalties:
        for endpoint in ('before', 'candidate'):
            close(row[endpoint] * row['weighting']['task_weight'], row[endpoint + '_mixture_weighted'])
    inventory = {str(Path(__file__).resolve()): digest(__file__),
        str(Path(__file__).resolve().parents[1] / 'tests/test_single_model_standardized_teacher_huber_probe_recheck.py'):
        digest(Path(__file__).resolve().parents[1] / 'tests/test_single_model_standardized_teacher_huber_probe_recheck.py')}
    for report in (config, audit, retention, mean_config, mean):
        merge_hashes(report, inventory)
    for path, expected in inventory.items():
        require(digest(path) == expected, f'Changed original evidence: {path}')
    used = set().union(*(probe.training.alignment.fixed.seeds_in(read(path)) for path in config['seed_inventory']))
    fresh = probe.training.alignment.fixed.seeds_in(config['batch'])
    require(len(fresh) == 56 and not fresh.intersection(used), 'Check original disjoint seed inventory')
    mean_adverse = sum(float(row['applied_change']) > 0 for row in tasks)
    require(mean_adverse == mean['adverse_teacher_mean_tasks'], 'Keep all adverse mean terms')
    paths = [ROOT / name for name in names] + [MEAN / name for name in ('config.json', 'audit.json', 'tasks.csv', 'columns.csv')]
    summary = {'complete': True, 'checked_unix': time.time(), 'config_sha256': CONFIG_SHA256,
        'scope': 'Read-only byte checks and saved-value regrouping; no inference or autograd replay',
        'accepted': record['accepted'], 'verified_files': len(inventory),
        'verified_hash_inventory_sha256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        'input_report_sha256': {str(path): digest(path) for path in paths},
        'review_source_sha256': {path: inventory[path] for path in inventory if path == str(Path(__file__).resolve())
            or path.endswith('test_single_model_standardized_teacher_huber_probe_recheck.py')},
        'historical_configs': len(config['seed_inventory']), 'historical_seed_values': len(used),
        'scheduled_seed_values': len(fresh), 'seed_overlap': 0,
        'individual_point_improvements': sum(row['change'] is not None and row['change'] < 0 for row in individual),
        'individual_adverse_changes': [row for row in individual if row['change'] is not None and row['change'] > 0],
        'individual_undefined': [row['comparison'] for row in individual if row['change'] is None],
        'grouped_point_improvements': sum(row['change'] is not None and row['change'] < 0 for row in grouped),
        'grouped_adverse_changes': [row for row in grouped if row['change'] is not None and row['change'] > 0],
        'grouped_undefined': [row['comparison'] for row in grouped if row['change'] is None],
        'mean_tasks': 28, 'mean_columns': len(columns), 'independently_summed_mean_terms': 112,
        'mean_adverse_tasks': mean_adverse,
        'mean_mixture_weighted_change': math.fsum(float(row['mixture_weighted_change']) for row in tasks),
        'retention_adverse_tasks': sum(row['candidate'] > row['before'] for row in penalties),
        'retention_mixture_weighted_before': math.fsum(row['before_mixture_weighted'] for row in penalties),
        'retention_mixture_weighted_candidate': math.fsum(row['candidate_mixture_weighted'] for row in penalties),
        'training_repeated': False, 'new_model_inference_performed': False, 'confirmation_evaluated': False,
        'real_benchmarks_evaluated': False, 'goal_achieved': False, 'full_trial_authorized': False}
    write_new('point_diagnostics.csv', csv_text(individual))
    write_new('grouped_point_guards.csv', csv_text(grouped))
    write_new('external_recheck.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    text = ['# Standardized teacher-Huber probe: terminal review', '',
        f'Original single-update acceptance: **{record["accepted"]}**. This is synthetic feasibility, not benchmark success.', '',
        'Actual V29 initialization; balanced mixed priors; standardized teacher-mean Huber on all26 imputation tasks. Ratings and retention remain unchanged. No inference-time ensemble.', '',
        f'Grouped improvements: {summary["grouped_point_improvements"]}/30. Individual improvements: {summary["individual_point_improvements"]}/56. '
        f'Individual adverse changes: {len(summary["individual_adverse_changes"])}. Undefined slots: {len(summary["individual_undefined"])}.', '',
        'All28 tasks/56 endpoint grids,52 imputation-teacher components,26 retention targets/50 components,84 raw gradients and77,998,496 parameter coordinates remain covered by the original audits.', '',
        f'Mean-term adverse tasks: {mean_adverse}/28; weighted change: {summary["mean_mixture_weighted_change"]:+.15g}. '
        f'Retention adverse tasks: {summary["retention_adverse_tasks"]}/26; weighted before/candidate: '
        f'{summary["retention_mixture_weighted_before"]:.15g}/{summary["retention_mixture_weighted_candidate"]:.15g}.', '',
        'All individual/grouped changes are preserved in point_diagnostics.csv and grouped_point_guards.csv. '
        'All teacher-mean columns/tasks are in the original sibling _mean_audit directory; all retention penalties remain in retention_audit.json.', '',
        'No scientific stage was repeated. No full trial is authorized by this review. Any separately registered trial must use fresh streams, start from actual V29 rather than this feasibility endpoint, and retain all22 development guards,32 confirmation comparisons and the complete frozen real-benchmark campaign.', '',
        'The single-model goal remains active and unmet.']
    write_new('OUTCOME.md', '\n'.join(text) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'input_report_sha256'}, indent=2))


if __name__ == '__main__':
    main()
