"""Decompose every saved point-metric change after the rejected native-coverage probe closes."""

from collections import defaultdict
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import preflight_single_model_balanced_native_coverage as probe
import single_model_point_metric_gradients as metrics
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = probe.OUTPUT
OUTPUT = probe.ROOT / 'balanced_native_coverage_probe_linearization_v1'
SCRATCH = Path('/mnt/matrixformer_scratch/balanced_native_coverage_probe_linearization_v1')
CONFIG_SHA256 = '1b3b5fae260b205702287b5fa858b199531b82d5fee138823f6b393e24e9416b'
POLICY = (
    'Read-only CPU decomposition after the original native-coverage probe completes rejected '
    'and all original teacher/retention/endpoint/mean audits and terminal review close normally. '
    'Preserve all28 tasks,56 point slots and30 original grouped guards. Use only original saved '
    'before/candidate grids, cached synthetic episodes and audited recorded84 parameter-gradient '
    'products; never evaluate another endpoint or fraction. Reconstruct both endpoint metrics '
    'with the frozen analytic output-subgradient definitions. Decompose actual change into the '
    'recorded parameter first-order prediction, parameter-to-output linearization residual and '
    'output-metric remainder. The middle term can include nonlinear, finite-precision or '
    'gradient/replay discrepancies; it is not a causal attribution or new parameter-autograd '
    'replay. Imputation metrics are convex in outputs with fixed masks/scales; clipped rating '
    'metrics need not be, so preserve negative rating remainders. Keep every adverse and '
    'undefined slot. No model inference, CUDA, new seed/episode, training, selection, repeated '
    'confirmation, benchmark scoring, changed metric/guard or full-trial authorization. '
    'Descriptive consumed-probe arithmetic is not qualification or evidence a new recipe works.'
)


def close(actual, expected):
    if (not math.isfinite(actual) or not math.isfinite(expected)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=2e-12)):
        raise ValueError(f'Preserve the original finite scalar: {actual} != {expected}')


def decompose(before, candidate, episode, parameter_changes):
    if before.device.type != 'cpu' or candidate.device.type != 'cpu':
        raise ValueError('Only saved CPU prediction grids are permitted')
    initial = metrics.point_metric_terms(before, episode)
    final = metrics.point_metric_terms(candidate, episode)
    if set(initial) != set(final) or set(parameter_changes) != set(initial):
        raise ValueError('Preserve every original point-metric slot')
    rows = []
    for metric, original in initial.items():
        endpoint = final[metric]
        row = {'metric': metric, 'before': None, 'candidate': None, 'change': None,
            'parameter_linear_change': parameter_changes[metric], 'output_linear_change': None,
            'parameter_to_output_residual': None, 'metric_remainder': None,
            'queries': int(episode.mask.sum()), 'defined_units': None}
        if original is None or endpoint is None:
            if original is not None or endpoint is not None or parameter_changes[metric] is not None:
                raise ValueError('Undefined common NRMSE cannot acquire an endpoint or a gradient')
            rows.append(row)
            continue
        parameter_change = parameter_changes[metric]
        if parameter_change is None or not math.isfinite(parameter_change):
            raise ValueError('Every defined metric needs its original finite recorded directional product')
        output_change = float((original['gradient'][episode.mask]
            * (candidate[episode.mask].double() - before[episode.mask].double())).sum())
        actual_change = endpoint['value'] - original['value']
        remainder = actual_change - output_change
        if episode.metadata['domain'] == 'imputation' and remainder < -2e-12:
            raise ValueError('Fixed-mask imputation norms must preserve the output convexity inequality')
        row.update(before=original['value'], candidate=endpoint['value'], change=actual_change,
            output_linear_change=output_change, parameter_to_output_residual=output_change - parameter_change,
            metric_remainder=remainder, defined_units=original['defined_units'])
        close(math.fsum(row[key] for key in ('parameter_linear_change', 'parameter_to_output_residual', 'metric_remainder')),
            actual_change)
        rows.append(row)
    return rows


def grouped(rows, record):
    groups = defaultdict(list)
    for row in rows:
        key = row['task'] if row['cohort'] == 'legacy' else row['cohort']
        groups[f'{key}/{row["metric"]}'].append(row)
    if set(groups) != set(record['finite']['changes']):
        raise ValueError('Never narrow the original grouped point guards')
    result = []
    for name, members in groups.items():
        if len(members) != (1 if name.startswith('legacy_draw_') else 2):
            raise ValueError('Retain both structured/rating replicas and separate native guards')
        defined = [row for row in members if row['change'] is not None]
        row = {'comparison': name, 'members': len(members), 'defined_members': len(defined)}
        for key in ('before', 'candidate', 'change', 'parameter_linear_change', 'output_linear_change',
                'parameter_to_output_residual', 'metric_remainder'):
            row[key] = math.fsum(member[key] for member in defined) / len(defined) if defined else None
        expected = record['finite']['changes'][name]
        if not defined:
            if expected is not None:
                raise ValueError('Never replace an undefined grouped metric')
        else:
            close(row['change'], expected)
        result.append(row)
    return result


def validate_reports(status, result, gate, audit, retention, review):
    if (status['state'] != 'complete' or status['retention_audited'] is not True
            or result['data_admission_only'] is not False or result['record']['accepted'] is not False
            or result['record']['accepted_updates'] != 0 or result['record']['reason'] != 'finite_metrics_rejected'
            or gate['passed'] is not False or gate['preflight_only'] is not True
            or gate['real_benchmark_success'] is not False or audit['complete'] is not True
            or audit['training_tasks'] != 28 or audit['prediction_grids'] != 56
            or audit['raw_endpoint']['gradient_vectors'] != 84 or audit['raw_endpoint']['accepted'] is not False
            or audit['raw_endpoint']['actual_final_parameters_checked'] is not True
            or audit['teacher_replay']['component_grids'] != 52 or audit['cuda_initialized'] is not False
            or retention['complete'] is not True or retention['component_grids'] != 50
            or retention['reference_grids'] != 26 or review['complete'] is not True or review['accepted'] is not False
            or review['config_sha256'] != CONFIG_SHA256 or review['full_trial_authorized'] is not False
            or review['goal_achieved'] is not False):
        raise ValueError('Require the original fully audited rejection without a scientific retry')


def require_closed():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Require CUDA-hidden CPU-only saved-value analysis')
    status, result, gate, audit, retention, review = [probe.read(ROOT / name) for name in (
        'status.json', 'result.json', 'gate.json', 'audit.json', 'retention_audit.json', 'external_recheck.json')]
    validate_reports(status, result, gate, audit, retention, review)
    for owner in (status['pid'], result['training_pid']):
        if probe.closed.original.process_identity(owner) is not None:
            raise ValueError('Original owners must be stopped')
    workers = {'preflight_single_model_balanced_native_coverage.py',
        'audit_single_model_balanced_native_coverage_probe.py', 'recheck_single_model_balanced_native_coverage_probe.py'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(Path(argument).name in workers for argument in arguments):
            raise ValueError('Wait for every original probe/audit/review worker, never duplicate it')
    verify_hashes({str(ROOT / 'config.json'): CONFIG_SHA256})
    verify_hashes(review['input_report_sha256'])
    verify_hashes(review['review_source_sha256'])
    if gate['audit_sha256'] != sha256(ROOT / 'audit.json') or retention['original_audit_sha256'] != gate['audit_sha256']:
        raise ValueError('Bind the exact original audit chain')
    return result


def main():
    torch.set_num_threads(4)
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, SCRATCH)):
        raise FileExistsError('Never repeat or overwrite a begun consumed-probe diagnosis')
    result = require_closed()
    if (shutil.disk_usage(probe.ROOT).free < 18 * 1024**3 or shutil.disk_usage('/').free < 6 * 1024**3
            or shutil.disk_usage(SCRATCH.parent).free < 50 * 1024**3 + 32 * 1024**2):
        raise OSError('Preserve all original reserves and a32 MiB scratch allowance')
    config = probe.read(ROOT / 'config.json')
    tasks = probe.read(ROOT / 'training_batch/manifest.json')['tasks']
    endpoints = {role: probe.read(ROOT / f'{role}_predictions.json')['rows'] for role in ('before', 'candidate')}
    if (len(tasks) != 28 or [task['index'] for task in tasks] != list(range(28))
            or any(len(rows) != 28 or [row['index'] for row in rows] != list(range(28)) for rows in endpoints.values())):
        raise ValueError('Require both complete ordered endpoint manifests and all28 tasks')
    paths = [ROOT / name for name in ('config.json', 'registration.json', 'status.json', 'result.json', 'audit.json',
        'gate.json', 'retention_audit.json', 'external_recheck.json', 'OUTCOME.md', 'training_batch/manifest.json',
        'before_predictions.json', 'candidate_predictions.json')]
    inputs = {str(path): sha256(path) for path in paths}
    for row in tasks:
        inputs[row['artifact']] = row['sha256']
    for rows in endpoints.values():
        for row in rows:
            inputs[row['prediction']] = row['sha256']
    sources = {**config['source_sha256'], **{str(path): sha256(path) for path in (
        Path(__file__).resolve(), REPO / 'tests/test_single_model_native_probe_linearization.py')}}
    verify_hashes(inputs)
    verify_hashes(sources)
    SCRATCH.mkdir()
    OUTPUT.symlink_to(SCRATCH, target_is_directory=True)
    write_json(OUTPUT / 'config.json', {'policy': POLICY, 'input_sha256': inputs, 'source_sha256': sources,
        'scratch_allowance_bytes': 32 * 1024**2, 'new_scientific_seed': False})
    write_json(OUTPUT / 'status.json', {'state': 'analyzing', 'pid': os.getpid(), 'updated_unix': time.time()})
    record = result['record']
    labels, names, grouping = probe.paired.gradient_layout(record['point_before'], 28)
    if len(labels) != 84 or len(names) != 30 or len(record['candidate_products']) != 84:
        raise ValueError('Retain every original directional product and grouped guard')
    parameter_changes = {f'{task}/{metric}': -product for (task, metric), product in zip(labels, record['candidate_products'])
        if metric != 'composite'}
    if set(parameter_changes) != {key for key, value in record['point_before'].items() if value is not None}:
        raise ValueError('Recorded parameter products must cover every defined point slot exactly')
    rows = []
    for saved, registered in zip(tasks, config['batch']['tasks']):
        payload = torch.load(saved['artifact'], map_location='cpu', weights_only=False, mmap=True)
        if payload['task'] != registered or any(saved[key] != value for key, value in registered.items()):
            raise ValueError('Never substitute the original consumed episode')
        episode = probe.training.alignment.mixture.backend.restore_episode(payload['episode'], 'cpu')
        predictions = {}
        for role, manifest in endpoints.items():
            row = manifest[registered['index']]
            if row['task'] != registered['name']:
                raise ValueError('Preserve the exact original task/endpoint identity')
            with np.load(row['prediction'], allow_pickle=False) as archive:
                predictions[role] = torch.from_numpy(archive['prediction'])
        kinds = ('rmse', 'mae' if registered['cohort'] == 'recsys' else 'common_nrmse')
        changes = {metric: parameter_changes.get(f'{registered["name"]}/{metric}') for metric in kinds}
        for row in decompose(predictions['before'], predictions['candidate'], episode, changes):
            comparison = f'{registered["name"]}/{row["metric"]}'
            for field, stored in (('before', 'point_before'), ('candidate', 'point_after')):
                if row[field] is None:
                    if record[stored][comparison] is not None:
                        raise ValueError('Preserve truth-defined null coverage')
                else:
                    close(row[field], record[stored][comparison])
            rows.append({'task_index': registered['index'], 'task': registered['name'], 'cohort': registered['cohort'],
                'domain': episode.metadata['domain'], 'rows': episode.target.shape[1], 'columns': episode.target.shape[2], **row})
    if len(rows) != 56:
        raise ValueError('Retain every original point-metric slot')
    groups = grouped(rows, record)
    expected_products = -(grouping @ np.asarray(record['candidate_products'], dtype=np.float64))
    for name, value in zip(names, expected_products):
        close(next(row for row in groups if row['comparison'] == name)['parameter_linear_change'], float(value))
    verify_hashes(inputs)
    verify_hashes(sources)
    if torch.cuda.is_initialized():
        raise ValueError('No CUDA or model inference is permitted')
    save_csv(OUTPUT / 'point_changes.csv', rows)
    save_csv(OUTPUT / 'grouped_changes.csv', groups)
    summary = {'complete': True, 'tasks': 28, 'point_slots': 56, 'grouped_guards': 30,
        'original_accepted': False, 'original_accepted_updates': 0,
        'adverse_points': [row for row in rows if row['change'] is not None and row['change'] > 0],
        'adverse_groups': [row for row in groups if row['change'] is not None and row['change'] > 0],
        'config_sha256': sha256(OUTPUT / 'config.json'),
        'output_sha256': {str(OUTPUT / name): sha256(OUTPUT / name) for name in ('point_changes.csv', 'grouped_changes.csv')},
        **{key: False for key in ('model_inference_performed', 'cuda_initialized', 'parameter_autograd_replayed',
            'new_scientific_seed', 'training_performed', 'checkpoint_selection_performed', 'confirmation_repeated',
            'real_data_used', 'guard_changed', 'full_trial_authorized', 'goal_achieved')}}
    write_json(OUTPUT / 'summary.json', summary)
    text = ['# Native-coverage probe: saved finite-step decomposition', '', POLICY, '',
        'Original proposal rejected with zero accepted updates. All56 original slots and30 guards are retained.', '',
        '| Guard | Actual change | Recorded parameter linear change | Parameter-to-output residual | Output-metric remainder |',
        '|---|---:|---:|---:|---:|']
    for row in groups:
        values = ['undefined' if row[key] is None else f'{row[key]:+.15g}' for key in (
            'change', 'parameter_linear_change', 'parameter_to_output_residual', 'metric_remainder')]
        text.append('|' + row['comparison'] + '|' + '|'.join(values) + '|')
    text.extend(['', 'These additive terms do not establish a causal mechanism or qualify a future recipe.',
        'Complete individual contributions, including every adverse direction, are in point_changes.csv.',
        'The full single-model goal remains active and unmet.'])
    (OUTPUT / 'OUTCOME.md').write_text('\n'.join(text) + '\n')
    write_json(OUTPUT / 'status.json', {'state': 'complete', 'pid': os.getpid(), 'updated_unix': time.time(),
        'summary_sha256': sha256(OUTPUT / 'summary.json')})
    print({key: value for key, value in summary.items() if key not in ('output_sha256', 'adverse_points')})


if __name__ == '__main__':
    main()
