"""Describe complete saved synthetic-development trajectories without changing selection."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

import analyze_single_model_teacher_alignment as metrics


ROOT = metrics.ROOT
BOUNDARIES = (0, 10, 25, 50, 64)
RUNS = {
    'all_imputation_blend_v45': 'b672f96797c02b4ca31988d142c3caa32fdc514702fe59c84c1de72df41d9f28',
    'stronger_mean_distillation_v46': 'ac98d26801b73579928c510b6e76a7b8479cd4fa05c5690a5599e69127733c49',
}
COUNTS = {'legacy': 40, 'structured': 96, 'ratings': 16}
HISTORICAL_AUDIT_SHA256 = 'de53f9b1158651670717092bbf45ab602605ccb5f34be1dc4a96d6b5dfcc5ff1'
POLICY = (
    'Read only complete, already published development boundaries for both V45 and V46, including '
    'every earlier registered boundary and all152 original synthetic cases at each boundary. Use '
    'the same original teacher, general parent, V8 retention and actual V29 initialization roles, '
    'keeping observed-source and original query masks distinct without removing overlaps. Record '
    'accepted-update counts and original published selection decisions, never select or promote '
    'from these diagnostics. All distances are normalized-network-coordinate, equal-case descriptive '
    'statistics, not original-unit/user-macro ratings or real benchmark results. Stronger teacher '
    'imitation is not truth accuracy. Trials have distinct training seeds and accepted trajectories: '
    'this is not a seed-controlled causal estimate of the coefficient change or a parameter-gradient '
    'conflict analysis. V45 has a completed independent audit; live V46 reports have bound hashes '
    'but the full independent audit remains pending unless separately completed. Do not infer audit '
    'completion from this diagnostic. No model inference, training, CUDA, fresh seeds, confirmation, '
    'real data, original-evidence writes, changed gates, scientific retry or inference ensemble.')


def requested_boundaries(through_step):
    if type(through_step) is not int or through_step not in BOUNDARIES[1:]:
        raise ValueError('Require an original trained boundary and every preceding boundary')
    return BOUNDARIES[:BOUNDARIES.index(through_step) + 1]


def published_prefix(records, boundaries):
    steps = [record['step'] for record in records]
    if steps != list(BOUNDARIES[:len(steps)]) or len(steps) < len(boundaries):
        raise ValueError('Require the complete ordered published development prefix without duplicates')
    return records[:len(boundaries)]


def validate_published(report, record, step):
    if (report['step'] != step or record['step'] != step
            or report['accepted_updates'] != record['accepted_updates']
            or report['student'] != record['student']
            or type(report['accepted_updates']) is not int or not 0 <= report['accepted_updates'] <= step):
        raise ValueError('Require the exact completed report and its original published trajectory entry')


def score_bank(bank, reports, reference, label, step, accepted_updates):
    if len(bank) != len(reference):
        raise ValueError('Never truncate the cached truth bank')
    rows, checks = [], 0
    for index, record in enumerate(reference):
        cached = bank[index]
        if (cached['metadata']['seed'] != record['seed']
                or cached['metadata']['gradient_data'] != 'synthetic'
                or list(cached['target'].shape) != [1, record['rows'], record['columns']]):
            raise ValueError('Bind every original synthetic truth case and its complete geometry')
        source, truth, query = (cached[key][0].numpy() for key in ('source', 'target', 'mask'))
        with np.load(record['prediction_file'], allow_pickle=False) as archive:
            arrays = {role: archive[role][0] for role in ('general', 'teacher')}
        for role, report in reports.items():
            if role != 'teacher':
                with np.load(report['cases'][index]['prediction_file'], allow_pickle=False) as archive:
                    arrays[role] = archive['student'][0]
        for score in metrics.score_case(source, truth, query, arrays):
            if record['domain'] == 'imputation' and score['split'] == 'query':
                role = score['role']
                expected = record[role]['rmse'] if role in ('general', 'teacher') else reports[role]['cases'][index]['student']['rmse']
                if not math.isclose(math.sqrt(score['target_mse']), expected, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError('Reproduce every original normalized imputation RMSE')
                checks += 1
            rows.append({'bank': f'{label}_step_{step:03d}', 'run': label, 'step': step,
                'accepted_updates': accepted_updates, 'index': index, 'seed': record['seed'],
                'cohort': metrics.cohort(record),
                'family': record.get('family', str(record.get('prior', record.get('component', 'ratings')))),
                'observed_query_overlap': int((query & np.isfinite(source)).sum()), **score})
    return rows, checks


def analyze(root, output, through_step):
    boundaries = requested_boundaries(through_step)
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized()
            or output.exists() or output.resolve().is_relative_to(root.resolve())):
        raise ValueError('Use hidden CUDA and a new external CPU-only diagnostic output')
    inputs, entries, snapshots = {}, [], {}
    directory = root / 'domain_clipping_v10'
    common_paths = {'teacher': directory / 'reference.json', 'v8': directory / 'validation_0.json'}
    common = {role: metrics.read(path) for role, path in common_paths.items()}
    bank_path = directory / 'synthetic_bank.pt'
    metrics.bind(bank_path, inputs, common['teacher']['bank_sha256'])
    for path in common_paths.values():
        metrics.bind(path, inputs)
    audit_path = root / 'all_imputation_blend_v45_audit/audit.json'
    metrics.bind(audit_path, inputs, HISTORICAL_AUDIT_SHA256)
    historical_audit = metrics.read(audit_path)
    if historical_audit['complete'] is not True:
        raise ValueError('Bind the original complete V45 audit without relabelling the live trial')
    for label, expected_config in RUNS.items():
        run = root / label
        configuration = metrics.read(run / 'config.json')
        metrics.bind(run / 'config.json', inputs, expected_config)
        metrics.bind(run / 'registration.json', inputs)
        if (metrics.read(run / 'registration.json')['config_sha256'] != expected_config
                or configuration['boundaries'] != list(BOUNDARIES)
                or configuration['parent_checkpoint'] != str(root / 'legacy_strata_v29/selected_checkpoint.pth')
                or configuration['input_sha256'].get(str(bank_path)) != common['teacher']['bank_sha256']):
            raise ValueError('Retain original registration, initializer and unchanged synthetic development bank')
        records = [metrics.json.loads(line) for line in (run / 'validation.jsonl').read_text().splitlines()]
        prefix = published_prefix(records, boundaries)
        snapshots[label] = prefix
        initializer_path = run / 'validation_0.json'
        metrics.bind(initializer_path, inputs)
        initializer = metrics.read(initializer_path)
        for step, record in zip(boundaries, prefix):
            path = run / f'validation_{step}.json'
            metrics.bind(path, inputs)
            report = metrics.read(path)
            validate_published(report, record, step)
            reports = {**common, 'actual_v29': initializer, 'student': report}
            reference = metrics.paired_cases(reports, COUNTS)
            for role_report in reports.values():
                for case in role_report['cases']:
                    metrics.bind(case['prediction_file'], inputs, case['prediction_sha256'])
                    if label == 'all_imputation_blend_v45':
                        known = historical_audit['artifact_sha256'].get(case['prediction_file'])
                        if known != case['prediction_sha256']:
                            raise ValueError('Every historical grid must remain bound to the completed audit')
            entries.append((label, step, report['accepted_updates'], reports, reference))
    sources = {str(Path(path).resolve()): metrics.digest(path) for path in
        (__file__, metrics.__file__, Path(__file__).resolve().parents[1] / 'tests/test_single_model_development_teacher_trajectory.py',
         Path(metrics.__file__).resolve().parents[1] / 'tests/test_single_model_teacher_alignment.py')}
    output.mkdir(parents=True)
    metrics.write(output / 'config.json', {'policy': POLICY, 'through_step': through_step,
        'boundaries': list(boundaries), 'input_sha256': inputs, 'source_sha256': sources,
        'published_trajectory_snapshots': snapshots, 'v46_full_independent_audit_claimed': False,
        'torch_version': str(torch.__version__), 'numpy_version': np.__version__})
    metrics.write(output / 'status.json', {'state': 'analyzing'})
    try:
        bank = torch.load(bank_path, map_location='cpu', weights_only=True, mmap=True)
        rows, checks = [], 0
        for label, step, accepted_updates, reports, reference in entries:
            scored, checked = score_bank(bank, reports, reference, label, step, accepted_updates)
            rows.extend(scored)
            checks += checked
        expected_cases = sum(COUNTS.values())
        imputation_cases = expected_cases - COUNTS.get('ratings', 0)
        if (len(rows) != len(RUNS) * len(boundaries) * expected_cases * 10
                or checks != len(RUNS) * len(boundaries) * imputation_cases * 5):
            raise ValueError('Keep every case, role, split and original imputation score check')
        for path, expected in {**inputs, **sources}.items():
            if metrics.digest(path) != expected:
                raise ValueError(f'Diagnostic evidence or source changed: {path}')
        if torch.cuda.is_initialized():
            raise ValueError('No CUDA initialization is permitted')
        metrics.save_csv(output / 'cases.csv', rows)
        metrics.save_csv(output / 'summary.csv', metrics.summarize(rows))
        result = {'complete': True, 'through_step': through_step, 'boundaries': list(boundaries),
            'case_role_split_rows': len(rows), 'raw_imputation_rmse_checks': checks,
            'cases_per_boundary_per_run': expected_cases, 'v46_full_independent_audit_claimed': False,
            'cuda_initialized': False, 'model_inference_performed': False, 'real_data_used': False,
            'training_performed': False, 'new_seeds_consumed': False, 'selection_performed': False, 'gate_changed': False,
            'config_sha256': metrics.digest(output / 'config.json'), 'cases_sha256': metrics.digest(output / 'cases.csv'),
            'summary_sha256': metrics.digest(output / 'summary.csv')}
        metrics.write(output / 'result.json', result)
        metrics.write(output / 'status.json', {'state': 'complete', 'result_sha256': metrics.digest(output / 'result.json')})
        print(metrics.json.dumps(result))
    except BaseException as error:
        metrics.write(output / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through-step', type=int, choices=BOUNDARIES[1:], required=True)
    arguments = parser.parse_args()
    analyze(arguments.root.absolute(), arguments.output.absolute(), arguments.through_step)
