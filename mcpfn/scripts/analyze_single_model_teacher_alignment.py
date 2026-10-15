"""Read-only output-space alignment on complete, already consumed synthetic banks."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path('/mnt/checkpoints_200/single_model_goal_20261005')
RUN = 'all_imputation_blend_v45'
POLICY = (
    'Use every original case in the completed V45 development and consumed confirmation banks. '
    'Compare saved outputs to the original observed-blend imputation teacher, specialist rating '
    'teacher, frozen V8 retention reference and general parent. Actual V29 initialization is '
    'available only on development and must never be conflated with V8. Keep observed-source '
    'reconstruction separate from original synthetic queries; overlapping masks stay unchanged. '
    'All distances use original normalized network coordinates, not real-benchmark metrics or '
    'original-unit user-macro rating errors. Query imputation RMSE must reproduce stored scores. '
    'Equal-case descriptive summaries and count-shrunk output distances are not parameter-gradient '
    'conflict estimates, causal evidence, selection rules or permission to change qualification. '
    'No inference, CUDA, fresh seeds, training, real data, gate changes or source-evidence writes.'
)
METRICS = ('target_mse', 'teacher_mse', 'v8_mse', 'general_mse', 'teacher_huber',
           'teacher_shrunk_mse', 'v8_shrunk_mse')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def bind(path, inputs, expected=None):
    path = Path(path).absolute()
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f'Original artifact identity changed: {path}')
    if str(path) in inputs and inputs[str(path)] != actual:
        raise ValueError('An input cannot carry contradictory identities')
    inputs[str(path)] = actual


def cohort(record):
    return 'ratings' if record['domain'] == 'recsys' else 'structured' if record.get('structured') else 'legacy'


def paired_cases(reports, expected_counts):
    reference = reports['teacher']['cases']
    if Counter(cohort(record) for record in reference) != expected_counts:
        raise ValueError('Retain complete original cohort coverage')
    identity = ('index', 'seed', 'rows', 'columns', 'domain', 'gradient_data')
    for report in reports.values():
        if len(report['cases']) != len(reference):
            raise ValueError('Retain every original case in every role')
        for index, (original, candidate) in enumerate(zip(reference, report['cases'])):
            if (original['index'] != index or original['gradient_data'] != 'synthetic'
                    or original['domain'] not in ('imputation', 'recsys')
                    or any(original[field] != candidate[field] for field in identity)
                    or bool(original.get('structured')) != bool(candidate.get('structured'))):
                raise ValueError('Do not reorder, relabel, replace or truncate synthetic cases')
    for original in reference:
        expected = 'specialist' if original['domain'] == 'recsys' else 'same-input observed blend'
        if original.get('routing', {}).get('target') != expected:
            raise ValueError('Keep the original domain-specific frozen teacher target')
    return reference


def squared_distances(prediction, reference, target, mask):
    if (mask.dtype != bool or mask.ndim != 2 or not mask.any()
            or any(values.shape != mask.shape for values in (prediction, reference, target))
            or any(not np.isfinite(values[mask]).all() for values in (prediction, reference, target))):
        raise ValueError('Require complete finite selected arrays and a nonempty original mask')
    residual = prediction.astype(np.float64) - reference.astype(np.float64)
    pooled = float(np.mean(residual[mask] ** 2))
    contributions = []
    for column_index in range(mask.shape[1]):
        selected = mask[:, column_index]
        count = int(selected.sum())
        if count:
            variance = float(np.var(target[selected, column_index].astype(np.float64)))
            denominator = max((count * variance + 1.) / (count + 1.), 1e-6)
            contributions.append(float(np.mean(residual[selected, column_index] ** 2)) / denominator)
    return pooled, math.fsum(contributions) / len(contributions)


def score_case(source, truth, query, arrays):
    if set(arrays) not in ({'general', 'teacher', 'v8', 'student'},
                           {'general', 'teacher', 'v8', 'student', 'actual_v29'}):
        raise ValueError('Keep explicit teacher, general, V8 and actual-initializer roles')
    rows = []
    for split, mask, target in [('query', query, truth), ('observed', np.isfinite(source), source)]:
        if not mask.any():
            raise ValueError('Keep an empty-evidence case explicit rather than silently omitting it')
        for role, prediction in arrays.items():
            teacher_mse, teacher_shrunk = squared_distances(prediction, arrays['teacher'], target, mask)
            v8_mse, v8_shrunk = squared_distances(prediction, arrays['v8'], target, mask)
            residual = np.abs(prediction[mask].astype(np.float64) - arrays['teacher'][mask].astype(np.float64))
            target_error = prediction[mask].astype(np.float64) - target[mask].astype(np.float64)
            general_error = prediction[mask].astype(np.float64) - arrays['general'][mask].astype(np.float64)
            rows.append({'split': split, 'role': role, 'selected_cells': int(mask.sum()),
                'selected_columns': int(mask.any(axis=0).sum()), 'target_mse': float(np.mean(target_error ** 2)),
                'teacher_mse': teacher_mse, 'v8_mse': v8_mse, 'general_mse': float(np.mean(general_error ** 2)),
                'teacher_huber': float(np.mean(np.where(residual <= 1., .5 * residual ** 2, residual - .5))),
                'teacher_shrunk_mse': teacher_shrunk, 'v8_shrunk_mse': v8_shrunk})
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for grouping, group in [('all', 'ALL'), ('family', row['family'])]:
            groups[row['bank'], row['cohort'], row['split'], row['role'], grouping, group].append(row)
    output = []
    for key, selected in sorted(groups.items()):
        if len({row['index'] for row in selected}) != len(selected):
            raise ValueError('Do not duplicate case weight within a group')
        output.append(dict(zip(('bank', 'cohort', 'split', 'role', 'grouping', 'group'), key),
            cases=len(selected), cells=sum(row['selected_cells'] for row in selected),
            **{metric: math.fsum(row[metric] for row in selected) / len(selected) for metric in METRICS}))
    return output


def save_csv(path, rows):
    with Path(path).open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output):
    if output.exists() or output.resolve().is_relative_to(root.resolve()) or torch.cuda.is_initialized():
        raise ValueError('Use a new external CPU-only diagnostic output')
    run = root / RUN
    pipeline = root / f'{RUN}_pipeline'
    confirmation = root / f'{RUN}_confirmation'
    review = root / f'{RUN}_review'
    if any(read(path / 'status.json')['state'] != 'complete' for path in (pipeline, confirmation, review)):
        raise ValueError('Require complete original qualification and real-review closure')
    gate = read(confirmation / 'gate.json')
    if not gate['passed'] or gate['selected_step'] != 50:
        raise ValueError('Bind the original sole selected step50; never choose another checkpoint')
    inputs = {}
    pipeline_result = read(pipeline / 'result.json')
    training_audit_path = root / f'{RUN}_audit/audit.json'
    bind(training_audit_path, inputs, pipeline_result['training_audit_sha256'])
    bind(confirmation / 'audit.json', inputs, gate['audit_sha256'])
    training_config = read(run / 'config.json')
    training_audit = read(training_audit_path)
    if (training_audit['training_initialization'] != str(root / 'legacy_strata_v29/selected_checkpoint.pth')
            or training_audit['retention_baseline'] != str(root / 'query_budget_v8/best_validation_checkpoint.pth')):
        raise ValueError('Do not conflate actual V29 initialization and V8 retention')
    identities = {}
    for recorded in (training_config['input_sha256'], training_audit['artifact_sha256'],
                     read(confirmation / 'audit.json')['artifact_sha256']):
        for path, expected in recorded.items():
            if path in identities and identities[path] != expected:
                raise ValueError('Original audits contain contradictory artifact identities')
            identities[path] = expected

    def original_digest(path):
        return identities.get(str(Path(path).absolute()), identities.get(str(Path(path).resolve())))

    for path in (run / 'config.json', run / 'registration.json', pipeline / 'config.json',
                 pipeline / 'result.json', confirmation / 'gate.json', confirmation / 'audit.json',
                 review / 'result.json'):
        bind(path, inputs)
    if digest(run / 'config.json') != read(run / 'registration.json')['config_sha256']:
        raise ValueError('Training registration changed')
    bind(run / 'selected_checkpoint.pth', inputs, gate['checkpoint_sha256'])
    banks = []
    for name, directory, expected_counts in [
            ('development', root / 'domain_clipping_v10', {'legacy': 40, 'structured': 96, 'ratings': 16}),
            ('consumed_confirmation', confirmation, {'legacy': 60, 'structured': 192, 'ratings': 32})]:
        paths = {'teacher': directory / 'reference.json',
                 'v8': directory / ('validation_0.json' if name == 'development' else 'initialization.json'),
                 'student': run / 'validation_50.json' if name == 'development' else directory / 'student.json'}
        if name == 'development':
            paths['actual_v29'] = run / 'validation_0.json'
        reports = {role: read(path) for role, path in paths.items()}
        reference = paired_cases(reports, expected_counts)
        bank = directory / 'synthetic_bank.pt'
        bind(bank, inputs, reports['teacher']['bank_sha256'])
        for path in paths.values():
            bind(path, inputs, original_digest(path))
        for report in reports.values():
            for record in report['cases']:
                expected = original_digest(record['prediction_file'])
                if expected is None or expected != record['prediction_sha256']:
                    raise ValueError('Every saved grid must remain bound to its original completed audit')
                bind(record['prediction_file'], inputs, record['prediction_sha256'])
        banks.append((name, bank, reports, reference))
    sources = {str(Path(__file__).resolve()): digest(__file__),
               str(Path(__file__).resolve().parents[1] / 'tests/test_single_model_teacher_alignment.py'):
                   digest(Path(__file__).resolve().parents[1] / 'tests/test_single_model_teacher_alignment.py')}
    output.mkdir(parents=True)
    write(output / 'config.json', {'policy': POLICY, 'input_sha256': inputs, 'source_sha256': sources,
        'selected_step': 50, 'checkpoint_sha256': gate['checkpoint_sha256'],
        'torch_version': str(torch.__version__), 'numpy_version': np.__version__})
    write(output / 'status.json', {'state': 'analyzing'})
    rows = []
    checks = 0
    try:
        for name, bank_path, reports, reference in banks:
            bank = torch.load(bank_path, map_location='cpu', weights_only=True, mmap=True)
            if len(bank) != len(reference):
                raise ValueError('Never truncate the cached truth bank')
            for index, record in enumerate(reference):
                cached = bank[index]
                if (cached['metadata']['seed'] != record['seed']
                        or cached['metadata']['gradient_data'] != 'synthetic'
                        or list(cached['target'].shape) != [1, record['rows'], record['columns']]):
                    raise ValueError('Bind complete original synthetic truth and geometry')
                source, truth, query = (cached[key][0].numpy() for key in ('source', 'target', 'mask'))
                with np.load(record['prediction_file'], allow_pickle=False) as archive:
                    arrays = {role: archive[role][0] for role in ('general', 'teacher')}
                for role, report in reports.items():
                    if role != 'teacher':
                        with np.load(report['cases'][index]['prediction_file'], allow_pickle=False) as archive:
                            arrays[role] = archive['student'][0]
                scores = score_case(source, truth, query, arrays)
                for score in scores:
                    if record['domain'] == 'imputation' and score['split'] == 'query':
                        role = score['role']
                        expected = record[role]['rmse'] if role in ('general', 'teacher') else reports[role]['cases'][index]['student']['rmse']
                        if not math.isclose(math.sqrt(score['target_mse']), expected, rel_tol=1e-6, abs_tol=1e-6):
                            raise ValueError('Normalized imputation RMSE must reproduce the original score')
                        checks += 1
                    rows.append({'bank': name, 'index': index, 'seed': record['seed'], 'cohort': cohort(record),
                        'family': record.get('family', str(record.get('prior', record.get('component', 'ratings')))),
                        'observed_query_overlap': int((query & np.isfinite(source)).sum()), **score})
        for path, expected in {**inputs, **sources}.items():
            if digest(path) != expected:
                raise ValueError(f'Diagnostic input or source changed: {path}')
        if torch.cuda.is_initialized():
            raise ValueError('No CUDA initialization is permitted')
        summaries = summarize(rows)
        save_csv(output / 'cases.csv', rows)
        save_csv(output / 'summary.csv', summaries)
        result = {'complete': True, 'policy': POLICY, 'development_cases': 152, 'consumed_confirmation_cases': 284,
            'raw_imputation_rmse_checks': checks, 'case_role_split_rows': len(rows),
            'cuda_initialized': False, 'model_inference_performed': False, 'real_data_used': False,
            'training_performed': False, 'selection_performed': False, 'gate_changed': False,
            'config_sha256': digest(output / 'config.json'), 'cases_sha256': digest(output / 'cases.csv'),
            'summary_sha256': digest(output / 'summary.csv')}
        write(output / 'result.json', result)
        write(output / 'status.json', {'state': 'complete', 'result_sha256': digest(output / 'result.json')})
        print(json.dumps({key: value for key, value in result.items() if key != 'policy'}))
    except BaseException as error:
        write(output / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.root.absolute(), arguments.output.absolute())
