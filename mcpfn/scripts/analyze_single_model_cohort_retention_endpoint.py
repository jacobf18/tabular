"""Describe all V42 endpoint retention penalties without inference or selection."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import single_model_cohort_anchor_retention as retention


ROOT = Path('/mnt/checkpoints_200/single_model_goal_20261005')
REPORTS = {
    'cohort_anchor_retention_v42_pipeline/result.json':
        '82a3da1d556d1ebd380b72642dd8254800c80ea4466da72cd99be7bc7920d1ca',
    'cohort_anchor_retention_v42_audit/audit.json':
        '30aaeb3f4397b59bab54157232555032784c26c2603702ef4745de26e2c74eeb',
    'cohort_anchor_retention_v42_endpoint_replay/audit.json':
        '54f6b00b203f2245f2aa4e4a77177f97fe8dc1f71c55af528bb03f621b206b6a',
}
POLICY = ('CPU-only descriptive scoring of every original V42 step64 endpoint task. '
    'Use only existing independently audited before/candidate prediction grids, original '
    'synthetic targets/masks and the registered frozen V8/V29 reference archives. '
    'Keep all26 imputation tasks and both rating tasks; ratings have no retention penalty. '
    'Report unweighted per-task penalties and original mixture-weighted sums separately. '
    'Do not infer gradient dominance or causation from loss magnitudes. No model loading, '
    'new examples, inference, optimization, checkpoint selection or changed qualification. '
    'This endpoint description is not a full-trajectory replay or a real benchmark result.')


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def bind(path, expected, identities):
    path = Path(path)
    if digest(path) != expected:
        raise ValueError(f'Archived input changed: {path}')
    identities[str(path)] = expected
    return path


def read(path):
    return json.loads(Path(path).read_text())


def same_tensor(actual, expected):
    return (actual.dtype == expected.dtype and actual.shape == expected.shape
        and torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['reference_role'] is None:
            continue
        for grouping, name in [('all', 'all'), ('reference', row['reference_role']),
                ('cohort', row['cohort'])]:
            groups[(grouping, name)].append(row)
    summaries = []
    for (grouping, name), selected in sorted(groups.items()):
        summaries.append({'grouping': grouping, 'name': name, 'tasks': len(selected),
            'increased': sum(row['after'] > row['before'] for row in selected),
            'decreased': sum(row['after'] < row['before'] for row in selected),
            'unchanged': sum(row['after'] == row['before'] for row in selected),
            'mean_before': math.fsum(row['before'] for row in selected) / len(selected),
            'mean_after': math.fsum(row['after'] for row in selected) / len(selected),
            'mixture_weighted_before': math.fsum(row['weight'] * row['before'] for row in selected),
            'mixture_weighted_after': math.fsum(row['weight'] * row['after'] for row in selected)})
    return summaries


def run(output):
    if torch.cuda.is_initialized():
        raise RuntimeError('This diagnostic must remain CPU-only')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Never replace a diagnostic result')
    identities = {}
    reports = [read(bind(ROOT / name, expected, identities)) for name, expected in REPORTS.items()]
    pipeline, audit, endpoint_audit = reports
    if (not pipeline['complete'] or pipeline['gate']['passed']
            or pipeline['gate']['confirmation_evaluated'] or audit['through_step'] != 64
            or not audit['complete'] or endpoint_audit['endpoint_grids'] != 56
            or not endpoint_audit['complete'] or endpoint_audit['selected_grids'] != 0):
        raise ValueError('Require the closed unqualified V42 trial and complete endpoint audit')
    config = read(bind(ROOT / 'cohort_anchor_retention_v42/config.json',
        'fec67da9d9de7a0127d9daa77b9ef68745d7a2feae4748407f3832d21b939b22', identities))
    for path, expected in config['source_sha256'].items():
        bind(path, expected, identities)
    training_path = ROOT / 'cohort_anchor_retention_v42/training.jsonl'
    bind(training_path, audit['training_prefix_sha256'], identities)
    rows = [json.loads(line) for line in training_path.read_text().splitlines()]
    if [row['step'] for row in rows] != list(range(1, 65)) or not rows[-1]['accepted']:
        raise ValueError('Require the complete original trajectory and accepted step64')
    terminal = rows[-1]
    tasks = read(bind(terminal['task_manifest'], terminal['task_manifest_sha256'], identities))['tasks']
    references = read(bind(terminal['retention_manifest'], terminal['retention_manifest_sha256'], identities))
    if not references['complete'] or len(tasks) != 28 or len(references['records']) != 28:
        raise ValueError('Never truncate endpoint task or reference coverage')
    records = {row['seed']: row for row in references['records']}
    if len(records) != 28:
        raise ValueError('Require distinct original endpoint seeds')
    scored = []
    for index, task in enumerate(tasks):
        if task['index'] != index:
            raise ValueError('Preserve original task ordering')
        bind(task['artifact'], task['sha256'], identities)
        task_path = ROOT / 'cohort_anchor_retention_v42_endpoint_replay/endpoint_tasks' / f'task_{index:02d}.pt'
        payload = torch.load(bind(task_path, endpoint_audit['artifact_sha256'][str(task_path)], identities),
            map_location='cpu', weights_only=True, mmap=True)
        if any(payload['task'][key] != task[key] for key in payload['task']):
            raise ValueError('Require the independently regenerated original endpoint task')
        episode = payload['episode']
        metadata = episode['metadata']
        record = records[metadata['seed']]
        role = retention.reference_role(metadata)
        if record['reference_role'] != role or record['metadata'] != metadata:
            raise ValueError('Require the original metadata-only reference assignment')
        current = {'index': index, 'seed': metadata['seed'], 'cohort': task['cohort'],
            'reference_role': role, 'weight': task['weight'], 'queries': int(episode['mask'].sum())}
        if role is not None:
            reference = torch.load(bind(record['artifact'], record['sha256'], identities),
                map_location='cpu', weights_only=True, mmap=True)
            if (reference['reference_role'] != role or reference['specification'] != retention.SPECIFICATION
                    or reference['identity']['sha256'] != retention.REFERENCE_SHA256S[role]
                    or not same_tensor(reference['source'], episode['source'])
                    or not same_tensor(reference['mask'], episode['mask'])):
                raise ValueError('Require the complete original reference/source/mask identity')
            for prefix, label in [('before', 'before'), ('candidate', 'after')]:
                path = ROOT / 'cohort_anchor_retention_v42_endpoint_replay' / f'{prefix}_task_{index:02d}.npz'
                bind(path, endpoint_audit['artifact_sha256'][str(path)], identities)
                with np.load(path, allow_pickle=False) as archive:
                    prediction = torch.from_numpy(archive['prediction'].copy())
                current[label] = retention.independent_penalty(episode['target'], episode['mask'],
                    prediction, reference['prediction'])
        scored.append(current)
    if sum(row['reference_role'] is not None for row in scored) != 26:
        raise ValueError('Require every original imputation task and both rating exclusions')
    for path, expected in identities.items():
        if digest(path) != expected:
            raise ValueError('Inputs changed during the CPU diagnostic')
    if torch.cuda.is_initialized():
        raise RuntimeError('The CPU diagnostic initialized CUDA')
    report = {'complete': True, 'policy': POLICY, 'endpoint_step': 64, 'accepted_updates': 58,
        'tasks': scored, 'summary': summarize(scored), 'artifact_sha256': identities,
        'source_sha256': {str(path): digest(path) for path in [Path(__file__).resolve(),
            Path(__file__).resolve().parent.parent / 'tests/test_single_model_cohort_retention_endpoint.py']},
        'cuda_initialized': False, 'training_performed': False, 'model_inference_performed': False,
        'checkpoint_selected': False, 'goal_success_not_assessed': True}
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'result': str(output / 'result.json'), 'sha256': digest(output / 'result.json'),
        'summary': report['summary']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
