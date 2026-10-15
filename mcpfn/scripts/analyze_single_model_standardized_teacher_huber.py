"""Describe the proposed Huber geometry on every saved V53 terminal task."""

from collections import Counter
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional

import single_model_standardized_teacher_huber as objective
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json, write_csv


ROOT = Path('/mnt/checkpoints_200/single_model_goal_20261005')
RUN = ROOT / 'balanced_structured_components_v53'
REPLAY = ROOT / 'balanced_structured_components_v53_endpoint_replay'
CLOSURE = ROOT / 'balanced_structured_components_v53_closure_audit/result.json'
OUTPUT = ROOT / 'standardized_teacher_huber_endpoint_analysis_v1'
POLICY = (
    'CPU-only descriptive analysis of all28 saved V53 terminal tasks and both original '
    'before/candidate prediction grids, after audited negative closure. Compare the raw '
    'inverse-variance Huber teacher term with the prospective standardized-residual term. '
    'Keep every query, task, role, family and component; independently check equal-column '
    'losses and output-coordinate derivatives. Ratings remain exactly unchanged. Preserve '
    'every adverse change and unequal variance regime. Different objective values are not '
    'accuracy gains. Coordinate-gradient norms are not shared-network parameter-gradient '
    'norms, optimization dominance or causal explanations of held-out regressions. No '
    'model checkpoint loading, model inference, CUDA initialization, new synthetic seed '
    'or data, optimizer update, selection, confirmation, gate change or real-data use.'
)


def read(path):
    import json
    return json.loads(Path(path).read_text())


def diagnostics(target, mask, prediction, teacher, metadata, weight):
    objective.validate_columns(target, mask)
    if (prediction.shape != target.shape or teacher.shape != target.shape
            or any(tensor.device.type != 'cpu' for tensor in (target, mask, prediction, teacher))
            or not torch.isfinite(prediction[mask]).all() or not torch.isfinite(teacher[mask]).all()
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError('Keep every finite original CPU query and positive task weight')
    role = objective.reference_role(metadata)
    residual = prediction[mask].float() - teacher.detach()[mask].float()
    raw_huber = functional.huber_loss(residual, torch.zeros_like(residual), reduction='none')
    queries = int(mask.sum())
    if role is None:
        coefficients = masses = scales = torch.ones_like(residual)
    else:
        coefficients = objective.raw_mean.column_coefficients(target, mask)[mask]
        masses_grid, scales_grid = objective.column_geometry(target, mask)
        masses, scales = masses_grid[mask], scales_grid[mask]
        torch.testing.assert_close(objective.column_coefficients(target, mask),
            objective.raw_mean.column_coefficients(target, mask), rtol=0, atol=0)
    old_loss = 20. * (coefficients * raw_huber).mean()
    normalized = residual / scales
    new_loss = 20. * (masses * functional.huber_loss(normalized, torch.zeros_like(normalized), reduction='none')).mean()
    old_gradient = 20. * coefficients.double() * residual.double().clamp(-1., 1.) / queries
    new_gradient = 20. * masses.double() * normalized.double().clamp(-1., 1.) / (scales.double() * queries)
    point = prediction[mask].detach().clone().requires_grad_()
    base = 20. * functional.huber_loss(point, teacher.detach()[mask], reduction='none')
    actual = objective.reweight_teacher_mean(base, point, teacher[mask], masses, scales).mean()
    actual_gradient = torch.autograd.grad(actual, point)[0]
    torch.testing.assert_close(actual, new_loss, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(actual_gradient.double(), new_gradient, rtol=2e-5, atol=2e-7)
    if role is None:
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        torch.testing.assert_close(old_gradient, new_gradient, rtol=0, atol=0)
    else:
        independent = objective.independent_teacher_mean(target, mask, prediction, teacher)
        if not math.isclose(float(new_loss), independent, rel_tol=3e-6, abs_tol=3e-6):
            raise ValueError('Independent float64 standardized teacher term must match every original query')
    old_norm, new_norm = float(old_gradient.norm()), float(new_gradient.norm())
    return {'queries': queries, 'queried_columns': int(mask.any(dim=1).sum()),
        'teacher_huber_raw': float(old_loss), 'teacher_huber_standardized': float(new_loss),
        'coordinate_gradient_norm_raw': old_norm, 'coordinate_gradient_norm_standardized': new_norm,
        'weighted_coordinate_gradient_norm_raw': weight * old_norm,
        'weighted_coordinate_gradient_norm_standardized': weight * new_norm,
        'gradient_norm_ratio': new_norm / old_norm if old_norm else None,
        'gradient_magnitude_decreased_cells': int((new_gradient.abs() < old_gradient.abs()).sum()),
        'gradient_magnitude_increased_cells': int((new_gradient.abs() > old_gradient.abs()).sum()),
        'gradient_magnitude_equal_cells': int((new_gradient.abs() == old_gradient.abs()).sum()),
        'raw_linear_cells': int((residual.abs() > 1.).sum()),
        'standardized_linear_cells': int((normalized.abs() > 1.).sum()),
        'scale_min': float(scales.min()), 'scale_max': float(scales.max()),
        'task_weight': weight, 'retention_reference': role or 'none',
        'rating_term_exactly_unchanged': role is None}


def validate_case_inventory(cases):
    identities = [(case['role'], case['task']) for case in cases]
    cohorts = ('legacy', *objective.FAMILIES, 'recsys')
    expected = {(role, f'{cohort}_draw_{replicate}') for role in ('before', 'candidate')
        for replicate in (0, 1) for cohort in cohorts}
    if len(identities) != 56 or len(set(identities)) != 56 or set(identities) != expected:
        raise ValueError('Require all28 original terminal tasks and both original prediction roles')


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError('Never overwrite a registered endpoint analysis')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Require hidden uninitialized CUDA for saved-grid analysis')
    torch.set_num_threads(4)
    if (shutil.disk_usage(ROOT).free - 16 * 1024**2 < 18 * 1024**3
            or shutil.disk_usage('/').free - 16 * 1024**2 < 6 * 1024**3):
        raise OSError('Preserve both original reserves and small diagnostic write allowances')
    closure, endpoint, audit = read(CLOSURE), read(REPLAY / 'result.json'), read(REPLAY / 'audit.json')
    if (closure.get('complete') is not True or closure.get('unqualified') is not True
            or closure['gate']['passed'] is not False or closure['confirmation_cases_generated'] != 0
            or closure['endpoint_audit_sha256'] != sha256(REPLAY / 'audit.json')
            or audit['complete'] is not True or audit['endpoint_grids'] != 56 or audit['selected_grids'] != 0
            or endpoint['endpoint_step'] != 64 or endpoint['selected'] is not None):
        raise ValueError('Require the completed original V53 negative closure and full endpoint audit')
    validate_case_inventory(endpoint['cases'])
    inputs = {str(path): sha256(path) for path in (CLOSURE, REPLAY / 'result.json', REPLAY / 'audit.json')}
    for case in endpoint['cases']:
        for field in ('artifact', 'prediction'):
            path, digest = case[field], case[field + '_sha256']
            if audit['artifact_sha256'].get(path) != digest:
                raise ValueError('Every saved task and prediction must retain its independent original audit binding')
            inputs[path] = digest
    sources = dict(read(RUN / 'config.json')['source_sha256'])
    for path in (Path(__file__).resolve(), Path(objective.__file__).resolve(),
            REPO / 'tests/test_single_model_standardized_teacher_huber.py',
            REPO / 'tests/test_single_model_standardized_teacher_huber_analysis.py'):
        sources[str(path)] = sha256(path)
    verify_hashes(inputs)
    verify_hashes(sources)
    config = {'policy': POLICY, 'mean_specification': objective.MEAN_SPECIFICATION,
        'input_sha256': inputs, 'source_sha256': sources, 'tasks': 28, 'roles': ['before', 'candidate']}
    OUTPUT.mkdir()
    write_json(OUTPUT / 'config.json', config)
    rows = []
    for case in endpoint['cases']:
        payload = torch.load(case['artifact'], map_location='cpu', weights_only=False, mmap=True)
        episode, task = payload['episode'], payload['task']
        if task['name'] != case['task'] or episode['metadata'].get('gradient_data') != 'synthetic':
            raise ValueError('Keep the original complete synthetic task identities')
        with np.load(case['prediction'], allow_pickle=False) as stored:
            if stored.files != ['prediction'] or stored['prediction'].dtype != np.float32:
                raise ValueError('Use only the exact complete original FP32 prediction grid')
            prediction = torch.from_numpy(stored['prediction'].copy())
        row = {'role': case['role'], 'task': task['name'], 'task_index': task['index'],
            'cohort': task['cohort'], 'component': str(episode['metadata'].get('component', 'ratings')),
            'seed': episode['metadata']['seed'], 'rows': episode['target'].shape[1],
            'columns': episode['target'].shape[2]}
        row.update(diagnostics(episode['target'], episode['mask'], prediction, payload['teacher'],
            episode['metadata'], task['weight']))
        rows.append(row)
        del payload, episode, prediction
    if Counter(row['role'] for row in rows) != {'before': 28, 'candidate': 28}:
        raise ValueError('Never drop an original role or task')
    write_csv(OUTPUT / 'tasks.csv', rows)
    write_json(OUTPUT / 'tasks.json', rows)
    verify_hashes(inputs)
    verify_hashes(sources)
    write_json(OUTPUT / 'receipt.json', {'complete': True, 'task_role_rows': 56, 'imputation_rows': 52,
        'rating_rows': 4, 'config_sha256': sha256(OUTPUT / 'config.json'),
        'output_sha256': {str(OUTPUT / name): sha256(OUTPUT / name) for name in ('tasks.csv', 'tasks.json')},
        'retention_coefficients_bitwise_unchanged': True, 'rating_terms_and_coordinate_gradients_bitwise_unchanged': True,
        'model_inference_performed': False, 'new_scientific_seeds': False, 'training_performed': False,
        'checkpoint_selection_performed': False, 'confirmation_evaluated': False, 'real_data_used': False,
        'cuda_initialized': torch.cuda.is_initialized(), 'goal_success_not_assessed': True})


if __name__ == '__main__':
    main()
