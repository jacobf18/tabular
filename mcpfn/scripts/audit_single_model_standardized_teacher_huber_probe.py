"""Independently reconstruct standardized mean terms on every fresh probe column."""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import shutil
import types

import numpy as np
import torch

import audit_single_model_balanced_structured_components_probe as blueprint
import audit_single_model_native_column_mean_probe as original_math
import preflight_single_model_standardized_teacher_huber as probe
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + '_mean_audit')
CONFIG_SHA256 = None
POLICY = (
    'CPU-only supplementary independent teacher-mean audit for one prospectively registered '
    'standardized-Huber feasibility update. Require its stopped original complete proposal, '
    'teacher-component and retention-component audits and the explicit original configuration '
    'SHA. Reconstruct equal-column Huber of count-shrunk variance-standardized residuals on '
    'all26 imputation tasks, using independent float64 NumPy moments, column masses, scales '
    'and piecewise Huber arithmetic. Preserve both original uniform mean20 rating terms, '
    'every column including empty-query/singleton/constant columns, both original endpoints '
    'and every adverse contribution. Compare geometry and the complete pure-mean term with '
    'the registered compiled callback. No new inference, scientific seed, training, gradient, '
    'selection, gate, full-composite replay or historical autograd claim. Feasibility and '
    'objective arithmetic are not held-out or real-benchmark qualification.'
)
huber = blueprint.huber
imputation_task = types.FunctionType(blueprint.imputation_task.__code__, globals(), 'imputation_task',
    blueprint.imputation_task.__defaults__)


def describe(target, mask, teacher, predictions, *, native):
    if (target.ndim != 3 or target.shape[0] != 1 or mask.shape != target.shape or mask.dtype != torch.bool
            or teacher.shape != target.shape or set(predictions) != {'before', 'candidate'}
            or not mask.any() or target.device.type != 'cpu' or mask.device.type != 'cpu'
            or teacher.device.type != 'cpu' or any(value.shape != target.shape for value in predictions.values())
            or not torch.isfinite(target[mask]).all() or not torch.isfinite(teacher[mask]).all()
            or any(not np.isfinite(value[mask.numpy()]).all() for value in predictions.values())):
        raise ValueError('Use complete finite queried CPU grids and both original endpoints')
    labels, selected, means = target[0].double().numpy(), mask[0].numpy(), teacher[0].double().numpy()
    queries = int(selected.sum())
    queried_columns = int(selected.any(axis=0).sum())
    residuals = {role: prediction[0].astype(np.float64) - means for role, prediction in predictions.items()}
    if native:
        weights, scales = probe.retention.column_geometry(target, mask)
        actual_weights, actual_scales = weights[0].numpy(), scales[0].numpy()
    rows = []
    for column_index in range(target.shape[-1]):
        chosen = selected[:, column_index]
        count = int(chosen.sum())
        variance = float(np.var(labels[chosen, column_index], ddof=0)) if count else None
        denominator = max((count * variance + 1.) / (count + 1.), 1e-6) if count else None
        coefficient = queries / (queried_columns * count) if native and count else 1. if count else 0.
        scale = math.sqrt(denominator) if native and count else 1.
        if native and (not np.allclose(actual_weights[chosen, column_index], np.float32(coefficient), rtol=1e-6, atol=0.)
                or not np.allclose(actual_scales[chosen, column_index], np.float32(scale), rtol=1e-6, atol=0.)
                or np.any(actual_weights[~chosen, column_index] != 0.)
                or np.any(actual_scales[~chosen, column_index] != 1.)):
            raise ValueError('Every original mass and scale must match without omitting query cells')
        row = {'column': column_index, 'queries': count, 'variance': variance, 'shrunk_variance': denominator,
            'applied_coefficient': coefficient, 'residual_standardization_scale': scale,
            'single_query': count == 1, 'constant_multi_query': count > 1 and variance == 0.}
        for role in predictions:
            residual = residuals[role][chosen, column_index]
            row[role + '_uniform_mean20_contribution'] = 20. * float(np.sum(huber(residual))) / queries
            row[role + '_applied_mean20_contribution'] = 20. * coefficient * float(np.sum(huber(residual / scale))) / queries
        row['applied_change'] = row['candidate_applied_mean20_contribution'] - row['before_applied_mean20_contribution']
        rows.append(row)
    values = {role + '_' + weighting + '_mean20': math.fsum(row[role + '_' + weighting + '_mean20_contribution'] for row in rows)
        for role in predictions for weighting in ('uniform', 'applied')}
    if native:
        for role, prediction in predictions.items():
            point = torch.from_numpy(prediction)[mask].float()
            teacher_values = teacher[mask].float()
            unweighted = 20. * torch.nn.functional.huber_loss(point, teacher_values, delta=1., reduction='none')
            compiled = probe.retention.reweight_teacher_mean(unweighted, point, teacher_values, weights[mask], scales[mask])
            if not math.isclose(float(compiled.mean()), values[role + '_applied_mean20'], rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError('Independent standardized mean term must match the registered complete callback')
    values.update(queries=queries, columns=target.shape[-1], queried_columns=queried_columns,
        applied_change=values['candidate_applied_mean20'] - values['before_applied_mean20'])
    return rows, values


AUDIT_SOURCE = replace_once(blueprint.AUDIT_SOURCE,
    'test_single_model_balanced_structured_components_probe_audit.py',
    'test_single_model_standardized_teacher_huber_probe_audit.py')
exec(compile(AUDIT_SOURCE, __file__, 'exec'), globals())
_audit = main
audit = types.FunctionType(blueprint.audit.__code__, globals(), 'audit', blueprint.audit.__defaults__)
main = types.FunctionType(blueprint.main.__code__, globals(), 'main', blueprint.main.__defaults__)


if __name__ == '__main__':
    main()
