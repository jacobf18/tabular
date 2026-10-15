"""Prospective variance-scaled teacher residuals with unchanged inference and retention."""

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import hashlib
import math
from pathlib import Path
import types

import numpy as np
import torch
from torch.nn import functional

import single_model_all_imputation_column_mean as raw_mean
from single_model_all_imputation_column_mean import (
    predecessor, validate_columns, compute_context, query, column, query_budget_majorizer,
    TRAIN_OPTIONS, streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES, stronger,
)
from single_model_registered_adaptation import replace_once


REFERENCE_SHA256S = dict(raw_mean.REFERENCE_SHA256S)
STRUCTURED_MASS = raw_mean.STRUCTURED_MASS
REPLICAS = raw_mean.REPLICAS
STRUCTURED_TASK_MASS = raw_mean.STRUCTURED_TASK_MASS
SPECIFICATION = deepcopy(raw_mean.SPECIFICATION)
MEAN_SPECIFICATION = deepcopy(raw_mean.MEAN_SPECIFICATION)
MEAN_SPECIFICATION.update(
    residual_units='student-minus-detached-teacher residual divided by detached count-shrunk query-target standard deviation',
    coefficient='total_queries / (queried_columns * column_queries)',
    scale='sqrt(max((column_queries * population_variance + 1) / (column_queries + 1), 1e-6))',
    structured_and_rating_callbacks='all native and structured imputation use standardized Huber; ratings unchanged mean20',
    moments='detached float64 synthetic target moments; float32 callback weights and scales',
)
POLICY = (
    'Prospective objective hypothesis after fully audited negative V53 closure. Keep its balanced '
    'structured component schedule and every original native/structured/rating task. Replace only '
    'all-imputation inverse-variance-weighted raw-residual teacher Huber by equal-column Huber '
    'of the variance-standardized teacher residual, with delta1 and weight20. Use the same '
    'detached synthetic query-target variance, one unit-variance pseudocount and1e-6 floor. '
    'Every queried column and cell remains, including singleton/constant columns. Apply '
    'uniformly to all eight native components and all twelve structured families/four priors, '
    'never select by validation case, error, seed, real data or teacher advantage. This keeps '
    'the small-residual quadratic objective but makes the Huber transition proportional to '
    'the shrunk column scale: low-variance teacher disagreements may exert less influence. '
    'Large-variance transitions also change; downweighting teacher disagreement can harm '
    'useful corrections. The unit-variance pseudocount means this is not exact rescaling '
    'invariance. Output-gradient bounds are not parameter-gradient or generalization claims. '
    'Keep actual V29 initialization/native retention, fixed-half V8/V29 structured retention, '
    'observed-only blended imputation teachers, specialist rating teacher, native/rating priors, '
    'normalization, mean20 rating callbacks, NLL/CDF/query/point terms, task masses, optimizer '
    'protections and all original development/confirmation/real-benchmark requirements. No '
    'inference operation, model ensemble, family exception, reference swap or relaxed gate. '
    'This helper alone authorizes no fresh seed, GPU probe, training, checkpoint or selection; '
    'require a separate prospective registration and capacity admission.'
)


for name in ('reference_role', 'reference_weights', 'target_identity', 'combine_predictions',
        'eligible', 'validate_identities', 'make_references', 'validate_target', 'shrunk_variance',
        'column_coefficients', 'independent_penalty', 'retention_weighting', 'validate_task_mixture'):
    function = getattr(raw_mean, name)
    if function.__closure__ is not None:
        raise ValueError('Standardized-Huber callbacks must bind unchanged references without closures')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


def column_geometry(target, mask):
    validate_columns(target, mask)
    values = target.detach().double()
    counts = mask.sum(dim=1, keepdim=True)
    safe_counts = counts.clamp_min(1)
    means = torch.where(mask, values, 0.).sum(dim=1, keepdim=True) / safe_counts
    centered = torch.where(mask, values - means, 0.)
    variances = centered.square().sum(dim=1, keepdim=True) / safe_counts
    denominators = shrunk_variance(variances, counts).clamp_min(SPECIFICATION['variance_floor'])
    mass = mask.sum().double() / ((counts > 0).sum() * safe_counts)
    weights = torch.where(mask, mass.expand_as(values), 0.).float()
    scales = torch.where(mask, denominators.sqrt().expand_as(values), 1.).float()
    if (not torch.isfinite(weights).all() or not torch.isfinite(scales).all()
            or not (weights[mask] > 0).all() or not (scales > 0).all()):
        raise FloatingPointError('Every original query needs finite positive column mass and shrunk scale')
    return weights, scales


def teacher_mean_geometry(episode):
    if reference_role(episode.metadata) is None:
        return None, None
    if column.balanced.balanced_loss is not stronger.balanced_loss:
        raise ValueError('Standardized teacher Huber requires the unchanged active mean20 objective')
    weights, scales = column_geometry(episode.target, episode.mask)
    return weights.reshape(-1), scales.reshape(-1)


def teacher_mean_weights(episode):
    return teacher_mean_geometry(episode)[0]


def reweight_teacher_mean(original, point, teacher, coefficients, scales):
    if (original.shape != point.shape or teacher.shape != point.shape
            or coefficients.shape != point.shape or scales.shape != point.shape
            or coefficients.requires_grad or scales.requires_grad
            or not torch.isfinite(coefficients).all() or not torch.isfinite(scales).all()
            or not (coefficients > 0).all() or not (scales > 0).all()):
        raise ValueError('Require aligned detached finite positive column masses and scales')
    residual = point.float() - teacher.detach().float()
    raw = functional.huber_loss(residual, torch.zeros_like(residual), delta=1., reduction='none')
    standardized = functional.huber_loss(residual / scales.float(), torch.zeros_like(residual),
        delta=1., reduction='none')
    return original + stronger.MEAN_WEIGHT * (coefficients.float() * standardized - raw)


def independent_teacher_mean(target, mask, prediction, teacher):
    validate_columns(target, mask)
    if (prediction.shape != target.shape or teacher.shape != target.shape
            or prediction.device != target.device or teacher.device != target.device
            or not torch.isfinite(prediction[mask]).all() or not torch.isfinite(teacher[mask]).all()):
        raise ValueError('Require complete aligned finite original teacher/student query grids')
    labels = target.detach().cpu().double().numpy()[0]
    queried = mask.detach().cpu().numpy()[0]
    residual = (prediction.detach().double() - teacher.detach().double()).cpu().numpy()[0]
    terms = []
    for column_index in range(labels.shape[1]):
        selected = queried[:, column_index]
        count = int(selected.sum())
        if count:
            variance = float(np.var(labels[selected, column_index], ddof=0))
            scale = math.sqrt(max((count * variance + 1.) / (count + 1.), 1e-6))
            absolute = np.abs(residual[selected, column_index] / scale)
            terms.append(float(np.mean(np.where(absolute <= 1., .5 * absolute**2, absolute - .5))))
    result = 20. * math.fsum(terms) / len(terms)
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite standardized teacher-mean objective')
    return result


BACKWARD_SOURCE = replace_once(raw_mean.BACKWARD_SOURCE,
    '    native_mean_weights = teacher_mean_weights(episode)',
    '    native_mean_weights, native_mean_scales = teacher_mean_geometry(episode)')
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE,
    'reweight_teacher_mean(original, point, teacher_values[lookup], native_mean_weights[lookup])',
    'reweight_teacher_mean(original, point, teacher_values[lookup], native_mean_weights[lookup], native_mean_scales[lookup])')
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, 'exec'), globals())
RUNTIME_SOURCE = raw_mean.RUNTIME_SOURCE
exec(compile(RUNTIME_SOURCE, __file__, 'exec'), globals())
