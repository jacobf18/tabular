"""Retain native-anchor predictions with equal-column synthetic query-variance weighting."""

from contextlib import contextmanager
from functools import partial
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import torch

import single_model_anchor_retention as predecessor
from single_model_anchor_retention import (
    REFERENCE_SHA256, eligible, validate_target, compute_context, query, column,
    query_budget_majorizer, TRAIN_OPTIONS, streamed_training_loss, replace_once,
    verify_model, STREAM_OPTIONS, atomic_torch_save, streamed_point_predictions,
    sha256, write_json,
)


SPECIFICATION = {**predecessor.SPECIFICATION,
    "loss": "equal queried-column mean of prediction-drift MSE divided by floored synthetic query-target variance",
    "variance_floor": 1e-6, "population_variance": True,
    "column_weighting": "equal among every column with an original query",
    "constant_columns": "included with variance floor, never dropped",
    "coefficient_arithmetic": "float64 moments then float32 per-query coefficients",
    "retention_scaling": "synthetic training labels only; no input or inference normalization change"}
POLICY = ("After the fully audited negative V37 result, keep its weight10 native-anchor retention reference "
    "and generator-only eligibility, but average squared prediction drift equally across every queried column "
    "and divide each column by max(population variance of its original synthetic query targets,1e-6). "
    "Include all original queries and every queried column, including single-query and constant columns; "
    "only query-empty columns have no contribution. Use float64 detached moments and float32 per-query "
    "coefficients to match the existing streamed cell-mean loss. The reference is the unchanged frozen V29 "
    "initializer. Eligibility never uses errors, query counts, validation failures or selected columns. "
    "Preserve original teacher targets, Huber/NLL/CDF/query-majorizer/true-target point terms, observation-only "
    "input normalization, masks, missingness families, priors, inference, evaluation denominators and guards. "
    "Spectral and rating tasks retain their exact original callback. No development labels enter training, "
    "no columns are discarded, and no inference-time reference or ensemble is added. Register before any "
    "new synthetic task is consumed; a feasible probe is not evidence of benchmark improvement.")


def validate_columns(target, mask):
    if (not isinstance(target, torch.Tensor) or not target.is_floating_point()
            or target.ndim != 3 or target.shape[0] != 1 or mask.shape != target.shape
            or mask.dtype != torch.bool or mask.device != target.device or not mask.any()
            or not torch.isfinite(target[mask]).all()):
        raise ValueError("Require one original finite-query synthetic matrix and its unchanged Boolean mask")


def column_coefficients(target, mask):
    validate_columns(target, mask)
    values = target.detach().double()
    counts = mask.sum(dim=1, keepdim=True)
    safe_counts = counts.clamp_min(1)
    selected = torch.where(mask, values, 0.)
    means = selected.sum(dim=1, keepdim=True) / safe_counts
    centered = torch.where(mask, values - means, 0.)
    variances = centered.square().sum(dim=1, keepdim=True) / safe_counts
    queried_columns = (counts > 0).sum()
    per_column = mask.sum() / (queried_columns * safe_counts * variances.clamp_min(SPECIFICATION["variance_floor"]))
    coefficients = torch.where(mask, per_column.expand_as(values), 0.).float()
    if not torch.isfinite(coefficients).all() or not (coefficients[mask] > 0).all():
        raise FloatingPointError("Every original query requires a finite positive retention coefficient")
    return coefficients


def independent_penalty(target, mask, prediction, reference):
    validate_columns(target, mask)
    if (prediction.shape != target.shape or reference.shape != target.shape
            or prediction.device != target.device or reference.device != target.device
            or not torch.isfinite(prediction[mask]).all() or not torch.isfinite(reference[mask]).all()):
        raise ValueError("Retain complete finite before/candidate and frozen reference grids")
    labels = target.detach().cpu().double().numpy()[0]
    selected = mask.detach().cpu().numpy()[0]
    differences = (prediction.detach().double() - reference.detach().double()).cpu().numpy()[0]
    contributions = []
    for column_index in range(labels.shape[1]):
        queried = selected[:, column_index]
        if queried.any():
            variance = float(np.var(labels[queried, column_index], ddof=0))
            squared_drift = float(np.mean(np.square(differences[queried, column_index])))
            contributions.append(squared_drift / max(variance, SPECIFICATION["variance_floor"]))
    result = SPECIFICATION["weight"] * math.fsum(contributions) / len(contributions)
    if not math.isfinite(result):
        raise FloatingPointError("Independent retention scoring must remain finite without dropping columns")
    return result


BACKWARD_SOURCE = replace_once(predecessor.BACKWARD_SOURCE,
    '    retention_values = retention_target.detach().float().reshape(-1)',
    '    retention_values = retention_target.detach().float().reshape(-1)\n'
    '    retention_weights = column_coefficients(episode.target, episode.mask).reshape(-1)')
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE,
    'SPECIFICATION["weight"] * (point - retention_values[lookup]).square()',
    'SPECIFICATION["weight"] * retention_weights[lookup] * (point - retention_values[lookup]).square()')
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = inspect.getsource(predecessor.runtime)
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
