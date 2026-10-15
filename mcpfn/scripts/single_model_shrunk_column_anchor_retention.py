"""Prospective native-anchor retention with one unit-variance query pseudocount."""

from contextlib import contextmanager
from functools import partial
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

import single_model_column_anchor_retention as predecessor
from single_model_column_anchor_retention import (
    REFERENCE_SHA256, eligible, validate_target, validate_columns, compute_context, query, column,
    query_budget_majorizer, TRAIN_OPTIONS, streamed_training_loss, verify_model, STREAM_OPTIONS,
    atomic_torch_save, streamed_point_predictions, sha256, write_json,
)


SPECIFICATION = {**predecessor.SPECIFICATION,
    "loss": "equal queried-column prediction-drift MSE divided by query-count-shrunk synthetic target variance",
    "variance_estimator": "(query_count * population_variance + prior_count * prior_variance) / (query_count + prior_count)",
    "prior_count": 1., "prior_variance": 1.,
    "constant_columns": "included using the same count-shrunk variance and unchanged numerical floor",
    "count_dependence": "repeating query rows changes shrinkage strength; column permutation remains invariant"}
POLICY = ("Prospective alternative to the V38 native-anchor retention denominator, not an edit or retry of "
    "that registered experiment. Keep the weight10 frozen V29 reference and every original eligible native "
    "synthetic query. Replace only the population-variance estimate inside this training-only regularizer "
    "by (query_count * population_variance + 1) / (query_count + 1): one prospectively fixed unit-variance "
    "pseudocount in the unchanged normalized training units. Preserve the1e-6 numerical floor, equal queried-"
    "column weighting, single-query/constant columns, detached float64 moments and float32 callback "
    "coefficients. Repeated rows intentionally affect the count-dependent shrinkage. This is not a claim "
    "of Bayesian posterior inference or a change to normalization, true-target objectives, NRMSE "
    "denominators, teacher identities, eligibility, selection or confirmation guards. Spectral and rating "
    "callbacks remain exact originals. Hypothetical V38 coefficient concentration motivates a bounded "
    "test, not a proven causal diagnosis or efficacy claim. No new training seeds, checkpoint, GPU probe "
    "or scientific trial are authorized by this standalone implementation.")


def shrunk_variance(variance, counts):
    return ((counts * variance + SPECIFICATION["prior_count"] * SPECIFICATION["prior_variance"])
        / (counts + SPECIFICATION["prior_count"]))


def column_coefficients(target, mask):
    validate_columns(target, mask)
    values = target.detach().double()
    counts = mask.sum(dim=1, keepdim=True)
    safe_counts = counts.clamp_min(1)
    selected = torch.where(mask, values, 0.)
    means = selected.sum(dim=1, keepdim=True) / safe_counts
    centered = torch.where(mask, values - means, 0.)
    variances = centered.square().sum(dim=1, keepdim=True) / safe_counts
    denominators = shrunk_variance(variances, counts).clamp_min(SPECIFICATION["variance_floor"])
    queried_columns = (counts > 0).sum()
    per_column = mask.sum() / (queried_columns * safe_counts * denominators)
    coefficients = torch.where(mask, per_column.expand_as(values), 0.).float()
    if not torch.isfinite(coefficients).all() or not (coefficients[mask] > 0).all():
        raise FloatingPointError("Every original query requires a finite positive shrunk retention coefficient")
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
        count = int(queried.sum())
        if count:
            variance = float(np.var(labels[queried, column_index], ddof=0))
            denominator = ((count * variance + SPECIFICATION["prior_count"] * SPECIFICATION["prior_variance"])
                / (count + SPECIFICATION["prior_count"]))
            squared_drift = float(np.mean(np.square(differences[queried, column_index])))
            contributions.append(squared_drift / max(denominator, SPECIFICATION["variance_floor"]))
    result = SPECIFICATION["weight"] * math.fsum(contributions) / len(contributions)
    if not math.isfinite(result):
        raise FloatingPointError("Independent shrunk retention scoring must remain finite without dropping columns")
    return result


BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = predecessor.RUNTIME_SOURCE
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
