"""Reweight native teacher-mean distillation without changing retention or inference."""

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

import single_model_general_native_retention as predecessor
import single_model_stronger_mean_distillation as stronger
from single_model_general_native_retention import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES,
)
from single_model_registered_adaptation import replace_once


REFERENCE_SHA256S = dict(predecessor.REFERENCE_SHA256S)
STRUCTURED_MASS = predecessor.STRUCTURED_MASS
REPLICAS = predecessor.REPLICAS
STRUCTURED_TASK_MASS = predecessor.STRUCTURED_TASK_MASS
SPECIFICATION = deepcopy(predecessor.SPECIFICATION)
MEAN_SPECIFICATION = {
    "native_components": list(range(8)),
    "teacher_mean_huber_weight": stronger.MEAN_WEIGHT,
    "huber_delta": 1.,
    "residual_units": "unchanged normalized-network raw residual, not variance-standardized residual",
    "coefficient": "total_queries / (queried_columns * column_queries * max(shrunk_variance, 1e-6))",
    "shrunk_variance": "(column_queries * population_variance + 1) / (column_queries + 1)",
    "moments": "detached float64 synthetic target moments; float32 callback coefficients",
    "coverage": "all original queried columns, including singleton and constant columns",
    "structured_and_rating_callbacks": "unchanged mean20",
    "retention": "unchanged general-native/V8-structured targets and weights",
    "teacher_detached": True,
    "architecture_and_inference_unchanged": True,
}
POLICY = (
    "Prospective native teacher-mean weighting hypothesis after V47's fully audited negative "
    "outcome and complete five-boundary saved-grid diagnosis. Keep actual V29 initialization, "
    "general-native/V8-structured retention and mean20, but multiply only each native-imputation "
    "teacher-mean Huber term by the existing count-shrunk equal-column coefficient. Apply to all "
    "eight native components uniformly, never select by case, seed, error, column outcome or real "
    "dataset. Keep raw-residual Huber delta1, detached teacher, one unit-variance pseudocount,1e-6 "
    "floor and every query including singleton/constant columns. Retain all structured/rating "
    "callbacks, supervised/retention masses, priors, normalization, observed-blend imputation and "
    "specialist rating teachers, NLL/CDF/query/point terms and optimizer protections. This may "
    "strengthen low-support teacher supervision but can worsen priors where the blend is worse "
    "and does not ensure MAR retention. No reference swap, changed NRMSE, inference operation, "
    "ensemble, relaxed development/confirmation/benchmark requirement or consumed-bank selection. "
    "This standalone helper authorizes no new seeds, GPU probe, training or qualification."
)


for name in ("reference_role", "eligible", "validate_identities", "make_references", "validate_target", "shrunk_variance",
        "column_coefficients", "independent_penalty", "retention_weighting", "validate_task_mixture"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Native-column callbacks must not retain predecessor closures")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


def teacher_mean_weights(episode):
    role = reference_role(episode.metadata)
    if role is None or episode.metadata.get("structured", False):
        return None
    if column.balanced.balanced_loss is not stronger.balanced_loss:
        raise ValueError("Native column reweighting requires the unchanged active mean20 objective")
    return column_coefficients(episode.target, episode.mask).reshape(-1)


def reweight_teacher_mean(original, point, teacher, coefficients):
    if (original.shape != point.shape or teacher.shape != point.shape or coefficients.shape != point.shape
            or coefficients.requires_grad or not torch.isfinite(coefficients).all() or not (coefficients > 0).all()):
        raise ValueError("Use aligned detached finite positive coefficients on every original queried cell")
    huber = functional.huber_loss(point.float(), teacher.detach().float(), delta=1., reduction="none")
    return original + stronger.MEAN_WEIGHT * (coefficients.float() - 1.) * huber


BACKWARD_SOURCE = replace_once(predecessor.BACKWARD_SOURCE,
    "    validate_target(episode, retention_target)",
    "    validate_target(episode, retention_target)\n    native_mean_weights = teacher_mean_weights(episode)")
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE,
    '        original = original + column.COLUMN_OBJECTIVE["weight"]',
    "        if native_mean_weights is not None:\n"
    "            original = reweight_teacher_mean(original, point, teacher_values[lookup], native_mean_weights[lookup])\n"
    '        original = original + column.COLUMN_OBJECTIVE["weight"]')
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = predecessor.RUNTIME_SOURCE
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
