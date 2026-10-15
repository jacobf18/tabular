"""Preserve the actual V29 native reference with the unchanged structured mean."""

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

import single_model_dual_parent_retention as predecessor
from single_model_dual_parent_retention import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES, stronger,
)


REFERENCE_SHA256S = dict(predecessor.REFERENCE_SHA256S)
STRUCTURED_MASS = predecessor.STRUCTURED_MASS
REPLICAS = predecessor.REPLICAS
STRUCTURED_TASK_MASS = predecessor.STRUCTURED_TASK_MASS
SPECIFICATION = deepcopy(predecessor.SPECIFICATION)
SPECIFICATION.update(
    reference_assignment={"legacy": "v29", "all_structured": "structured_mean", "recsys": None},
    initialization="unchanged actual V29; every reference remains a separate frozen model")
SPECIFICATION["reference_combination"]["scope"] = (
    "every structured family and component unchanged; native uses frozen V29 and ratings unchanged")
MEAN_SPECIFICATION = deepcopy(predecessor.MEAN_SPECIFICATION)
MEAN_SPECIFICATION["retention"] = "unchanged weights; V29-native/fixed V8-V29 mean-structured targets"
POLICY = (
    "Prospective native-reference-only hypothesis after V50's independently verified negative "
    "no-selection closure. All trained boundaries fail original guards despite synthetic rating "
    "and structured aggregate improvements. Keep actual V29 initialization and replace only "
    "the native retention reference from general to frozen actual V29, uniformly for every "
    "native component, orientation and support regime. Structured retention remains the exact "
    "FP32 equal-weight V8/V29 mean for every family; ratings remain without retention. "
    "Keep original observed-blend imputation and specialist rating teachers, mixed priors, "
    ".5/.25/.25 masses, native count-shrunk mean20 Huber, NLL/CDF/query/point terms, weight10 "
    "count-shrunk equal-column retention, variance pseudocount, floor and optimizer protections. "
    "Reference choice uses validated cohort metadata only, never query errors, seeds, real "
    "dataset identities or family exceptions. Keep original V8/general qualification thresholds. "
    "General remains a separately verified frozen identity but supplies no retention targets. "
    "The actual initializer has stronger saved native development scores, but this descriptive "
    "fact does not prove a causal remedy and retaining it may obstruct useful improvement. "
    "Inference remains one ordinary unchanged checkpoint, without routing or an ensemble. "
    "This standalone helper authorizes no new seeds, GPU probe, training, selection or "
    "confirmation; a fresh separately registered experiment and capacity check are required."
)


def reference_role(metadata):
    role = predecessor.reference_role(metadata)
    return "v29" if role == "general" else role


def reference_weights(role):
    if role == "v29":
        return {"v29": 1.}
    if role == "structured_mean":
        return {"v8": .5, "v29": .5}
    raise ValueError("Require a native V29 or unchanged structured-mean reference role")


def target_identity(role, identities):
    validate_identities(identities)
    weights = reference_weights(role)
    if role == "v29":
        return deepcopy(identities[role])
    return {"components": {name: deepcopy(identities[name]) for name in weights},
        "weights": weights, "arithmetic": SPECIFICATION["reference_combination"]["arithmetic"]}


def combine_predictions(role, components):
    weights = reference_weights(role)
    if set(components) != set(weights):
        raise ValueError("Preserve every declared reference component without substitution")
    if role == "v29":
        return predecessor.combine_predictions("general", {"general": components["v29"]})
    return predecessor.combine_predictions(role, components)


for name in ("eligible", "validate_identities", "make_references", "validate_target", "shrunk_variance",
        "column_coefficients", "independent_penalty", "retention_weighting", "validate_task_mixture",
        "teacher_mean_weights", "reweight_teacher_mean"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Native-parent callbacks must not retain predecessor closures")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = predecessor.RUNTIME_SOURCE
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
