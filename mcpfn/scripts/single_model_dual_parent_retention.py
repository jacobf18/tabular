"""Use a fixed V8/V29 mean as a structured training-only retention target."""

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

import single_model_parent_structured_retention as predecessor
from single_model_parent_structured_retention import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES, stronger,
)
from single_model_registered_adaptation import replace_once


REFERENCE_SHA256S = {**predecessor.REFERENCE_SHA256S,
    "v8": "7b79b3b650c1e44b2a2b6f3b69839623871406b7a60ef5a17fd08d253125ba76"}
STRUCTURED_MASS = predecessor.STRUCTURED_MASS
REPLICAS = predecessor.REPLICAS
STRUCTURED_TASK_MASS = predecessor.STRUCTURED_TASK_MASS
SPECIFICATION = deepcopy(predecessor.SPECIFICATION)
SPECIFICATION.update(reference_sha256s=REFERENCE_SHA256S,
    reference_assignment={"legacy": "general", "all_structured": "structured_mean", "recsys": None},
    initialization="unchanged actual V29; general, V8 and V29 references are separate frozen models",
    reference_combination={"weights": {"v8": .5, "v29": .5},
        "arithmetic": "float32 half V8 plus float32 half V29; no coefficient fitting",
        "scope": "every structured family and component; native general and ratings unchanged"})
MEAN_SPECIFICATION = deepcopy(predecessor.MEAN_SPECIFICATION)
MEAN_SPECIFICATION["retention"] = "unchanged weights; general-native/fixed V8-V29 mean-structured targets"
POLICY = (
    "Prospective target-only hypothesis after V49's audited negative closure and full284-case "
    "actual-initializer attribution. Both MAR and skip-logic failures are inherited; MAR improves "
    "but skip-logic worsens, and eight of22 distinct metrics worsen. Keep unchanged actual V29 "
    "initialization and replace only the structured retention grid with an equal-weight FP32 "
    "mean of frozen original V8 and actual V29 predictions, uniformly across all original "
    "families/components. Native retention remains general; ratings have no retention. Choose "
    "neither coefficients nor references from query errors, validation, seeds or dataset identity. "
    "Preserve both complete component grids and their checkpoint identities; independently "
    "replay each component and the combined target. Keep native count-shrunk mean20 Huber, all "
    "NLL/CDF/query/point terms, priors, .5/.25/.25 masses, weight10 count-shrunk equal-column "
    "retention, one variance pseudocount,1e-6 floor and optimizer protections. Keep original "
    "V8/general qualification thresholds. The fixed mean may dilute useful predictions or "
    "retain weaknesses; consumed-bank attribution does not establish causality. All three "
    "references are separate frozen training-only instances, never the trainable student. "
    "Inference is one ordinary unchanged checkpoint without routing or an ensemble. This "
    "helper alone authorizes no seeds, GPU probe, training, reselection or qualification."
)


def reference_role(metadata):
    role = predecessor.reference_role(metadata)
    return "structured_mean" if role == "v29" else role


def reference_weights(role):
    if role == "general":
        return {"general": 1.}
    if role == "structured_mean":
        return {"v8": .5, "v29": .5}
    raise ValueError("Require an eligible native or structured reference role")


def validate_identities(identities):
    if set(identities) != set(REFERENCE_SHA256S):
        raise ValueError("Require all three original frozen references")
    for role, digest in REFERENCE_SHA256S.items():
        identity = identities[role]
        if set(identity) != {"checkpoint", "sha256"} or identity["sha256"] != digest or not identity["checkpoint"]:
            raise ValueError("Never substitute a reference checkpoint or invert identities")


def target_identity(role, identities):
    validate_identities(identities)
    weights = reference_weights(role)
    if role == "general":
        return deepcopy(identities[role])
    return {"components": {name: deepcopy(identities[name]) for name in weights},
        "weights": weights, "arithmetic": SPECIFICATION["reference_combination"]["arithmetic"]}


def combine_predictions(role, components):
    weights = reference_weights(role)
    if set(components) != set(weights):
        raise ValueError("Preserve every component of the fixed retention reference")
    prototype = components[next(iter(weights))]
    if not isinstance(prototype, torch.Tensor):
        raise ValueError("Require a tensor for every reference component")
    for prediction in components.values():
        if (not isinstance(prediction, torch.Tensor) or prediction.dtype != torch.float32
                or prediction.ndim not in (2, 3) or prediction.requires_grad
                or (prediction.ndim == 3 and prediction.shape[0] != 1)
                or prediction.shape != prototype.shape or prediction.device != prototype.device):
            raise ValueError("Require detached FP32 full component grids with identical shapes and devices")
    if role == "general":
        return components["general"]
    return components["v8"] * .5 + components["v29"] * .5


for name in ("eligible", "make_references", "validate_target", "shrunk_variance",
        "column_coefficients", "independent_penalty", "retention_weighting", "validate_task_mixture",
        "teacher_mean_weights", "reweight_teacher_mean"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Dual-parent callbacks must not retain predecessor closures")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = replace_once(predecessor.RUNTIME_SOURCE,
    'references["v29"] is references["general"]',
    'len({id(model) for model in references.values()}) != len(references)')
RUNTIME_SOURCE = replace_once(RUNTIME_SOURCE,
    '            reference, identity = references[role], identities[role]',
    '            identity = target_identity(role, identities)')
RUNTIME_SOURCE = replace_once(RUNTIME_SOURCE,
    '                prediction = streamed_point_predictions(reference, episode.source, criterion, **STREAM_OPTIONS).detach()',
    '                components = {name: streamed_point_predictions(references[name], episode.source,\n'
    '                    criterion, **STREAM_OPTIONS).detach().float() for name in reference_weights(role)}\n'
    '            for component in components.values():\n'
    '                validate_target(episode, component)\n'
    '            prediction = combine_predictions(role, components)')
RUNTIME_SOURCE = replace_once(RUNTIME_SOURCE,
    '                "mask": episode.mask.detach().cpu(), "identity": identity, "reference_role": role,',
    '                "mask": episode.mask.detach().cpu(), "identity": identity, "reference_role": role,\n'
    '                "components": {name: component.cpu() for name, component in components.items()},\n'
    '                "component_weights": reference_weights(role),')
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
