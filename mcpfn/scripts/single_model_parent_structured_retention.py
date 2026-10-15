"""Retain the actual V29 parent on every structured task without changing loss math."""

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

import single_model_native_column_mean as predecessor
from single_model_native_column_mean import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES, stronger,
)
from single_model_registered_adaptation import replace_once


REFERENCE_SHA256S = {
    "general": predecessor.REFERENCE_SHA256S["general"],
    "v29": "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32",
}
STRUCTURED_MASS = predecessor.STRUCTURED_MASS
REPLICAS = predecessor.REPLICAS
STRUCTURED_TASK_MASS = predecessor.STRUCTURED_TASK_MASS
SPECIFICATION = deepcopy(predecessor.SPECIFICATION)
SPECIFICATION.update(reference_sha256s=REFERENCE_SHA256S,
    reference_assignment={"legacy": "general", "all_structured": "v29", "recsys": None},
    initialization="unchanged actual V29, verified independently; structured reference is a separate frozen model")
MEAN_SPECIFICATION = deepcopy(predecessor.MEAN_SPECIFICATION)
MEAN_SPECIFICATION["retention"] = "unchanged weights; general-native/V29-structured targets"
POLICY = (
    "Prospective reference-only hypothesis after V48's audited failed confirmation and complete "
    "284-case frozen-initializer attribution. V48 introduces two-phase and skip-logic threshold "
    "failures while improving but not repairing two inherited general-reference RMSE gaps; seven "
    "of22 distinct metrics worsen versus actual V29. Keep actual V29 initialization and replace "
    "only the structured retention reference from V8 to the actual frozen V29 parent, uniformly "
    "across every original family and component. Native retention remains general; ratings remain "
    "without retention. Use validated synthetic cohort metadata only, never query errors, seeds, "
    "family exceptions or real dataset identity. Keep V48's native count-shrunk teacher-mean "
    "weighting, raw-residual mean20 Huber, all NLL/CDF/query/point terms, supervised task masses, "
    "weight10 count-shrunk equal-column retention, structured mass.25, all queries, one variance "
    "pseudocount,1e-6 floor, priors and optimizer protections. Preserve original V8/general "
    "development and confirmation guards; changing a training reference does not change those "
    "thresholds. This may retain useful parent predictions but may inhibit improvements or "
    "worsen other cohorts; consumed-bank attribution is not causal evidence. Both training-only "
    "references are distinct frozen models, never aliases of the trainable student. Inference "
    "remains one unchanged ordinary network, without references, routing or ensembling. This "
    "standalone helper authorizes no seeds, GPU probe, training, reselection or qualification."
)


def reference_role(metadata):
    role = predecessor.reference_role(metadata)
    return "v29" if role == "v8" else role


for name in ("eligible", "validate_identities", "make_references", "validate_target", "shrunk_variance",
        "column_coefficients", "independent_penalty", "retention_weighting", "validate_task_mixture",
        "teacher_mean_weights", "reweight_teacher_mean"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Parent-structured callbacks must not retain predecessor closures")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = replace_once(predecessor.RUNTIME_SOURCE,
    'references["v8"] is references["general"]', 'references["v29"] is references["general"]')
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
