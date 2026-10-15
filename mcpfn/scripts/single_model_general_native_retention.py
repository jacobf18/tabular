"""Use the general parent for native retention and V8 for every structured family."""

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

import single_model_uniform_v8_retention as predecessor
from single_model_uniform_v8_retention import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES,
)
from single_model_registered_adaptation import replace_once


REFERENCE_SHA256S = {
    "v8": predecessor.REFERENCE_SHA256S["v8"],
    "general": "c0ddf75274c8ff92ca5fd22c1263369ffa7e55eb7cb85fe80961b0afa101f7a9",
}
STRUCTURED_MASS = predecessor.STRUCTURED_MASS
REPLICAS = predecessor.REPLICAS
STRUCTURED_TASK_MASS = predecessor.STRUCTURED_TASK_MASS
SPECIFICATION = deepcopy(predecessor.SPECIFICATION)
SPECIFICATION.pop("unused_v29_reference")
SPECIFICATION.update(reference_sha256s=REFERENCE_SHA256S,
    reference_assignment={"legacy": "general", "all_structured": "v8", "recsys": None},
    assignment_inputs="validated synthetic native/structured domain metadata only; no seed, error or family exception",
    initialization="unchanged actual V29, verified separately; not a retention target")
POLICY = (
    "Prospective native-reference change after V46's fully audited negative confirmation and "
    "all284-case V29 attribution. All four failures were inherited; V46 reduced the native general-"
    "reference gap but worsened the inherited MAR gap, and ten of22 distinct metrics worsened versus "
    "V29. Keep actual V29 initialization and change only native-imputation training retention from "
    "V8 to the original general parent. Every structured family retains V8 uniformly; ratings retain "
    "their original callback without retention. Assignment uses synthetic cohort metadata, not "
    "query truth, errors, seeds, component performance or real dataset identity. Preserve weight10 "
    "count-shrunk equal-column retention, all queries, one variance pseudocount,1e-6 floor, equal-"
    "family structured mass.25, original supervised .5/.25/.25 masses, stronger mean20 teacher "
    "distillation, observed-blend imputation/specialist rating targets, NLL/CDF/query/point terms, "
    "priors, optimizer protections and all original held-out guards. This may improve native "
    "retention but does not establish that MAR or benchmark performance will improve; shared "
    "parameter changes can sacrifice other cohorts. Replace only the unused V29 reference-model "
    "allocation with the general model; verify the actual initializer independently. One unchanged "
    "ordinary network at inference, without any reference, router or ensemble. This helper alone "
    "authorizes no seed, GPU probe, training, confirmation, reselection or real evaluation."
)


def reference_role(metadata):
    role = predecessor.reference_role(metadata)
    if role is None:
        return None
    return "v8" if metadata.get("structured", False) else "general"


for name in ("eligible", "validate_identities", "make_references", "validate_target", "shrunk_variance",
             "column_coefficients", "independent_penalty", "retention_weighting", "validate_task_mixture"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Do not retain predecessor reference or weighting closures")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = replace_once(predecessor.RUNTIME_SOURCE,
    'references["v8"] is references["v29"]', 'references["v8"] is references["general"]')
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
