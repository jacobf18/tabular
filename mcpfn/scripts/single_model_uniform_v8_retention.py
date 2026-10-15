"""Use one uniform V8 imputation reference without changing V43's retention weights."""

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

import single_model_family_balanced_retention as predecessor
from single_model_family_balanced_retention import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES,
)


REFERENCE_SHA256S = dict(predecessor.REFERENCE_SHA256S)
STRUCTURED_MASS = predecessor.STRUCTURED_MASS
REPLICAS = predecessor.REPLICAS
STRUCTURED_TASK_MASS = predecessor.STRUCTURED_TASK_MASS
SPECIFICATION = deepcopy(predecessor.SPECIFICATION)
SPECIFICATION.update(reference_assignment={"legacy": "v8", "all_structured": "v8", "recsys": None},
    assignment_inputs="validated synthetic imputation domain only; no family-specific exception",
    unused_v29_reference="identity retained for compatibility and initializer verification; never a retention target")
POLICY = ("Prospective reference-only change after V43's audited failed confirmation and complete284-case "
    "initializer diagnosis. Keep the actual V29 initializer, but use the original frozen V8 as the retention "
    "target for every original imputation component and every structured missingness family, without "
    "exceptions chosen from seeds, cases, errors or columns. Ratings retain their original callback without "
    "retention. Keep V43's weight10 count-shrunk equal-column loss, all queries, one unit-variance pseudocount, "
    "1e-6 floor, equal-family structured retention mass.25 and original supervised .5/.25/.25 masses and "
    "unequal supervised family weights. Preserve general/observed-blend/specialist targets, all priors, "
    "Huber/NLL/CDF/query/point losses, normalizations, optimizer protections and original held-out guards. "
    "The V29 identity remains verified but is no longer selected as a retention target. Moving every "
    "structured family toward V8 may sacrifice V29's existing improvements; the consumed-bank diagnosis "
    "does not establish that this policy repairs generalization. No model ensemble or new operation at "
    "inference, changed metric, gate relaxation, checkpoint reselection or reused fresh seed. This standalone "
    "implementation authorizes no GPU probe, training, confirmation or benchmark evaluation.")


def reference_role(metadata):
    role = predecessor.reference_role(metadata)
    return None if role is None else "v8"


for name in ("eligible", "validate_identities", "make_references", "validate_target", "shrunk_variance",
        "column_coefficients", "independent_penalty", "retention_weighting", "validate_task_mixture"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Uniform-reference functions must not retain predecessor closures")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = predecessor.RUNTIME_SOURCE
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
