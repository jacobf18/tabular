"""Prospective equal-column teacher-mean weighting for every imputation task."""

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

import single_model_native_parent_retention as original
from single_model_native_parent_retention import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES, stronger,
)


predecessor = original.predecessor
REFERENCE_SHA256S = dict(original.REFERENCE_SHA256S)
STRUCTURED_MASS = original.STRUCTURED_MASS
REPLICAS = original.REPLICAS
STRUCTURED_TASK_MASS = original.STRUCTURED_TASK_MASS
SPECIFICATION = deepcopy(original.SPECIFICATION)
MEAN_SPECIFICATION = deepcopy(original.MEAN_SPECIFICATION)
MEAN_SPECIFICATION.update(
    structured_components=list(SPECIFICATION["structured_components"]),
    structured_families=list(FAMILIES),
    structured_and_rating_callbacks="structured uses the same column coefficient as native; ratings unchanged mean20",
    scope="every original native and structured imputation query; no ratings",
)
POLICY = (
    "Prospective structured teacher-mean weighting hypothesis after V51's complete audited negative "
    "closure. Extend the existing native equal-column count-shrunk mean20 Huber weighting to every "
    "structured imputation task, uniformly over all twelve families and four components. Keep "
    "native callbacks bitwise identical, rating callbacks unchanged, raw-residual Huber delta1, "
    "detached teachers and coefficients, one unit-variance pseudocount,1e-6 floor and all original "
    "queried columns, including singletons and constants. No additional prediction pass or changed "
    "input normalization. Preserve actual V29 initialization/native retention, exact fixed-half "
    "V8/V29 structured retention, observed-only blended imputation teachers, rating specialist, "
    "all priors and .5/.25/.25 task masses, NLL/CDF/query/point terms, optimizer protections and "
    "every original development/confirmation/real-benchmark requirement. This changes only the "
    "structured teacher-mean aggregation; it may amplify low-variance teacher error and worsen "
    "other cohorts. Output-coordinate gradients do not establish parameter-gradient dominance "
    "or efficacy. No family exception, truth-based teacher choice, gate relaxation, inference "
    "ensemble or reference swap. This standalone helper authorizes no scientific seeds, GPU "
    "probe, training, selection or confirmation; a fresh registration and capacity check are required."
)


for name in ("reference_role", "reference_weights", "target_identity", "combine_predictions",
        "eligible", "validate_identities", "make_references", "validate_target", "shrunk_variance",
        "column_coefficients", "independent_penalty", "retention_weighting", "validate_task_mixture",
        "reweight_teacher_mean"):
    function = getattr(original, name)
    if function.__closure__ is not None:
        raise ValueError("All-imputation mean callbacks must preserve closure-free reference behavior")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


def teacher_mean_weights(episode):
    if reference_role(episode.metadata) is None:
        return None
    if column.balanced.balanced_loss is not stronger.balanced_loss:
        raise ValueError("All-imputation column reweighting requires the unchanged active mean20 objective")
    return column_coefficients(episode.target, episode.mask).reshape(-1)


BACKWARD_SOURCE = original.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = original.RUNTIME_SOURCE
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
