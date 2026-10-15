"""Extend count-shrunk V29 retention to every original synthetic imputation prior."""

from contextlib import contextmanager
from functools import partial
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import torch

import single_model_shrunk_column_anchor_retention as predecessor
from single_model_shrunk_column_anchor_retention import (
    REFERENCE_SHA256, validate_columns, compute_context, query, column,
    query_budget_majorizer, TRAIN_OPTIONS, streamed_training_loss, verify_model, STREAM_OPTIONS,
    atomic_torch_save, streamed_point_predictions, sha256, write_json,
)


SPECIFICATION = {key: value for key, value in predecessor.SPECIFICATION.items()
    if key not in {"legacy_components", "structured_component"}}
SPECIFICATION.update(legacy_components=list(range(8)),
    structured_components=["low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"],
    retention_scope="all original synthetic imputation tasks; no ratings")
POLICY = ("Prospective scope-only extension after the independently audited negative V39 trial. "
    "Keep its weight10 frozen V29 reference and exact count-shrunk equal-column prediction-drift loss, "
    "including one unit-variance pseudocount, the1e-6 numerical floor, every original query, single-query "
    "and constant columns. Extend reference retention from native anchors to all eight original legacy "
    "imputation components and all four original structured components, for every original missingness "
    "family. Preserve the original synthetic seed/domain/component validation; eligibility never uses "
    "errors, query counts, target variance, validation failures or individual columns. Ratings retain "
    "the exact original callback. The frozen reference remains the unchanged V29 initializer, not an "
    "eligible benchmark winner, a changed general teacher, the failed V39 endpoint or a probe endpoint. "
    "Keep the original general/specialist teachers, Huber/NLL/CDF/query/point losses, priors, observation-"
    "only normalization, fixed-half projected update and all development/confirmation/benchmark checks "
    "unchanged in a separately registered experiment. Complete grids and every eligibility decision "
    "require independent replay. This adds no inference operation or ensemble and is not evidence that "
    "broader retention fixes either native low-variance drift or spectral MAR regressions. This standalone "
    "implementation authorizes no new seeds, GPU probe, training, selection or evaluation.")


def eligible(metadata):
    predecessor.eligible(metadata)
    return metadata["domain"] == "imputation"


VALIDATION_SOURCE = inspect.getsource(predecessor.validate_target).replace("native-anchor query", "imputation query")
exec(compile(VALIDATION_SOURCE, __file__, "exec"), globals())
LOSS_SOURCE = "\n\n".join(inspect.getsource(getattr(predecessor, name)) for name in
    ("shrunk_variance", "column_coefficients", "independent_penalty"))
exec(compile(LOSS_SOURCE, __file__, "exec"), globals())
BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())
RUNTIME_SOURCE = predecessor.RUNTIME_SOURCE
exec(compile(RUNTIME_SOURCE, __file__, "exec"), globals())
