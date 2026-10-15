"""Bind general-native and actual V29-structured references to full grid replay."""

from collections import Counter
from pathlib import Path
import types

import single_model_native_column_mean_training as predecessor
import single_model_parent_structured_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


INITIALIZER = predecessor.INITIALIZER
INITIALIZER_SHA256 = predecessor.INITIALIZER_SHA256
EVALUATION_BASELINE = INITIALIZER.parent.parent / "query_budget_v8/best_validation_checkpoint.pth"
EVALUATION_BASELINE_SHA256 = predecessor.cohort_retention.REFERENCE_SHA256S["v8"]
POLICY_TEXT = ("Full-trial integration of parent-structured retention. " + cohort_retention.POLICY
    + " Preserve every original task, supervised weight, retention multiplier and scale1 raw "
    "gradient. Bind native mean weighting explicitly and replay two general-native and24 "
    "actual V29-structured reference grids per admitted batch. Retain all teacher/component "
    "grids and original V8/general qualification thresholds. No historical full-autograd or "
    "optimizer-trajectory replay claim. Actual execution needs separate trial registration.")


def reference_identities(parent, base):
    if str(parent) != str(INITIALIZER):
        raise ValueError("Initialize only from actual V29, separate from the frozen reference instance")
    if (base["parent_checkpoint"] != str(EVALUATION_BASELINE)
            or base["input_sha256"].get(str(EVALUATION_BASELINE)) != EVALUATION_BASELINE_SHA256):
        raise ValueError("Changing the training reference must not change the original V8 evaluation baseline")
    identities = {"v29": {"checkpoint": str(INITIALIZER), "sha256": INITIALIZER_SHA256},
                  "general": {"checkpoint": str(GENERAL), "sha256": cohort_retention.REFERENCE_SHA256S["general"]}}
    cohort_retention.validate_identities(identities)
    return identities


def verify_references(references, identities):
    cohort_retention.validate_identities(identities)
    if set(references) != set(identities) or references["v29"] is references["general"]:
        raise ValueError("Independently replay both distinct frozen training references")
    return {role: cohort_retention.verify_model(reference, identities[role])
            for role, reference in references.items()}


for name, original in (("original_identity", "original_identity"), ("original_cohort_identity", "original_cohort_identity"),
                       ("native_cohort_identity", "cohort_identity")):
    function = getattr(predecessor, original)
    if function.__closure__ is not None:
        raise ValueError("Parent-structured integration must not retain predecessor closures")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


def cohort_identity(config):
    if config["input_sha256"].get(str(EVALUATION_BASELINE)) != EVALUATION_BASELINE_SHA256:
        raise ValueError("Bind the original V8 evaluation baseline independently of the new training references")
    return native_cohort_identity(config)


RETENTION_TRAINING_SOURCE = replace_once(predecessor.RETENTION_TRAINING_SOURCE,
    "single_model_native_column_mean as retention", "single_model_parent_structured_retention as retention")
RETENTION_TRAINING_SOURCE = replace_assignment(RETENTION_TRAINING_SOURCE, "POLICY", POLICY_TEXT)
exec(compile(RETENTION_TRAINING_SOURCE, __file__, "exec"), globals())
