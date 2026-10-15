"""Bind native-general and structured-V8 references to full training and replay."""

from collections import Counter
from pathlib import Path
import types

import single_model_uniform_v8_retention_training as predecessor
import single_model_general_native_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment
from train_single_matrixformer import GENERAL


POLICY_TEXT = ("Full-trial integration of native-general retention. " + cohort_retention.POLICY
    + " Bind all original tasks, weights and scale1 gradients. Independently replay general on both "
    "native draws and V8 on all24 structured draws per admitted batch. Preserve both rating callbacks, "
    "every teacher/component grid and all unchanged qualification requirements.")
INITIALIZER = Path("/mnt/checkpoints_200/single_model_goal_20261005/legacy_strata_v29/selected_checkpoint.pth")
INITIALIZER_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"


def reference_identities(parent, base):
    if str(parent) != str(INITIALIZER):
        raise ValueError("The actual V29 initializer is separate from both training references")
    identities = {"v8": {"checkpoint": base["parent_checkpoint"],
                           "sha256": base["input_sha256"][base["parent_checkpoint"]]},
                  "general": {"checkpoint": str(GENERAL),
                              "sha256": cohort_retention.REFERENCE_SHA256S["general"]}}
    cohort_retention.validate_identities(identities)
    return identities


def verify_references(references, identities):
    cohort_retention.validate_identities(identities)
    if set(references) != set(identities) or references["v8"] is references["general"]:
        raise ValueError("Independently replay both distinct frozen training references")
    return {role: cohort_retention.verify_model(reference, identities[role])
            for role, reference in references.items()}


for name in ("original_identity",):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Do not retain predecessor reference closures")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)


def cohort_identity(config):
    identities = original_identity(config)
    if config["input_sha256"].get(str(INITIALIZER)) != INITIALIZER_SHA256:
        raise ValueError("Bind the actual V29 initializer independently of general and V8")
    for batch in config["batches"]:
        retention.validate_task_mixture(batch["tasks"])
    return identities

RETENTION_TRAINING_SOURCE = predecessor.SOURCE.replace(
    "single_model_uniform_v8_retention as retention", "single_model_general_native_retention as retention")
RETENTION_TRAINING_SOURCE = replace_assignment(RETENTION_TRAINING_SOURCE, "POLICY", POLICY_TEXT)
exec(compile(RETENTION_TRAINING_SOURCE, __file__, "exec"), globals())
