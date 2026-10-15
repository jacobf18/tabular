"""Bind the fixed native-column teacher objective to full training and replay."""

from collections import Counter
from pathlib import Path
import types

import single_model_general_native_retention_training as predecessor
import single_model_native_column_mean as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


INITIALIZER = predecessor.INITIALIZER
INITIALIZER_SHA256 = predecessor.INITIALIZER_SHA256
POLICY_TEXT = ("Full-trial integration of native column-balanced teacher-mean distillation. " + cohort_retention.POLICY
    + " Preserve every original task, reference assignment, task mass and scale1 gradient. Bind the "
    "native mean specification explicitly; replay all general-native and V8-structured references "
    "and every original teacher/component grid without changing qualification requirements.")


for name in ("reference_identities", "verify_references", "original_identity"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Native-column integration must not retain predecessor closures")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
original_cohort_identity = types.FunctionType(predecessor.cohort_identity.__code__, globals(), "original_cohort_identity")


def cohort_identity(config):
    if config["native_teacher_mean"] != cohort_retention.MEAN_SPECIFICATION:
        raise ValueError("Bind the complete fixed native teacher-mean weighting rule")
    return original_cohort_identity(config)


RETENTION_TRAINING_SOURCE = replace_once(predecessor.RETENTION_TRAINING_SOURCE,
    "single_model_general_native_retention as retention", "single_model_native_column_mean as retention")
RETENTION_TRAINING_SOURCE = replace_assignment(RETENTION_TRAINING_SOURCE, "POLICY", POLICY_TEXT)
exec(compile(RETENTION_TRAINING_SOURCE, __file__, "exec"), globals())
