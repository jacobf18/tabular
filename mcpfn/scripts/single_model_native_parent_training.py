"""Bind native V29 and structured fixed-mean retention to complete trial replay."""

from collections import Counter
from pathlib import Path
import types

import single_model_dual_parent_training as predecessor
import single_model_native_parent_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


INITIALIZER = predecessor.INITIALIZER
INITIALIZER_SHA256 = predecessor.INITIALIZER_SHA256
EVALUATION_BASELINE = predecessor.EVALUATION_BASELINE
EVALUATION_BASELINE_SHA256 = predecessor.EVALUATION_BASELINE_SHA256
POLICY_TEXT = ("Full-trial integration of native-parent retention. " + cohort_retention.POLICY
    + " Preserve all original tasks, teachers, supervised/retention weights and scale1 gradients. "
    "For each admitted28-task batch independently replay26 targets and50 separate components: "
    "24 V8 and26 V29, with zero general retention inference. Preserve both structured components "
    "so offsetting errors cannot hide in their mean. Verify all three distinct frozen identities. "
    "Data-only rejections and ratings create no retention inference. Keep original V8/general "
    "qualification thresholds and all original teacher-component checks. No historical full "
    "autograd or optimizer-trajectory replay claim. Execution requires a new full registration.")


for name in ("reference_identities", "verify_references", "original_identity", "original_cohort_identity",
        "native_cohort_identity", "cohort_identity", "validate_saved_components", "require_report_components"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Native-parent integration must bind callbacks to its own reference policy")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


def expected_component_counts(admitted_batches):
    if type(admitted_batches) is not int or admitted_batches < 0:
        raise ValueError("Require a nonnegative exact admitted-batch count")
    return {role: count * admitted_batches for role, count in (("v8", 24), ("v29", 26)) if admitted_batches}


RETENTION_TRAINING_SOURCE = replace_once(predecessor.RETENTION_TRAINING_SOURCE,
    "single_model_dual_parent_retention as retention", "single_model_native_parent_retention as retention")
RETENTION_TRAINING_SOURCE = replace_assignment(RETENTION_TRAINING_SOURCE, "POLICY", POLICY_TEXT)
exec(compile(RETENTION_TRAINING_SOURCE, __file__, "exec"), globals())
