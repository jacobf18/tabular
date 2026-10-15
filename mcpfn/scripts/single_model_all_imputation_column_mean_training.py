"""Replay every original reference with the all-imputation column-mean objective."""

from collections import Counter
from pathlib import Path
import types

import single_model_native_parent_training as predecessor
import single_model_all_imputation_column_mean as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


INITIALIZER = predecessor.INITIALIZER
INITIALIZER_SHA256 = predecessor.INITIALIZER_SHA256
EVALUATION_BASELINE = predecessor.EVALUATION_BASELINE
EVALUATION_BASELINE_SHA256 = predecessor.EVALUATION_BASELINE_SHA256
POLICY_TEXT = (
    "Full-trial integration of equal-column teacher-mean weighting for every imputation task. "
    + cohort_retention.POLICY
    + " Preserve all original teacher grids, supervised/retention weights and scale1 gradients. "
    "Independently replay26 retention targets and50 separate components per admitted batch: "
    "24 V8 and26 V29, with no general retention inference. Keep the original V8/general "
    "development/confirmation requirements, all teacher-component checks and every adverse "
    "result. No historical full-autograd or optimizer-trajectory replay claim. Execution "
    "requires a separate full-trial registration, never promotion of a feasibility checkpoint."
)


for name in ("reference_identities", "verify_references", "original_identity", "original_cohort_identity",
        "native_cohort_identity", "cohort_identity", "validate_saved_components", "require_report_components",
        "expected_component_counts"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Full-trial callbacks must bind the new objective and unchanged references")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

RETENTION_TRAINING_SOURCE = replace_once(predecessor.RETENTION_TRAINING_SOURCE,
    "single_model_native_parent_retention as retention", "single_model_all_imputation_column_mean as retention")
RETENTION_TRAINING_SOURCE = replace_assignment(RETENTION_TRAINING_SOURCE, "POLICY", POLICY_TEXT)
exec(compile(RETENTION_TRAINING_SOURCE, __file__, "exec"), globals())
