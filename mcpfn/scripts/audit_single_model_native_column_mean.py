"""Audit every V48 task, teacher/reference grid and optimizer endpoint."""

import audit_single_model_general_native_retention as audit_blueprint
from audit_single_model_general_native_retention import fixed_trial, admission_source
import single_model_native_column_mean_training as retention_training
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_native_column_mean_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


FULL_AUDIT_SOURCE = audit_blueprint.FULL_AUDIT_SOURCE
for old, new in {
    "train_single_matrixformer_general_native_retention": "train_single_matrixformer_native_column_mean",
    "general_native_retention_v47": "native_column_mean_v48",
    "V47": "V48",
}.items():
    if old not in FULL_AUDIT_SOURCE:
        raise ValueError(f"Missing explicit native-column full-audit adaptation: {old}")
    FULL_AUDIT_SOURCE = FULL_AUDIT_SOURCE.replace(old, new)
exec(compile(FULL_AUDIT_SOURCE, __file__, "exec"), globals())
