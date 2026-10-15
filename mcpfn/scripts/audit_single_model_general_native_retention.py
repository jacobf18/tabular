"""Audit every V47 task, teacher component, reference and optimizer endpoint."""

import audit_single_model_stronger_mean as audit_blueprint
from audit_single_model_stronger_mean import fixed_trial, admission_source
import single_model_general_native_retention_training as retention_training
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_general_native_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


FULL_AUDIT_SOURCE = audit_blueprint.SOURCE
for old, new in {
    "train_single_matrixformer_stronger_mean": "train_single_matrixformer_general_native_retention",
    "stronger_mean_distillation_v46": "general_native_retention_v47",
    "V46": "V47",
}.items():
    if old not in FULL_AUDIT_SOURCE:
        raise ValueError(f"Missing explicit native-general audit adaptation: {old}")
    FULL_AUDIT_SOURCE = FULL_AUDIT_SOURCE.replace(old, new)
exec(compile(FULL_AUDIT_SOURCE, __file__, "exec"), globals())
