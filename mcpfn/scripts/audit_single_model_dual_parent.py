"""Audit every V50 task, component, averaged reference and saved endpoint."""

import audit_single_model_parent_structured as audit_blueprint
from audit_single_model_parent_structured import fixed_trial, admission_source
import single_model_dual_parent_training as retention_training
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_dual_parent_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


FULL_AUDIT_SOURCE = audit_blueprint.FULL_AUDIT_SOURCE
for old, new in {
    "train_single_matrixformer_parent_structured": "train_single_matrixformer_dual_parent",
    "parent_structured_v49": "dual_parent_v50",
    "V49": "V50",
}.items():
    if old not in FULL_AUDIT_SOURCE:
        raise ValueError(f"Missing explicit dual-parent full-audit adaptation: {old}")
    FULL_AUDIT_SOURCE = FULL_AUDIT_SOURCE.replace(old, new)
exec(compile(FULL_AUDIT_SOURCE, __file__, "exec"), globals())
