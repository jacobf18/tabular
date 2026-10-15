"""Qualify V51 using all original synthetic guards and frozen real suites."""

import run_single_model_dual_parent_pipeline as pipeline_blueprint
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_native_parent_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = pipeline_blueprint.ARITHMETIC_PATCH
PIPELINE_ADAPTATION_SOURCE = pipeline_blueprint.PIPELINE_ADAPTATION_SOURCE
for old, new in {
    "train_single_matrixformer_dual_parent": "train_single_matrixformer_native_parent",
    "audit_single_model_dual_parent": "audit_single_model_native_parent",
    "single_model_dual_parent_training": "single_model_native_parent_training",
    "single_model_dual_parent_storage": "single_model_native_parent_storage",
    "dual_parent_v50": "native_parent_v51",
    "test_single_model_dual_parent_trial.py": "test_single_model_native_parent_trial.py",
    "test_single_model_dual_parent_pipeline.py": "test_single_model_native_parent_pipeline.py",
    "V50": "V51",
}.items():
    if old not in PIPELINE_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit native-parent pipeline adaptation: {old}")
    PIPELINE_ADAPTATION_SOURCE = PIPELINE_ADAPTATION_SOURCE.replace(old, new)
exec(compile(PIPELINE_ADAPTATION_SOURCE, __file__, "exec"), globals())
