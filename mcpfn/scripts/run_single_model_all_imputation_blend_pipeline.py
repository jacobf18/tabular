"""Qualify V45 with all original gates and complete imputation-component replay."""

import run_single_model_uniform_v8_retention_pipeline as previous_pipeline
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_all_imputation_blend_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = previous_pipeline.ARITHMETIC_PATCH
ADAPTATION_SOURCE = previous_pipeline.ADAPTATION_SOURCE
for old, new in {
    "train_single_matrixformer_uniform_v8_retention": "train_single_matrixformer_all_imputation_blend",
    "audit_single_model_uniform_v8_retention": "audit_single_model_all_imputation_blend",
    "uniform_v8_retention_v44": "all_imputation_blend_v45",
    "V44": "V45",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit all-imputation-blend qualification adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
