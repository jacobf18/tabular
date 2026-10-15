"""Qualify V42 only after complete original and dual-reference audits."""

import run_single_model_legacy_blend_teacher_pipeline as previous_pipeline
import single_model_legacy_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_cohort_anchor_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ADAPTATION_SOURCE = previous_pipeline.ADAPTATION_SOURCE
for old, new in {
    "single_model_all_imputation_retention_training": "single_model_cohort_anchor_retention_training",
    "train_single_matrixformer_legacy_blend_teacher": "train_single_matrixformer_cohort_anchor_retention",
    "audit_single_model_legacy_blend_teacher": "audit_single_model_cohort_anchor_retention",
    "legacy_blend_teacher_v41": "cohort_anchor_retention_v42",
    "test_single_model_legacy_blend_teacher_trial": "test_single_model_cohort_anchor_retention_trial",
    "test_single_model_legacy_blend_teacher_pipeline": "test_single_model_cohort_anchor_retention_pipeline",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V42 cohort qualification adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V41", "V42")
ARITHMETIC_PATCH = 'ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, '\
    + repr('"OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"') + ', '\
    + repr('"OPENBLAS_NUM_THREADS": "24", "MKL_NUM_THREADS": "24"') + ')\n'\
    + 'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()'
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'exec(compile(SOURCE, __file__, "exec"), globals())',
    'SOURCE = replace_once(SOURCE, "ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()", ARITHMETIC_PATCH)\n'
    'SOURCE = cohort_trial.wrap_cli(SOURCE)\n'
    'exec(compile(SOURCE, __file__, "exec"), globals())')
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
