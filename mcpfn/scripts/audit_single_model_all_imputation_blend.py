"""Audit V45 targets, all raw imputation components and unchanged V8 retention."""

import audit_single_model_uniform_v8_retention as previous_audit
from audit_single_model_uniform_v8_retention import fixed_trial, admission_source
import single_model_uniform_v8_retention_training as retention_training
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_all_imputation_blend_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


SOURCE = previous_audit.SOURCE.replace("train_single_matrixformer_uniform_v8_retention", "train_single_matrixformer_all_imputation_blend")
SOURCE = SOURCE.replace("audit_single_model_uniform_v8_retention", "audit_single_model_all_imputation_blend")
SOURCE = SOURCE.replace("uniform_v8_retention_v44", "all_imputation_blend_v45").replace("V44", "V45")
exec(compile(SOURCE, __file__, "exec"), globals())
