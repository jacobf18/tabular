"""Audit every V44 task, weighting decision, teacher and frozen reference grid."""

import audit_single_model_family_balanced_retention as previous_audit
from audit_single_model_family_balanced_retention import fixed_trial, admission_source
import single_model_uniform_v8_retention_training as retention_training
import single_model_legacy_blend_full_replay as blend_replay
import single_model_uniform_v8_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


SOURCE = previous_audit.SOURCE.replace('family_balanced_retention', 'uniform_v8_retention')
SOURCE = SOURCE.replace('uniform_v8_retention_v43', 'uniform_v8_retention_v44').replace('V43', 'V44')
exec(compile(SOURCE, __file__, 'exec'), globals())
