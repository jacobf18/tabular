"""Audit all V43 tasks, weights, original teachers and frozen references."""

import audit_single_model_cohort_anchor_retention as previous_audit
from audit_single_model_cohort_anchor_retention import fixed_trial, admission_source
import single_model_family_balanced_retention_training as retention_training
import single_model_legacy_blend_full_replay as blend_replay
import single_model_family_balanced_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


SOURCE = previous_audit.SOURCE.replace('cohort_anchor_retention', 'family_balanced_retention')
SOURCE = SOURCE.replace('family_balanced_retention_v42', 'family_balanced_retention_v43').replace('V42', 'V43')
exec(compile(SOURCE, __file__, 'exec'), globals())
