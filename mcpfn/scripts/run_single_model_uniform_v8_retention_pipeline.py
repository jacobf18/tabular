"""Qualify V44 only after complete original, component and V8-reference audits."""

import run_single_model_family_balanced_retention_pipeline as previous_pipeline
import single_model_legacy_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_uniform_v8_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = previous_pipeline.ARITHMETIC_PATCH
ADAPTATION_SOURCE = previous_pipeline.ADAPTATION_SOURCE.replace('family_balanced_retention', 'uniform_v8_retention')
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace('uniform_v8_retention_v43', 'uniform_v8_retention_v44').replace('V43', 'V44')
exec(compile(ADAPTATION_SOURCE, __file__, 'exec'), globals())
