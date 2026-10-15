"""Qualify V43 only after complete original and weighted-reference audits."""

import run_single_model_cohort_anchor_retention_pipeline as previous_pipeline
import single_model_legacy_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_family_balanced_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = previous_pipeline.ARITHMETIC_PATCH
ADAPTATION_SOURCE = previous_pipeline.ADAPTATION_SOURCE.replace('cohort_anchor_retention', 'family_balanced_retention')
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace('family_balanced_retention_v42', 'family_balanced_retention_v43')
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace('V42', 'V43')
exec(compile(ADAPTATION_SOURCE, __file__, 'exec'), globals())
