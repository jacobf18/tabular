"""Qualify V46 against unchanged synthetic guards and all frozen real benchmarks."""

import run_single_model_all_imputation_blend_pipeline as pipeline_blueprint
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_stronger_mean_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = pipeline_blueprint.ARITHMETIC_PATCH
ADAPTATION_SOURCE = pipeline_blueprint.ADAPTATION_SOURCE
for old, new in {
    'train_single_matrixformer_all_imputation_blend': 'train_single_matrixformer_stronger_mean',
    'audit_single_model_all_imputation_blend': 'audit_single_model_stronger_mean',
    'all_imputation_blend_v45': 'stronger_mean_distillation_v46',
    'V45': 'V46',
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f'Missing explicit stronger-mean qualification adaptation: {old}')
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
exec(compile(ADAPTATION_SOURCE, __file__, 'exec'), globals())
