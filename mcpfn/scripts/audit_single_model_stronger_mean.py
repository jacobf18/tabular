"""Audit every V46 original task, component, reference and optimizer endpoint."""

import audit_single_model_all_imputation_blend as audit_blueprint
from audit_single_model_all_imputation_blend import fixed_trial, admission_source
import single_model_uniform_v8_retention_training as retention_training
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_stronger_mean_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


SOURCE = audit_blueprint.SOURCE
for old, new in {
    'train_single_matrixformer_all_imputation_blend': 'train_single_matrixformer_stronger_mean',
    'all_imputation_blend_v45': 'stronger_mean_distillation_v46',
    'V45': 'V46',
}.items():
    if old not in SOURCE:
        raise ValueError(f'Missing explicit stronger-mean audit adaptation: {old}')
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, 'exec'), globals())
