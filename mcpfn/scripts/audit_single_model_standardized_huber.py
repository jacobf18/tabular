"""Audit every V54 task, teacher, retention component and saved endpoint."""

import audit_single_model_balanced_structured as audit_blueprint
from audit_single_model_balanced_structured import fixed_trial, admission_source
import single_model_standardized_huber_training as retention_training
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_standardized_huber_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


FULL_AUDIT_SOURCE = audit_blueprint.FULL_AUDIT_SOURCE
for old, new in {
    'train_single_matrixformer_balanced_structured': 'train_single_matrixformer_standardized_huber',
    'balanced_structured_components_v53': 'standardized_teacher_huber_v54',
    'V53': 'V54',
}.items():
    if old not in FULL_AUDIT_SOURCE:
        raise ValueError(f'Missing standardized-Huber full-audit adaptation: {old}')
    FULL_AUDIT_SOURCE = FULL_AUDIT_SOURCE.replace(old, new)
exec(compile(FULL_AUDIT_SOURCE, __file__, 'exec'), globals())
