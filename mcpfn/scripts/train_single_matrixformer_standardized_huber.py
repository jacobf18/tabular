"""Train one V54 checkpoint using balanced priors and standardized teacher Huber."""

import train_single_matrixformer_balanced_structured as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_standardized_huber_trial as all_trial
import single_model_balanced_structured_components as components
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once


CAPACITY_RECEIPT_SHA256 = '6e449d933ceeeead08da81f8d2eef11420f3307cff9a30b38372cff0984984b2'
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    'relocate_single_model_balanced_structured_capacity': 'relocate_single_model_standardized_huber_capacity',
    'single_model_all_imputation_column_mean_training': 'single_model_standardized_huber_training',
    'single_model_all_imputation_column_mean as retention': 'single_model_standardized_teacher_huber as retention',
    'balanced_structured_components_v53': 'standardized_teacher_huber_v54',
    '1391100901, 1392100901': '1421100901, 1422100901',
    '1401100901, 1402100901': '1431100901, 1432100901',
    'preflight_single_model_balanced_structured_components': 'preflight_single_model_standardized_teacher_huber',
    'audit_single_model_balanced_structured.py': 'audit_single_model_standardized_huber.py',
    'run_single_model_balanced_structured_pipeline.py': 'run_single_model_standardized_huber_pipeline.py',
    'test_single_model_balanced_structured_trial.py': 'test_single_model_standardized_huber_trial.py',
    'test_single_model_balanced_structured_full_audit.py': 'test_single_model_standardized_huber_full_audit.py',
    'test_single_model_balanced_structured_pipeline.py': 'test_single_model_standardized_huber_pipeline.py',
    'V53': 'V54',
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f'Missing standardized-Huber trial adaptation: {old}')
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, 'ANCHOR_POLICY', all_trial.POLICY)
BATCH_WRAPPER = training_blueprint.BATCH_WRAPPER
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, 'exec'), globals())
