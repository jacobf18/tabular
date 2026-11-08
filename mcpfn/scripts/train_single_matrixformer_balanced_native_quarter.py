"""Train one V55 checkpoint with balanced native coverage and quarter displacement."""

import train_single_matrixformer_standardized_huber as training_blueprint
import single_model_fixed_quarter_trial as fixed_trial
import single_model_balanced_native_quarter_trial as all_trial
import single_model_balanced_native_coverage as native
import single_model_balanced_structured_components as components
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once


CAPACITY_RECEIPT_SHA256 = 'a17e0402d32f284b1285705e75b4036fb4b86d9f3d11cb710101f678d9abd087'
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    'relocate_single_model_standardized_huber_capacity': 'relocate_single_model_balanced_native_quarter_capacity',
    'standardized_teacher_huber_v54': 'balanced_native_quarter_v55',
    '1421100901, 1422100901': '1461100901, 1462100901',
    '1431100901, 1432100901': '1471100901, 1472100901',
    'preflight_single_model_standardized_teacher_huber': 'preflight_single_model_balanced_native_quarter',
    'audit_single_model_standardized_huber.py': 'audit_single_model_balanced_native_quarter.py',
    'run_single_model_standardized_huber_pipeline.py': 'run_single_model_balanced_native_quarter_pipeline.py',
    'test_single_model_standardized_huber_trial.py': 'test_single_model_balanced_native_quarter_trial.py',
    'test_single_model_standardized_huber_full_audit.py': 'test_single_model_balanced_native_quarter_full_audit.py',
    'test_single_model_standardized_huber_pipeline.py': 'test_single_model_balanced_native_quarter_pipeline.py',
    'V54': 'V55',
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f'Missing native-quarter trial adaptation: {old}')
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, 'ANCHOR_POLICY', all_trial.POLICY)
BATCH_WRAPPER = replace_once(training_blueprint.BATCH_WRAPPER, 'components.balance(batch)', 'native.balance(batch)')
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, 'exec'), globals())
