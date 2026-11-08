"""Train one V56 checkpoint with a fixed256-attempt version of V55's recipe."""

import train_single_matrixformer_balanced_native_quarter as training_blueprint
import single_model_fixed_quarter_trial as fixed_trial
import single_model_balanced_native_long_trial as all_trial
import single_model_balanced_native_long_budget as budget
import single_model_balanced_native_coverage as native
import single_model_balanced_structured_components as components
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once


CAPACITY_RECEIPT_SHA256 = '69e8fb865c10e8a17327a295b62a96f68db90058d1fd568bea3deb0bc889d7ad'
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    'relocate_single_model_balanced_native_quarter_capacity': 'relocate_single_model_balanced_native_long_capacity',
    'import single_model_standardized_huber_training as retention_training':
        'import single_model_balanced_native_long_retention as retention_training',
    'balanced_native_quarter_v55': 'balanced_native_long_v56',
    '1461100901, 1462100901': '1481100901, 1482100901',
    '1471100901, 1472100901': '1491100901, 1492100901',
    'audit_single_model_balanced_native_quarter.py': 'audit_single_model_balanced_native_long.py',
    'run_single_model_balanced_native_quarter_pipeline.py': 'run_single_model_balanced_native_long_pipeline.py',
    'test_single_model_balanced_native_quarter_trial.py': 'test_single_model_balanced_native_long_trial.py',
    'test_single_model_balanced_native_quarter_full_audit.py': 'test_single_model_balanced_native_long_full_audit.py',
    'test_single_model_balanced_native_quarter_pipeline.py': 'test_single_model_balanced_native_long_pipeline.py',
    'V55': 'V56',
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f'Missing long-budget training adaptation: {old}')
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, 'ANCHOR_POLICY', all_trial.POLICY)
BATCH_WRAPPER = (
    '\ndef replicated_batches():\n'
    '    return budget.replicated_batches(previous_training, DATA_SEED, GRADIENT_SEED)\n'
    + training_blueprint.BATCH_WRAPPER)
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, 'exec'), globals())
