"""Train one V53 checkpoint with component-balanced structured synthetic priors."""

import train_single_matrixformer_all_imputation_column_mean as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_balanced_structured_trial as all_trial
import single_model_balanced_structured_components as components
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once


CAPACITY_RECEIPT_SHA256 = 'ea43b960546285301030e4be4444268c760d30af9038e5c6c24286223f116742'
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    'relocate_single_model_all_imputation_column_mean_capacity': 'relocate_single_model_balanced_structured_capacity',
    'all_imputation_column_mean_v52': 'balanced_structured_components_v53',
    '1361100901, 1362100901': '1391100901, 1392100901',
    '1371100901, 1372100901': '1401100901, 1402100901',
    'preflight_single_model_all_imputation_column_mean': 'preflight_single_model_balanced_structured_components',
    'audit_single_model_all_imputation_column_mean.py': 'audit_single_model_balanced_structured.py',
    'run_single_model_all_imputation_column_mean_pipeline.py': 'run_single_model_balanced_structured_pipeline.py',
    'test_single_model_all_imputation_column_mean_trial.py': 'test_single_model_balanced_structured_trial.py',
    'test_single_model_all_imputation_column_mean_full_audit.py': 'test_single_model_balanced_structured_full_audit.py',
    'test_single_model_all_imputation_column_mean_pipeline.py': 'test_single_model_balanced_structured_pipeline.py',
    'V52': 'V53',
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f'Missing explicit balanced-prior full-trial adaptation: {old}')
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, 'ANCHOR_POLICY', all_trial.POLICY)
BATCH_WRAPPER = replace_once(training_blueprint.BATCH_WRAPPER,
    'result = unvalidated_batches()', 'result = [components.balance(batch) for batch in unvalidated_batches()]')
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, 'exec'), globals())
