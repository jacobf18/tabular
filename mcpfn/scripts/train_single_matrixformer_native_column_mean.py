"""Train one V48 checkpoint with the fixed native-column mean objective."""

import train_single_matrixformer_general_native_retention as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_native_column_mean_trial as all_trial
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = "b15d13d06fe8598ac3bb3ab672746966043b522322691e6b6d0d7125fab0492d"
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    "relocate_single_model_general_native_capacity": "relocate_single_model_native_column_capacity",
    "single_model_general_native_retention_training": "single_model_native_column_mean_training",
    "single_model_general_native_retention as retention": "single_model_native_column_mean as retention",
    "general_native_retention_v47": "native_column_mean_v48",
    "1211100901, 1212100901": "1241100901, 1242100901",
    "1221100901, 1222100901": "1251100901, 1252100901",
    "preflight_single_model_general_native_retention": "preflight_single_model_native_column_mean",
    "audit_single_model_general_native_retention.py": "audit_single_model_native_column_mean.py",
    "run_single_model_general_native_retention_pipeline.py": "run_single_model_native_column_mean_pipeline.py",
    "test_single_model_general_native_retention.py": "test_single_model_native_column_mean.py",
    "test_single_model_native_column_mean_training.py": "test_single_model_native_column_training.py",
    "test_single_model_general_native_trial.py": "test_single_model_native_column_trial.py",
    "test_single_model_general_native_full_audit.py": "test_single_model_native_column_full_audit.py",
    "test_single_model_general_native_pipeline.py": "test_single_model_native_column_pipeline.py",
    "V47": "V48",
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit native-column full-trial adaptation: {old}")
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, "ANCHOR_POLICY", all_trial.POLICY)
BATCH_WRAPPER = training_blueprint.BATCH_WRAPPER
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, "exec"), globals())
