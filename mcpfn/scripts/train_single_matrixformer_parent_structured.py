"""Train one V49 checkpoint with fixed parent-structured retention."""

import train_single_matrixformer_native_column_mean as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_parent_structured_trial as all_trial
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = "de3f132f725b09c4d8a4225b14e41187f63a668921213b37cb181409ba2b5e49"
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    "relocate_single_model_native_column_capacity": "relocate_single_model_parent_structured_capacity",
    "single_model_native_column_mean_training": "single_model_parent_structured_training",
    "single_model_native_column_mean as retention": "single_model_parent_structured_retention as retention",
    "native_column_mean_v48": "parent_structured_v49",
    "1241100901, 1242100901": "1271100901, 1272100901",
    "1251100901, 1252100901": "1281100901, 1282100901",
    "preflight_single_model_native_column_mean": "preflight_single_model_parent_structured_retention",
    "audit_single_model_native_column_mean.py": "audit_single_model_parent_structured.py",
    "run_single_model_native_column_mean_pipeline.py": "run_single_model_parent_structured_pipeline.py",
    "test_single_model_native_column_mean.py": "test_single_model_parent_structured_retention.py",
    "test_single_model_native_column_training.py": "test_single_model_parent_structured_training.py",
    "test_single_model_native_column_trial.py": "test_single_model_parent_structured_trial.py",
    "test_single_model_native_column_full_audit.py": "test_single_model_parent_structured_full_audit.py",
    "test_single_model_native_column_pipeline.py": "test_single_model_parent_structured_pipeline.py",
    "V48": "V49",
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit parent-structured full-trial adaptation: {old}")
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, "ANCHOR_POLICY", all_trial.POLICY)
BATCH_WRAPPER = training_blueprint.BATCH_WRAPPER
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, "exec"), globals())
