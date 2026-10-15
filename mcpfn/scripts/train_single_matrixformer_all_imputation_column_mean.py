"""Train one V52 checkpoint with equal-column teacher means on all imputation."""

import train_single_matrixformer_native_parent as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_all_imputation_column_mean_trial as all_trial
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = "77f920e6ff03910cf965e59578f65d35495f27f9b8a20574dd8563f27736a8dc"
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    "relocate_single_model_native_parent_capacity": "relocate_single_model_all_imputation_column_mean_capacity",
    "test_single_model_native_parent_training.py": "test_single_model_all_imputation_column_mean_training.py",
    "single_model_native_parent_training": "single_model_all_imputation_column_mean_training",
    "single_model_native_parent_retention as retention": "single_model_all_imputation_column_mean as retention",
    "native_parent_v51": "all_imputation_column_mean_v52",
    "1331100901, 1332100901": "1361100901, 1362100901",
    "1341100901, 1342100901": "1371100901, 1372100901",
    "preflight_single_model_native_parent_retention": "preflight_single_model_all_imputation_column_mean",
    "audit_single_model_native_parent.py": "audit_single_model_all_imputation_column_mean.py",
    "run_single_model_native_parent_pipeline.py": "run_single_model_all_imputation_column_mean_pipeline.py",
    "test_single_model_native_parent_retention.py": "test_single_model_all_imputation_column_mean.py",
    "test_single_model_native_parent_trial.py": "test_single_model_all_imputation_column_mean_trial.py",
    "test_single_model_native_parent_full_audit.py": "test_single_model_all_imputation_column_mean_full_audit.py",
    "test_single_model_native_parent_pipeline.py": "test_single_model_all_imputation_column_mean_pipeline.py",
    "V51": "V52",
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit all-imputation full-trial adaptation: {old}")
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, "ANCHOR_POLICY", all_trial.POLICY)
BATCH_WRAPPER = training_blueprint.BATCH_WRAPPER
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, "exec"), globals())
