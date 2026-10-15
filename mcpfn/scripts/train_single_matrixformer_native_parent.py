"""Train one V51 checkpoint with native V29 and fixed-mean structured retention."""

import train_single_matrixformer_dual_parent as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_native_parent_trial as all_trial
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = "9d9379e6a38f1ac661843b5c076059b4e2a7e50a5c63d5c4cf5aa6f02cd9983e"
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    "relocate_single_model_dual_parent_capacity": "relocate_single_model_native_parent_capacity",
    "test_single_model_dual_parent_training.py": "test_single_model_native_parent_training.py",
    "single_model_dual_parent_training": "single_model_native_parent_training",
    "single_model_dual_parent_retention as retention": "single_model_native_parent_retention as retention",
    "dual_parent_v50": "native_parent_v51",
    "1301100901, 1302100901": "1331100901, 1332100901",
    "1311100901, 1312100901": "1341100901, 1342100901",
    "preflight_single_model_dual_parent_retention": "preflight_single_model_native_parent_retention",
    "audit_single_model_dual_parent.py": "audit_single_model_native_parent.py",
    "run_single_model_dual_parent_pipeline.py": "run_single_model_native_parent_pipeline.py",
    "test_single_model_dual_parent_retention.py": "test_single_model_native_parent_retention.py",
    "test_single_model_dual_parent_trial.py": "test_single_model_native_parent_trial.py",
    "test_single_model_dual_parent_full_audit.py": "test_single_model_native_parent_full_audit.py",
    "test_single_model_dual_parent_pipeline.py": "test_single_model_native_parent_pipeline.py",
    "V50": "V51",
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit native-parent full-trial adaptation: {old}")
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, "ANCHOR_POLICY", all_trial.POLICY)
BATCH_WRAPPER = training_blueprint.BATCH_WRAPPER
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, "exec"), globals())
