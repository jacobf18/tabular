"""Train one V50 checkpoint with fixed dual-parent structured retention."""

import train_single_matrixformer_parent_structured as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_dual_parent_trial as all_trial
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = "055bc901667df789527eef698c22085dcb81fbab795577f0c42d055f66f0213c"
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.TRAINING_ADAPTATION_SOURCE
for old, new in {
    "relocate_single_model_parent_structured_capacity": "relocate_single_model_dual_parent_capacity",
    "test_single_model_parent_structured_training.py": "test_single_model_dual_parent_training.py",
    "single_model_parent_structured_training": "single_model_dual_parent_training",
    "single_model_parent_structured_retention as retention": "single_model_dual_parent_retention as retention",
    "parent_structured_v49": "dual_parent_v50",
    "1271100901, 1272100901": "1301100901, 1302100901",
    "1281100901, 1282100901": "1311100901, 1312100901",
    "preflight_single_model_parent_structured_retention": "preflight_single_model_dual_parent_retention",
    "audit_single_model_parent_structured.py": "audit_single_model_dual_parent.py",
    "run_single_model_parent_structured_pipeline.py": "run_single_model_dual_parent_pipeline.py",
    "test_single_model_parent_structured_retention.py": "test_single_model_dual_parent_retention.py",
    "test_single_model_parent_structured_trial.py": "test_single_model_dual_parent_trial.py",
    "test_single_model_parent_structured_full_audit.py": "test_single_model_dual_parent_full_audit.py",
    "test_single_model_parent_structured_pipeline.py": "test_single_model_dual_parent_pipeline.py",
    "V49": "V50",
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit dual-parent full-trial adaptation: {old}")
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, "ANCHOR_POLICY", all_trial.POLICY)
BATCH_WRAPPER = training_blueprint.BATCH_WRAPPER
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, "exec"), globals())
