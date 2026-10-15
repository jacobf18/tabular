"""Train one V47 checkpoint with native-general and structured-V8 retention."""

import train_single_matrixformer_stronger_mean as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_general_native_trial as all_trial
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = "1c511a6d3bd32c2f2ae09729699410e83e050ca2208f8a456243a29e200015b8"
previous_training = training_blueprint
TRAINING_ADAPTATION_SOURCE = training_blueprint.ADAPTATION_SOURCE
for old, new in {
    "relocate_single_model_stronger_mean_capacity": "relocate_single_model_general_native_capacity",
    "single_model_uniform_v8_retention": "single_model_general_native_retention",
    "stronger_mean_distillation_v46": "general_native_retention_v47",
    "1181100901, 1182100901": "1211100901, 1212100901",
    "1191100901, 1192100901": "1221100901, 1222100901",
    "preflight_single_model_stronger_mean_distillation": "preflight_single_model_general_native_retention",
    "audit_single_model_stronger_mean.py": "audit_single_model_general_native_retention.py",
    "run_single_model_stronger_mean_pipeline.py": "run_single_model_general_native_retention_pipeline.py",
    "test_single_model_stronger_mean_trial.py": "test_single_model_general_native_trial.py",
    "test_single_model_stronger_mean_full_audit.py": "test_single_model_general_native_full_audit.py",
    "test_single_model_stronger_mean_pipeline.py": "test_single_model_general_native_pipeline.py",
    "V46": "V47",
}.items():
    if old not in TRAINING_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit native-general full-trial adaptation: {old}")
    TRAINING_ADAPTATION_SOURCE = TRAINING_ADAPTATION_SOURCE.replace(old, new)
TRAINING_ADAPTATION_SOURCE = replace_assignment(TRAINING_ADAPTATION_SOURCE, "ANCHOR_POLICY", all_trial.POLICY)
BATCH_WRAPPER = training_blueprint.BATCH_WRAPPER
exec(compile(TRAINING_ADAPTATION_SOURCE, __file__, "exec"), globals())
