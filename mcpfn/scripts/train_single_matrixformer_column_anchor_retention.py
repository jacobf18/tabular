"""Train V38 with equal-column normalized anchor retention and unchanged qualification."""

import inspect

import train_single_matrixformer_anchor_retention as previous_trial


ADAPTATION_SOURCE = inspect.getsource(previous_trial)
for old, new in {
    "import single_model_anchor_retention as retention": "import single_model_column_anchor_retention as retention",
    "single_model_anchor_retention_training": "single_model_column_anchor_retention_training",
    "relocate_single_model_anchor_retention_capacity": "relocate_single_model_column_anchor_capacity",
    "anchor_retention_v37": "column_anchor_retention_v38",
    "881100901, 882100901": "911100901, 912100901",
    "891100901, 892100901": "921100901, 922100901",
    "preflight_single_model_anchor_retention": "preflight_single_model_column_anchor_retention",
    "audit_single_model_anchor_retention": "audit_single_model_column_anchor_retention",
    "run_single_model_anchor_retention_full_pipeline": "run_single_model_column_anchor_retention_full_pipeline",
    "test_single_model_anchor_retention": "test_single_model_column_anchor_retention",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V38 training adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V37", "V38")
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
