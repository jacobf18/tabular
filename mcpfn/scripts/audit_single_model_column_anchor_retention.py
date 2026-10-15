"""Independently audit V38 with complete task and frozen-reference replay."""

import inspect

import audit_single_model_anchor_retention as previous_audit


ADAPTATION_SOURCE = inspect.getsource(previous_audit)
for old, new in {
    "single_model_anchor_retention_training": "single_model_column_anchor_retention_training",
    "train_single_matrixformer_anchor_retention": "train_single_matrixformer_column_anchor_retention",
    "anchor_retention_v37_audit": "column_anchor_retention_v38_audit",
    "single_model_anchor_retention_audit_replay": "single_model_column_anchor_retention_audit_replay",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V38 independent-audit adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V37", "V38")
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
