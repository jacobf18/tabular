"""Qualify V38 only through all original and additional independent audits."""

import inspect

import run_single_model_anchor_retention_full_pipeline as previous_pipeline


ADAPTATION_SOURCE = inspect.getsource(previous_pipeline)
for old, new in {
    "single_model_anchor_retention_training": "single_model_column_anchor_retention_training",
    "train_single_matrixformer_anchor_retention": "train_single_matrixformer_column_anchor_retention",
    "audit_single_model_anchor_retention": "audit_single_model_column_anchor_retention",
    "anchor_retention_v37": "column_anchor_retention_v38",
    "test_single_model_anchor_retention": "test_single_model_column_anchor_retention",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V38 qualification adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V37", "V38")
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
