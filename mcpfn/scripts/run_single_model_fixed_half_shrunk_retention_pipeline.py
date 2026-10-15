"""Qualify V39 only through complete fixed-half, reference and frozen-benchmark audits."""

import run_single_model_column_anchor_retention_full_pipeline as previous_pipeline
import single_model_fixed_half_trial as fixed_trial
from single_model_registered_adaptation import replace_once


ADAPTATION_SOURCE = previous_pipeline.ADAPTATION_SOURCE
for old, new in {
    "single_model_column_anchor_retention_training": "single_model_fixed_half_shrunk_retention_training",
    "train_single_matrixformer_column_anchor_retention": "train_single_matrixformer_fixed_half_shrunk_retention",
    "audit_single_model_column_anchor_retention": "audit_single_model_fixed_half_shrunk_retention",
    "column_anchor_retention_v38": "fixed_half_shrunk_retention_v39",
    "test_single_model_column_anchor_retention_trial": "test_single_model_fixed_half_shrunk_retention_trial",
    "test_single_model_column_anchor_retention_full_pipeline": "test_single_model_fixed_half_shrunk_retention_pipeline",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V39 qualification adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V38", "V39")
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'exec(compile(SOURCE, __file__, "exec"), globals())',
    'SOURCE = fixed_trial.adapt_pipeline_recipe(SOURCE)\nexec(compile(SOURCE, __file__, "exec"), globals())')
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
