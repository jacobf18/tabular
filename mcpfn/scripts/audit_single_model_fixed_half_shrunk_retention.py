"""Independently audit every V39 task, fixed-half update and frozen reference grid."""

import audit_single_model_column_anchor_retention as previous_audit
import single_model_fixed_half_trial as fixed_trial
from single_model_registered_adaptation import replace_once


ADAPTATION_SOURCE = previous_audit.ADAPTATION_SOURCE
for old, new in {
    "single_model_column_anchor_retention_training": "single_model_fixed_half_shrunk_retention_training",
    "train_single_matrixformer_column_anchor_retention": "train_single_matrixformer_fixed_half_shrunk_retention",
    "column_anchor_retention_v38_audit": "fixed_half_shrunk_retention_v39_audit",
    "single_model_column_anchor_retention_audit_replay": "single_model_fixed_half_shrunk_retention_audit_replay",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V39 independent-audit adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V38", "V39")
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'exec(compile(SOURCE, __file__, "exec"), globals())',
    'SOURCE = fixed_trial.adapt_audit_recipe(SOURCE)\nexec(compile(SOURCE, __file__, "exec"), globals())')
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
