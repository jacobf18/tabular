"""Audit every V40 task, fixed-half optimizer record and imputation reference."""

import audit_single_model_fixed_half_shrunk_retention as previous_audit
import single_model_fixed_half_trial as fixed_trial


ADAPTATION_SOURCE = previous_audit.ADAPTATION_SOURCE
for old, new in {
    "single_model_fixed_half_shrunk_retention_training": "single_model_all_imputation_retention_training",
    "train_single_matrixformer_fixed_half_shrunk_retention": "train_single_matrixformer_all_imputation_retention",
    "fixed_half_shrunk_retention_v39_audit": "all_imputation_retention_v40_audit",
    "single_model_fixed_half_shrunk_retention_audit_replay": "single_model_all_imputation_retention_audit_replay",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V40 independent-audit adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V39", "V40")
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
