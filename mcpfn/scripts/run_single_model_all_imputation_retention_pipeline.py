"""Qualify V40 without changing any original guard or frozen benchmark."""

import run_single_model_fixed_half_shrunk_retention_pipeline as previous_pipeline
import single_model_fixed_half_trial as fixed_trial
from single_model_registered_adaptation import replace_once


ADAPTATION_SOURCE = previous_pipeline.ADAPTATION_SOURCE
for old, new in {
    "single_model_fixed_half_shrunk_retention_training": "single_model_all_imputation_retention_training",
    "train_single_matrixformer_fixed_half_shrunk_retention": "train_single_matrixformer_all_imputation_retention",
    "audit_single_model_fixed_half_shrunk_retention": "audit_single_model_all_imputation_retention",
    "fixed_half_shrunk_retention_v39": "all_imputation_retention_v40",
    "test_single_model_fixed_half_shrunk_retention_trial": "test_single_model_all_imputation_retention_trial",
    "test_single_model_fixed_half_shrunk_retention_pipeline": "test_single_model_all_imputation_retention_pipeline",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V40 qualification adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V39", "V40")
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'exec(compile(SOURCE, __file__, "exec"), globals())',
    'SOURCE = replace_once(SOURCE, "import single_model_full_network_storage as diagnostics",\n'
    '    "import single_model_all_imputation_storage as diagnostics")\n'
    'exec(compile(SOURCE, __file__, "exec"), globals())')
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
