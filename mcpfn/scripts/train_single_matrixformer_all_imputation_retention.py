"""Train one ordinary V40 checkpoint with all-imputation reference retention."""

import train_single_matrixformer_fixed_half_shrunk_retention as previous_training
import single_model_fixed_half_trial as fixed_trial
import single_model_all_imputation_trial as all_trial
from single_model_registered_adaptation import replace_once


ADAPTATION_SOURCE = previous_training.ADAPTATION_SOURCE
for old, new in {
    "single_model_shrunk_column_anchor_retention": "single_model_all_imputation_retention",
    "single_model_fixed_half_shrunk_retention_training": "single_model_all_imputation_retention_training",
    "relocate_single_model_fixed_half_capacity": "relocate_single_model_all_imputation_capacity",
    "fixed_half_shrunk_retention_v39": "all_imputation_retention_v40",
    "971100901, 972100901": "1001100901, 1002100901",
    "981100901, 982100901": "1011100901, 1012100901",
    "preflight_single_model_fixed_half_shrunk_retention": "preflight_single_model_all_imputation_retention",
    "audit_single_model_fixed_half_shrunk_retention": "audit_single_model_all_imputation_retention",
    "run_single_model_fixed_half_shrunk_retention_pipeline": "run_single_model_all_imputation_retention_pipeline",
    "test_single_model_fixed_half_shrunk_retention_trial": "test_single_model_all_imputation_retention_trial",
    "test_single_model_fixed_half_shrunk_retention_audit": "test_single_model_all_imputation_retention_audit",
    "test_single_model_fixed_half_shrunk_retention_pipeline": "test_single_model_all_imputation_retention_pipeline",
    'receipt["canonical_artifacts_checked"] != 6': 'receipt["canonical_artifacts_checked"] != 3',
    "66a6cdeb08870a99ea7aa241084eff16f20d1fece5657798a5a91931cb26e709":
        "b56202d69f52ed8373132a3f94193bb733fc21c42bf254b2485f3e0494c93e20",
    "six-file preservation receipt": "three-file preservation receipt",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V40 training adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V39", "V40")
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'exec(compile(SOURCE, __file__, "exec"), globals())',
    'SOURCE = all_trial.adapt_recipe(SOURCE)\nexec(compile(SOURCE, __file__, "exec"), globals())')
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
