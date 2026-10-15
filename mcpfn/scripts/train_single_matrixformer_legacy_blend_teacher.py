"""Train one ordinary V41 student with legacy observed-blend distillation."""

import train_single_matrixformer_all_imputation_retention as previous_training
import single_model_fixed_half_trial as fixed_trial
import single_model_legacy_blend_trial as all_trial


ADAPTATION_SOURCE = previous_training.ADAPTATION_SOURCE
for old, new in {
    "relocate_single_model_all_imputation_capacity": "relocate_single_model_legacy_blend_capacity",
    "all_imputation_retention_v40": "legacy_blend_teacher_v41",
    "1001100901, 1002100901": "1031100901, 1032100901",
    "1011100901, 1012100901": "1041100901, 1042100901",
    "preflight_single_model_all_imputation_retention": "preflight_single_model_legacy_blend_teacher",
    "audit_single_model_all_imputation_retention": "audit_single_model_legacy_blend_teacher",
    "run_single_model_all_imputation_retention_pipeline": "run_single_model_legacy_blend_teacher_pipeline",
    "test_single_model_all_imputation_retention_trial": "test_single_model_legacy_blend_teacher_trial",
    "test_single_model_all_imputation_retention_audit": "test_single_model_legacy_blend_teacher_audit",
    "test_single_model_all_imputation_retention_pipeline": "test_single_model_legacy_blend_teacher_pipeline",
    "b56202d69f52ed8373132a3f94193bb733fc21c42bf254b2485f3e0494c93e20":
        "1e4d520861c2fc0490e68e14396e62ffd25ccfdaeaf623550bfb0bcc1f0ed891",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V41 legacy-blend training adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V40", "V41")
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
