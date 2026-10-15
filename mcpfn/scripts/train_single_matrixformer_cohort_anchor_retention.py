"""Train one V42 student with frozen V8/V29 cohort-specific training references."""

import train_single_matrixformer_legacy_blend_teacher as previous_training
import single_model_fixed_half_trial as fixed_trial
import single_model_cohort_anchor_trial as all_trial
from single_model_registered_adaptation import replace_once


CAPACITY_RECEIPT_SHA256 = "5917eccbe3458fb81c327d7e3b8adade493fc7fe0a05314b9cd510d6a7f4a522"
ADAPTATION_SOURCE = previous_training.ADAPTATION_SOURCE
for old, new in {
    "single_model_all_imputation_retention": "single_model_cohort_anchor_retention",
    "relocate_single_model_legacy_blend_capacity": "relocate_single_model_cohort_capacity",
    "legacy_blend_teacher_v41": "cohort_anchor_retention_v42",
    "1031100901, 1032100901": "1061100901, 1062100901",
    "1041100901, 1042100901": "1071100901, 1072100901",
    "preflight_single_model_legacy_blend_teacher": "preflight_single_model_cohort_anchor_retention",
    "audit_single_model_legacy_blend_teacher": "audit_single_model_cohort_anchor_retention",
    "run_single_model_legacy_blend_teacher_pipeline": "run_single_model_cohort_anchor_retention_pipeline",
    "test_single_model_legacy_blend_teacher_trial": "test_single_model_cohort_anchor_retention_trial",
    "test_single_model_legacy_blend_teacher_audit": "test_single_model_cohort_anchor_retention_audit",
    "test_single_model_legacy_blend_teacher_pipeline": "test_single_model_cohort_anchor_retention_pipeline",
    'receipt["canonical_artifacts_checked"] != 3': 'receipt["canonical_artifacts_checked"] != 7',
    '"1e4d520861c2fc0490e68e14396e62ffd25ccfdaeaf623550bfb0bcc1f0ed891"': "CAPACITY_RECEIPT_SHA256",
    "three-file preservation receipt": "seven-file preservation receipt",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V42 cohort training adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V41", "V42")
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'def retention_evidence():',
    'def retention_evidence():\n'
    '    if CAPACITY_RECEIPT_SHA256 is None:\n'
    '        raise ValueError("Do not register V42 before binding its completed storage-preservation receipt")')
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
