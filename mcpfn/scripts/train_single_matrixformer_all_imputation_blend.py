"""Train one V45 student with uniform observed-blend imputation targets."""

import ast

import train_single_matrixformer_uniform_v8_retention as previous_training
import single_model_fixed_half_trial as fixed_trial
import single_model_all_imputation_blend_trial as all_trial
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = "0ca932be8427b0d7933f353425abe89a5d0f4b45e7a157ddc75bcc85eabcca23"
ADAPTATION_SOURCE = previous_training.ADAPTATION_SOURCE
for old, new in {
    "relocate_single_model_uniform_v8_capacity": "relocate_single_model_all_blend_capacity",
    "uniform_v8_retention_v44": "all_imputation_blend_v45",
    "1121100901, 1122100901": "1151100901, 1152100901",
    "1131100901, 1132100901": "1161100901, 1162100901",
    "preflight_single_model_uniform_v8_retention": "preflight_single_model_all_imputation_blend_teacher",
    "audit_single_model_uniform_v8_retention.py": "audit_single_model_all_imputation_blend.py",
    "run_single_model_uniform_v8_retention_pipeline.py": "run_single_model_all_imputation_blend_pipeline.py",
    "test_single_model_uniform_v8_retention_trial.py": "test_single_model_all_imputation_blend_trial.py",
    "test_single_model_uniform_v8_retention_audit.py": "test_single_model_all_imputation_blend_audit.py",
    "test_single_model_uniform_v8_retention_pipeline.py": "test_single_model_all_imputation_blend_pipeline.py",
    "receipt['canonical_artifacts_checked'] != 3": "receipt['canonical_artifacts_checked'] != 6",
    "receipt['bytes_relocated'] != 1652558524": "receipt['bytes_relocated'] != 2496316664",
    "three-file": "six-file",
    "require_completed_v43": "require_completed_v44",
    "V44": "V45",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit all-imputation-blend trial adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = replace_assignment(ADAPTATION_SOURCE, "ANCHOR_POLICY", all_trial.POLICY)
node = next(node for node in ast.parse(ADAPTATION_SOURCE).body
    if isinstance(node, ast.FunctionDef) and node.name == "retention_evidence")
RETENTION_EVIDENCE_SOURCE = ast.get_source_segment(ADAPTATION_SOURCE, node)
BATCH_WRAPPER = previous_training.BATCH_WRAPPER
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
