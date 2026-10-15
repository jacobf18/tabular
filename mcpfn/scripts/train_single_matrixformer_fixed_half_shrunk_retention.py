"""Train V39 from V29 with count-shrunk retention and fixed-half projected updates."""

import train_single_matrixformer_column_anchor_retention as previous_trial
import single_model_fixed_half_trial as fixed_trial
from single_model_registered_adaptation import replace_once


ADAPTATION_SOURCE = previous_trial.ADAPTATION_SOURCE
for old, new in {
    "import single_model_column_anchor_retention as retention": "import single_model_shrunk_column_anchor_retention as retention",
    "single_model_column_anchor_retention_training": "single_model_fixed_half_shrunk_retention_training",
    "relocate_single_model_column_anchor_capacity": "relocate_single_model_fixed_half_capacity",
    "column_anchor_retention_v38": "fixed_half_shrunk_retention_v39",
    "911100901, 912100901": "971100901, 972100901",
    "921100901, 922100901": "981100901, 982100901",
    "preflight_single_model_column_anchor_retention": "preflight_single_model_fixed_half_shrunk_retention",
    "audit_single_model_column_anchor_retention": "audit_single_model_fixed_half_shrunk_retention",
    "run_single_model_column_anchor_retention_full_pipeline": "run_single_model_fixed_half_shrunk_retention_pipeline",
    "test_single_model_column_anchor_retention_trial": "test_single_model_fixed_half_shrunk_retention_trial",
    "test_single_model_column_anchor_retention_audit": "test_single_model_fixed_half_shrunk_retention_audit",
    "test_single_model_column_anchor_retention_full_pipeline": "test_single_model_fixed_half_shrunk_retention_pipeline",
    "test_single_model_column_anchor_retention.py": "test_single_model_shrunk_column_anchor_retention.py",
    'receipt["canonical_artifacts_checked"] != 3': 'receipt["canonical_artifacts_checked"] != 6',
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V39 training adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V38", "V39")
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'def retention_evidence():',
    'ANCHOR_POLICY += " " + fixed_trial.POLICY\n\ndef retention_evidence():')
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE,
    '    return {**evidence, str(path): capacity.sha256(path), str(configuration): capacity.sha256(configuration)}',
    '    if capacity.sha256(path) != "66a6cdeb08870a99ea7aa241084eff16f20d1fece5657798a5a91931cb26e709":\n'
    '        raise ValueError("Bind the exact completed six-file preservation receipt")\n'
    '    configuration_record = capacity.previous.copying.read(configuration)\n'
    '    capacity.verify_hashes(receipt["artifact_sha256"])\n'
    '    capacity.verify_hashes(configuration_record["input_sha256"])\n'
    '    capacity.verify_hashes(configuration_record["source_sha256"])\n'
    '    return {**evidence, str(path): capacity.sha256(path), str(configuration): capacity.sha256(configuration)}')
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'exec(compile(SOURCE, __file__, "exec"), globals())',
    'SOURCE = fixed_trial.adapt_training_recipe(SOURCE)\nexec(compile(SOURCE, __file__, "exec"), globals())')
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
