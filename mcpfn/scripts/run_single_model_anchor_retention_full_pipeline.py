"""Qualify V37 only after complete original and added-reference audits."""

import run_single_model_full_network_cap90_v2_pipeline as predecessor
from run_single_model_full_network_cap90_v2_pipeline import memory_evidence
import single_model_anchor_retention_training as retention_training
from single_model_registered_adaptation import replace_once


SOURCE = predecessor.SOURCE
for old, new in {
    "import train_single_matrixformer_full_network_cap90_v2 as training": "import train_single_matrixformer_anchor_retention as training",
    "import audit_single_model_full_network_cap90_v2 as training_audit": "import audit_single_model_anchor_retention as training_audit",
    "full_network_cap90_v36": "anchor_retention_v37",
    "train_single_matrixformer_full_network_cap90_v2.py": "train_single_matrixformer_anchor_retention.py",
    "audit_single_model_full_network_cap90_v2.py": "audit_single_model_anchor_retention.py",
    "test_single_model_full_network_cap90_v2_trial.py": "test_single_model_anchor_retention_trial.py",
    "test_single_model_full_network_cap90_v2_pipeline.py": "test_single_model_anchor_retention_full_pipeline.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V37 pipeline adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, 'POLICY_TEXT += " " + memory_evidence.POLICY.replace("V" + "35", "V36")',
    'POLICY_TEXT += " " + retention_training.POLICY')
SOURCE = replace_once(SOURCE, 'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()',
    'ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, "    require_compact_audit(audit)", '
    '"    require_compact_audit(audit)\\n    retention_training.require_report(training, audit)")\n'
    'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()')
SOURCE = SOURCE.replace("V36", "V37")
exec(compile(SOURCE, __file__, "exec"), globals())
