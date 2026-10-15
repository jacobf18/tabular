"""Qualify V36 under the unchanged full-network synthetic and frozen benchmark comparisons."""

import run_single_model_full_network_cap90_pipeline as predecessor
import single_model_full_network_cap90_evidence as memory_evidence
from single_model_registered_adaptation import replace_once


SOURCE = predecessor.SOURCE
for old, new in {
    "import train_single_matrixformer_full_network_cap90 as training": "import train_single_matrixformer_full_network_cap90_v2 as training",
    "import audit_single_model_full_network_cap90 as training_audit": "import audit_single_model_full_network_cap90_v2 as training_audit",
    "full_network_cap90_v35": "full_network_cap90_v36",
    "train_single_matrixformer_full_network_cap90.py": "train_single_matrixformer_full_network_cap90_v2.py",
    "audit_single_model_full_network_cap90.py": "audit_single_model_full_network_cap90_v2.py",
    "test_single_model_full_network_cap90_trial.py": "test_single_model_full_network_cap90_v2_trial.py",
    "test_single_model_full_network_cap90_pipeline.py": "test_single_model_full_network_cap90_v2_pipeline.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V36 coordinator adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, 'POLICY_TEXT += " " + memory_evidence.POLICY',
    'POLICY_TEXT += " " + memory_evidence.POLICY.replace("V" + "35", "V36")')
SOURCE = SOURCE.replace("V35", "V36")
exec(compile(SOURCE, __file__, "exec"), globals())
