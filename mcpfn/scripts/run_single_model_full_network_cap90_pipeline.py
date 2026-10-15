"""Qualify a single V35 checkpoint without changing any synthetic or frozen-real comparison."""

import inspect

import run_single_model_full_network_pipeline as predecessor
import single_model_full_network_cap90_evidence as memory_evidence
from single_model_registered_adaptation import replace_once


SOURCE = inspect.getsource(predecessor)
for old, new in {
    "import train_single_matrixformer_full_network as training": "import train_single_matrixformer_full_network_cap90 as training",
    "import audit_single_model_full_network as training_audit": "import audit_single_model_full_network_cap90 as training_audit",
    "full_network_v34": "full_network_cap90_v35",
    "train_single_matrixformer_full_network.py": "train_single_matrixformer_full_network_cap90.py",
    "audit_single_model_full_network.py": "audit_single_model_full_network_cap90.py",
    "test_single_model_full_network_trial.py": "test_single_model_full_network_cap90_trial.py",
    "test_single_model_full_network_pipeline.py": "test_single_model_full_network_cap90_pipeline.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V35 coordinator adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, 'for name in ("require_stage_capacity",',
    'POLICY_TEXT += " " + memory_evidence.POLICY\n\nfor name in ("require_stage_capacity",')
SOURCE = SOURCE.replace("V34", "V35")
exec(compile(SOURCE, __file__, "exec"), globals())
