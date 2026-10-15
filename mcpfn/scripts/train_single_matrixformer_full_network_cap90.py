"""Train fresh V35 synthetic batches with the unchanged full network and measured.9 allocator cap."""

import inspect

import train_single_matrixformer_full_network as predecessor
import single_model_full_network_cap90_evidence as memory_evidence
from single_model_registered_adaptation import replace_once


SOURCE = inspect.getsource(predecessor)
for old, new in {
    "full_network_v34": "full_network_cap90_v35",
    "811100901, 812100901": "831100901, 832100901",
    "821100901, 822100901": "841100901, 842100901",
    "audit_single_model_full_network.py": "audit_single_model_full_network_cap90.py",
    "run_single_model_full_network_pipeline.py": "run_single_model_full_network_cap90_pipeline.py",
    "test_single_model_full_network_trial.py": "test_single_model_full_network_cap90_trial.py",
    "test_single_model_full_network_audit.py": "test_single_model_full_network_cap90_audit.py",
    "test_single_model_full_network_pipeline.py": "test_single_model_full_network_cap90_pipeline.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V35 adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, 'TRIAL_QUALIFICATION = template.TRIAL_QUALIFICATION',
    'TRIAL_POLICY += " " + memory_evidence.POLICY\nTRIAL_QUALIFICATION = template.TRIAL_QUALIFICATION')
SOURCE = replace_once(SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    'REGISTER_SOURCE = replace_once(REGISTER_SOURCE, '\
    '\'base = read(original.alignment.fixed.PRODUCER / "config.json")\', '\
    '\'base = read(original.alignment.fixed.PRODUCER / "config.json")\\n    base["memory_fraction"] = .9\')\n'
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())\n\n'
    'original_require_evidence = require_evidence\n'
    'def require_evidence():\n'
    '    return {**original_require_evidence(), **memory_evidence.require_evidence()}\n')
SOURCE = replace_once(SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    'REGISTER_SOURCE = replace_once(REGISTER_SOURCE, '\
    '\'sources = {Path(__file__).resolve(),\', '\
    '\'sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_full_network_cap90_evidence.py",\')\n'
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
SOURCE = SOURCE.replace("V34", "V35")
exec(compile(SOURCE, __file__, "exec"), globals())
