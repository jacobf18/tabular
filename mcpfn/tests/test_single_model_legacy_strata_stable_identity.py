"""Check code identity independently of marshal's process-local reference encoding."""

from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import recover_single_model_legacy_strata_stable_identity as stable


def test_real_inventory_stable_across_interpreters_entry_modes_and_hash_seeds():
    expected = stable.function_inventory()
    for seed, mode in (("17", "import"), ("81", "runpy")):
        command = ("import json; import recover_single_model_legacy_strata_stable_identity as module; "
                   "print(json.dumps(module.function_inventory(), sort_keys=True))" if mode == "import" else
                   "import json, runpy; namespace = runpy.run_path('scripts/recover_single_model_legacy_strata_stable_identity.py', "
                   "run_name='identity_test'); print(json.dumps(namespace['function_inventory'](), sort_keys=True))")
        result = subprocess.check_output([sys.executable, "-c", command], cwd=stable.previous.REPO,
            env={**os.environ, "PYTHONHASHSEED": seed, "CUDA_VISIBLE_DEVICES": ""}, text=True)
        assert json.loads(result) == expected


def test_stable_identity_detects_scientific_code_or_constant_change():
    def first():
        return {"guard": 22, "cases": (152, 284)}

    def second():
        return {"guard": 21, "cases": (152, 284)}

    original = first.__code__
    changed = original.replace(co_consts=second.__code__.co_consts)
    assert stable.canonical_constant(original) != stable.canonical_constant(changed)
    assert stable.canonical_constant(original) != stable.canonical_constant(original.replace(co_code=bytes(len(original.co_code))))
    assert stable.canonical_constant(frozenset({"rating", "legacy", "structured"})) == stable.canonical_constant(
        frozenset({"structured", "legacy", "rating"}))
    assert stable.canonical_constant(True) != stable.canonical_constant(1)


def test_isolated_operational_namespace_preserves_all_science_and_originals():
    before = dict(vars(stable.previous))
    original_before = dict(vars(stable.previous.original))
    namespace = stable.namespace()
    science = namespace["execution_namespace"]()
    assert all(vars(stable.previous)[name] is value for name, value in before.items())
    assert all(vars(stable.previous.original)[name] is value for name, value in original_before.items())
    for name, function in stable.previous.owned_functions().items():
        if name != "verify":
            assert inspect.unwrap(science[name]).__code__ is inspect.unwrap(function).__code__
    assert science["REPLAY"] == stable.REPLAY
    assert science["OUTPUT"] == stable.OUTPUT
    assert science["CONFIRMATION"] == stable.previous.original.CONFIRMATION
    assert science["deepcopy"] is deepcopy
    assert science["verify"] is namespace["verify"]
    assert namespace["function_inventory"]() == stable.function_inventory()
    assert stable.previous.missing_globals(science) == {}


@pytest.mark.parametrize("artifact", ["run.lock", "result.json", "endpoint_tasks", "selected.npz"])
def test_previous_recovery_must_remain_completely_unused(tmp_path, monkeypatch, artifact):
    output = tmp_path / "first"
    output.mkdir()
    for name in ("config.json", "registration.json", "status.json"):
        stable.previous.write_json(output / name, {"state": "registered"})
    (output / artifact).touch()
    monkeypatch.setattr(stable.previous, "OUTPUT", output)
    with pytest.raises(ValueError, match="unused first recovery"):
        stable.require_previous_unused()
