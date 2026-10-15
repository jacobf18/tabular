import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import recover_single_model_legacy_blend_blas_stable as stable


def test_stable_inventory_matches_across_fresh_entry_modes():
    path = str(Path(stable.__file__).resolve())
    commands = [
        "import json; import recover_single_model_legacy_blend_blas_stable as module; "
        "print(json.dumps(module.function_inventory(), sort_keys=True))",
        f"import json, runpy; namespace = runpy.run_path({path!r}, run_name='identity_test'); "
        "print(json.dumps(namespace['function_inventory'](), sort_keys=True))",
    ]
    inventories = [json.loads(subprocess.check_output([sys.executable, "-c", command],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True)) for command in commands]
    assert inventories[0] == inventories[1] == stable.function_inventory()


def test_namespace_preserves_original_science_and_runtime_repair():
    original = stable.previous.original
    before = dict(vars(original))
    recovery = stable.namespace()
    scientific = recovery["execution_namespace"]()
    assert all(vars(original)[name] is value for name, value in before.items())
    assert recovery["POLICY"] == stable.previous.POLICY
    assert recovery["DIAGNOSIS"] == stable.previous.DIAGNOSIS
    assert recovery["BLAS_THREADS"] == 24
    assert recovery["function_inventory"] is stable.function_inventory
    assert scientific["REPLAY"] == stable.REPLAY
    assert scientific["CONFIRMATION"] == original.CONFIRMATION
    for name, function in stable.previous.owned_functions().items():
        if name != "verify":
            assert inspect.unwrap(scientific[name]).__code__ is inspect.unwrap(function).__code__


def test_registration_binding_includes_previous_evidence_and_all_new_dependencies():
    code = stable.namespace()["register"].__code__
    constants = json.dumps(stable.identities.canonical_constant(code.co_consts))
    assert "require_previous_unused" in code.co_names
    assert '"identity_policy"' in constants
    assert '"tests/test_single_model_legacy_blend_blas_stable.py"' in constants
    assert "identities" in code.co_names and "previous" in code.co_names


@pytest.mark.parametrize("change", ["lock", "result", "state", "live"])
def test_previous_recovery_must_remain_unused(tmp_path, monkeypatch, change):
    for name in ("OUTPUT", "REPLAY"):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setattr(stable.previous, name, directory)
        for filename in ("config.json", "registration.json"):
            (directory / filename).write_text("{}")
        stable.previous.write_json(directory / "status.json", {"state": "registered", "pid": 100})
    monkeypatch.setattr(stable.previous.original, "process_identity", lambda pid: {"pid": pid} if change == "live" else None)
    if change in {"lock", "result"}:
        (stable.previous.REPLAY / ("run.lock" if change == "lock" else "result.json")).touch()
    elif change == "state":
        stable.previous.write_json(stable.previous.REPLAY / "status.json", {"state": "complete", "pid": 100})
    with pytest.raises(ValueError):
        stable.require_previous_unused()
