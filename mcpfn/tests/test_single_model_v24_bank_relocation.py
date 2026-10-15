import ast
import inspect

import pytest

import relocate_single_model_v24_confirmation_bank as relocation


def test_v24_scope_does_not_mutate_v23_or_move_any_checkpoint():
    assert relocation.OWNER.name == "top_block_v24_confirmation"
    assert relocation.OUTPUT.name == "v24_failed_confirmation_bank_relocation_v1"
    assert relocation.DESTINATION.name == "synthetic_bank.pt"
    assert relocation.DESTINATION.is_relative_to(relocation.scratch.SCRATCH)
    assert relocation.template.OWNER.name == "compact_continuation_v23_confirmation"
    assert "without starting or interrupting any scientific run" in relocation.POLICY
    assert "not provider-persistent storage or an external backup" in relocation.POLICY
    assert "Path(template.__file__).resolve()" in relocation.ADAPTED_SOURCE
    assert "scripts/single_model_registered_adaptation.py" in relocation.ADAPTED_SOURCE
    assert relocation.main.__globals__ is relocation.__dict__


def test_original_bank_completeness_and_capacity_function_bodies_are_unchanged():
    actual_tree = ast.parse(relocation.ADAPTED_SOURCE)
    original_tree = ast.parse(inspect.getsource(relocation.template))
    for name in ("check_bank", "require_capacity"):
        actual = next(node for node in actual_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        original = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual) == ast.dump(original)
    with pytest.raises(ValueError, match="complete284-case"):
        relocation.check_bank([])


def test_exact_completed_and_failed_v24_owners_are_mandatory(tmp_path, monkeypatch):
    monkeypatch.setattr(relocation, "ROOT", tmp_path)
    monkeypatch.setattr(relocation, "OWNER", tmp_path / "top_block_v24_confirmation")
    for directory in (relocation.OWNER, tmp_path / "top_block_v24_pipeline"):
        relocation.write_json(directory / "status.json", {"state": "complete", "pid": 999999999})
    relocation.write_json(relocation.OWNER / "audit.json", {"complete": True, "cases": 284, "prediction_grids": 1136})
    gate = {"passed": False, "confirmation_evaluated": True,
            "audit_sha256": relocation.sha256(relocation.OWNER / "audit.json")}
    relocation.write_json(relocation.OWNER / "gate.json", gate)
    assert relocation.require_owner()[0] == gate
    relocation.write_json(relocation.OWNER / "gate.json", {**gate, "passed": True})
    with pytest.raises(ValueError, match="failed V24 confirmation"):
        relocation.require_owner()
    relocation.write_json(relocation.OWNER / "gate.json", gate)
    relocation.write_json(relocation.OWNER / "status.json", {"state": "confirming", "pid": 999999999})
    with pytest.raises(ValueError, match="absent original V24"):
        relocation.require_owner()
