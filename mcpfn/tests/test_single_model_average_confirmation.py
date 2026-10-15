from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import confirm_single_model_weight_average as confirmation
from recsys_evaluation_common import sha256, write_json


def window():
    steps = list(range(300, 401, 10))
    config = {"steps": steps, "registered_at_producer_step": 224, "registered_unix": 1.}
    result = {"source_updates": steps, "snapshots": len(steps), "single_parameter_set": True}
    state = {"source_updates": steps, "step": 400, "derived_checkpoint": True, "gradient_updates_performed_here": 0}
    receipts = {"expected_steps": steps, "snapshots": [{"step": step, "checkpoint_sha256": "a" * 64, "captured_unix": 2.} for step in steps]}
    return config, result, state, receipts


def test_window_accepts_only_the_complete_prospective_average():
    values = window()
    confirmation.check_window(*values)
    altered = deepcopy(values)
    altered[3]["snapshots"].pop(2)
    with pytest.raises(ValueError, match="receipts"):
        confirmation.check_window(*altered)
    altered = deepcopy(values)
    altered[0]["registered_at_producer_step"] = 300
    with pytest.raises(ValueError, match="prospectively"):
        confirmation.check_window(*altered)
    altered = deepcopy(values)
    altered[2]["gradient_updates_performed_here"] = 400
    with pytest.raises(ValueError, match="semantics"):
        confirmation.check_window(*altered)


def test_inventory_rejects_extra_branches_changed_dtypes_and_nonfinite_weights():
    original = {"weight": torch.ones(2)}
    assert confirmation.inventory({"weight": torch.zeros(2)}, original)["changed_tensors"] == 1
    for state in ({"weight": torch.ones(2), "teacher": torch.ones(2)},
                  {"weight": torch.ones(2, dtype=torch.float64)}, {"weight": torch.tensor([1., float("nan")])}):
        with pytest.raises(ValueError):
            confirmation.inventory(state, original)


def test_alias_is_published_only_for_a_passed_hash_bound_candidate(tmp_path):
    checkpoint = tmp_path / "averaged_checkpoint.pth"
    torch.save({"model": {"weight": torch.ones(2)}}, checkpoint)
    gate = {"passed": False, "checkpoint_sha256": sha256(checkpoint)}
    with pytest.raises(ValueError, match="failed candidate"):
        confirmation.publish_alias(tmp_path, gate)
    gate["passed"] = True
    confirmation.publish_alias(tmp_path, gate)
    alias = tmp_path / "best_validation_checkpoint.pth"
    assert alias.is_symlink() and alias.resolve() == checkpoint
    confirmation.publish_alias(tmp_path, gate)
    gate["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="changed"):
        confirmation.publish_alias(tmp_path, gate)


def test_end_to_end_confirmation_uses_one_averaged_state_and_qualifies_it(tmp_path, monkeypatch):
    average, output = tmp_path / "average", tmp_path / "confirmation"
    average.mkdir()
    output.mkdir()
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    state = {"model": {"weight": torch.full((1, 1), .8)}}
    checkpoint = average / "averaged_checkpoint.pth"
    torch.save(state, checkpoint)
    audit = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "selected_step": 400}
    config = {"average": str(average), "base": {}, "cases": [{"domain": "imputation"}], "seed_start": 20100501, "policy": "strict"}
    write_json(output / "config.json", config)
    torch.save([], output / "synthetic_bank.pt")
    metric = lambda value: {"imputation": {"rmse": value, "mae": value}, "recsys": {"rmse": value, "mae": value}}
    reference = {"references": metric(.9), "general": metric(.9), "teacher": metric(.9),
                 "bank_sha256": sha256(output / "synthetic_bank.pt")}
    monkeypatch.setattr(confirmation, "verify_registration", lambda *args: None)
    monkeypatch.setattr(confirmation, "audit_candidate", lambda *args: (state, audit))
    monkeypatch.setattr(confirmation, "setup", lambda *args: (model, None, None, [], SimpleNamespace()))
    monkeypatch.setattr(confirmation, "prepare_bank", lambda *args: ([{}], reference))
    monkeypatch.setattr(confirmation, "evaluate_bank", lambda model, *args: (metric(float(model.weight.detach())), []))
    confirmation.confirm(config, output)
    gate = confirmation.read(output / "gate.json")
    assert gate["passed"] and gate["derived_checkpoint"] and gate["selected_step"] == 400
    assert (average / "best_validation_checkpoint.pth").is_symlink()


def test_readiness_does_not_accept_failed_or_missing_jobs(tmp_path, monkeypatch):
    write_json(tmp_path / "status.json", {"state": "complete"})
    assert confirmation.ready(tmp_path)
    write_json(tmp_path / "status.json", {"state": "failed"})
    with pytest.raises(RuntimeError, match="did not complete"):
        confirmation.ready(tmp_path)
    write_json(tmp_path / "status.json", {"state": "waiting", "pid": 123})
    monkeypatch.setattr(confirmation.os, "kill", lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    with pytest.raises(RuntimeError, match="missing or terminal"):
        confirmation.ready(tmp_path)
