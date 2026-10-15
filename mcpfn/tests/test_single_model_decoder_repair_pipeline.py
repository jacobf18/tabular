from copy import deepcopy

import pytest

import run_single_model_decoder_repair_pipeline as pipeline
from run_single_model_decoder_repair_pipeline import no_confirmation_reason


def test_initialization_never_consumes_confirmation():
    assert no_confirmation_reason({"step": 0}, {"selected_validation_passed": True}) is not None


def test_selected_replay_failure_has_no_fallback():
    assert "no fallback" in no_confirmation_reason({"step": 10}, {"selected_validation_passed": False})


def test_audited_trained_selection_may_reach_fresh_confirmation():
    assert no_confirmation_reason({"step": 10}, {"selected_validation_passed": True}) is None


@pytest.mark.parametrize("confirmation", [False, True])
def test_reservation_binds_policy_cases_and_parent_configs(confirmation):
    reservation = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
                   "confirmation_cases": pipeline.training.confirmation_cases() if confirmation else None}
    pipeline.verify_reservation(reservation, "pipeline", "run", confirmation)
    for key in reservation:
        changed = deepcopy(reservation)
        changed[key] = "changed"
        with pytest.raises(ValueError, match="reservation"):
            pipeline.verify_reservation(changed, "pipeline", "run", confirmation)


@pytest.mark.parametrize("confirmation_evaluated,passed", [(False, False), (True, False), (True, True)])
def test_pipeline_runs_real_campaign_only_after_independent_confirmation(monkeypatch, tmp_path, confirmation_evaluated, passed):
    calls = []
    gate = {"passed": passed, "confirmation_evaluated": confirmation_evaluated}
    reports = {pipeline.training.OUTPUT / "status.json": {"state": "training_complete"},
               pipeline.CONFIRMATION / "status.json": {"state": "awaiting_independent_audit" if confirmation_evaluated else "complete"},
               pipeline.CONFIRMATION / "gate.json": gate}
    original_read = pipeline.read
    monkeypatch.setattr(pipeline, "read", lambda path: reports[path])
    monkeypatch.setattr(pipeline, "verify", lambda output: None)
    monkeypatch.setattr(pipeline, "sha256", lambda path: "audit-digest")
    monkeypatch.setattr(pipeline, "run_child", lambda output, label, script, arguments, cpu=False: calls.append((label, cpu)))
    pipeline.run(tmp_path)
    expected = [("training_audit", True), ("replay", False), ("audit-replay", True), ("confirm", False)]
    if confirmation_evaluated:
        expected.append(("audit-confirmation", True))
    if passed:
        expected.extend([("benchmarks", False), ("review", True)])
    assert calls == expected
    result = original_read(tmp_path / "result.json")
    assert result["gate"] == gate
    assert result["campaign"] == (str(pipeline.CAMPAIGN) if passed else None)
    assert result["goal_success_not_assessed"] is True


def test_terminal_training_failure_never_starts_qualification(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "verify", lambda output: None)
    monkeypatch.setattr(pipeline, "read", lambda path: {"state": "failed"})
    monkeypatch.setattr(pipeline, "run_child", lambda *args, **kwargs: pytest.fail("Stopped training must never qualify"))
    with pytest.raises(RuntimeError, match="Training stopped"):
        pipeline.run(tmp_path)


def test_confirmation_setup_uses_v8_not_the_v13_training_input(monkeypatch):
    import torch
    model = torch.nn.Linear(2, 2)
    baseline = deepcopy(model.state_dict())
    calls = []
    monkeypatch.setattr(pipeline.training.original.alignment.mixture.backend, "setup", lambda config: (model, None, None, [], None))
    monkeypatch.setattr(pipeline.torch, "load", lambda path, **kwargs: calls.append(path) or {"model": baseline})
    assert pipeline.confirmation_setup({"base": {}})[0] is model
    assert calls == [pipeline.training.original.alignment.fixed.PARENT]
    with torch.no_grad():
        model.weight.add_(.01)
    with pytest.raises(ValueError, match="V8"):
        pipeline.confirmation_setup({"base": {}})


def test_endpoint_setup_loads_v13_backbone_and_assigns_decoder_coordinates(monkeypatch):
    import torch
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.decoder = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model.register_buffer("fixed_buffer", torch.ones(2))
    parent = deepcopy(model.state_dict())
    parent["encoder.weight"].add_(.01)
    inventory = [[name, list(value.shape), value.numel()] for name, value in model.named_parameters()]
    selected = [item for item in inventory if item[0].startswith("decoder.")]
    monkeypatch.setattr(pipeline.training.original.alignment.mixture, "original_setup", lambda config: (model, None, None, [], None))
    calls = []
    monkeypatch.setattr(pipeline.torch, "load", lambda path, **kwargs: calls.append(path) or {"model": parent})
    monkeypatch.setattr(pipeline, "read", lambda path: selected if path.name == "decoder_inventory.json" else inventory)
    def check(actual, subset, expected):
        assert actual == expected == inventory
        assert subset == selected
    monkeypatch.setattr(pipeline.training_audit, "check_inventories", check)
    current, view, criterion, loaded, mutable = pipeline.endpoint_setup({"base": {}})
    assert calls == [pipeline.training.PARENT]
    assert torch.equal(current.encoder.weight, parent["encoder.weight"])
    assert mutable == {name for name, value in view.named_parameters()}
    vector = torch.cat([value.detach().flatten() for value in view.parameters()]) + .02
    pipeline.training.original.alignment.preflight.assign(view, vector)
    frozen = pipeline.decoder.frozen_state_equal(parent, current.state_dict(), mutable)
    assert frozen["backbone_and_buffers_bitwise_unchanged"]
    assert len(frozen["changed_decoder_tensors"]) == 6
    assert not current.encoder.weight.requires_grad
    assert all(value.requires_grad for value in view.parameters())


def test_nested_comparator_covers_metric_decision_lists():
    decision = {"accepted": True, "comparisons": [{"metric": "rmse", "before": 1.0, "after": .9}]}
    pipeline.compare_nested(decision, deepcopy(decision))
    changed = deepcopy(decision)
    changed["comparisons"][0]["after"] = 1.1
    with pytest.raises((ValueError, AssertionError)):
        pipeline.compare_nested(decision, changed)
