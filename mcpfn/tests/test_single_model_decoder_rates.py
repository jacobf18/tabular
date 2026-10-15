from copy import deepcopy

import pytest
import torch

import probe_single_model_decoder_rates as probe


def test_compact_endpoint_reuses_only_exact_duplicate_gradients():
    gradients = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original = {"task_gradients": gradients[:2].clone(), "before": torch.ones(4)}
    compact = probe.compact_snapshot(original, gradients, 2)
    assert "task_gradients" not in compact
    assert "task_gradients" in original
    restored = probe.restore_snapshot(compact, gradients, 2)
    assert torch.equal(restored["task_gradients"], original["task_gradients"])
    assert restored["task_gradients"].data_ptr() == gradients.data_ptr()
    original["task_gradients"][0, 0] += .01
    with pytest.raises(ValueError, match="exactly duplicated"):
        probe.compact_snapshot(original, gradients, 2)


@pytest.mark.parametrize("count", [0, 3, 4])
def test_shared_gradient_reference_rejects_incomplete_inventory(count):
    with pytest.raises(ValueError, match="complete shared"):
        probe.restore_snapshot({}, torch.ones(3, 4), count)


def test_repeated_feasibility_cannot_hide_one_bad_repeat():
    before = {"0": {"rmse": 1., "constant": None}, "1": {"rmse": 1., "constant": None}}
    after = {"0": {"rmse": .9, "constant": None}, "1": {"rmse": .9, "constant": None}}
    assert probe.repeat_decision(before, after)["finite_feasible"]
    after["1"]["rmse"] = 1. + 1e-14
    assert not probe.repeat_decision(before, after)["finite_feasible"]
    with pytest.raises(ValueError, match="Both"):
        probe.repeat_decision(before, {"0": after["0"]})
    after["1"]["constant"] = 0.
    with pytest.raises(ValueError, match="coverage"):
        probe.repeat_decision(before, after)


def test_fresh_optimizer_never_inherits_another_rates_moments():
    model = torch.nn.Linear(2, 1)
    first = probe.fresh_optimizer(model, .02)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    first.step()
    second = probe.fresh_optimizer(model, .02)
    assert first.state and not second.state
    assert second.param_groups[0]["lr"] == probe.source.RATE == 2e-6
    assert second.param_groups[0]["fused"]
    assert second.param_groups[0]["weight_decay"] == .02


def test_probe_does_not_change_the_existing_training_guard_policy():
    assert probe.RATES == (2e-6, 2e-5, 1e-4)
    assert probe.source.training.guarded.MARGIN == .1
    assert probe.WAIT_FOR.name == "targeted_decoder_v15_pipeline_recovery1"
    assert "not training or model selection" in probe.POLICY


@pytest.mark.parametrize("rate", probe.RATES)
@pytest.mark.parametrize("accept", [True, False])
def test_actual_fresh_adam_and_compact_endpoint_roundtrip(rate, accept):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    parent = deepcopy(model.state_dict())
    gradients = torch.tensor([[1., 1.], [1., 0.]])
    optimizer = probe.fresh_optimizer(model, .02)
    record, snapshot = probe.source.training.guarded.protected_update(model, optimizer, gradients, [1.], rate,
        {"rmse": 1.}, lambda current: {"rmse": float(current.weight[0, 0].detach()) if accept else 1.1})
    record.update(step=1, accepted_updates=int(record["accepted"]))
    assert record["accepted"] is accept
    compact = probe.compact_snapshot(snapshot, gradients, 1)
    restored = probe.restore_snapshot(compact, gradients, 1)
    probe.checks.check_record(record, [1.], {"rmse": 1.}, 2)
    for name in ("before", "tentative", "after", "task_gradients", "combined_gradient", "applied_gradient"):
        assert torch.equal(restored[name], snapshot[name])
    assert not restored["optimizer_before"]["state"]
    if accept:
        probe.checks.optimizer_equal(optimizer.state_dict(), restored["optimizer_after"])
        assert all(int(state["step"]) == 1 for state in optimizer.state.values())
    if not accept:
        assert torch.equal(model.weight, parent["weight"])
        assert not optimizer.state
        assert optimizer.param_groups[0]["lr"] == probe.source.RATE
    second = probe.fresh_optimizer(model, .02)
    assert not second.state
