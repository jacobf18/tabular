from copy import deepcopy

import pytest
import torch

import single_model_fixed_half_protection as protection
from single_model_finite_guarded_audit import optimizer_equal
import test_single_model_point_only_protection as previous_tests


def attempt(*, accepted=True, existing_moments=False):
    model, optimizer, gradients = previous_tests.fixture()
    if existing_moments:
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    original_state = deepcopy(optimizer.state_dict())
    before = previous_tests.metrics()
    after = {name: .9 for name in before}
    if not accepted:
        after["legacy_draw_0/common_nrmse"] = 1.1
    record, snapshot = protection.protected_update(model, optimizer, gradients, [.5, .5], .01, before, lambda current: after)
    record.update(step=1, accepted_updates=int(existing_moments) + int(record["accepted"]))
    return model, optimizer, gradients, original_state, before, record, snapshot


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("existing_moments", [False, True])
def test_actual_update_and_independent_audit_preserve_original_adam_and_exact_rollback(accepted, existing_moments):
    model, optimizer, gradients, original_state, before, record, snapshot = attempt(accepted=accepted, existing_moments=existing_moments)
    assert record["accepted"] == accepted and record["displacement_fraction"] == .5
    assert record["proposal_weights"] == protection.previous.proposal_weights(before, [.5, .5])
    assert torch.equal(snapshot["contracted"], protection.independent_half_step(snapshot["before"], snapshot["uncontracted"]))
    assert record["uncontracted_norm"] > record["candidate_norm"] > 0
    assert not torch.equal(snapshot["contracted"], snapshot["uncontracted"])
    assert record["displacement"]["individual_adverse_directions"] == 2
    if accepted:
        assert torch.equal(snapshot["after"], snapshot["contracted"])
        optimizer_equal(optimizer.state_dict(), snapshot["optimizer_after"])
    else:
        assert torch.equal(snapshot["after"], snapshot["before"])
        optimizer_equal(optimizer.state_dict(), original_state)
    protection.check_record(record, [.5, .5], before, len(gradients))
    report = protection.check_endpoint({"record": record, "snapshot": snapshot, "gradients": gradients},
        record, [2], .02, .003, optimizer.state_dict())
    assert report["fixed_half_step"]["actual_saved_displacements_checked"]
    assert report["fixed_half_step"]["contracted_non_bitwise_elements"] == 0
    assert report["adam"]["gradient_vectors"] == len(gradients)


@pytest.mark.parametrize("field", ["uncontracted", "contracted", "after"])
def test_corrupt_saved_vectors_are_rejected(field):
    model, optimizer, gradients, original_state, before, record, snapshot = attempt()
    snapshot[field] = snapshot[field].clone() + .01
    with pytest.raises((ValueError, AssertionError)):
        protection.check_endpoint({"record": record, "snapshot": snapshot, "gradients": gradients},
            record, [2], .02, .003, optimizer.state_dict())


def test_fixed_fraction_and_geometry_records_cannot_change():
    model, optimizer, gradients, original_state, before, record, snapshot = attempt()
    record["displacement_fraction"] = .25
    with pytest.raises(ValueError):
        protection.check_record(record, [.5, .5], before, len(gradients))
    with pytest.raises(ValueError):
        protection.check_contraction(snapshot, gradients, record)
    record["displacement_fraction"] = .5
    record["uncontracted_products"][0] += 1.
    with pytest.raises(AssertionError):
        protection.check_contraction(snapshot, gradients, record)


def test_callback_failure_keeps_both_vectors_and_restores_parameters_and_moments():
    model, optimizer, gradients = previous_tests.fixture()
    before, state = model.weight.detach().clone(), deepcopy(optimizer.state_dict())

    def fail(current):
        raise RuntimeError("injected forward failure")

    with pytest.raises(protection.StepEvaluationError) as raised:
        protection.protected_update(model, optimizer, gradients, [.5, .5], .01, previous_tests.metrics(), fail)
    assert torch.equal(model.weight, before)
    optimizer_equal(optimizer.state_dict(), state)
    assert set(("uncontracted", "contracted")).issubset(raised.value.snapshot)
    assert raised.value.record["reason"] == "evaluation_failed"


@pytest.mark.parametrize("bad", [torch.ones((1, 2)), torch.ones(3), torch.ones(2, dtype=torch.float64), torch.tensor([1., float('nan')])])
def test_contraction_rejects_invalid_full_vectors(bad):
    with pytest.raises(ValueError):
        protection.half_step(torch.zeros(2), bad)


def test_original_protection_and_namespaces_are_unchanged():
    assert protection.metric_decision is protection.previous.metric_decision
    assert protection.paired_solve is protection.previous.paired_solve
    assert protection.protected_update.__globals__["half_step"] is protection.half_step
    assert "half_step" not in protection.previous.protected_update.__globals__
    assert protection.check_endpoint.__globals__["check_contraction"] is protection.check_contraction
    assert "check_contraction" not in protection.previous.check_endpoint.__globals__


def test_contraction_preserves_the_full_rate_adam_proposal_not_a_recomputed_half_rate():
    model, optimizer, gradients, original_state, before, record, snapshot = attempt()
    original_model, original_optimizer, original_gradients = previous_tests.fixture()
    original_record, original_snapshot = protection.previous.protected_update(original_model, original_optimizer,
        original_gradients, [.5, .5], .01, before, lambda current: {name: .9 for name in before})
    assert original_record["accepted"]
    assert torch.equal(snapshot["tentative"], original_snapshot["tentative"])
    assert torch.equal(snapshot["uncontracted"], original_snapshot["after"])
    optimizer_equal(snapshot["optimizer_after"], original_snapshot["optimizer_after"])
    assert record["rate"] == original_record["rate"] == .01


def test_infeasible_projection_has_no_fabricated_contraction_and_rolls_back(monkeypatch):
    model, optimizer, gradients = previous_tests.fixture()
    parameters, state = model.weight.detach().clone(), deepcopy(optimizer.state_dict())

    def fail(*arguments):
        raise ValueError("injected infeasible projection")

    monkeypatch.setitem(protection._update_namespace, "paired_solve", fail)
    record, snapshot = protection.protected_update(model, optimizer, gradients, [.5, .5], .01,
        previous_tests.metrics(), lambda current: pytest.fail("No candidate forward for an infeasible projection"))
    record.update(step=1, accepted_updates=0)
    assert record["reason"] == "projection_infeasible"
    assert torch.equal(model.weight, parameters)
    optimizer_equal(optimizer.state_dict(), state)
    assert protection.check_contraction(snapshot, gradients, record) == {"evaluated": False, "fraction": .5}
    protection.check_record(record, [.5, .5], previous_tests.metrics(), len(gradients))
