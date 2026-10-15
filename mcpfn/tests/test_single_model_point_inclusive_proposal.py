from copy import deepcopy

import pytest
import torch

import single_model_paired_protection as paired
import single_model_point_inclusive_proposal as inclusive
from single_model_finite_guarded_audit import optimizer_equal


def metrics():
    return {f"legacy_draw_{draw}/{metric}": 1. for draw in (0, 1) for metric in ("rmse", "common_nrmse")}


def fixture():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    return model, optimizer, torch.ones((6, 2))


def test_mass_is_half_distillation_half_defined_point_metrics():
    before = metrics()
    assert inclusive.proposal_weights(before, [.5, .5]) == [.25, .25, .125, .125, .125, .125]
    before["legacy_draw_0/common_nrmse"] = None
    assert inclusive.proposal_weights(before, [.5, .5]) == [.25, .25, .25, .125, .125]
    before["legacy_draw_1/common_nrmse"] = None
    assert inclusive.proposal_weights(before, [.5, .5]) == [.25, .25, .25, .25]


def test_priors_and_ratings_point_mass_remain_explicit():
    before = {}
    for draw in (0, 1):
        for cohort in ("legacy", "structured", "recsys"):
            for metric in ("rmse", "mae" if cohort == "recsys" else "common_nrmse"):
                before[f"{cohort}_draw_{draw}/{metric}"] = 1.
    weights = [.125, .125, .25] * 2
    proposal = inclusive.proposal_weights(before, weights)
    labels, names, matrix = paired.gradient_layout(before, len(weights))
    for cohort, expected in (("legacy", .25), ("structured", .25), ("recsys", .5)):
        assert sum(weight for (task, kind), weight in zip(labels, proposal) if task.startswith(cohort)) == expected


@pytest.mark.parametrize("weights", [[.4, .6], [.5, .4], [0., 1.], [float("nan"), .5]])
def test_changed_or_invalid_task_masses_are_rejected(weights):
    with pytest.raises(ValueError):
        inclusive.proposal_weights(metrics(), weights)


def test_actual_update_and_independent_optimizer_audit():
    model, optimizer, gradients = fixture()
    before = metrics()
    record, snapshot = inclusive.protected_update(model, optimizer, gradients, [.5, .5], .01, before,
        lambda current: {name: float(current.weight[0, 0].detach()) for name in before})
    record.update(step=1, accepted_updates=int(record["accepted"]))
    assert record["accepted"]
    assert snapshot["task_gradients"].shape == gradients.shape
    inclusive.check_record(record, [.5, .5], before, 6)
    payload = {"record": record, "snapshot": snapshot, "gradients": gradients}
    audit = inclusive.check_endpoint(payload, record, [2], .02, .003, optimizer.state_dict())
    assert audit["accepted"] and audit["gradient_vectors"] == 6
    assert audit["adam"]["gradient_vectors"] == 6


def test_original_proposal_does_not_silently_replace_point_inclusive_adam():
    model, optimizer, gradients = fixture()
    gradients[2:, 1] = 4.
    record, snapshot = inclusive.protected_update(model, optimizer, gradients, [.5, .5], .01, metrics(),
        lambda current: {name: .9 for name in metrics()})
    combined, information = paired.original.original.projection.combined_gradient(
        gradients, inclusive.proposal_weights(metrics(), [.5, .5]))
    composite, information = paired.original.original.projection.combined_gradient(gradients[:2], [.5, .5])
    torch.testing.assert_close(snapshot["combined_gradient"], combined.cpu(), rtol=0, atol=0)
    assert not torch.equal(combined, composite)


@pytest.mark.parametrize("existing_moments", [False, True])
def test_rejection_restores_parameters_moments_and_rate(existing_moments):
    model, optimizer, gradients = fixture()
    if existing_moments:
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    parameters, state = model.weight.detach().clone(), deepcopy(optimizer.state_dict())
    record, snapshot = inclusive.protected_update(model, optimizer, gradients, [.5, .5], .01, metrics(),
        lambda current: {name: 1.1 for name in metrics()})
    record.update(step=1, accepted_updates=int(existing_moments))
    assert not record["accepted"] and record["reason"] == "finite_metrics_rejected"
    torch.testing.assert_close(model.weight, parameters, rtol=0, atol=0)
    optimizer_equal(optimizer.state_dict(), state)
    inclusive.check_record(record, [.5, .5], metrics(), 6)
    payload = {"record": record, "snapshot": snapshot, "gradients": gradients}
    inclusive.check_endpoint(payload, record, [2], .02, .003, optimizer.state_dict())


def test_tampered_proposal_weights_or_policy_are_rejected():
    model, optimizer, gradients = fixture()
    record, snapshot = inclusive.protected_update(model, optimizer, gradients, [.5, .5], .01, metrics(),
        lambda current: {name: .9 for name in metrics()})
    for field, value in (("proposal_weights", [.5] * 6), ("proposal_policy", "composite only")):
        changed = {**record, field: value}
        with pytest.raises(ValueError, match="proposal policy or weights"):
            inclusive.check_record(changed, [.5, .5], metrics(), 6)


def test_protection_is_unchanged_and_templates_remain_immutable():
    assert "gradients[:len(weights)]" in paired.PROTECTED_SOURCE
    assert "paired_solve" in inclusive.PROTECTED_SOURCE
    assert "paired_displacement" in inclusive.PROTECTED_SOURCE
    assert "metric_decision(before_metrics, after_metrics)" in inclusive.PROTECTED_SOURCE
    assert "proposal_policy" not in paired.PROTECTED_SOURCE
