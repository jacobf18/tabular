from copy import deepcopy

import numpy as np
import pytest
import torch

import single_model_paired_protection as paired
from test_single_model_finite_guarded_update import assert_optimizer_equal


def metrics():
    return {f"legacy_draw_{draw}/{kind}": 1. for draw in (0, 1) for kind in ("rmse", "common_nrmse")}


def fixture():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    return model, optimizer, torch.ones((6, 2))


def test_actual_paired_update_and_independent_full_endpoint_audit():
    model, optimizer, gradients = fixture()
    before = metrics()

    def evaluate(current):
        return {name: float(current.weight[0, 0].detach()) for name in before}

    record, snapshot = paired.protected_update(model, optimizer, gradients, [.5, .5], .01, before, evaluate)
    record.update(step=1, accepted_updates=int(record["accepted"]))
    assert record["accepted"]
    paired.check_record(record, [.5, .5], before, 6)
    payload = {"record": record, "snapshot": snapshot, "gradients": gradients}
    audit = paired.check_endpoint(payload, record, [2], .02, .003, optimizer.state_dict())
    assert audit["accepted"] and audit["gradient_vectors"] == 6
    assert record["displacement"]["paired_directions"] == 3


def test_mean_training_acceptance_does_not_hide_individual_regression():
    before = metrics()
    after = {name: (.9 if "draw_0" in name else 1.05) for name in before}
    decision = paired.metric_decision(before, after)
    assert decision["accepted"]
    assert not decision["individual_draw_diagnostics"]["accepted"]
    assert sum(change > 0 for change in decision["individual_draw_diagnostics"]["changes"].values()) == 2
    assert not paired.metric_decision(before, before)["accepted"]
    assert not paired.metric_decision(before, {name: 1. + 1e-14 for name in before})["accepted"]


@pytest.mark.parametrize("existing_moments", [False, True])
def test_rejected_pair_means_restore_every_parameter_moment_and_rate(existing_moments):
    model, optimizer, gradients = fixture()
    if existing_moments:
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    parameters, state = model.weight.detach().clone(), deepcopy(optimizer.state_dict())
    record, snapshot = paired.protected_update(model, optimizer, gradients, [.5, .5], .01, metrics(),
                                               lambda current: {name: 1.1 for name in metrics()})
    assert not record["accepted"] and record["reason"] == "finite_metrics_rejected"
    torch.testing.assert_close(model.weight, parameters, rtol=0, atol=0)
    assert_optimizer_equal(optimizer.state_dict(), state)
    torch.testing.assert_close(snapshot["before"], snapshot["after"], rtol=0, atol=0)


def test_undefined_truth_coverage_is_preserved_without_fabricated_gradients():
    before = metrics()
    before["legacy_draw_0/common_nrmse"] = None
    labels, names, matrix = paired.gradient_layout(before, 2)
    assert len(labels) == 5 and len(names) == 3
    assert matrix.sum(axis=1).tolist() == [1., 1., 1.]
    assert paired.mean_metrics(before)["legacy/common_nrmse"] == 1.
    before["legacy_draw_1/common_nrmse"] = None
    assert paired.mean_metrics(before)["legacy/common_nrmse"] is None
    labels, names, matrix = paired.gradient_layout(before, 2)
    assert len(labels) == 4 and len(names) == 2
    with pytest.raises(ValueError, match="coverage"):
        paired.metric_decision(before, metrics())


def test_order_and_weights_must_bind_the_saved_raw_gradient_manifest():
    before = metrics()
    labels, names, matrix = paired.gradient_layout(before, 2)
    entries = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    tasks = [{"index": draw, "name": f"legacy_draw_{draw}", "weight": .5} for draw in (0, 1)]
    np.testing.assert_array_equal(paired.validate_layout(entries, tasks, before), matrix)
    with pytest.raises(ValueError, match="order"):
        paired.validate_layout(entries[::-1], tasks, before)
    tasks[0]["weight"] = .4
    with pytest.raises(ValueError, match="equally"):
        paired.validate_layout(entries, tasks, before)


@pytest.mark.parametrize("mutation", ["draw", "metric", "missing", "null", "nonfinite"])
def test_invalid_or_partial_pairs_are_rejected(mutation):
    before = metrics()
    if mutation == "draw":
        before["legacy_draw_2/rmse"] = before.pop("legacy_draw_0/rmse")
    elif mutation == "metric":
        before["legacy_draw_0/unknown"] = before.pop("legacy_draw_0/rmse")
    elif mutation == "missing":
        before.pop("legacy_draw_0/rmse")
    elif mutation == "null":
        before["legacy_draw_0/rmse"] = None
    else:
        before["legacy_draw_0/rmse"] = float("nan")
    with pytest.raises(ValueError):
        paired.gradient_layout(before, 2)


def test_projection_certificate_tampering_is_rejected():
    before = metrics()
    gradients, direction = np.ones((6, 2)), np.asarray([-1., -1.])
    gram, products, norm = gradients @ gradients.T, gradients @ direction, np.linalg.norm(direction)
    solution = paired.paired_solve(gram, products, norm, before, 2, .1)
    paired.paired_verify(gram, products, norm, solution, .1, before, 2)
    solution["raw_gradient_coefficients"][0] += .01
    with pytest.raises(AssertionError):
        paired.paired_verify(gram, products, norm, solution, .1, before, 2)
