from copy import deepcopy

import numpy as np
import pytest
import torch

import single_model_four_draw_protection as protection
from single_model_finite_guarded_audit import optimizer_equal


def metrics():
    return {f"{cohort}_draw_{draw}/{kind}": 1. for draw in range(4) for cohort in protection.COHORTS
        for kind in ("rmse", "mae" if cohort == "recsys" else "common_nrmse")}


def tasks():
    return [{"name": f"{cohort}_draw_{draw}", "index": 14 * draw + index,
        "weight": .125 if cohort == "legacy" else .0625 if cohort == "recsys" else
        .025 if cohort in ("mar", "mnar_skip_logic") else .00125}
        for draw in range(4) for index, cohort in enumerate(protection.COHORTS)]


def fixture(before, existing_moments=False):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    if existing_moments:
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    labels, names, matrix = protection.gradient_layout(before, 56)
    gradients = torch.ones((len(labels), 2))
    gradients[:56] = -.25
    return model, optimizer, gradients


def test_all168_gradients_and34_guards_have_expected_equal_draw_support():
    before = metrics()
    labels, names, matrix = protection.gradient_layout(before, 56)
    assert len(labels) == 168 and matrix.shape == (34, 168)
    assert not matrix[:, :56].any()
    for index, name in enumerate(names):
        expected = 1 if name.startswith("legacy_draw_") else 4
        assert np.count_nonzero(matrix[index]) == expected
        np.testing.assert_array_equal(matrix[index, matrix[index] > 0], np.full(expected, 1. / expected))
    entries = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    np.testing.assert_array_equal(protection.validate_layout(entries, tasks(), before), matrix)
    proposal = protection.proposal_weights(before, [task["weight"] for task in tasks()])
    assert len(proposal) == 168 and sum(proposal) == pytest.approx(1.)
    assert sum(proposal[:56]) == pytest.approx(.5)


def test_native_regression_cannot_be_hidden_by_other_native_draws():
    before, after = metrics(), metrics()
    after["legacy_draw_3/common_nrmse"] = 1.1
    after["legacy_draw_0/common_nrmse"] = .5
    decision = protection.metric_decision(before, after)
    assert not decision["accepted"] and decision["defined_slots"] == 34
    assert decision["changes"]["legacy_draw_3/common_nrmse"] > 0


def test_structured_four_draw_mean_and_individual_regression_remain_explicit():
    before, after = metrics(), metrics()
    after["mar_draw_3/common_nrmse"] = 1.1
    after["mar_draw_0/common_nrmse"] = .5
    decision = protection.metric_decision(before, after)
    assert decision["accepted"] and decision["changes"]["mar/common_nrmse"] < 0
    assert not decision["individual_draw_diagnostics"]["all_retained"]


@pytest.mark.parametrize("undefined", [(), ("legacy_draw_0/common_nrmse",),
    tuple(f"mar_draw_{draw}/common_nrmse" for draw in range(4))])
def test_nulls_keep_slots_without_fabricating_gradients_or_measured_passes(undefined):
    before = metrics()
    before.update(dict.fromkeys(undefined))
    labels, names, matrix = protection.gradient_layout(before, 56)
    after = {name: .9 if value is not None else None for name, value in before.items()}
    decision = protection.metric_decision(before, after)
    assert len(labels) == 168 - len(undefined)
    assert len(decision["changes"]) == 34
    assert decision["defined_slots"] == (33 if undefined else 34)
    assert sum(protection.proposal_weights(before, [task["weight"] for task in tasks()])) == pytest.approx(1.)
    if undefined:
        after[undefined[0]] = 0.
        with pytest.raises(ValueError, match="coverage cannot change"):
            protection.metric_decision(before, after)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("existing_moments", [False, True])
@pytest.mark.parametrize("undefined", [False, True])
def test_actual_update_quarter_contraction_and_independent_audit(accepted, existing_moments, undefined):
    before = metrics()
    if undefined:
        before["legacy_draw_2/common_nrmse"] = None
    model, optimizer, gradients = fixture(before, existing_moments)
    state, parameters = deepcopy(optimizer.state_dict()), model.weight.detach().clone()
    after = {name: .9 if value is not None else None for name, value in before.items()}
    if not accepted:
        after["legacy_draw_3/rmse"] = 1.1
    weights = [task["weight"] for task in tasks()]
    record, snapshot = protection.protected_update(model, optimizer, gradients, weights, .01, before,
        lambda current: after)
    record.update(step=1 + int(existing_moments), accepted_updates=int(existing_moments) + int(accepted))
    assert record["accepted"] is accepted
    assert record["displacement_fraction"] == .25
    assert record["projection"]["scope"] == protection.SCOPE
    assert record["displacement"]["paired_directions"] == (33 if undefined else 34)
    assert record["displacement"]["individual_adverse_directions"] == 56
    assert torch.equal(snapshot["contracted"], protection.quarter_step(snapshot["before"], snapshot["uncontracted"]))
    if not accepted:
        assert torch.equal(model.weight, parameters)
        optimizer_equal(optimizer.state_dict(), state)
    protection.check_record(record, weights, before, len(gradients))
    report = protection.check_endpoint({"record": record, "snapshot": snapshot, "gradients": gradients},
        record, [2], .02, .003, optimizer.state_dict())
    assert report["fixed_quarter_step"]["actual_saved_displacements_checked"]
    assert report["adam"]["gradient_vectors"] == (167 if undefined else 168)


@pytest.mark.parametrize("mutation", ["missing", "unknown_cohort", "fifth_draw", "wrong_metric", "undefined_rmse"])
def test_incomplete_or_reinterpreted_metrics_are_rejected(mutation):
    before = metrics()
    if mutation == "missing":
        del before["mar_draw_3/common_nrmse"]
    elif mutation == "unknown_cohort":
        before["other_draw_0/rmse"] = before.pop("mar_draw_0/rmse")
    elif mutation == "fifth_draw":
        before["mar_draw_4/rmse"] = before.pop("mar_draw_0/rmse")
    elif mutation == "wrong_metric":
        before["recsys_draw_0/common_nrmse"] = before.pop("recsys_draw_0/mae")
    elif mutation == "undefined_rmse":
        before["mar_draw_0/rmse"] = None
    with pytest.raises(ValueError):
        protection.gradient_layout(before, 56)


def test_unequal_weights_and_reordered_capture_are_rejected():
    before, original_tasks = metrics(), tasks()
    labels, names, matrix = protection.gradient_layout(before, 56)
    entries = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    weights = [task["weight"] for task in original_tasks]
    weights[0] += .01
    weights[14] -= .01
    with pytest.raises(ValueError, match="equally weighted"):
        protection.proposal_weights(before, weights)
    entries[0], entries[1] = entries[1], entries[0]
    with pytest.raises(ValueError, match="ordered"):
        protection.validate_layout(entries, original_tasks, before)


def test_new_callbacks_bind_both_update_and_audit_without_mutating_predecessors():
    assert protection.PROTECTED_SOURCE_SHA256 == protection.quarter.PROTECTED_SOURCE_SHA256
    assert protection.AUDIT_SOURCE_SHA256 == protection.quarter.AUDIT_SOURCE_SHA256
    assert protection.quarter.protected_update.__globals__["metric_decision"] is not protection.metric_decision
    for function in (protection.protected_update, protection.check_record, protection.check_endpoint):
        assert function.__globals__["proposal_weights"] is protection.proposal_weights
        assert function.__globals__["paired_displacement"] is protection.paired_displacement
        assert function.__globals__["metric_decision"] is protection.metric_decision
    assert protection.check_contraction.__globals__["paired_displacement"] is protection.paired_displacement


@pytest.mark.parametrize("failure", ["projection", "evaluation"])
def test_failed_proposals_preserve_exact_parameter_moment_and_rate_rollback(monkeypatch, failure):
    before = metrics()
    model, optimizer, gradients = fixture(before, existing_moments=True)
    state, parameters = deepcopy(optimizer.state_dict()), model.weight.detach().clone()
    weights = [task["weight"] for task in tasks()]

    def fail_projection(*arguments):
        raise ValueError("injected infeasible four-draw projection")

    def fail_evaluation(current):
        raise RuntimeError("injected four-draw forward failure")

    if failure == "projection":
        monkeypatch.setitem(protection._update_namespace, "paired_solve", fail_projection)
        record, snapshot = protection.protected_update(model, optimizer, gradients, weights, .01, before,
            lambda current: pytest.fail("No forward after infeasible projection"))
        assert record["reason"] == "projection_infeasible"
        assert protection.check_contraction(snapshot, gradients, record) == {"evaluated": False, "fraction": .25}
    else:
        with pytest.raises(protection.StepEvaluationError) as raised:
            protection.protected_update(model, optimizer, gradients, weights, .01, before, fail_evaluation)
        record, snapshot = raised.value.record, raised.value.snapshot
        assert record["reason"] == "evaluation_failed"
        assert {"uncontracted", "contracted"}.issubset(snapshot)
    assert torch.equal(model.weight, parameters)
    optimizer_equal(optimizer.state_dict(), state)


@pytest.mark.parametrize("field", ["uncontracted", "contracted", "after"])
def test_independent_auditor_rejects_corrupt_four_draw_saved_vectors(field):
    before = metrics()
    model, optimizer, gradients = fixture(before)
    weights = [task["weight"] for task in tasks()]
    record, snapshot = protection.protected_update(model, optimizer, gradients, weights, .01, before,
        lambda current: {name: .9 for name in before})
    record.update(step=1, accepted_updates=1)
    snapshot[field] = snapshot[field].clone() + .01
    with pytest.raises((ValueError, AssertionError)):
        protection.check_endpoint({"record": record, "snapshot": snapshot, "gradients": gradients},
            record, [2], .02, .003, optimizer.state_dict())
