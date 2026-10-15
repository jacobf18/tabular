from copy import deepcopy

import numpy as np
import pytest
import torch

import single_model_point_only_protection as protection
from single_model_finite_guarded_audit import optimizer_equal


def metrics():
    return {f"legacy_draw_{draw}/{kind}": 1. for draw in range(2) for kind in ("rmse", "common_nrmse")}


def fixture():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    gradients = torch.ones((6, 2))
    gradients[:2] = -.25
    return model, optimizer, gradients


def test_complete_layout_keeps84_raw_rows_and_all30_point_constraints():
    cohorts = ["legacy", *[f"structured_{index}" for index in range(12)], "recsys"]
    before = {f"{cohort}_draw_{draw}/{kind}": 1. for draw in range(2) for cohort in cohorts
              for kind in ("rmse", "mae" if cohort == "recsys" else "common_nrmse")}
    labels, names, matrix = protection.gradient_layout(before, 28)
    old_labels, old_names, old_matrix = protection.strata.gradient_layout(before, 28)
    assert labels == old_labels and len(labels) == 84
    assert matrix.shape == (30, 84) and not matrix[:, :28].any()
    assert all(not name.endswith("/composite") for name in names)
    for index, name in enumerate(names):
        np.testing.assert_array_equal(matrix[index], old_matrix[old_names.index(name)])


@pytest.mark.parametrize("undefined", [False, True])
def test_actual_point_only_update_allows_but_records_adverse_composite_directions(undefined):
    model, optimizer, gradients = fixture()
    before = metrics()
    if undefined:
        before["legacy_draw_0/common_nrmse"] = None
        gradients = gradients[:-1]
    after = {name: .9 if value is not None else None for name, value in before.items()}
    record, snapshot = protection.protected_update(model, optimizer, gradients, [.5, .5], .01, before,
                                                   lambda current: after)
    record.update(step=1, accepted_updates=int(record["accepted"]))
    assert record["accepted"] and record["projection"]["scope"] == protection.SCOPE
    assert record["displacement"]["individual_adverse_directions"] == 2
    assert record["displacement"]["paired_directions"] == (3 if undefined else 4)
    assert all(value < 0 for value in record["candidate_products"][:2])
    assert all(value > 0 for value in record["candidate_products"][2:])
    torch.testing.assert_close(snapshot["task_gradients"], gradients, rtol=0, atol=0)
    assert record["proposal_weights"] == protection.previous.proposal_weights(before, [.5, .5])
    protection.check_record(record, [.5, .5], before, len(gradients))
    audit = protection.check_endpoint({"record": record, "snapshot": snapshot, "gradients": gradients},
                                      record, [2], .02, .003, optimizer.state_dict())
    assert audit["adam"]["gradient_vectors"] == len(gradients)
    with pytest.raises(ValueError):
        protection.previous.check_record(record, [.5, .5], before, len(gradients))


@pytest.mark.parametrize("existing_moments", [False, True])
def test_original_separate_legacy_finite_gate_and_exact_rollback_remain(existing_moments):
    model, optimizer, gradients = fixture()
    if existing_moments:
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    parameters, state = model.weight.detach().clone(), deepcopy(optimizer.state_dict())
    before, after = metrics(), metrics()
    after["legacy_draw_0/common_nrmse"], after["legacy_draw_1/common_nrmse"] = 1.1, .8
    record, snapshot = protection.protected_update(model, optimizer, gradients, [.5, .5], .01, before,
                                                   lambda current: after)
    record.update(step=1, accepted_updates=int(existing_moments))
    assert not record["accepted"] and record["reason"] == "finite_metrics_rejected"
    torch.testing.assert_close(model.weight, parameters, rtol=0, atol=0)
    optimizer_equal(optimizer.state_dict(), state)
    protection.check_record(record, [.5, .5], before, len(gradients))
    protection.check_endpoint({"record": record, "snapshot": snapshot, "gradients": gradients},
                              record, [2], .02, .003, optimizer.state_dict())


def test_update_and_audit_bind_identical_new_projection_and_unchanged_finite_gate():
    assert protection.metric_decision is protection.strata.metric_decision
    for function in (protection.protected_update, protection.check_record, protection.check_endpoint):
        assert function.__globals__["paired_displacement"] is protection.paired_displacement
        assert function.__globals__["metric_decision"] is protection.strata.metric_decision
    assert protection.protected_update.__globals__["paired_solve"] is protection.paired_solve
    assert protection.check_record.__globals__["paired_verify"] is protection.paired_verify
