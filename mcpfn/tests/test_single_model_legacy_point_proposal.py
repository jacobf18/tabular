from copy import deepcopy

import numpy as np
import pytest
import torch

import single_model_legacy_point_proposal as proposal
import single_model_legacy_stratum_protection as strata
import single_model_point_inclusive_proposal as previous
from single_model_finite_guarded_audit import optimizer_equal


def metrics():
    return {f"{cohort}_draw_{draw}/{kind}": 1.
            for draw in range(2) for cohort in ("legacy", "mnar", "recsys")
            for kind in ("rmse", "mae" if cohort == "recsys" else "common_nrmse")}


def fixture():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    return model, torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02), torch.ones((18, 2))


def test_preserves_new_protection_namespace_and_full_proposal_mass():
    before, weights = metrics(), [.25, .125, .125] * 2
    assert proposal.proposal_weights(before, weights) == previous.proposal_weights(before, weights)
    assert proposal.gradient_layout is strata.gradient_layout
    assert proposal.validate_layout is strata.validate_layout
    for function in (proposal.protected_update, proposal.check_record, proposal.check_endpoint):
        assert function.__globals__["metric_decision"] is strata.metric_decision
        assert function.__globals__["paired_displacement"] is strata.paired_displacement
    assert proposal.protected_update.__globals__["paired_solve"] is strata.paired_solve
    assert proposal.check_record.__globals__["paired_verify"] is strata.paired_verify


@pytest.mark.parametrize("null_metric", [False, True])
def test_actual_update_preserves_all_rows_and_independent_optimizer_audit(null_metric):
    model, optimizer, gradients = fixture()
    before, weights = metrics(), [.25, .125, .125] * 2
    if null_metric:
        before["legacy_draw_0/common_nrmse"] = None
        gradients = gradients[:-1]
    gradients[6:, 1] = 4.
    after = {name: .9 if value is not None else None for name, value in before.items()}
    record, snapshot = proposal.protected_update(model, optimizer, gradients, weights, .01, before,
                                                 lambda current: after)
    record.update(step=1, accepted_updates=int(record["accepted"]))
    assert record["accepted"]
    assert record["projection"]["scope"] == "legacy_strata_other_paired_means"
    assert snapshot["task_gradients"].shape == gradients.shape
    proposal.check_record(record, weights, before, len(gradients))
    audit = proposal.check_endpoint({"record": record, "snapshot": snapshot, "gradients": gradients},
                                    record, [2], .02, .003, optimizer.state_dict())
    assert audit["adam"]["gradient_vectors"] == len(gradients)
    expected, information = strata.original.original.projection.combined_gradient(
        gradients, proposal.proposal_weights(before, weights))
    torch.testing.assert_close(snapshot["combined_gradient"], expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("existing_moments", [False, True])
def test_legacy_regression_cannot_hide_and_rollback_is_exact(existing_moments):
    model, optimizer, gradients = fixture()
    if existing_moments:
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    parameters, state = model.weight.detach().clone(), deepcopy(optimizer.state_dict())
    before, after, weights = metrics(), metrics(), [.25, .125, .125] * 2
    after["legacy_draw_0/common_nrmse"], after["legacy_draw_1/common_nrmse"] = 1.1, .8
    assert strata.template.metric_decision(before, after)["accepted"]
    record, snapshot = proposal.protected_update(model, optimizer, gradients, weights, .01, before,
                                                 lambda current: after)
    record.update(step=1, accepted_updates=int(existing_moments))
    assert not record["accepted"] and record["reason"] == "finite_metrics_rejected"
    torch.testing.assert_close(model.weight, parameters, rtol=0, atol=0)
    optimizer_equal(optimizer.state_dict(), state)
    proposal.check_record(record, weights, before, len(gradients))
    proposal.check_endpoint({"record": record, "snapshot": snapshot, "gradients": gradients},
                            record, [2], .02, .003, optimizer.state_dict())


def test_complete_family_layout_preserves45_protected_directions():
    cohorts = ["legacy", *[f"structured_{index}" for index in range(12)], "recsys"]
    before = {f"{cohort}_draw_{draw}/{kind}": 1. for draw in range(2) for cohort in cohorts
              for kind in ("rmse", "mae" if cohort == "recsys" else "common_nrmse")}
    labels, names, matrix = proposal.gradient_layout(before, 28)
    assert matrix.shape == (45, 84) and len(labels) == 84
    for index, name in enumerate(names):
        assert np.count_nonzero(matrix[index]) == (1 if name.startswith("legacy_draw_") else 2)


@pytest.mark.parametrize("field,value", [("proposal_policy", "wrong"), ("proposal_weights", [.5, .5])])
def test_auditor_rejects_changed_proposal(field, value):
    model, optimizer, gradients = fixture()
    before, weights = metrics(), [.25, .125, .125] * 2
    record, snapshot = proposal.protected_update(model, optimizer, gradients, weights, .01, before,
                                                 lambda current: {name: .9 for name in before})
    record[field] = value
    with pytest.raises(ValueError):
        proposal.check_record(record, weights, before, len(gradients))
