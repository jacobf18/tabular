from copy import deepcopy

import pytest
import torch

from single_model_finite_guarded_audit import check_endpoint, check_record
from single_model_finite_guarded_update import protected_update
from test_single_model_finite_guarded_update import fixture


def capture(accepted):
    model, optimizer, gradients = fixture()
    optimizer.param_groups[0]["lr"] = .003
    record, snapshot = protected_update(model, optimizer, gradients, [1.], .01, {"rmse": 1.},
        lambda current: {"rmse": float(current.weight[0, 0].detach()) if accepted else 1.1})
    record.update(step=1, accepted_updates=int(accepted))
    payload = {"snapshot": snapshot, "gradients": gradients, "record": deepcopy(record)}
    return record, payload, optimizer


@pytest.mark.parametrize("accepted", [True, False])
def test_log_and_endpoint_independently_reproduce_acceptance_or_rollback(accepted):
    record, payload, optimizer = capture(accepted)
    check_record(record, [1.], {"rmse": 1.}, 2)
    result = check_endpoint(payload, record, [2], .02, .003, optimizer.state_dict())
    assert result["accepted"] == accepted and result["gradient_vectors"] == 2
    assert result["proposed_optimizer_count"] == 1


def test_changed_metric_decision_is_rejected():
    record, _, _ = capture(True)
    record["point_after"]["rmse"] = 1.1
    with pytest.raises(ValueError, match="acceptance"):
        check_record(record, [1.], {"rmse": 1.}, 2)


def test_changed_raw_gradient_or_rollback_rate_is_rejected():
    record, payload, optimizer = capture(False)
    payload["gradients"] = payload["gradients"].clone()
    payload["gradients"][0, 0] += .1
    with pytest.raises(ValueError, match="full FP32"):
        check_endpoint(payload, record, [2], .02, .003, optimizer.state_dict())
    record, payload, optimizer = capture(False)
    with pytest.raises(ValueError, match="learning rate"):
        check_endpoint(payload, record, [2], .02, .004, optimizer.state_dict())


def test_endpoint_uses_accepted_optimizer_count_not_attempted_batch_index():
    model, optimizer, gradients = fixture()
    for attempt in range(2):
        before = float(model.weight[0, 0].detach())
        protected_update(model, optimizer, gradients, [1.], .01, {"rmse": before},
                         lambda current: {"rmse": float(current.weight[0, 0].detach())})
    before = float(model.weight[0, 0].detach())
    record, snapshot = protected_update(model, optimizer, gradients, [1.], .01, {"rmse": before},
                                       lambda current: {"rmse": before + .1})
    record.update(step=7, accepted_updates=2)
    payload = {"snapshot": snapshot, "gradients": gradients, "record": deepcopy(record)}
    result = check_endpoint(payload, record, [2], .02, .01, optimizer.state_dict())
    assert result["proposed_optimizer_count"] == 3 and result["accepted_updates"] == 2
