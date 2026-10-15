from copy import deepcopy
import inspect

import pytest

import preflight_single_model_anchor_small_step as preflight


def test_half_rate_preserves_parent_tasks_teachers_and_original_guard_with_disjoint_seeds():
    assert preflight.RATE == preflight.template.RATE / 2 == 5e-6
    assert preflight.PARENT == preflight.template.PARENT
    assert preflight.PARENT_SHA256 == preflight.template.PARENT_SHA256
    assert preflight.teacher_policy is preflight.template.teacher_policy
    assert preflight.PREFIXES == preflight.template.PREFIXES
    actual, previous = preflight.batch(), preflight.template.batch()
    for task, source in zip(actual["tasks"], previous["tasks"]):
        normalized = deepcopy(task)
        normalized["gradient_seed"] = source["gradient_seed"]
        normalized["specification"].get("structured", normalized["specification"])["seed"] = (
            source["specification"].get("structured", source["specification"])["seed"])
        assert normalized == source
    fresh = preflight.training.alignment.fixed.seeds_in(actual)
    assert len(fresh) == 56
    assert not fresh.intersection(preflight.training.alignment.fixed.seeds_in(previous))
    assert not fresh.intersection(preflight.training.alignment.fixed.seeds_in(preflight.preceding.training.confirmation_cases()))
    assert "paired.protected_update(" in preflight.ADAPTED_SOURCE
    assert '"selection_qualified": False' in preflight.ADAPTED_SOURCE


def rejected_probe():
    result = {"preflight_only": True, "record": {"accepted": False, "accepted_updates": 0,
              "rate": 1e-5, "reason": "finite_metrics_rejected"}}
    audit = {"complete": True, "training_tasks": 28, "prediction_grids": 56,
             "teacher_replay": {"complete": True, "full_grids": 28, "bitwise_equal": True},
             "raw_endpoint": {"accepted": False, "accepted_updates": 0, "gradient_vectors": 84},
             "autograd_replayed": False, "frozen": {"frozen_tensors": 192, "changed_trainable_tensors": []}}
    gate = {"passed": False, "preflight_only": True, "reason": "finite_metrics_rejected"}
    current = {"state": "complete", "accepted": False}
    return result, audit, gate, current


def test_original_rejection_is_preserved_not_reinterpreted_as_success():
    preflight.validate_failed_probe(*rejected_probe())


@pytest.mark.parametrize("mutation", ["accepted", "incomplete", "omitted_task", "omitted_grid", "teacher",
    "raw", "rollback", "gate", "state", "rate", "autograd"])
def test_unpreserved_or_incomplete_rejection_is_not_admitted(mutation):
    result, audit, gate, current = rejected_probe()
    if mutation == "accepted":
        result["record"]["accepted"] = True
    elif mutation == "incomplete":
        audit["complete"] = False
    elif mutation == "omitted_task":
        audit["training_tasks"] -= 1
    elif mutation == "omitted_grid":
        audit["prediction_grids"] -= 1
    elif mutation == "teacher":
        audit["teacher_replay"]["bitwise_equal"] = False
    elif mutation == "raw":
        audit["raw_endpoint"]["gradient_vectors"] -= 1
    elif mutation == "rollback":
        audit["frozen"]["changed_trainable_tensors"] = ["decoder.weight"]
    elif mutation == "gate":
        gate["passed"] = True
    elif mutation == "state":
        current["state"] = "running"
    elif mutation == "rate":
        result["record"]["rate"] = 5e-6
    else:
        audit["autograd_replayed"] = True
    with pytest.raises(ValueError):
        preflight.validate_failed_probe(result, audit, gate, current)


def test_audit_functions_use_new_config_namespace_without_mutating_registered_predecessor():
    for name in ("require_space", "frozen_inventory", "save_raw", "audit_raw", "check_routing", "replay_targets", "audit_teacher_replay"):
        assert inspect.unwrap(getattr(preflight, name)).__globals__ is preflight.__dict__
    assert "RATE = 1e-5" in preflight.template.ADAPTED_SOURCE
    assert "RATE = 5e-6" in preflight.ADAPTED_SOURCE
    assert "test_single_model_anchor_small_step.py" in preflight.REGISTER_SOURCE
