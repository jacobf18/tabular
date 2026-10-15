from copy import deepcopy
import inspect

import pytest

import preflight_single_model_all_imputation_column_mean as probe
import preflight_single_model_native_parent_retention as predecessor
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_fresh_batch_changes_only_seed_block_and_structured_mean_weighting():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1351100901, 1352100901)
    assert probe.OUTPUT.name == "all_imputation_column_mean_preflight_v1"
    expected = deepcopy(predecessor.batch())
    for task in expected["tasks"]:
        task["gradient_seed"] += 30000000
        specification = task["specification"]
        specification.get("structured", specification)["seed"] += 30000000
    assert probe.batch() == expected and len(expected["tasks"]) == 28
    assert (probe.TRAINABLE_PARAMETERS, probe.TRAINABLE_TENSORS, probe.FROZEN_TENSORS) == (77998496, 238, 16)
    assert probe.PARENT == predecessor.PARENT
    assert probe.RATE == predecessor.RATE == 5e-6
    for name in ("RESERVES", "WRITE_BUDGETS", "ARITHMETIC"):
        assert getattr(probe, name) == getattr(predecessor, name)
    assert probe.retention.SPECIFICATION == predecessor.retention.SPECIFICATION
    assert probe.teacher_policy is predecessor.teacher_policy and probe.stronger is predecessor.stronger


def test_new_objective_sources_and_all_original_components_are_bound():
    assert probe.retention is probe.cohort_retention
    assert probe.retention.MEAN_SPECIFICATION["scope"].startswith("every original native and structured")
    assert probe.reference_identities() == predecessor.reference_identities()
    for name in ("all_imputation_column_mean", "all_imputation_column_mean_preflight"):
        assert f'test_single_model_{name}.py' in probe.REGISTER_SOURCE
    assert "require_completed_v51" in probe.require_predecessor.__code__.co_names
    assert '"native_teacher_mean": retention.MEAN_SPECIFICATION' in probe.REGISTER_SOURCE
    for function in vars(probe).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(probe):
            assert not unresolved_globals(function), function.__name__


def closed_reports():
    gate = {"passed": False, "confirmation_evaluated": False, "selected_step": 0, "real_benchmark_success": False}
    return {
        "training": {"complete": True, "complete_training_audit": True, "through_step": 64,
            "accepted_updates": 58, "best": {"step": 0}, "independently_regenerated_training_tasks": 1792,
            "independently_regenerated_rating_contexts": 128, "independently_regenerated_structured_draws": 1536,
            "admission_attempts": {"data_rejected_steps": []}, "autograd_replayed": False,
            "full_optimizer_trajectory_replayed": False, "confirmation_evaluated": False},
        "endpoint": {"complete": True, "endpoint_grids": 56, "selected_grids": 0, "cuda_initialized": False},
        "pipeline": {"complete": True, "gate": gate, "campaign": None, "review": None}, "gate": gate,
        "external": {"complete": True, "outcome": "negative_no_eligible_checkpoint", "training_attempts": 64,
            "accepted_updates": 58, "rejected_attempts": [32, 35, 42, 44, 55, 59], "original_workers_terminal": True,
            "goal_achieved": False, "confirmation_evaluated": False, "candidate_real_benchmarks_run": False,
            "model_inference_performed_by_closure": False, "gate_changed": False},
        "retention": {"complete": True, "admitted_batches": 64, "reference_grids": 1664,
            "component_grids": 3200, "component_role_counts": {"v8": 1536, "v29": 1664}},
        "blend": {"complete": True, "training_tasks_checked": 1792, "blend_tasks": 1664,
            "raw_component_grids": 3328, "query_truth_used_for_routing": False},
    }


@pytest.mark.parametrize("change", [None, ("training", "accepted_updates", 56),
    ("training", "independently_regenerated_training_tasks", 1764), ("training", "autograd_replayed", True),
    ("training", "confirmation_evaluated", True), ("endpoint", "endpoint_grids", 55),
    ("endpoint", "selected_grids", 152), ("endpoint", "cuda_initialized", True),
    ("pipeline", "complete", False), ("pipeline", "campaign", "/real"), ("pipeline", "review", {}),
    ("gate", "passed", True), ("gate", "confirmation_evaluated", True), ("gate", "selected_step", 64),
    ("external", "goal_achieved", True), ("external", "outcome", "qualified"),
    ("external", "rejected_attempts", []), ("external", "original_workers_terminal", False),
    ("external", "gate_changed", True), ("retention", "component_grids", 1664),
    ("retention", "component_role_counts", {"general": 128, "v8": 1536, "v29": 1536}),
    ("blend", "raw_component_grids", 3327), ("blend", "query_truth_used_for_routing", True)])
def test_predecessor_requires_complete_exact_negative_closure(change):
    reports = closed_reports()
    if change is None:
        probe.validate_closed_v51(reports)
    else:
        report, key, value = change
        reports[report][key] = value
        with pytest.raises(ValueError):
            probe.validate_closed_v51(reports)


def test_registration_is_not_a_full_training_trial():
    assert "No retry" in probe.POLICY_TEXT
    assert "all26 imputation tasks" in probe.POLICY_TEXT
    assert probe.retention.BACKWARD_SOURCE == predecessor.retention.BACKWARD_SOURCE
    assert probe.RETENTION_REPLAY_SOURCE == predecessor.RETENTION_REPLAY_SOURCE
