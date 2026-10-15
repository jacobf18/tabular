from copy import deepcopy
import inspect

import pytest

import preflight_single_model_legacy_blend_teacher as probe
import test_single_model_all_imputation_retention_preflight as previous_tests
import test_single_model_anchor_retention_preflight as original_tests


def latest():
    records = previous_tests.latest()
    records[1].update(accepted_updates=58, independently_regenerated_training_tasks=1764,
        independently_regenerated_rating_contexts=126,
        admission_attempts={"data_rejected_steps": [40], "optimizer_attempts": 63})
    return records


@pytest.mark.parametrize("entry,path,value", [
    (0, ("complete",), False), (0, ("campaign",), "campaign"), (0, ("review",), "review"),
    (1, ("accepted_updates",), 59), (1, ("attempted_batches",), 63), (1, ("through_step",), 63),
    (1, ("fixed_half_training",), False), (1, ("displacement_fraction",), 1.),
    (1, ("prediction_grids",), 1216), (1, ("best", "step"), 10),
    (1, ("independently_regenerated_training_tasks",), 1792),
    (1, ("independently_regenerated_rating_contexts",), 128),
    (1, ("independently_regenerated_structured_draws",), 1535),
    (1, ("admission_attempts", "data_rejected_steps"), []),
    (1, ("admission_attempts", "optimizer_attempts"), 64),
    (1, ("raw_endpoint", "adam", "gradient_vectors"), 28),
    (2, ("endpoint_grids",), 55), (2, ("selected_grids",), 152),
    (3, ("endpoint_reproduced",), False), (4, ("confirmation_evaluated",), True),
    (4, ("passed",), True), (4, ("real_benchmark_success",), True)])
def test_requires_complete_negative_v40_including_the_original_data_rejection(entry, path, value):
    records = latest()
    probe.validate_latest(*records)
    target = records[entry]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        probe.validate_latest(*records)


def test_only_legacy_training_targets_change_and_all_original_guards_remain():
    previous = probe.previous_probe
    assert probe.latest is probe.completed and probe.latest.training.OUTPUT.name == "all_imputation_retention_v40"
    assert probe.PARENT == previous.PARENT == probe.latest.training.PARENT
    assert probe.PARENT_SHA256 == probe.retention.REFERENCE_SHA256
    assert probe.retention is previous.retention and probe.paired is previous.paired
    assert probe.RATE == previous.RATE == 5e-6
    assert probe.RESERVES == previous.RESERVES and probe.WRITE_BUDGETS == previous.WRITE_BUDGETS
    assert probe.STORAGE_ACCOUNTING == {**previous.STORAGE_ACCOUNTING,
        "checkpoint_artifact_allowance_bytes": previous.STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] + 4 * 1024**2}
    assert probe.STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] == 517514880
    assert probe.STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] < probe.WRITE_BUDGETS["checkpoint"]
    assert probe.DATA_SEED == 1021100901 and probe.GRADIENT_SEED == 1022100901
    assert probe.OUTPUT.name == "legacy_blend_teacher_preflight_v1"
    assert probe.teacher_policy is probe.new_teacher and previous.teacher_policy is not probe.new_teacher
    assert previous.teacher_policy.POLICY["legacy"] == "general"
    assert probe.teacher_policy.POLICY == {"legacy": "observed_blend", "structured": "general", "recsys": "specialist"}
    current, original = probe.batch(), previous.batch()
    seeds = probe.training.alignment.fixed.seeds_in(current)
    assert len(seeds) == 56 and not seeds.intersection(probe.training.alignment.fixed.seeds_in(original))
    roles = [probe.expected_role(task) for task in current["tasks"]]
    assert roles.count("general") == 24 and roles.count("specialist") == roles.count("observed_blend") == 2
    for current_task, original_task in zip(current["tasks"], original["tasks"]):
        restored = deepcopy(current_task)
        restored["gradient_seed"] = original_task["gradient_seed"]
        specification = restored["specification"].get("structured", restored["specification"])
        original_specification = original_task["specification"].get("structured", original_task["specification"])
        specification["seed"] = original_specification["seed"]
        assert restored == original_task
    assert probe.blend_replay.storage_accounting(current) == {"legacy_tasks": 2, "component_grids": 4,
        "array_bytes": 524288, "serialization_allowance_bytes": 1024**2, "reserved_checkpoint_bytes": 4194304}


def test_generated_functions_use_new_policy_without_mutating_ancestor_modules(monkeypatch):
    for name in ("run", "verify", "audit", "audit_original", "register_trial", "replay_admitted", "audit_teacher_replay"):
        function = inspect.unwrap(getattr(probe, name))
        assert function.__globals__ is vars(probe) and function.__closure__ is None
        assert function.__globals__["teacher_policy"] is probe.new_teacher
    assert probe.check_routing is probe.blend_replay.check_routing
    assert probe.expected_role is probe.blend_replay.expected_role
    assert probe.previous_probe.check_routing is not probe.check_routing
    assert "require_exact_predecessor" in probe.require_predecessor.__code__.co_names
    assert "require_displacement_panel" not in probe.require_predecessor.__code__.co_names
    assert 'expected = payload["snapshot"]["contracted"]' in probe.ADAPTED_SOURCE
    assert 'config["displacement_fraction"] != paired.FRACTION' in probe.ADAPTED_SOURCE
    for name in ("blend_replay_storage", "blend_replay_policy", "teacher_replay_source_sha256", "teacher_replay_audit_source_sha256"):
        assert name in probe.REGISTER_SOURCE and name in probe.ADAPTED_SOURCE
    for name in ("legacy_blend_teacher", "legacy_blend_replay", "legacy_blend_teacher_preflight"):
        assert f"tests/test_single_model_{name}.py" in probe.REGISTER_SOURCE
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_generated_run_reaches_real_retention_context_and_does_not_modify_inference()


def test_binds_exact_complete_v40_artifacts(monkeypatch):
    captured = []
    monkeypatch.setattr(probe.previous_probe, "verify_hashes", lambda expected: captured.append(expected))
    expected = probe.require_exact_predecessor()
    assert captured == [expected] and len(expected) == 6
    assert expected[str(probe.completed.OUTPUT / "result.json")] == (
        "13d9ef09ba2cf33d524493200b77dba6333d447a64de9587e8eab3d32292b7ed")


@pytest.mark.parametrize("mutation", [None, "omit", "hide_spectral", "duplicate", "source", "mask", "identity", "nonfinite", "shape", "task"])
def test_all26_retention_references_remain_mandatory(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(previous_tests, "probe", probe)
    previous_tests.test_all26_imputation_references_are_mandatory_and_hash_bound(tmp_path, mutation)


@pytest.mark.parametrize("mutation", [None, "same_process", "different_prediction", "missing_spectral_reference"])
def test_retention_audit_still_checks_all26_full_grids(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(previous_tests, "probe", probe)
    previous_tests.test_independent_audit_covers_every_spectral_grid_and_original_penalty(tmp_path, monkeypatch, mutation)


def test_rejected_admission_has_no_training_or_retention_work(tmp_path, monkeypatch):
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_rejected_admission_does_no_retention_or_model_work(tmp_path, monkeypatch)
