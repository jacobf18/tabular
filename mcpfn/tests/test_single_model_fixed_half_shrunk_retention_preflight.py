from copy import deepcopy
import inspect

import pytest

import preflight_single_model_fixed_half_shrunk_retention as probe
import test_single_model_anchor_retention_preflight as original_tests


def panel_evidence():
    rows = [{"fraction": fraction, "locally_feasible": fraction in (.5, .25, .125),
        "finite": {"repetitions": {str(repeat): {"defined_slots": 30} for repeat in (0, 1)}}}
        for fraction in probe.panel.FRACTIONS]
    result = {"complete": True, "prediction_grids": 336, "checkpoint_selected": False, "optimizer_updates": 0,
        "new_data": False, "real_benchmark_success": False, "config_sha256": probe.PANEL_CONFIG_SHA256, "fractions": deepcopy(rows)}
    audit = {"complete": True, "prediction_grids": 336, "training_tasks": 28, "parameter_vectors": 6,
        "gradient_vectors_per_fraction": 84, "protected_directions": 30, "frozen_buffers_per_fraction": 16,
        "cuda_initialized": False, "research_selection_tolerance": 0, "checkpoint_selected": False,
        "real_benchmark_success": False, "fractions": deepcopy(rows)}
    status = {"state": "complete", "checkpoint_selected": False, "locally_feasible_fractions": [.5, .25, .125]}
    return result, audit, status


@pytest.mark.parametrize("entry,field,value", [(0, "complete", False), (0, "prediction_grids", 335),
    (0, "checkpoint_selected", True), (0, "optimizer_updates", 1), (0, "new_data", True),
    (1, "parameter_vectors", 5), (1, "gradient_vectors_per_fraction", 83), (1, "protected_directions", 29),
    (1, "frozen_buffers_per_fraction", 15), (1, "cuda_initialized", True), (1, "research_selection_tolerance", 1e-12),
    (2, "state", "running"), (2, "locally_feasible_fractions", [.25, .125])])
def test_requires_complete_audited_panel_without_promoting_a_diagnostic(entry, field, value):
    evidence = panel_evidence()
    probe.validate_panel(*evidence)
    evidence[entry][field] = value
    with pytest.raises(ValueError):
        probe.validate_panel(*evidence)


@pytest.mark.parametrize("mutation", ["omitted_fraction", "wrong_outcome", "omitted_repeat", "omitted_guard"])
def test_never_hides_negative_controls_or_missing_coverage(mutation):
    result, audit, status = panel_evidence()
    if mutation == "omitted_fraction":
        result["fractions"].pop()
    elif mutation == "wrong_outcome":
        result["fractions"][1]["locally_feasible"] = True
    elif mutation == "omitted_repeat":
        audit["fractions"][0]["finite"]["repetitions"].pop("1")
    else:
        audit["fractions"][0]["finite"]["repetitions"]["1"]["defined_slots"] = 29
    with pytest.raises(ValueError):
        probe.validate_panel(result, audit, status)


def test_preserves_prior_teacher_parent_and_every_guard_except_explicit_fixed_contraction():
    previous = probe.predecessor
    assert probe.paired is probe.half_protection and probe.paired is not previous.paired
    assert probe.retention is previous.retention and probe.teacher_policy is previous.teacher_policy
    assert probe.PARENT == previous.PARENT and probe.PARENT_SHA256 == previous.PARENT_SHA256
    assert probe.RATE == previous.RATE == 5e-6
    assert probe.mixture.MASSES == previous.mixture.MASSES
    assert probe.RESERVES == previous.RESERVES and probe.WRITE_BUDGETS == previous.WRITE_BUDGETS
    assert probe.STORAGE_ACCOUNTING["snapshot_vector_allowance_bytes"] == 10 * 4 * probe.TRAINABLE_PARAMETERS
    scratch_total = sum(value for key, value in probe.STORAGE_ACCOUNTING.items() if key != "checkpoint_artifact_allowance_bytes")
    assert scratch_total < probe.WRITE_BUDGETS["scratch"]
    assert previous.STORAGE_ACCOUNTING["snapshot_vector_allowance_bytes"] == 8 * 4 * probe.TRAINABLE_PARAMETERS
    assert probe.OUTPUT.name == "fixed_half_shrunk_retention_preflight_v1"
    assert probe.DATA_SEED == 961100901 and probe.GRADIENT_SEED == 962100901
    current, original = probe.batch(), previous.batch()
    assert len(probe.training.alignment.fixed.seeds_in(current)) == 56
    assert not probe.training.alignment.fixed.seeds_in(current).intersection(probe.training.alignment.fixed.seeds_in(original))
    for current_task, original_task in zip(current["tasks"], original["tasks"]):
        restored = deepcopy(current_task)
        restored["gradient_seed"] = original_task["gradient_seed"]
        specification = restored["specification"].get("structured", restored["specification"])
        original_specification = original_task["specification"].get("structured", original_task["specification"])
        specification["seed"] = original_specification["seed"]
        assert restored == original_task


def test_generated_training_and_all_auditors_resolve_the_new_contraction_namespace(monkeypatch):
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_cloned_helpers_resolve_new_trial_globals_not_old_owners()
    original_tests.test_generated_run_reaches_real_retention_context_and_does_not_modify_inference()
    for name in ("run", "audit", "audit_original", "audit_admitted", "register_trial"):
        assert inspect.unwrap(getattr(probe, name)).__globals__["paired"] is probe.half_protection
    assert 'expected = payload["snapshot"]["contracted"]' in probe.ADAPTED_SOURCE
    assert 'if not torch.equal(candidate, expected)' in probe.ADAPTED_SOURCE
    assert '"displacement_fraction": paired.FRACTION' in probe.REGISTER_SOURCE
    assert 'config["displacement_fraction"] != paired.FRACTION' in probe.ADAPTED_SOURCE
    assert "require_displacement_panel" in probe.require_predecessor.__code__.co_names
