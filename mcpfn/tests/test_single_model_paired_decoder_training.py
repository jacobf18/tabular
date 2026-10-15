from copy import deepcopy
import math

import pytest

import audit_single_model_paired_decoder as audit
import train_single_matrixformer_paired_decoder as training
import train_single_matrixformer_replicated_decoder as previous


def test_trial_keeps_every_recipe_and_changes_only_seeds_and_training_acceptance():
    assert training.PARENT == previous.PARENT == training.original.alignment.fixed.PARENT
    assert training.STEPS == 64 and training.BOUNDARIES == previous.BOUNDARIES
    assert training.QUALIFICATION == previous.QUALIFICATION
    for batch, old in zip(training.batches(), previous.batches()):
        assert len(batch["tasks"]) == 28
        assert math.fsum(task["weight"] for task in batch["tasks"]) == 1.
        for task, reference in zip(batch["tasks"], old["tasks"]):
            copied = deepcopy(task)
            copied["gradient_seed"] = reference["gradient_seed"]
            copied["specification"].get("structured", copied["specification"])["seed"] = reference["specification"].get("structured", reference["specification"])["seed"]
            assert copied == reference
    seeds = training.original.alignment.fixed.seeds_in
    fresh = seeds([training.batches(), training.confirmation_cases()])
    assert len(fresh) == 3868
    assert fresh.isdisjoint(seeds([previous.batches(), previous.confirmation_cases(), training.preflight.batch()]))
    assert len(training.confirmation_cases()) == 284
    assert {training.constant_rate(step) for step in range(1, 65)} == {2e-5}


def test_both_live_training_and_cpu_audit_bind_the_complete_paired_layout():
    assert "paired.protected_update(" in training.ADAPTED_SOURCE
    assert 'paired.validate_layout(entries, batch["tasks"], before)' in training.ADAPTED_SOURCE
    assert 'config["guarded_policy"] != paired.POLICY' in training.ADAPTED_SOURCE
    assert '"guarded_policy": paired.POLICY' in training.ADAPTED_SOURCE
    assert "original.guarded.protected_update(" not in training.ADAPTED_SOURCE
    assert audit.checks is training.paired
    assert 'checks.validate_layout(entries, batch["tasks"], point_before)' in audit.ADAPTED_SOURCE
    assert 'len(manifest["tasks"]) != 28' in audit.ADAPTED_SOURCE
    assert '"training_episodes": 28 * through_step' in audit.ADAPTED_SOURCE
    assert '152 * (3 + len(boundaries))' in audit.ADAPTED_SOURCE
    assert 'len(decision["comparisons"]) != 22' in audit.ADAPTED_SOURCE
    assert 'decision["research_selection_tolerance"] != 0' in audit.ADAPTED_SOURCE


def test_no_changed_training_acceptance_can_relax_selection_or_earlier_ties():
    baseline = {"step": 0, "score": 1.146}
    assert training.selected_boundary(baseline, 0, 0, {"passed": True, "score": 1.146}) == baseline
    selected = training.selected_boundary(baseline, 10, 9, {"passed": True, "score": 1.1})
    assert selected == {"step": 10, "score": 1.1}
    assert training.selected_boundary(selected, 25, 24, {"passed": True, "score": 1.1}) == selected
    assert training.selected_boundary(selected, 50, 49, {"passed": False, "score": 1.}) == selected


def test_the_original_v8_grid_and_whole_training_storage_reserve_are_mandatory(monkeypatch):
    baseline = {"student": {"score": 1.146}, "cases": [
        {"index": index, "student": {"rmse": 1.}, "prediction_sha256": str(index)} for index in range(152)]}
    audit.check_initialization(baseline, deepcopy(baseline))
    changed = deepcopy(baseline)
    changed["cases"][0]["prediction_sha256"] = "changed"
    with pytest.raises(ValueError, match="original V8 prediction"):
        audit.check_initialization(changed, baseline)
    assert 'shutil.disk_usage(ROOT).free < 31 * 1024**3' in training.ADAPTED_SOURCE
