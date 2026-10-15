from copy import deepcopy
import math

import pytest
import torch

import audit_single_model_replicated_decoder as audit
import train_single_matrixformer_replicated_decoder as training
import train_single_matrixformer_v8_decoder as previous


def test_v18_retains_v8_initialization_and_doubles_every_fresh_cohort_draw():
    assert training.PARENT == previous.PARENT == training.original.alignment.fixed.PARENT
    assert training.OUTPUT.name == "replicated_decoder_v18"
    assert training.BOUNDARIES == previous.BOUNDARIES == (0, 10, 25, 50, 64)
    current, original = training.batches(), previous.batches()
    assert len(current) == 64
    for batch, reference in zip(current, original):
        assert len(batch["tasks"]) == 28
        assert math.fsum(task["weight"] for task in batch["tasks"]) == 1.
        for task in batch["tasks"]:
            source = reference["tasks"][task["index"] % 14]
            specification = deepcopy(task["specification"])
            specification.get("structured", specification)["seed"] = source["specification"].get("structured", source["specification"])["seed"]
            assert specification == source["specification"]
            assert task["weight"] == source["weight"] / 2
            assert task["cohort"] == source["name"]
    seeds = training.original.alignment.fixed.seeds_in
    fresh = seeds([current, training.confirmation_cases()])
    assert len(fresh) == 3868
    assert fresh.isdisjoint(seeds([original, previous.confirmation_cases(), training.preflight.batch()]))
    assert len(training.confirmation_cases()) == 284
    assert {training.constant_rate(step) for step in range(1, 65)} == {2e-5}


def test_every_replicated_gradient_task_and_grid_is_required():
    assert 'replicated.capture_batch(' in training.ADAPTED_SOURCE
    assert 'gradients = torch.empty((84,' in training.ADAPTED_SOURCE
    assert 'audit["training_tasks"] != 28' in training.ADAPTED_SOURCE
    assert 'audit["prediction_grids"] != 56' in training.ADAPTED_SOURCE
    assert 'audit["raw_endpoint"]["gradient_vectors"] != 84' in training.ADAPTED_SOURCE
    assert 'len(manifest["tasks"]) != 28' in audit.ADAPTED_SOURCE
    assert '"training_episodes": 28 * through_step' in audit.ADAPTED_SOURCE
    assert '152 * (3 + len(boundaries))' in audit.ADAPTED_SOURCE
    assert '(parent_report, ("student",))' not in audit.ADAPTED_SOURCE
    assert 'status(output, "initializing_training")\n    wait_for_gpu' in training.ADAPTED_SOURCE


def test_original_gate_initialization_and_earlier_ties_do_not_change():
    baseline = {"step": 0, "score": 1.146}
    assert training.selected_boundary(baseline, 0, 0, {"passed": True, "score": 1.146}) == baseline
    selected = training.selected_boundary(baseline, 10, 9, {"passed": True, "score": 1.1})
    assert selected == {"step": 10, "score": 1.1}
    assert training.selected_boundary(selected, 25, 24, {"passed": True, "score": 1.1}) == selected
    assert training.selected_boundary(selected, 50, 49, {"passed": False, "score": 1.0}) == selected


def test_auditor_rejects_initialization_and_frozen_backbone_drift():
    baseline = {"student": {"score": 1.146}, "cases": [
        {"index": index, "student": {"rmse": 1.}, "prediction_sha256": str(index)} for index in range(152)]}
    audit.check_initialization(baseline, deepcopy(baseline))
    changed = deepcopy(baseline)
    changed["cases"][0]["prediction_sha256"] = "wrong"
    with pytest.raises(ValueError, match="original V8 prediction"):
        audit.check_initialization(changed, baseline)
    parent = {"backbone": torch.ones(2), "decoder.weight": torch.ones(3)}
    current = deepcopy(parent)
    current["backbone"] += .01
    with pytest.raises(ValueError, match="Frozen model state"):
        audit.decoder.frozen_state_equal(parent, current, {"decoder.weight"})
