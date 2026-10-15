from copy import deepcopy
from itertools import product
import math

import pytest
import torch

import audit_single_model_v8_decoder as audit
import train_single_matrixformer_stronger_decoder as previous
import train_single_matrixformer_v8_decoder as training


def test_v8_is_both_actual_training_input_and_unchanged_retention_baseline():
    assert training.PARENT == training.original.alignment.fixed.PARENT
    assert training.PARENT == training.preflight.PARENT
    assert training.INITIAL_RUN == training.original.alignment.fixed.PRODUCER
    assert training.PARENT != previous.PARENT
    assert training.OUTPUT.name == "v8_decoder_v17"
    assert '"parent_training_updates": 0' in training.ADAPTED_SOURCE
    assert '"parent_selection_qualified": True' in training.ADAPTED_SOURCE
    assert 'read(INITIAL_RUN / "validation_0.json")["student"]' in training.ADAPTED_SOURCE
    assert training.original is previous.original


def test_both_priors_and_every_context_combination_are_retained():
    current, prior = training.batches(), previous.batches()
    assert len(current) == 64
    for actual, expected in zip(current, prior):
        restored = deepcopy(actual)
        for task, reference in zip(restored["tasks"], expected["tasks"]):
            task["gradient_seed"] = reference["gradient_seed"]
            task["specification"].get("structured", task["specification"])["seed"] = reference["specification"].get("structured", reference["specification"])["seed"]
        assert restored == expected
        assert math.fsum(task["weight"] for task in actual["tasks"]) == 1.
    expected = set(product(((240, 80), (1024, 256)), ("legacy", "area_scaled"), (.2, .6)))
    observed = {family: {component: set() for component in set(cycle)} for family, cycle in training.COMPONENT_CYCLES.items()}
    for batch in current:
        for task in batch["tasks"]:
            if task["name"] in observed:
                specification = task["specification"]["structured"]
                observed[task["name"]][specification["component"]].add(
                    (tuple(specification["shape"]), specification["regime"], specification["requested_missingness"]))
    assert all(coverage == expected for components in observed.values() for coverage in components.values())


def test_all_training_and_confirmation_seeds_are_fresh():
    seeds = training.original.alignment.fixed.seeds_in
    fresh = seeds([training.batches(), training.confirmation_cases()])
    assert len(fresh) == 2076
    assert fresh.isdisjoint(seeds([previous.batches(), previous.confirmation_cases(), training.preflight.batch()]))
    assert seeds(training.batches()).isdisjoint(seeds(training.confirmation_cases()))
    assert len(training.confirmation_cases()) == 284
    assert {training.constant_rate(step) for step in range(1, 65)} == {2e-5}
    for value in (0, 65, True, 1.5):
        with pytest.raises(ValueError):
            training.constant_rate(value)


def test_initialization_never_becomes_an_eligible_trained_selection():
    best = {"step": 0, "score": 1.146}
    assert training.selected_boundary(best, 0, 0, {"passed": True, "score": 1.146}) == best
    selected = training.selected_boundary(best, 10, 9, {"passed": True, "score": 1.1})
    assert selected == {"step": 10, "score": 1.1}
    assert training.selected_boundary(selected, 25, 24, {"passed": True, "score": 1.1}) == selected


def test_reference_audit_does_not_double_count_the_identical_v8_parent():
    assert audit.training is training
    assert '152 * (3 + len(boundaries))' in audit.ADAPTED_SOURCE
    assert '(parent_report, ("student",))' not in audit.ADAPTED_SOURCE
    assert 'parent_report_path = training.INITIAL_RUN / "validation_0.json"' in audit.ADAPTED_SOURCE
    assert 'row["rate"] != training.constant_rate(step)' in audit.ADAPTED_SOURCE
    parent = {"student": {"score": 1.146}, "cases": [
        {"index": index, "student": {"rmse": 1.}, "prediction_file": f"old/{index}", "prediction_sha256": str(index)} for index in range(152)]}
    current = deepcopy(parent)
    for row in current["cases"]:
        row["prediction_file"] = f'new/{row["index"]}'
    audit.check_initialization(current, parent)
    current["cases"][-1]["prediction_sha256"] = "wrong"
    with pytest.raises(ValueError, match="original V8 prediction"):
        audit.check_initialization(current, parent)


def test_checkpoint_lineage_counts_only_updates_since_v8():
    parent = {"encoder.weight": torch.ones(2), "decoder.weight": torch.ones(3), "buffer": torch.zeros(1)}
    model = deepcopy(parent)
    model["decoder.weight"] += .1
    selected = [["decoder.weight", [3], 3]]
    config = {"input_sha256": {str(training.PARENT): "v8"}}
    best = {"step": 10, "score": 1.}
    saved = {"model": model, "step": 10, "epoch": 10, "attempted_batches": 10, "gradient_updates_performed": 9,
             "parent_training_updates": 0, "config_sha256": "config", "bank_sha256": "bank", "preflight_only": False,
             "independent_confirmation_passed": False, "decoder_only_training": True,
             "parent_checkpoint_sha256": "v8", "retention_baseline_sha256": "v8", "best_step": 10, "best_score": 1.}
    assert audit.check_checkpoint(saved, parent, selected, 10, 9, best, config, "config", "bank")["backbone_and_buffers_bitwise_unchanged"]
    saved["parent_training_updates"] = 95
    with pytest.raises(ValueError, match="provenance"):
        audit.check_checkpoint(saved, parent, selected, 10, 9, best, config, "config", "bank")
