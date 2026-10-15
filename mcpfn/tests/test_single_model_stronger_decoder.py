from copy import deepcopy
from itertools import product
import math

import pytest

import audit_single_model_stronger_decoder as audit
from single_model_registered_adaptation import replace_assignment, replace_once
import train_single_matrixformer_stronger_decoder as training
import train_single_matrixformer_targeted_decoder as original


def test_training_changes_only_declared_rate_horizon_and_fresh_seeds():
    assert training.STEPS == 64 and training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert original.STEPS == 100 and original.BOUNDARIES[-1] == 100
    assert training.PARENT == original.PARENT and training.INITIAL_RUN == original.INITIAL_RUN
    assert training.original is original.original
    assert training.original.alignment.fixed.PARENT.name == "best_validation_checkpoint.pth"
    assert training.original.alignment.fixed.PARENT.parent.name == "query_budget_v8"
    assert training.OUTPUT != original.OUTPUT
    assert training.POLICY != original.POLICY
    assert training.COMPONENT_CYCLES == original.COMPONENT_CYCLES
    current, prior = training.batches(), original.batches()[:64]
    for actual, expected in zip(current, prior):
        restored = deepcopy(actual)
        for task, reference in zip(restored["tasks"], expected["tasks"]):
            task["gradient_seed"] = reference["gradient_seed"]
            task["specification"].get("structured", task["specification"])["seed"] = reference["specification"].get("structured", reference["specification"])["seed"]
        assert restored == expected
        assert math.fsum(task["weight"] for task in actual["tasks"]) == 1.


def test_every_emphasized_prior_component_keeps_complete_context_coverage():
    expected = set(product(((240, 80), (1024, 256)), ("legacy", "area_scaled"), (.2, .6)))
    observed = {family: {component: set() for component in set(cycle)} for family, cycle in training.COMPONENT_CYCLES.items()}
    for batch in training.batches():
        for task in batch["tasks"]:
            if task["name"] in observed:
                specification = task["specification"]["structured"]
                observed[task["name"]][specification["component"]].add(
                    (tuple(specification["shape"]), specification["regime"], specification["requested_missingness"]))
    assert all(coverage == expected for components in observed.values() for coverage in components.values())


def test_all_reserved_seeds_remain_fresh_and_confirmation_separate():
    seeds = training.original.alignment.fixed.seeds_in
    current = seeds([training.batches(), training.confirmation_cases()])
    assert len(current) == 2076
    assert current.isdisjoint(seeds([original.batches(), original.confirmation_cases()]))
    assert seeds(training.batches()).isdisjoint(seeds(training.confirmation_cases()))
    assert len(training.confirmation_cases()) == 284


def test_no_annealing_warmup_or_rate_search_hides_in_the_new_trial():
    assert {training.constant_rate(step) for step in range(1, 65)} == {2e-5}
    for value in (0, 65, True, 1.5):
        with pytest.raises(ValueError):
            training.constant_rate(value)
    assert 'rate = constant_rate(step)' in training.ADAPTED_SOURCE
    assert 'rate = learning_rate(step' not in training.ADAPTED_SOURCE
    assert 'row["rate"] != training.constant_rate(step)' in audit.ADAPTED_SOURCE
    assert audit.training is training


def test_registered_variant_rejects_undeclared_source_drift():
    with pytest.raises(ValueError, match="one exact"):
        replace_once("twice twice", "twice", "once")
    with pytest.raises(ValueError, match="one top-level"):
        replace_assignment("OTHER = 1\n", "POLICY", "new")
    assert replace_assignment('POLICY = ("old"\n          " policy")\nOTHER = 1\n', "POLICY", "new") == "POLICY = 'new'\nOTHER = 1\n"


def test_original_selection_rejects_step_zero_and_preserves_earlier_ties():
    best = {"step": 0, "score": 1.2}
    decision = {"passed": True, "score": 1.1}
    assert training.selected_boundary(best, 0, 0, decision) == best
    selected = training.selected_boundary(best, 10, 9, decision)
    assert selected == {"step": 10, "score": 1.1}
    assert training.selected_boundary(selected, 25, 24, decision) == selected
