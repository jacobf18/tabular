from collections import Counter, defaultdict
from copy import deepcopy
import math

import numpy as np
import pytest

import single_model_balanced_native_coverage as native
import single_model_fixed_half_protection as protection
from test_single_model_balanced_structured_components import original_batch, schedule


@pytest.mark.parametrize("step", range(1, 65))
def test_only_uncaptured_native_components_and_orientations_change(step):
    original = original_batch(step)
    saved = deepcopy(original)
    previous = native.structured.balance(original)
    actual = native.balance(original)
    assert original == saved and actual is not original
    orientations = []
    for task, before in zip(actual["tasks"], previous["tasks"]):
        restored = deepcopy(task)
        if task["cohort"] == "legacy":
            assert task["specification"]["component"] in range(8)
            assert type(task["specification"]["transpose"]) is bool
            orientations.append(task["specification"]["transpose"])
            for field in ("component", "transpose"):
                restored["specification"][field] = before["specification"][field]
        assert restored == before
    assert sorted(orientations) == [False, True]
    assert schedule.original.alignment.fixed.seeds_in(actual) == schedule.original.alignment.fixed.seeds_in(original)
    with pytest.raises(ValueError):
        native.balance(actual)


@pytest.mark.parametrize("start", [1, 17, 33, 49])
def test_every_complete_period_covers_all32_joint_strata(start):
    counts, weights = Counter(), defaultdict(list)
    for step in range(start, start + 16):
        for task in native.balance(original_batch(step))["tasks"]:
            if task["cohort"] == "legacy":
                specification = task["specification"]
                identity = specification["component"], tuple(specification["shape"]), specification["transpose"]
                counts[identity] += 1
                weights[identity].append(task["weight"])
    expected = {(component, shape, transpose) for component in range(8)
        for shape in ((256, 128), (2000, 1000)) for transpose in (False, True)}
    assert set(counts) == expected and set(counts.values()) == {1}
    assert all(math.fsum(values) == .25 for values in weights.values())


def test_full_schedule_keeps_domain_masses_and_uniform_native_components():
    components, orientations, original_components = Counter(), Counter(), Counter()
    component_weights = defaultdict(list)
    for step in range(1, 65):
        original = original_batch(step)
        for task in original["tasks"]:
            if task["cohort"] == "legacy":
                original_components[task["specification"]["component"]] += 1
        batch = native.balance(original)
        weights = defaultdict(list)
        for task in batch["tasks"]:
            cohort = task["cohort"]
            weights[cohort if cohort in ("legacy", "recsys") else "structured"].append(task["weight"])
            if cohort == "legacy":
                component = task["specification"]["component"]
                components[component] += 1
                orientations[task["specification"]["transpose"]] += 1
                component_weights[component].append(task["weight"])
        assert {name: math.fsum(values) for name, values in weights.items()} == native.SPECIFICATION["supervised_domain_masses"]
    assert original_components == {component: 72 if component == 6 else 8 for component in range(8)}
    assert components == {component: 16 for component in range(8)}
    assert orientations == {False: 64, True: 64}
    assert all(math.fsum(values) / 64 == .0625 for values in component_weights.values())


@pytest.mark.parametrize("step", range(1, 65))
def test_requested_shapes_follow_eight_attempt_blocks_not_replica(step):
    expected = [256, 128] if ((step - 1) // 8) % 2 == 0 else [2000, 1000]
    native_tasks = [task for task in original_batch(step)["tasks"] if task["cohort"] == "legacy"]
    assert len(native_tasks) == 2
    assert all(task["specification"]["shape"] == expected for task in native_tasks)


@pytest.mark.parametrize("mutation", ["missing", "order", "step", "weight", "replicate", "captured",
    "native_component", "native_anchor", "native_orientation", "orientation_type", "shape", "dimension", "rating"])
def test_invalid_or_already_changed_batches_are_rejected_without_mutation(mutation):
    batch = original_batch(1)
    tasks = batch["tasks"]
    if mutation == "missing":
        tasks.pop()
    elif mutation == "order":
        tasks.reverse()
    elif mutation == "step":
        batch["step"] = True
    elif mutation == "weight":
        tasks[0]["weight"] = .1
    elif mutation == "replicate":
        tasks[0]["replicate"] = True
    elif mutation == "captured":
        tasks[0]["artifact"] = "captured.pt"
    elif mutation == "native_component":
        tasks[0]["specification"]["component"] = 8
    elif mutation == "native_anchor":
        tasks[14]["specification"]["component"] = 7
    elif mutation == "native_orientation":
        tasks[0]["specification"]["transpose"] = True
    elif mutation == "orientation_type":
        tasks[0]["specification"]["transpose"] = 0
    elif mutation == "shape":
        tasks[0]["specification"]["shape"] = [256]
    elif mutation == "dimension":
        tasks[0]["specification"]["shape"] = [True, 128]
    else:
        tasks[13]["specification"]["domain"] = "imputation"
    saved = deepcopy(batch)
    with pytest.raises(ValueError):
        native.balance(batch)
    assert batch == saved


def test_all84_gradients_and30_point_protections_remain_unchanged():
    previous = native.structured.balance(original_batch(1))["tasks"]
    current = native.balance(original_batch(1))["tasks"]
    before = {f'{task["name"]}/{metric}': 1. for task in current for metric in
        (("rmse", "mae") if task["cohort"] == "recsys" else ("rmse", "common_nrmse"))}
    labels, names, matrix = protection.gradient_layout(before, 28)
    assert len(labels) == 84 and len(names) == 30 and matrix.shape == (30, 84)
    current_entries = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    assert np.array_equal(protection.validate_layout(current_entries, current, before), matrix)
    assert np.array_equal(protection.validate_layout(current_entries, previous, before), matrix)
    after = dict(before)
    after["legacy_draw_0/rmse"], after["legacy_draw_1/rmse"] = 1.1, .8
    assert not protection.metric_decision(before, after)["all_retained"]
