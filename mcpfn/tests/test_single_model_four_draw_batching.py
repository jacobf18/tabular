from collections import Counter
from copy import deepcopy

import pytest

import single_model_balanced_native_long_budget as budget
import single_model_four_draw_batching as batching
import train_single_matrixformer_balanced_native_quarter as previous


@pytest.fixture(scope="module")
def source_batches():
    return budget.batches(previous)


def test_complete_schedule_preserves_every_task_and_does_not_mutate_inputs(source_batches):
    saved = deepcopy(source_batches)
    paired = batching.pair_batches(source_batches)
    assert source_batches == saved
    assert batching.validate_schedule(paired) == {"attempts": 128, "tasks": 7168,
        "native_tasks": 512, "structured_tasks": 6144, "rating_tasks": 512, "seed_identities": 14336}
    for batch in paired:
        for index, task in enumerate(batch["tasks"]):
            member, original_index = divmod(index, 28)
            original = saved[batch["source_steps"][member] - 1]["tasks"][original_index]
            restored = deepcopy(task)
            restored.update(index=original_index, replicate=original["replicate"],
                name=original["name"], weight=2 * task["weight"])
            assert restored == original
    paired[0]["tasks"][0]["specification"]["shape"][0] += 1
    assert source_batches == saved


def test_each_update_has_all_components_and_preserves_mixture_masses(source_batches):
    paired = batching.pair_batches(source_batches)
    strata = Counter()
    for batch in paired:
        tasks = batch["tasks"]
        assert len({task["name"] for task in tasks}) == 56
        assert Counter(task["cohort"] for task in tasks) == dict.fromkeys(batching.COHORTS, 4)
        assert sum(task["weight"] for task in tasks if task["cohort"] == "legacy") == .5
        assert sum(task["weight"] for task in tasks if task["cohort"] == "recsys") == .25
        assert sum(task["weight"] for task in tasks if task["cohort"] in batching.FAMILIES) == pytest.approx(.25)
        native = [task["specification"] for task in tasks if task["cohort"] == "legacy"]
        assert len({specification["component"] for specification in native}) == 4
        assert Counter(specification["transpose"] for specification in native) == {False: 2, True: 2}
        for family in batching.FAMILIES:
            assert Counter(task["specification"]["structured"]["component"]
                for task in tasks if task["cohort"] == family) == dict.fromkeys(batching.COMPONENTS, 1)
        for task in tasks:
            if task["cohort"] == "legacy":
                specification = task["specification"]
                strata[specification["component"], tuple(specification["shape"]), specification["transpose"]] += 1
    assert len(strata) == 32 and set(strata.values()) == {16}
    assert batching.SPECIFICATION["potential_gradient_slots_per_attempt"] == 168
    assert batching.SPECIFICATION["grouped_metric_slots_per_attempt"] == 34


@pytest.mark.parametrize("mutation", ["missing", "reordered", "step_bool", "task_missing", "captured",
    "index", "replicate", "cohort", "name", "weight", "native_component", "transpose",
    "structured_component", "domain", "shape", "seed_bool", "seed_collision", "strata"])
def test_rejects_incomplete_altered_or_captured_sources(source_batches, mutation):
    batches = deepcopy(source_batches)
    task = batches[0]["tasks"][0]
    if mutation == "missing":
        batches.pop()
    elif mutation == "reordered":
        batches[0], batches[1] = batches[1], batches[0]
    elif mutation == "step_bool":
        batches[0]["step"] = True
    elif mutation == "task_missing":
        batches[0]["tasks"].pop()
    elif mutation == "captured":
        task["artifact"] = "already_captured.pt"
    elif mutation == "index":
        task["index"] = 1
    elif mutation == "replicate":
        task["replicate"] = True
    elif mutation == "cohort":
        task["cohort"] = "recsys"
    elif mutation == "name":
        task["name"] = "legacy_draw_1"
    elif mutation == "weight":
        task["weight"] = float("nan")
    elif mutation == "native_component":
        task["specification"]["component"] = 1
    elif mutation == "transpose":
        task["specification"]["transpose"] = 0
    elif mutation == "structured_component":
        batches[0]["tasks"][1]["specification"]["structured"]["component"] = "steep_tail_mcar"
    elif mutation == "domain":
        task["specification"]["domain"] = "recsys"
    elif mutation == "shape":
        task["specification"]["shape"] = [0, 128]
    elif mutation == "seed_bool":
        task["gradient_seed"] = True
    elif mutation == "seed_collision":
        task["gradient_seed"] = task["specification"]["seed"]
    elif mutation == "strata":
        task["specification"]["shape"] = [257, 128]
    with pytest.raises(ValueError):
        batching.pair_batches(batches)


@pytest.mark.parametrize("mutation", ["missing", "pair", "pair_bool", "index", "name", "replicate", "weight", "component"])
def test_derived_schedule_validation_rejects_tampering(source_batches, mutation):
    batches = batching.pair_batches(source_batches)
    task = batches[0]["tasks"][28]
    if mutation == "missing":
        batches.pop()
    elif mutation == "pair":
        batches[0]["source_steps"] = [2, 1]
    elif mutation == "pair_bool":
        batches[0]["source_steps"] = [True, 2]
    elif mutation == "index":
        task["index"] = 0
    elif mutation == "name":
        task["name"] = "legacy_draw_0"
    elif mutation == "replicate":
        task["replicate"] = 0
    elif mutation == "weight":
        task["weight"] *= 2
    elif mutation == "component":
        batches[0]["tasks"][29]["specification"]["structured"]["component"] = "low_rank_mcar"
    with pytest.raises(ValueError):
        batching.validate_schedule(batches)
