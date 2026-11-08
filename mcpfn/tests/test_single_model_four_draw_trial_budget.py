from collections import Counter
from copy import deepcopy
from math import prod

import pytest

import single_model_four_draw_trial_budget as budget
import train_single_matrixformer_balanced_native_quarter as template


@pytest.fixture(scope="module")
def schedule():
    return budget.batches(template)


def test_full_schedule_retains_all_original_specifications_not_probe_stress_shapes(schedule):
    original = budget.original_budget.batches(template, budget.DATA_SEED, budget.GRADIENT_SEED)
    for batch in schedule:
        for task in batch["tasks"]:
            member, index = divmod(task["index"], 28)
            source = original[batch["source_steps"][member] - 1]["tasks"][index]
            restored = deepcopy(task)
            restored.update(index=index, replicate=source["replicate"], name=source["name"], weight=2 * task["weight"])
            assert restored == source
    shapes = {tuple(task["specification"]["shape"]) for batch in schedule for task in batch["tasks"] if task["cohort"] == "legacy"}
    assert shapes == {(256, 128), (2000, 1000)}
    assert budget.SPECIFICATION["probe_native_shape_override_applied"] is False


def test_scheduled_tasks_targets_components_and_areas_are_complete(schedule):
    tasks = [task for batch in schedule for task in batch["tasks"]]
    counts = Counter(task["cohort"] for task in tasks)
    assert len(schedule) == budget.STEPS == 128 and len(tasks) == 7168
    assert counts == dict.fromkeys(budget.batching.COHORTS, 512)
    imputation = [task for task in tasks if task["specification"]["domain"] == "imputation"]
    assert len(imputation) == budget.SPECIFICATION["imputation_teacher_targets"] == 6656
    assert 2 * len(imputation) == budget.SPECIFICATION["imputation_teacher_components"] == 13312
    reference_counts = {"v8": sum(task["cohort"] in budget.batching.FAMILIES for task in tasks), "v29": len(imputation)}
    assert reference_counts == budget.SPECIFICATION["retention_component_roles"]
    assert sum(reference_counts.values()) == budget.SPECIFICATION["retention_components"] == 12800
    old_tasks = [task for batch in budget.original_budget.batches(template) for task in batch["tasks"]]
    area = lambda task: prod(task["specification"].get("structured", task["specification"])["shape"])
    assert sum(map(area, tasks)) == sum(map(area, old_tasks))


def test_eight_boundaries_match_task_budget_with_one_explicit_upward_rounding():
    assert budget.BOUNDARIES == tuple((step + 1) // 2 for step in budget.original_budget.BOUNDARIES)
    assert len(budget.BOUNDARIES) == len(budget.original_budget.BOUNDARIES) == 8
    differences = [paired - original for paired, original in zip(
        budget.SPECIFICATION["paired_source_boundaries"], budget.SPECIFICATION["reference_source_boundaries"])]
    assert differences == [0, 0, 1, 0, 0, 0, 0, 0]
    assert "ineligible step0" in budget.POLICY and "all32 comparisons" in budget.POLICY


def test_prospective_training_streams_are_distinct_from_prior_and_probe_metadata(schedule):
    def seeds(batches):
        return {seed for batch in batches for task in batch["tasks"] for seed in (
            task["specification"].get("structured", task["specification"])["seed"], task["gradient_seed"])}

    current = seeds(schedule)
    assert len(current) == 14336
    assert current.isdisjoint(seeds(budget.original_budget.batches(template)))
    probe_metadata = budget.original_budget.batches(template, 1501100901, 1502100901)
    assert current.isdisjoint(seeds(probe_metadata))
    assert all(seed not in current for seed in budget.CONFIRMATION_SEEDS)
