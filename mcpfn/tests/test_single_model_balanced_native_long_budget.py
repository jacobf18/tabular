from collections import Counter
from copy import deepcopy
import inspect

import pytest

import single_model_all_imputation_trial as references
import single_model_balanced_native_long_budget as budget
import single_model_balanced_native_long_retention as retention
import single_model_balanced_native_long_blend_replay as replay
import single_model_balanced_native_long_storage as storage
import train_single_matrixformer_balanced_native_quarter as previous
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def batches():
    return budget.batches(previous)


def test_fixed_budget_and_complete_unchanged_template_cycles():
    original = previous.baseline.previous.batches()
    saved = deepcopy(original)
    extended = budget.extend_base_batches(original)
    assert budget.STEPS == 256 and budget.BOUNDARIES == (0, 10, 25, 50, 64, 128, 192, 256)
    assert len(extended) == 256 and [batch["step"] for batch in extended] == list(range(1, 257))
    for index, batch in enumerate(extended):
        copied = deepcopy(batch)
        copied["step"] = saved[index % 64]["step"]
        assert copied == saved[index % 64]
    extended[-1]["tasks"][0]["weight"] = -1
    assert original == saved


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "task"])
def test_partial_or_reordered_templates_are_rejected(mutation):
    original = previous.baseline.previous.batches()
    if mutation == "missing":
        original.pop()
    elif mutation == "duplicate":
        original[-1] = deepcopy(original[-2])
    elif mutation == "reordered":
        original[0], original[1] = original[1], original[0]
    else:
        original[0]["tasks"].pop()
    with pytest.raises(ValueError):
        budget.extend_base_batches(original)


def test_all_tasks_seeds_weights_and_native_strata_remain_complete():
    current = batches()
    assert len(current) == 256 and sum(len(batch["tasks"]) for batch in current) == 7168
    seeds = previous.original.alignment.fixed.seeds_in
    assert len(seeds(current)) == 14336
    assert not seeds(current).intersection(seeds([previous.batches(), previous.confirmation_cases()]))
    strata = Counter()
    for batch in current:
        previous.retention.validate_task_mixture(batch["tasks"])
        for task in batch["tasks"]:
            if task["cohort"] == "legacy":
                specification = task["specification"]
                strata[specification["component"], tuple(specification["shape"]), specification["transpose"]] += 1
        assert Counter(task["specification"]["structured"]["component"] for task in batch["tasks"]
            if "structured" in task["specification"]) == dict.fromkeys(previous.components.COMPONENTS, 6)
    assert len(strata) == 32 and set(strata.values()) == {16}


def test_every_original_specification_is_preserved_apart_from_fresh_streams():
    original = previous.batches()
    for index, current in enumerate(batches()):
        expected = original[index % 64]
        copied = deepcopy(current)
        copied["step"] = expected["step"]
        for task, prior in zip(copied["tasks"], expected["tasks"]):
            task["gradient_seed"] = prior["gradient_seed"]
            task["specification"].get("structured", task["specification"])["seed"] = (
                prior["specification"].get("structured", prior["specification"])["seed"])
        assert copied == expected


def test_complete_requested_area_storage_and_original_plans_are_preserved():
    current = batches()
    components = storage.component_storage(current)
    assert components["tasks"] == 6656 and components["components"] == 12800
    assert storage.RESERVES == previous.diagnostics.RESERVES
    assert previous.diagnostics.WRITE_BUDGETS["training"]["scratch"] == 48 * 1024**3
    assert storage.WRITE_BUDGETS["training"] == {"checkpoint": 4096 * 1024**2,
        "root": 64 * 1024**2, "scratch": 192 * 1024**3}
    for stage in storage.WRITE_BUDGETS:
        if stage != "training":
            assert storage.WRITE_BUDGETS[stage] == previous.diagnostics.WRITE_BUDGETS[stage]
    namespace = {"prod": __import__("math").prod}
    source = budget.adapt_reference_storage(inspect.getsource(references.reference_storage))
    exec(compile(source, "<long_reference_storage_test>", "exec"), namespace)
    reference = namespace["reference_storage"](current)
    assert reference["tasks"] == 6656
    assert reference["cells"] == 4 * references.reference_storage(previous.batches())["cells"]
    blend = replay.storage_accounting(current)
    assert blend["blend_tasks"] == 6656 and blend["raw_component_grids"] == 13312
    with pytest.raises(ValueError):
        storage.component_storage(current[:-1])
    with pytest.raises(ValueError):
        replay.storage_accounting(current[:-1])


def test_extended_step_paths_do_not_mutate_original_retention_scope(tmp_path):
    assert retention.directory(tmp_path, 256).name == "step_256"
    assert previous.retention_training.directory(tmp_path, 64).name == "step_064"
    with pytest.raises(ValueError):
        previous.retention_training.directory(tmp_path, 65)
    for step in (0, 257, True, 1.5):
        with pytest.raises(ValueError):
            retention.directory(tmp_path, step)
    assert retention.cohort_retention is previous.retention
    assert retention.reference_identities.__code__ is previous.retention_training.reference_identities.__code__
    for name in ("INITIALIZER", "INITIALIZER_SHA256", "EVALUATION_BASELINE",
            "EVALUATION_BASELINE_SHA256", "GENERAL"):
        assert getattr(retention, name) == getattr(previous.retention_training, name)


def test_extended_callbacks_resolve_all_globals():
    for module in (retention, replay, storage):
        for value in vars(module).values():
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(module):
                assert unresolved_globals(value) == []
