from copy import deepcopy

import pytest

import single_model_four_draw_trial_storage as storage
import single_model_balanced_native_long_trial as original_trial
import single_model_balanced_native_long_blend_replay as original_blend
import train_single_matrixformer_balanced_native_quarter as training


@pytest.fixture(scope="module")
def schedules():
    original = storage.budget.original_budget.batches(training, storage.budget.DATA_SEED, storage.budget.GRADIENT_SEED)
    return original, storage.budget.batching.pair_batches(original)


def test_all_requested_target_and_component_bytes_equal_original_task_budget(schedules):
    original, paired = schedules
    assert storage.reference_storage(paired) == original_trial.reference_storage(original)
    assert storage.component_storage(paired) == storage.predecessor.component_storage(original)
    assert storage.teacher_storage(paired) == original_blend.storage_accounting(original)
    assert storage.teacher_storage(paired)["raw_component_grids"] == 13312
    assert storage.component_storage(paired)["components"] == 12800


def test_capacity_covers_full_gradients_simultaneous_atomic_write_and_all_metadata(schedules):
    result = storage.accounting(schedules[1])
    vector = 4 * 77998496
    assert result["scratch"]["raw_gradient_bytes"] == 168 * vector == 52414989312
    assert result["scratch"]["atomic_raw_allowance"] == (168 + 12) * vector
    assert result["scratch"]["snapshot_vector_allowance"] == 12 * vector
    for volume, fields in result.items():
        assert sum(fields.values()) <= storage.WRITE_BUDGETS["training"][volume]
    assert storage.WRITE_BUDGETS == storage.predecessor.WRITE_BUDGETS
    assert storage.RESERVES == storage.predecessor.RESERVES
    assert storage.WRITE_BUDGETS is not storage.predecessor.WRITE_BUDGETS
    assert all(storage.WRITE_BUDGETS[stage] is not storage.predecessor.WRITE_BUDGETS[stage] for stage in storage.WRITE_BUDGETS)


@pytest.mark.parametrize("mutation", ["attempt", "task", "index", "source", "shape", "weight", "domain"])
def test_missing_reordered_or_invalid_requests_cannot_reduce_budget(schedules, mutation):
    paired = deepcopy(schedules[1])
    if mutation == "attempt":
        paired.pop()
    elif mutation == "task":
        paired[0]["tasks"].pop()
    elif mutation == "index":
        paired[0]["tasks"][0]["index"] = 1
    elif mutation == "source":
        paired[0]["source_steps"] = [1, 3]
    elif mutation == "shape":
        paired[0]["tasks"][0]["specification"]["shape"] = [True, 1000]
    elif mutation == "weight":
        paired[0]["tasks"][0]["weight"] *= 2
    else:
        paired[0]["tasks"][0]["specification"]["domain"] = "unknown"
    with pytest.raises(ValueError):
        storage.accounting(paired)
