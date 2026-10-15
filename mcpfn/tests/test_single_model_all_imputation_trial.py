from copy import deepcopy

import pytest

import single_model_all_imputation_trial as recipe
import train_single_matrixformer_all_imputation_retention as training


def test_full_reference_storage_counts_all_tasks_and_replay_not_native_subset():
    accounting = recipe.reference_storage(training.batches())
    assert accounting == {"tasks": 1664, "cells": 310341632, "archived_bytes": 2793074688,
        "replay_bytes": 1241366528, "atomic_reference_bytes": 18000000,
        "serialization_and_manifest_allowance": 67108864}
    assert recipe.reference_bytes(training.batches()) == 4119550080
    assert training.STORAGE_ACCOUNTING["scratch"]["all_imputation_reference_allowance"] == 4119550080
    for volume, values in training.STORAGE_ACCOUNTING.items():
        assert sum(values.values()) <= training.diagnostics.WRITE_BUDGETS["training"][volume]
    assert training.capacity.TRAINING_ADMISSION_BYTES == (
        training.diagnostics.RESERVES["checkpoint"] + training.diagnostics.WRITE_BUDGETS["training"]["checkpoint"])


@pytest.mark.parametrize("mutation", ["omit_batch", "omit_task", "negative", "noninteger"])
def test_storage_never_omits_tasks_or_assumes_unknown_shapes(mutation):
    batches = deepcopy(training.batches())
    if mutation == "omit_batch":
        batches.pop()
    elif mutation == "omit_task":
        batches[0]["tasks"].pop(0)
    else:
        batches[0]["tasks"][0]["specification"]["shape"][0] = -1 if mutation == "negative" else 1.5
    with pytest.raises(ValueError):
        recipe.reference_storage(batches)


def test_only_scope_budget_and_unused_seeds_change():
    previous = training.previous_training
    assert training.PARENT == previous.PARENT
    assert training.teacher_policy is previous.teacher_policy
    assert training.paired is previous.paired
    assert training.retention.SPECIFICATION["weight"] == previous.retention.SPECIFICATION["weight"]
    assert training.POLICY == recipe.POLICY
    assert training.capacity.probe is training.preflight
    assert training.retention_training.retention is training.retention is training.preflight.retention
    assert training.QUALIFICATION == previous.QUALIFICATION
    assert training.BOUNDARIES == previous.BOUNDARIES == (0, 10, 25, 50, 64)
    assert training.TRAINABLE_TENSORS == 238 and training.FROZEN_TENSORS == 16
    assert 'config["displacement_fraction"] != paired.FRACTION' in training.ADAPTED_SOURCE
    assert training.diagnostics is not previous.diagnostics
    assert previous.diagnostics.WRITE_BUDGETS["training"]["checkpoint"] == 2304 * 1024**2
    previous_seeds = training.original.alignment.fixed.seeds_in([previous.batches(), previous.confirmation_cases(), training.preflight.batch()])
    new_seeds = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    assert len(new_seeds) == 3868 and not new_seeds.intersection(previous_seeds)
