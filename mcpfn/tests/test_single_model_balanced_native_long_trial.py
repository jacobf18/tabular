from copy import deepcopy
import hashlib
import types

import pytest

import train_single_matrixformer_balanced_native_long as training
import train_single_matrixformer_balanced_native_quarter as previous
import single_model_balanced_native_long_budget as budget
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_all_256_batches_and_confirmation_seeds_are_fresh_and_complete():
    batches = training.batches()
    assert batches == budget.batches(previous)
    assert training.STEPS == 256 and training.BOUNDARIES == budget.BOUNDARIES
    assert training.OUTPUT.name == "balanced_native_long_v56"
    assert training.DATA_SEED == budget.DATA_SEED and training.GRADIENT_SEED == budget.GRADIENT_SEED
    assert training.CONFIRMATION_SEEDS == budget.CONFIRMATION_SEEDS
    assert len(training.confirmation_cases()) == 284
    seeds = training.original.alignment.fixed.seeds_in
    fresh = seeds([batches, training.confirmation_cases()])
    assert len(fresh) == 14620
    assert not fresh.intersection(seeds([previous.batches(), previous.confirmation_cases(), training.preflight.batch()]))
    original_cases = previous.confirmation_cases()
    for current, original in zip(training.confirmation_cases(), original_cases):
        copied = deepcopy(current)
        copied.get("structured", copied)["seed"] = original.get("structured", original)["seed"]
        assert copied == original


def test_initializer_model_losses_and_optimizer_are_unchanged():
    for name in ("PARENT", "PARENT_SHA256", "RATE", "PARENT_UPDATES", "TRAINABLE_PARAMETERS",
            "TRAINABLE_TENSORS", "FROZEN_TENSORS", "paired", "teacher_policy", "stronger", "retention", "preflight"):
        assert getattr(training, name) == getattr(previous, name)
    assert training.RATE == 5e-6 and training.paired.FRACTION == .25
    assert training.retention_training.cohort_retention is previous.retention
    assert training.retention_training.expected_component_counts(256) == {"v8": 6144, "v29": 6656}
    assert training.all_trial.require_probe is previous.all_trial.require_probe
    assert training.capacity.probe is training.preflight
    assert training.diagnostics is training.all_trial.component_storage
    assert 'duration_extension' in training.REGISTER_SOURCE and 'duration_extension' in training.ADAPTED_SOURCE


def test_complete_storage_accounts_for_every_extended_task():
    for volume, amounts in training.STORAGE_ACCOUNTING.items():
        assert sum(amounts.values()) <= training.diagnostics.WRITE_BUDGETS["training"][volume]
    assert training.all_trial.reference_storage(training.batches())["tasks"] == 6656
    assert training.diagnostics.component_storage(training.batches())["components"] == 12800
    assert training.blend_replay.storage_accounting(training.batches())["raw_component_grids"] == 13312
    assert "previous_training.batches()" not in training.SOURCE


def test_original_selection_and_failure_handling_remain_registered():
    assert "all22" in training.QUALIFICATION and "all32" in training.POLICY
    assert "1824" in training.QUALIFICATION and "128/192/256" in training.QUALIFICATION
    assert "ineligible step0" in training.QUALIFICATION and "earlier-boundary ties" in training.QUALIFICATION
    assert "480-comparison" in training.POLICY and "392 required/18 primary" in training.POLICY
    assert training.ADAPTED_SOURCE_SHA256 == hashlib.sha256(training.ADAPTED_SOURCE.encode()).hexdigest()
    for suffix in ("budget", "trial", "full_audit", "pipeline", "capacity"):
        assert f"test_single_model_balanced_native_long_{suffix}.py" in training.REGISTER_SOURCE


@pytest.mark.parametrize("receipt", [None, "", "wrong", 1234])
def test_unregistered_capacity_evidence_cannot_start_training(receipt):
    with pytest.raises(ValueError, match="completed independent"):
        training.all_trial.storage_evidence(None, receipt)


def test_capacity_receipt_is_bound_to_independent_validation():
    seen = []
    capacity = types.SimpleNamespace(require_completed=lambda receipt: seen.append(receipt) or {"evidence": receipt})
    assert training.all_trial.storage_evidence(capacity, "a" * 64) == {"evidence": "a" * 64}
    assert seen == ["a" * 64]


def test_every_local_lifecycle_callback_resolves_globals():
    for value in vars(training).values():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(training):
            assert not unresolved_globals(value), value.__name__
    for name in ("replicated_batches", "batches", "train", "register", "verify_registration"):
        assert getattr(training, name).__globals__ is vars(training)
