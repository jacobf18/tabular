import ast
from copy import deepcopy
import hashlib
import inspect
import types

import pytest

import train_single_matrixformer_four_draw as training
import single_model_four_draw_full_storage as storage
import single_model_four_draw_fixed_trial as fixed
import single_model_four_draw_full_trial as trial
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def function_ast(source, name):
    return ast.dump(next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name))


def test_full_schedule_initializer_and_fresh_metadata():
    batches = training.batches()
    assert training.STEPS == 128 and training.BOUNDARIES == (0, 5, 13, 25, 32, 64, 96, 128)
    assert len(batches) == 128 and all(len(batch["tasks"]) == 56 for batch in batches)
    assert batches == training.budget.batches(training.blueprint.previous_training)
    assert training.OUTPUT.name == training.SCRATCH.name == "four_draw_v57"
    assert training.PARENT_SHA256 == "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
    assert training.PARENT == training.blueprint.PARENT
    assert training.TRAINABLE_PARAMETERS == 77998496 and training.TRAINABLE_TENSORS == 238
    assert training.FROZEN_TENSORS == 16 and training.RATE == 5e-6
    assert training.paired is fixed.protection and training.paired.FRACTION == .25
    assert training.preflight.retention is training.retention
    assert training.retention_training.cohort_retention is training.retention
    assert len(training.confirmation_cases()) == 284
    seeds = training.original.alignment.fixed.seeds_in
    fresh = seeds([batches, training.confirmation_cases()])
    assert len(fresh) == 14620
    assert not fresh.intersection(seeds([training.blueprint.batches(), training.blueprint.confirmation_cases(), training.preflight.batch()]))
    for current, previous in zip(training.confirmation_cases(), training.blueprint.confirmation_cases()):
        copied = deepcopy(current)
        copied.get("structured", copied)["seed"] = previous.get("structured", previous)["seed"]
        assert copied == previous


def test_original_selection_validation_and_checkpoint_code_is_unchanged():
    for name in ("validate", "checkpoint", "confirmation_cases", "selected_boundary", "setup"):
        assert function_ast(training.ADAPTED_SOURCE, name) == function_ast(training.blueprint.ADAPTED_SOURCE, name)
    assert training.ADAPTED_SOURCE_SHA256 == hashlib.sha256(training.ADAPTED_SOURCE.encode()).hexdigest()
    assert 'all22' in training.QUALIFICATION and 'all32' in training.POLICY
    assert '0/5/13/25/32/64/96/128' in training.QUALIFICATION
    assert '392 required/18 primary' in training.POLICY
    assert 'test_single_model_four_draw*.py' in training.REGISTER_SOURCE
    assert '*four_draw*.py' in training.REGISTER_SOURCE
    assert 'audit_single_model_four_draw.py' in training.REGISTER_SOURCE
    assert 'run_single_model_four_draw_pipeline.py' in training.REGISTER_SOURCE
    assert 'validate_host_admission' in training.verify_registration.__code__.co_names
    assert training.blueprint.STEPS == 256 and len(training.blueprint.batches()[0]["tasks"]) == 28


def test_complete_storage_and_bound_capacity_receipt():
    assert training.diagnostics is storage
    assert storage.WRITE_BUDGETS["training"] == {"checkpoint": 4 * 1024**3, "root": 64 * 1024**2, "scratch": 192 * 1024**3}
    for volume, fields in training.STORAGE_ACCOUNTING.items():
        assert sum(fields.values()) <= storage.WRITE_BUDGETS["training"][volume]
    plan = storage.layout(training.OUTPUT, training.SCRATCH, training.BOUNDARIES)
    for phase in ("training", "qualification"):
        storage.require_plan(plan, phase)
    assert plan["stage_write_budget_bytes"] == storage.WRITE_BUDGETS
    assert len(plan["directories"]["training"]) == 8
    calls = []
    capacity = types.SimpleNamespace(require_completed=lambda value: calls.append(value) or {"complete": True})
    assert trial.storage_evidence(capacity, trial.CAPACITY_RECEIPT_SHA256) == {"complete": True}
    assert calls == [trial.CAPACITY_RECEIPT_SHA256]
    with pytest.raises(ValueError):
        trial.storage_evidence(capacity, "a" * 64)


@pytest.mark.parametrize("mutation", [None, "working", "reserve", "available", "boolean", "source"])
def test_original_host_admission_is_mandatory(mutation):
    memory = {"working_allowance_bytes": training.preflight.HOST_WORKING_BYTES,
        "reserve_bytes": training.preflight.HOST_RESERVE_BYTES,
        "available_bytes": training.preflight.HOST_WORKING_BYTES + training.preflight.HOST_RESERVE_BYTES,
        "source": "/proc/meminfo:MemAvailable"}
    if mutation in ("working", "reserve", "available"):
        memory[{"working": "working_allowance_bytes", "reserve": "reserve_bytes", "available": "available_bytes"}[mutation]] -= 1
    elif mutation == "boolean":
        memory["available_bytes"] = True
    elif mutation == "source":
        memory["source"] = "inferred"
    if mutation:
        with pytest.raises(ValueError):
            trial.validate_host_admission(memory)
    else:
        trial.validate_host_admission(memory)


def test_all_runtime_callbacks_resolve_in_the_correct_namespace():
    for module in (training, storage, fixed):
        for name, function in vars(module).items():
            if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(module):
                assert not unresolved_globals(inspect.unwrap(function)), (module.__name__, name)
    for name in ("register", "verify_registration", "train", "require_evidence", "validate"):
        assert inspect.unwrap(getattr(training, name)).__globals__ is vars(training)
    for name in ("check_preflight_evidence", "require_report", "reconstruct_candidate", "check_replayed_candidate"):
        assert inspect.unwrap(getattr(fixed, name)).__code__ is inspect.unwrap(getattr(fixed.template, name)).__code__
    assert hasattr(fixed.reconstruct_candidate, "__wrapped__")


@pytest.mark.parametrize("accepted", [False, True])
def test_exact_quarter_reconstruction_uses_new_protection_without_changing_arithmetic(accepted):
    import torch
    import test_single_model_fixed_quarter_trial as original_tests

    original_training, payload = original_tests.fixture(accepted)
    expected, gradients = original_tests.integration.reconstruct_candidate(original_training, payload, "cpu")
    original_training.paired = fixed.protection
    candidate, transferred = fixed.reconstruct_candidate(original_training, payload, "cpu")
    assert torch.equal(candidate, expected) and torch.equal(gradients, transferred)
    assert torch.equal(candidate, payload["snapshot"]["contracted"])
    fixed.check_replayed_candidate(payload, candidate)


@pytest.mark.parametrize("stage", ["register", "train"])
def test_cli_original_arithmetic_and_single_shot_lock(tmp_path, monkeypatch, stage):
    from contextlib import contextmanager

    calls = []

    @contextmanager
    def arithmetic():
        calls.append("enter")
        yield
        calls.append("exit")

    monkeypatch.setattr(training.arithmetic_recovery.previous, "original_arithmetic", arithmetic)
    monkeypatch.setattr(training, "OUTPUT", tmp_path)
    monkeypatch.setattr(training, "register", lambda output: calls.append("register"))
    monkeypatch.setattr(training, "train", lambda output: calls.append("train"))
    monkeypatch.setattr(training, "read", lambda path: {"state": "registered"})
    monkeypatch.setattr(training.fcntl, "flock", lambda descriptor, flags: calls.append("lock"))
    monkeypatch.setattr(training.sys, "argv", ["trainer", "--stage", stage])
    training.main()
    assert calls == (["enter", "register", "exit"] if stage == "register" else ["enter", "lock", "train", "exit"])
    if stage == "train":
        monkeypatch.setattr(training, "read", lambda path: {"state": "failed"})
        with pytest.raises(ValueError, match="Never restart"):
            training.main()
