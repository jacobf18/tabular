from copy import deepcopy
import inspect
import types

import pytest

import test_single_model_native_column_trial as blueprint
import train_single_matrixformer_native_column_mean as previous_training
import train_single_matrixformer_parent_structured as training
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


fixed_preflight = types.FunctionType(blueprint.fixed_preflight.__code__, globals(), "fixed_preflight", blueprint.fixed_preflight.__defaults__)
ADAPTED_TEST_SOURCE = blueprint.ADAPTED_TEST_SOURCE
for old, new in (("train_single_matrixformer_native_column_mean", "train_single_matrixformer_parent_structured"),
        ("1241100901", "1271100901"), ("1242100901", "1272100901"),
        ("1251100901", "1281100901"), ("1252100901", "1282100901"), ('("v8", "general")', '("v29", "general")')):
    ADAPTED_TEST_SOURCE = ADAPTED_TEST_SOURCE.replace(old, new)
exec(compile(ADAPTED_TEST_SOURCE, __file__, "exec"), globals())

source = inspect.getsource(blueprint.test_full_trial_requires_complete_probe_and_supplementary_evidence)
source = source.replace("require_completed_v47", "require_completed_v48").replace("closed_v47", "closed_v48")
exec(compile(source, __file__, "exec"), globals())


def test_only_structured_reference_changes_with_fresh_full_trial_and_confirmation_seeds():
    assert training.OUTPUT.name == "parent_structured_v49"
    for name in ("PARENT", "PARENT_SHA256", "RATE", "PARENT_UPDATES", "BOUNDARIES", "STEPS", "TRIAL_QUALIFICATION",
            "STORAGE_ACCOUNTING", "teacher_policy", "stronger", "diagnostics", "paired"):
        assert getattr(training, name) == getattr(previous_training, name)
    assert training.retention is training.preflight.retention
    assert training.capacity.probe is training.preflight
    assert training.retention_training.retention is training.retention
    assert training.CAPACITY_RECEIPT_SHA256 == "de3f132f725b09c4d8a4225b14e41187f63a668921213b37cb181409ba2b5e49"
    current = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    previous = training.original.alignment.fixed.seeds_in([previous_training.batches(), previous_training.confirmation_cases(), training.preflight.batch()])
    assert len(current) == 3868 and not current.intersection(previous)
    for batch, original in zip(training.batches(), previous_training.batches()):
        training.retention.validate_task_mixture(batch["tasks"])
        for task, prior in zip(batch["tasks"], original["tasks"]):
            copied = deepcopy(task)
            copied["gradient_seed"] = prior["gradient_seed"]
            copied["specification"].get("structured", copied["specification"])["seed"] = prior["specification"].get("structured", prior["specification"])["seed"]
            assert copied == prior
    assert training.retention.BACKWARD_SOURCE == previous_training.retention.BACKWARD_SOURCE
    assert set(training.preflight.reference_identities()) == {"v29", "general"}
    assert training.original.alignment.fixed.PARENT == previous_training.original.alignment.fixed.PARENT


def test_every_actual_binding_keeps_mean_and_reference_identity_checks():
    for module in (training, training.retention_training, training.all_trial):
        for value in vars(module).values():
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(module):
                assert not unresolved_globals(value)
    assert '"native_teacher_mean": retention.MEAN_SPECIFICATION' in training.REGISTER_SOURCE
    assert 'config["native_teacher_mean"] != retention.MEAN_SPECIFICATION' in training.ADAPTED_SOURCE
    assert 'with stronger.training_objective(), teacher_policy.runtime(' in training.ADAPTED_SOURCE
    assert training.require_evidence.__globals__ is vars(training)
    assert "1922740672" in training.all_trial.STORAGE_EVIDENCE_SOURCE
    assert "'canonical_artifacts_checked'] != 5" in training.all_trial.STORAGE_EVIDENCE_SOURCE
