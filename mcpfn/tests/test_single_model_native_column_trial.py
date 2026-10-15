from copy import deepcopy
import inspect
import types

import pytest

import test_single_model_general_native_trial as blueprint
import train_single_matrixformer_general_native_retention as previous_training
import train_single_matrixformer_native_column_mean as training
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


fixed_preflight = types.FunctionType(blueprint.fixed_preflight.__code__, globals(), "fixed_preflight", blueprint.fixed_preflight.__defaults__)
ADAPTED_TEST_SOURCE = blueprint.ADAPTED_TEST_SOURCE
for old, new in (("train_single_matrixformer_general_native_retention", "train_single_matrixformer_native_column_mean"),
        ("1211100901", "1241100901"), ("1212100901", "1242100901"),
        ("1221100901", "1251100901"), ("1222100901", "1252100901")):
    ADAPTED_TEST_SOURCE = ADAPTED_TEST_SOURCE.replace(old, new)
exec(compile(ADAPTED_TEST_SOURCE, __file__, "exec"), globals())


def test_only_native_teacher_mean_weighting_changes_scientifically():
    assert training.OUTPUT.name == "native_column_mean_v48"
    for name in ("PARENT", "PARENT_SHA256", "RATE", "PARENT_UPDATES", "BOUNDARIES", "STEPS", "TRIAL_QUALIFICATION",
            "STORAGE_ACCOUNTING", "teacher_policy", "stronger", "diagnostics", "paired"):
        assert getattr(training, name) == getattr(previous_training, name)
    assert training.retention is training.preflight.retention
    assert training.retention.SPECIFICATION == previous_training.retention.SPECIFICATION
    assert training.capacity.probe is training.preflight
    assert training.retention_training.retention is training.retention
    assert training.CAPACITY_RECEIPT_SHA256 == "b15d13d06fe8598ac3bb3ab672746966043b522322691e6b6d0d7125fab0492d"
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


def test_native_mean_rule_and_all_actual_global_bindings_are_registered():
    for module in (training, training.retention_training, training.all_trial):
        for value in vars(module).values():
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(module):
                assert not unresolved_globals(value)
    assert '"native_teacher_mean": retention.MEAN_SPECIFICATION' in training.REGISTER_SOURCE
    assert 'config["native_teacher_mean"] != retention.MEAN_SPECIFICATION' in training.ADAPTED_SOURCE
    assert 'with stronger.training_objective(), teacher_policy.runtime(' in training.ADAPTED_SOURCE
    assert training.require_evidence.__globals__ is vars(training)


@pytest.mark.parametrize("mutation", [None, "live", "reference", "components", "weights", "objective", "penalty", "native_mean",
    "partial_mean", "hidden_adverse", "mean_inference"])
def test_full_trial_requires_complete_probe_and_supplementary_evidence(monkeypatch, mutation):
    probe = training.preflight
    documents = {name: deepcopy(probe.read(probe.OUTPUT / name)) for name in
        ("result.json", "audit.json", "gate.json", "retention_audit.json", "config.json", "status.json")}
    mean_directory = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")
    mean_documents = {name: deepcopy(probe.read(mean_directory / name)) for name in ("audit.json", "config.json")}
    monkeypatch.setattr(probe, "verify", lambda output: None)
    monkeypatch.setattr(probe, "require_completed_v47", lambda: {"closed_v47": "digest"})
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "sha256", lambda path: mean_documents["audit.json"]["config_sha256"] if path == mean_directory / "config.json" else "digest")
    monkeypatch.setattr(probe, "read", lambda path: (mean_documents if path.parent == mean_directory else documents)[path.name])
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: {"pid": pid} if mutation == "live" else None)
    if mutation == "reference":
        documents["retention_audit.json"]["reference_role_counts"] = {"v8": 26}
    elif mutation == "components":
        documents["audit.json"]["teacher_replay"]["component_grids"] = 51
    elif mutation == "weights":
        documents["audit.json"]["teacher_replay"]["observed_weights_refitted"] = False
    elif mutation in ("objective", "penalty", "native_mean"):
        documents["config.json"][{"objective": "mean_distillation", "penalty": "anchor_retention", "native_mean": "native_teacher_mean"}[mutation]] = {}
    elif mutation == "partial_mean":
        mean_documents["audit.json"]["tasks"] = 27
    elif mutation == "hidden_adverse":
        mean_documents["audit.json"]["adverse_teacher_mean_tasks"] = 0
    elif mutation == "mean_inference":
        mean_documents["audit.json"]["model_inference_performed"] = True
    validation = []
    if mutation is None:
        evidence = training.all_trial.require_probe(probe, lambda *args: validation.append(args))
        assert evidence["closed_v47"] == "digest" and len(validation) == 1
        assert evidence[str(mean_directory / "audit.json")] == training.all_trial.MEAN_AUDIT_SHA256
    else:
        with pytest.raises(ValueError):
            training.all_trial.require_probe(probe, lambda *args: validation.append(args))
