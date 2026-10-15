from copy import deepcopy
import inspect
import types

import pytest

import test_single_model_cohort_anchor_retention_trial as complete_trial_blueprint
import train_single_matrixformer_stronger_mean as previous_training
import train_single_matrixformer_general_native_retention as training
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


function = complete_trial_blueprint.fixed_preflight
fixed_preflight = types.FunctionType(function.__code__, globals(), "fixed_preflight", function.__defaults__)
ADAPTED_TEST_SOURCE = complete_trial_blueprint.ADAPTED_TEST_SOURCE.replace(
    "cohort_anchor_retention", "general_native_retention")
for old, new in {"1061100901": "1211100901", "1062100901": "1212100901",
                 "1071100901": "1221100901", "1072100901": "1222100901",
                 'training.retention.REFERENCE_SHA256S["v29"]': 'training.PARENT_SHA256',
                 '("v8", "v29")': '("v8", "general")'}.items():
    ADAPTED_TEST_SOURCE = ADAPTED_TEST_SOURCE.replace(old, new)
exec(compile(ADAPTED_TEST_SOURCE, __file__, "exec"), globals())


def test_only_native_training_reference_changes_scientifically():
    assert training.OUTPUT.name == "general_native_retention_v47"
    for name in ("PARENT", "PARENT_SHA256", "RATE", "PARENT_UPDATES", "BOUNDARIES", "STEPS", "TRIAL_QUALIFICATION",
                 "STORAGE_ACCOUNTING", "teacher_policy", "stronger", "diagnostics", "paired"):
        assert getattr(training, name) == getattr(previous_training, name)
    assert training.retention is training.preflight.retention
    assert training.capacity.probe is training.preflight
    assert training.retention_training.retention is training.retention
    current = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    previous = training.original.alignment.fixed.seeds_in([
        previous_training.batches(), previous_training.confirmation_cases(), training.preflight.batch()])
    assert len(current) == 3868 and not current.intersection(previous)
    for batch, original in zip(training.batches(), previous_training.batches()):
        training.retention.validate_task_mixture(batch["tasks"])
        for task, prior in zip(batch["tasks"], original["tasks"]):
            copied = deepcopy(task)
            copied["gradient_seed"] = prior["gradient_seed"]
            copied["specification"].get("structured", copied["specification"])["seed"] = prior["specification"].get(
                "structured", prior["specification"])["seed"]
            assert copied == prior


def test_every_actual_function_has_resolved_globals_and_new_references():
    for module in (training, training.retention_training, training.all_trial):
        for value in vars(module).values():
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(module):
                assert not unresolved_globals(value)
    assert '"retention_references": retention_training.reference_identities(PARENT, base)' in training.REGISTER_SOURCE
    assert '"mean_distillation": stronger.SPECIFICATION' in training.REGISTER_SOURCE
    assert 'with stronger.training_objective(), teacher_policy.runtime(' in training.ADAPTED_SOURCE
    assert 'config["mean_distillation"] != stronger.SPECIFICATION' in training.ADAPTED_SOURCE
    assert training.require_evidence.__globals__ is vars(training)


@pytest.mark.parametrize("mutation", [None, "live", "reference", "components", "weights", "objective", "penalty"])
def test_full_trial_consumes_only_completed_new_probe(monkeypatch, mutation):
    probe = training.preflight
    documents = {name: deepcopy(probe.read(probe.OUTPUT / name)) for name in
        ("result.json", "audit.json", "gate.json", "retention_audit.json", "config.json", "status.json")}
    monkeypatch.setattr(probe, "verify", lambda output: None)
    monkeypatch.setattr(probe, "require_completed_v46", lambda: {"closed_v46": "digest"})
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "sha256", lambda path: "digest")
    monkeypatch.setattr(probe, "read", lambda path: documents[path.name])
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: {"pid": pid} if mutation == "live" else None)
    if mutation == "reference":
        documents["retention_audit.json"]["reference_role_counts"] = {"v8": 26}
    elif mutation == "components":
        documents["audit.json"]["teacher_replay"]["component_grids"] = 51
    elif mutation == "weights":
        documents["audit.json"]["teacher_replay"]["observed_weights_refitted"] = False
    elif mutation == "objective":
        documents["config.json"]["mean_distillation"] = {}
    elif mutation == "penalty":
        documents["config.json"]["anchor_retention"] = {}
    validation = []
    validator = lambda *args: validation.append(args)
    if mutation is None:
        evidence = training.all_trial.require_probe(probe, validator)
        assert evidence["closed_v46"] == "digest" and len(validation) == 1
        assert str(probe.OUTPUT / "retention_audit.json") in evidence
    else:
        with pytest.raises(ValueError):
            training.all_trial.require_probe(probe, validator)
