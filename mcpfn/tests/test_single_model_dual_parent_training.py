from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import single_model_dual_parent_training as integration
import preflight_single_model_dual_parent_retention as probe
import test_single_model_parent_structured_training as parent_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def fixture(tmp_path, monkeypatch):
    training, config, counts, references, backend, ignored = parent_tests.fixture(tmp_path, monkeypatch)
    references["v8"] = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    identities = probe.reference_identities()
    config.update(retention_references=identities, anchor_retention=integration.retention.SPECIFICATION,
        native_teacher_mean=integration.retention.MEAN_SPECIFICATION,
        input_sha256={identity["checkpoint"]: identity["sha256"] for identity in identities.values()})
    probe.write_json(training.OUTPUT / "config.json", config)
    folder = integration.directory(training.SCRATCH, 1)
    manifest = probe.read(folder / "manifest.json")
    verified = {role: {**identity, "state_tensors": 254, "bitwise_equal": True} for role, identity in identities.items()}
    manifest.update(policy=integration.retention.POLICY, specification=integration.retention.SPECIFICATION,
        reference=verified)
    for record in manifest["records"]:
        role = integration.retention.reference_role(record["metadata"])
        record["reference_role"] = role
        if not record["eligible"]:
            continue
        path = Path(record["artifact"])
        cached = torch.load(path, weights_only=True)
        prediction = cached["prediction"]
        components = {name: prediction + {"general": 0., "v8": .5, "v29": -.5}[name]
            for name in integration.retention.reference_weights(role)}
        cached.update(reference_role=role, identity=integration.retention.target_identity(role, identities),
            specification=integration.retention.SPECIFICATION, components=components,
            component_weights=integration.retention.reference_weights(role),
            prediction=integration.retention.combine_predictions(role, components))
        torch.save(cached, path)
        record["sha256"] = probe.sha256(path)
    probe.write_json(folder / "manifest.json", manifest)
    trace = json.loads((training.OUTPUT / "training.jsonl").read_text())
    trace["retention_manifest_sha256"] = probe.sha256(folder / "manifest.json")
    (training.OUTPUT / "training.jsonl").write_text(json.dumps(trace) + "\n")
    monkeypatch.setattr(integration.retention, "verify_model", lambda model, identity:
        {**identity, "state_tensors": 254, "bitwise_equal": True})
    calls = []
    base_predict = parent_tests.integration.retention.streamed_point_predictions

    def predict(model, values, criterion, **options):
        assert not torch.is_grad_enabled()
        role = next(name for name, reference in references.items() if reference is model)
        calls.append(role)
        return base_predict(model, values, criterion, **options) + {"general": 0., "v8": .5, "v29": -.5}[role]

    monkeypatch.setattr(integration.retention, "streamed_point_predictions", predict)
    return training, config, counts, references, backend, calls


def replay_all(state):
    for path in state.tasks:
        state.check(Path(path), torch.load(path, weights_only=False))


def test_full_trial_replays_all50_components_and26_targets(tmp_path, monkeypatch):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    replay_all(state)
    hashes = state.finish()
    report = integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
    assert report["component_grids"] == 50 and report["reference_grids"] == 26
    assert report["component_role_counts"] == {"general": 2, "v8": 24, "v29": 24}
    assert report["reference_role_counts"] == {"general": 2, "structured_mean": 24}
    assert {name: calls.count(name) for name in references} == report["component_role_counts"]
    assert report["historical_full_autograd_replayed"] is False
    assert report["training_performed"] is False and report["no_inference_ensemble"] is True
    for row in map(json.loads, state.rows_path.read_text().splitlines()):
        if row["eligible"]:
            with np.load(row["prediction"], allow_pickle=False) as archive:
                expected = {"prediction", *(f"component_{name}" for name in row["component_roles"])}
                assert set(archive.files) == expected
    for function in vars(integration).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(integration):
            assert unresolved_globals(function) == []


def test_data_only_rejection_performs_no_component_inference(tmp_path, monkeypatch):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    directory = integration.directory(training.SCRATCH, 1)
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()
    (training.OUTPUT / "training.jsonl").write_text(json.dumps({"step": 1, "attempt_kind": "data_admission_rejected"}) + "\n")
    counts = {"admitted_steps": [], "training_tasks": 0}
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    hashes = state.finish()
    report = integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
    assert not calls and report["component_grids"] == report["reference_grids"] == 0
    assert report["component_role_counts"] == {}


@pytest.mark.parametrize("mutation", ["reference", "base", "top_level", "native_mean", "mean_rule", "initializer"])
def test_full_identity_cannot_change_original_baseline_or_objective(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    config = deepcopy(config)
    if mutation == "reference":
        config["retention_references"].pop("v8")
    elif mutation == "base":
        config["base"]["input_sha256"][str(integration.EVALUATION_BASELINE)] = "changed"
    elif mutation == "top_level":
        config["input_sha256"].pop(str(integration.EVALUATION_BASELINE))
    elif mutation == "native_mean":
        config["native_teacher_mean"]["retention"] = "changed"
    elif mutation == "mean_rule":
        config["anchor_retention"]["reference_combination"]["weights"]["v8"] = .6
    else:
        config["parent_checkpoint"] = "/probe/candidate.pth"
    with pytest.raises(ValueError):
        integration.identity(config)


@pytest.mark.parametrize("mutation", ["component", "offsetting", "weights", "identity", "missing", "prediction"])
def test_full_replay_rejects_rehashed_component_corruption(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    path = next(path for path, (task, decision, folder) in state.tasks.items() if decision["reference_role"] == "structured_mean")
    task, decision, folder = state.tasks[path]
    payload = torch.load(decision["artifact"], weights_only=True)
    if mutation in {"component", "offsetting"}:
        payload["components"]["v8"] += 1
        if mutation == "offsetting":
            payload["components"]["v29"] -= 1
        payload["prediction"] = integration.retention.combine_predictions("structured_mean", payload["components"])
    elif mutation == "weights":
        payload["component_weights"]["v8"] = .6
    elif mutation == "identity":
        payload["identity"]["components"]["v8"]["sha256"] = "changed"
    elif mutation == "missing":
        payload["components"].pop("v8")
    else:
        payload["prediction"] += 1
    torch.save(payload, decision["artifact"])
    decision["sha256"] = probe.sha256(decision["artifact"])
    with pytest.raises(ValueError):
        state.check(Path(path), torch.load(path, weights_only=False))
    assert not state.summary_path.exists()


@pytest.mark.parametrize("mutation", ["count", "component_role_counts", "drop_component", "false_component_check"])
def test_full_qualification_requires_component_coverage_even_if_rehashed(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    replay_all(state)
    hashes = state.finish()
    report = probe.read(state.summary_path)
    if mutation == "count":
        report["component_grids"] -= 1
    elif mutation == "component_role_counts":
        report["component_role_counts"]["v8"] -= 1
    else:
        rows = [json.loads(line) for line in state.rows_path.read_text().splitlines()]
        row = next(row for row in rows if row["reference_role"] == "structured_mean")
        if mutation == "drop_component":
            row["component_roles"].pop()
        else:
            row["components_bitwise_equal"] = False
        state.rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes[str(state.rows_path)] = probe.sha256(state.rows_path)
        report["artifact_sha256"][str(state.rows_path)] = probe.sha256(state.rows_path)
    probe.write_json(state.summary_path, report)
    hashes[str(state.summary_path)] = probe.sha256(state.summary_path)
    with pytest.raises(ValueError):
        integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
