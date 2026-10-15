from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import preflight_single_model_legacy_blend_teacher as probe
import single_model_distillation as distillation
import single_model_legacy_blend_replay as replay


def fixture(tmp_path, monkeypatch):
    identities = {}
    for role in ("initialization", "general", "specialist"):
        path = tmp_path / f"{role}.pth"
        path.write_bytes(role.encode())
        identities[role] = {"checkpoint": str(path), "sha256": probe.sha256(path)}
    config = {"batch": probe.batch(), "teacher_identities": identities, "base": {}}
    probe.write_json(tmp_path / "config.json", config)
    probe.write_json(tmp_path / "result.json", {"training_pid": -123})
    probe.write_json(tmp_path / "status.json", {"state": "awaiting_independent_audit"})
    (tmp_path / "training_batch").mkdir()
    source = torch.tensor([[[1., float("nan")], [2., -1.], [float("nan"), 4.]]])
    center = torch.tensor([[[1., 30.], [2., -1.], [50., 4.]]])
    predictions = {"general": center - .5, "specialist": center + 1.5}
    tasks = []
    for task in config["batch"]["tasks"]:
        role = replay.expected_role(task)
        if role == "observed_blend":
            prediction, decision = distillation.blend_predictions(source, **predictions)
        else:
            prediction, decision = predictions[role].clone(), None
        episode = {"source": source.clone(), "metadata": {"domain": task["specification"]["domain"],
            "structured": "structured" in task["specification"], "gradient_data": "synthetic"}}
        cached = {"teacher": prediction, "routing": replay.routing_record(role, config, decision), "episode": episode}
        path = tmp_path / "training_batch" / f'task_{task["index"]:02d}.pt'
        torch.save(cached, path)
        tasks.append({"artifact": str(path), "sha256": probe.sha256(path)})
    probe.write_json(tmp_path / "training_batch/manifest.json", {"tasks": tasks})
    models = {role: torch.nn.Linear(1, 1).eval().requires_grad_(False) for role in identities}
    calls, states = [], []

    def predict(model, actual_source, criterion, **options):
        assert not torch.is_grad_enabled() and options == probe.teacher_policy.STREAM_OPTIONS
        assert torch.allclose(actual_source, source, equal_nan=True)
        role = next(role for role, current in models.items() if model is current)
        calls.append(role)
        return predictions[role].clone()

    def verify_model(model, identity):
        states.append(identity)
        return {**identity, "bitwise_equal": True, "state_tensors": 2}

    monkeypatch.setattr(probe, "verify", lambda output: config)
    monkeypatch.setattr(probe, "wait_for_gpu", lambda *args: None)
    monkeypatch.setattr(probe, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(probe, "teacher_predictions", predict)
    monkeypatch.setattr(probe.task_audit, "check_task", lambda *args: {})
    monkeypatch.setattr(probe.teacher_policy, "verify_model", verify_model)
    monkeypatch.setattr(probe.training.alignment.mixture, "original_setup", lambda base:
        (models["initialization"], None, None, [models["general"], models["specialist"]], None))
    monkeypatch.setattr(probe.training.alignment.mixture.backend, "restore_episode",
        lambda episode, device: SimpleNamespace(**episode))
    return config, calls, states


def test_actual_generated_replay_and_cpu_audit_preserve_all_targets_and_components(tmp_path, monkeypatch):
    config, calls, states = fixture(tmp_path, monkeypatch)
    probe.replay_admitted(tmp_path)
    result = probe.audit_teacher_replay(tmp_path, config)
    assert result["complete"] and result["full_grids"] == 28 and result["bitwise_equal"]
    assert result["blend_tasks"] == 2 and result["component_grids"] == 4
    assert result["observed_weights_refitted"] and len(result["artifact_sha256"]) == 61
    assert calls.count("general") == 26 and calls.count("specialist") == 4
    assert states == list(config["teacher_identities"].values())
    report = probe.read(tmp_path / "teacher_replay/result.json")
    for row in report["rows"]:
        with np.load(row["prediction"], allow_pickle=False) as saved:
            assert set(saved.files) == ({"teacher", "general", "specialist"}
                if row["role"] == "observed_blend" else {"teacher"})
    cached = torch.load(tmp_path / "training_batch/task_00.pt", weights_only=False)
    assert cached["routing"]["selected"] == "general"
    assert cached["routing"]["specialist_weight"] == pytest.approx(.25)
    assert cached["routing"]["evidence_cells"] == 4


@pytest.mark.parametrize("mutation", ["missing_component", "query_component", "observed_component", "weight",
    "source", "identity", "role", "count", "same_process", "hash", "model_state", "missing_task",
    "single_teacher_grid", "wrong_dtype", "omitted_grid", "checkpoint"])
def test_generated_cpu_audit_rejects_corrupt_full_replay(tmp_path, monkeypatch, mutation):
    config, _, _ = fixture(tmp_path, monkeypatch)
    probe.replay_admitted(tmp_path)
    report_path = tmp_path / "teacher_replay/result.json"
    report = probe.read(report_path)
    row = report["rows"][1 if mutation == "single_teacher_grid" else 0]
    prediction_path = Path(row["prediction"])
    if mutation in {"missing_component", "query_component", "observed_component", "wrong_dtype", "single_teacher_grid"}:
        with np.load(prediction_path, allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        if mutation == "missing_component":
            arrays.pop("specialist")
        elif mutation == "query_component":
            arrays["general"][0, 0, 1] += 1.
        elif mutation == "observed_component":
            arrays["general"][0, 0, 0] += 1.
        elif mutation == "wrong_dtype":
            arrays["general"] = arrays["general"].astype(np.float64)
        else:
            arrays["teacher"][0, 0, 1] += 1.
        probe.atomic_npz(prediction_path, **arrays)
        row["prediction_sha256"] = probe.sha256(prediction_path)
    elif mutation in {"weight", "source", "identity"}:
        manifest_path = tmp_path / "training_batch/manifest.json"
        manifest = probe.read(manifest_path)
        record = manifest["tasks"][0]
        cached = torch.load(record["artifact"], weights_only=False)
        if mutation == "weight":
            cached["routing"]["specialist_weight"] = .5
        elif mutation == "source":
            cached["episode"]["source"][0, 0, 0] += .1
        else:
            cached["routing"]["teacher_identity"]["specialist"]["sha256"] = "wrong"
        torch.save(cached, record["artifact"])
        row["task_sha256"] = record["sha256"] = probe.sha256(record["artifact"])
        probe.write_json(manifest_path, manifest)
    elif mutation == "role":
        row["role"] = "general"
    elif mutation == "count":
        report["component_grids"] = 2
    elif mutation == "same_process":
        report["pid"] = report["training_pid"]
    elif mutation == "hash":
        row["prediction_sha256"] = "wrong"
    elif mutation == "model_state":
        report["model_states"]["specialist"]["bitwise_equal"] = False
    elif mutation == "missing_task":
        report["rows"].pop()
    elif mutation == "omitted_grid":
        prediction_path.unlink()
    else:
        Path(config["teacher_identities"]["specialist"]["checkpoint"]).write_bytes(b"replacement")
    probe.write_json(report_path, report)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        probe.audit_teacher_replay(tmp_path, config)


def test_replay_rejects_changed_target_before_writing_components(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    original = probe.teacher_predictions
    monkeypatch.setattr(probe, "teacher_predictions", lambda *args, **kwargs: original(*args, **kwargs) + .1)
    with pytest.raises(ValueError):
        probe.replay_admitted(tmp_path)
    assert not (tmp_path / "teacher_replay/result.json").exists()


def test_observed_fit_never_uses_query_predictions_and_preserves_no_evidence_fallback():
    source = torch.tensor([[[1., float("nan")], [2., 3.]]])
    general, specialist = torch.nan_to_num(source) - .5, torch.nan_to_num(source) + 1.5
    prediction, decision = replay.blend_components(source, general, specialist)
    changed = general.clone()
    changed[0, 0, 1] += 100.
    changed_prediction, changed_decision = replay.blend_components(source, changed, specialist)
    assert decision == changed_decision and not torch.equal(prediction, changed_prediction)
    _, fallback = replay.blend_components(torch.full_like(source, float("nan")), general, specialist)
    assert fallback == {"evidence_cells": 0, "general_mse": None, "specialist_mse": None,
        "selected": "general", "specialist_weight": 0., "fallback": "no evidence"}


@pytest.mark.parametrize("mutation", ["missing_task", "large_shape", "wrong_domain", "structured_recsys"])
def test_storage_does_not_allow_missing_or_unbudgeted_components(mutation):
    batch = deepcopy(probe.batch())
    if mutation == "missing_task":
        batch["tasks"].pop(0)
    elif mutation == "large_shape":
        batch["tasks"][0]["specification"]["shape"] = [10000, 10000]
    elif mutation == "wrong_domain":
        batch["tasks"][0]["specification"]["domain"] = "real"
    else:
        batch["tasks"][13]["specification"]["structured"] = {}
    with pytest.raises(ValueError):
        replay.storage_accounting(batch)
