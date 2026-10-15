from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import preflight_single_model_legacy_blend_teacher as probe
import single_model_legacy_blend_full_replay as integration
import single_model_distillation as original


def fixture(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    batches = [{**deepcopy(probe.batch()), "step": step} for step in range(1, 65)]
    identities, models = {}, []
    for role in ("general", "specialist"):
        path = tmp_path / f"{role}.pth"
        path.write_bytes(role.encode())
        identities[role] = {"checkpoint": str(path), "sha256": integration.sha256(path)}
        models.append(torch.nn.Linear(1, 1).eval().requires_grad_(False))
    config = {"batches": batches, "teacher_policy": integration.teacher_policy.POLICY, "teacher_identities": identities,
        "legacy_blend_replay": integration.specification(scratch, batches)}
    probe.write_json(tmp_path / "config.json", config)
    training = SimpleNamespace(OUTPUT=tmp_path, SCRATCH=scratch, read=probe.read)
    counts = {"admitted_steps": [1], "training_tasks": 28}
    source = torch.tensor([[[1., float("nan")], [2., -1.], [float("nan"), 4.]]])
    center = torch.tensor([[[1., 30.], [2., -1.], [50., 4.]]])
    predictions = {"general": center - .5, "specialist": center + 1.5}
    payloads = {}
    (tmp_path / "training_batches/step_001").mkdir(parents=True)
    for path, (step, task) in integration.tasks_for(training, config, counts).items():
        role = integration.replay.expected_role(task)
        if role == "observed_blend":
            target, decision = original.blend_predictions(source, **predictions)
        else:
            target, decision = predictions[role].clone(), None
        payload = {"task": task, "episode": {"source": source.clone(), "scaling": None,
            "target": torch.full_like(source, 987654.),
            "metadata": {"domain": task["specification"]["domain"], "gradient_data": "synthetic",
                "structured": "structured" in task["specification"]}}, "teacher": target,
            "routing": integration.replay.routing_record(role, config, decision)}
        torch.save(integration.compact_cache.capsule_for(payload), path)
        payloads[path] = payload
    calls = []

    def predict(model, actual_source, criterion, **options):
        assert not torch.is_grad_enabled() and torch.allclose(actual_source, source, equal_nan=True)
        assert options == integration.teacher_policy.STREAM_OPTIONS
        role = "general" if model is models[0] else "specialist"
        calls.append(role)
        return predictions[role].clone()

    monkeypatch.setattr(integration, "teacher_predictions", predict)
    monkeypatch.setattr(integration, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(integration.teacher_policy, "verify_model", lambda model, identity:
        {**identity, "bitwise_equal": True, "state_tensors": 2})
    return training, config, counts, models, payloads, calls


def replay_all(state, payloads):
    for path, payload in payloads.items():
        state.check(Path(path), payload)


def test_actual_full_replay_reopens_every_component_and_never_archives_hidden_truth(tmp_path, monkeypatch):
    training, config, counts, models, payloads, calls = fixture(tmp_path, monkeypatch)
    state = integration.ComponentReplay(training, config, counts, models, None)
    replay_all(state, payloads)
    hashes = state.finish()
    report = integration.require_report(training, {"admission_attempts": counts, "artifact_sha256": hashes})
    assert len(state.seen) == 28 and calls == ["specialist", "general"] * 2
    assert report["training_tasks_checked"] == 28 and report["legacy_tasks"] == 2 and report["raw_component_grids"] == 4
    assert report["observed_weights_refitted_on_cpu"] and report["separate_cpu_only_process"] is False
    assert report["training_performed"] is report["inference_ensemble"] is report["query_truth_used_for_routing"] is False
    assert len(report["artifact_sha256"]) == 34
    for row in state.rows:
        if row["role"] == "observed_blend":
            with np.load(row["archive"], allow_pickle=False) as saved:
                assert set(saved.files) == {"source", "teacher", "general", "specialist"}
                assert not any(np.any(saved[name] == 987654.) for name in saved.files)
            assert row["routing"]["specialist_weight"] == pytest.approx(.25)
            assert row["routing"]["selected"] == "general"
    with pytest.raises(ValueError, match="exactly one"):
        state.finish()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order", "source", "query_component", "observed_component",
    "target", "component_missing", "extra_truth", "weight", "identity", "role", "capsule", "digest", "metadata"])
def test_reopened_components_reject_every_omission_or_changed_target(tmp_path, monkeypatch, mutation):
    training, config, counts, models, payloads, calls = fixture(tmp_path, monkeypatch)
    state = integration.ComponentReplay(training, config, counts, models, None)
    replay_all(state, payloads)
    state.finish()
    rows = deepcopy(state.rows)
    record = rows[0]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(record)
    elif mutation == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation in {"source", "query_component", "observed_component", "target", "component_missing", "extra_truth"}:
        with np.load(record["archive"], allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        if mutation == "source":
            arrays["source"][0, 0, 0] += .1
        elif mutation == "query_component":
            arrays["specialist"][0, 0, 1] += 1.
        elif mutation == "observed_component":
            arrays["general"][0, 0, 0] += 1.
        elif mutation == "target":
            arrays["teacher"][0, 0, 1] += 1.
        elif mutation == "component_missing":
            arrays.pop("general")
        else:
            arrays["truth"] = arrays["teacher"].copy()
        integration.atomic_npz(Path(record["archive"]), **arrays)
        record["archive_sha256"] = integration.sha256(record["archive"])
    elif mutation == "weight":
        record["routing"]["specialist_weight"] = .5
    elif mutation == "identity":
        record["routing"]["teacher_identity"]["specialist"]["sha256"] = "wrong"
    elif mutation == "role":
        record["role"] = "general"
    elif mutation == "capsule":
        capsule = torch.load(record["task"], weights_only=True)
        capsule["task"]["weight"] *= 2
        torch.save(capsule, record["task"])
        record["task_sha256"] = integration.sha256(record["task"])
    elif mutation == "digest":
        record["content_sha256"] = "wrong"
    else:
        record["metadata"]["domain"] = "recsys"
    with pytest.raises((ValueError, RuntimeError)):
        integration.verify_records(training, config, counts, rows)


def test_data_only_noop_has_no_component_inference(tmp_path, monkeypatch):
    training, config, counts, models, payloads, calls = fixture(tmp_path, monkeypatch)
    counts = {"admitted_steps": [], "training_tasks": 0}
    state = integration.ComponentReplay(training, config, counts, models, None)
    hashes = state.finish()
    report = integration.require_report(training, {"admission_attempts": counts, "artifact_sha256": hashes})
    assert not calls and report["legacy_tasks"] == report["training_tasks_checked"] == report["raw_component_grids"] == 0
    assert (state.output / "tasks.jsonl").read_text() == ""


@pytest.mark.parametrize("mutation", ["duplicate", "payload", "teacher", "incomplete", "frozen_teacher"])
def test_live_replay_cannot_hide_changed_targets_or_teacher_state(tmp_path, monkeypatch, mutation):
    training, config, counts, models, payloads, calls = fixture(tmp_path, monkeypatch)
    state = integration.ComponentReplay(training, config, counts, models, None)
    path = next(iter(payloads))
    if mutation == "payload":
        payloads[path]["teacher"][0, 0, 1] += 1.
    elif mutation == "teacher":
        original_predict = integration.teacher_predictions
        monkeypatch.setattr(integration, "teacher_predictions", lambda *args, **kwargs: original_predict(*args, **kwargs) + .1)
    elif mutation == "duplicate":
        state.check(Path(path), payloads[path])
    elif mutation == "frozen_teacher":
        replay_all(state, payloads)
        monkeypatch.setattr(integration.teacher_policy, "verify_model", lambda *args: {})
    with pytest.raises(ValueError):
        if mutation in {"incomplete", "frozen_teacher"}:
            state.finish()
        else:
            state.check(Path(path), payloads[path])


@pytest.mark.parametrize("mutation", [None, "batch_count", "task_count", "legacy_count", "shape"])
def test_full64_attempt_storage_accounts_for_source_target_and_both_components(mutation):
    batches = [{**deepcopy(probe.batch()), "step": step} for step in range(1, 65)]
    if mutation == "batch_count":
        batches.pop()
    elif mutation == "task_count":
        batches[0]["tasks"].pop()
    elif mutation == "legacy_count":
        batches[0]["tasks"][0]["specification"]["structured"] = {}
    elif mutation == "shape":
        batches[0]["tasks"][0]["specification"]["shape"] = [0, 128]
    if mutation:
        with pytest.raises(ValueError):
            integration.storage_accounting(batches)
    else:
        assert integration.storage_accounting(batches) == {"legacy_tasks": 128, "raw_component_grids": 256,
            "cells": 4194304, "array_bytes": 67108864, "atomic_archive_bytes": 524288,
            "serialization_and_manifest_allowance": 8388608}
        assert integration.storage_bytes(batches) == 76021760


def test_real_full_trial_shapes_fit_the_unchanged40_gib_budget():
    import train_single_matrixformer_all_imputation_retention as previous

    accounting = integration.storage_accounting(previous.batches())
    assert accounting == {"legacy_tasks": 128, "raw_component_grids": 256, "cells": 130097152,
        "array_bytes": 2081554432, "atomic_archive_bytes": 32000000, "serialization_and_manifest_allowance": 8388608}
    extra = integration.storage_bytes(previous.batches())
    assert extra == 2121943040
    assert sum(previous.STORAGE_ACCOUNTING["scratch"].values()) + extra == 38340399232
    assert 38340399232 < previous.diagnostics.WRITE_BUDGETS["training"]["scratch"] == 40 * 1024**3


@pytest.mark.parametrize("field,value", [("complete", False), ("legacy_tasks", 1), ("raw_component_grids", 2),
    ("query_truth_used_for_routing", True), ("separate_cpu_only_process", True), ("training_performed", True),
    ("inference_ensemble", True), ("observed_weights_refitted_on_cpu", False)])
def test_qualification_rechecks_scope_even_with_a_new_summary_hash(tmp_path, monkeypatch, field, value):
    training, config, counts, models, payloads, calls = fixture(tmp_path, monkeypatch)
    state = integration.ComponentReplay(training, config, counts, models, None)
    replay_all(state, payloads)
    hashes = state.finish()
    summary = state.output / "summary.json"
    report = probe.read(summary)
    report[field] = value
    probe.write_json(summary, report)
    hashes[str(summary)] = integration.sha256(summary)
    with pytest.raises(ValueError):
        integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
