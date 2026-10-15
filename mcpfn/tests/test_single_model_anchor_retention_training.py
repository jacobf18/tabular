from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import single_model_anchor_retention_training as integration
import preflight_single_model_anchor_retention as probe
from test_single_model_anchor_retention_preflight import retention_fixture


def fixture(tmp_path, monkeypatch):
    retention_fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    integration.prepare(scratch)
    target = integration.directory(scratch, 1)
    (tmp_path / "retention_targets").rename(target)
    manifest = probe.read(target / "manifest.json")
    for row in manifest["records"]:
        if row["eligible"]:
            row["artifact"] = str(target / Path(row["artifact"]).name)
    probe.write_json(target / "manifest.json", manifest)
    (tmp_path / "training_batches").mkdir()
    (tmp_path / "training_batch").rename(tmp_path / "training_batches/step_001")
    config = probe.read(tmp_path / "config.json")
    batch = {"step": 1, **config.pop("batch")}
    config.update(batches=[batch], scratch=str(scratch), anchor_retention=integration.retention.SPECIFICATION,
        retention_backward_sha256=integration.retention.BACKWARD_SOURCE_SHA256,
        retention_storage=integration.layout(scratch), parent_checkpoint=str(probe.PARENT),
        input_sha256={str(probe.PARENT): probe.PARENT_SHA256})
    probe.write_json(tmp_path / "config.json", config)
    row = {"step": 1, "attempt_kind": "optimizer_attempt", "retention_manifest": str(target / "manifest.json"),
           "retention_manifest_sha256": probe.sha256(target / "manifest.json")}
    (tmp_path / "training.jsonl").write_text(json.dumps(row) + "\n")
    training = SimpleNamespace(OUTPUT=tmp_path, SCRATCH=scratch, STEPS=1, read=probe.read)
    counts = {"admitted_steps": [1], "training_tasks": 28}
    reference = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    monkeypatch.setattr(integration.retention, "verify_model", lambda model, identity:
        {**identity, "state_tensors": 254, "bitwise_equal": True})
    prediction = torch.tensor([[[0., 1.], [2., 3.], [4., 5.]]]) + .1
    calls = []

    def predict(model, values, criterion, **options):
        assert model is reference and not torch.is_grad_enabled()
        calls.append(values.clone())
        return prediction.clone()

    monkeypatch.setattr(integration.retention, "streamed_point_predictions", predict)
    backend = probe.training.alignment.mixture.backend
    return training, config, counts, reference, backend, calls


def replay_all(state):
    for path in state.tasks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state.check(Path(path), payload)


def test_full_audit_replays_every_eligible_grid_without_changing_original_teachers(tmp_path, monkeypatch):
    training, config, counts, reference, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, reference, None, backend)
    original = {path: probe.sha256(Path(path)) for path in state.tasks}
    replay_all(state)
    hashes = state.finish()
    assert len(calls) == 2 and state.reference_grids == 2 and len(state.seen) == 28
    assert all(probe.sha256(Path(path)) == digest for path, digest in original.items())
    audit = {"artifact_sha256": hashes, "admission_attempts": counts}
    report = integration.require_report(training, audit)
    assert report["eligibility_decisions"] == 28 and report["admitted_batches"] == 1
    assert report["reference_grids"] == 2 and report["training_performed"] is False
    assert report["historical_full_autograd_replayed"] is False and report["no_inference_ensemble"] is True
    with pytest.raises(ValueError, match="exactly one"):
        state.finish()


def test_data_only_rejection_has_no_reference_inference_or_fabricated_grid(tmp_path, monkeypatch):
    training, config, counts, reference, backend, calls = fixture(tmp_path, monkeypatch)
    directory = integration.directory(training.SCRATCH, 1)
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()
    (training.OUTPUT / "training.jsonl").write_text(json.dumps({"step": 1, "attempt_kind": "data_admission_rejected"}) + "\n")
    counts = {"admitted_steps": [], "training_tasks": 0}
    state = integration.RetentionReplay(training, config, counts, reference, None, backend)
    hashes = state.finish()
    report = integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
    assert not calls and report["reference_grids"] == 0 and report["eligibility_decisions"] == 0
    assert report["admitted_batches"] == 0 and state.rows_path.read_text() == ""


@pytest.mark.parametrize("mutation", ["drop_manifest", "wrong_reference", "omit_decision", "record_hash", "configuration_steps", "target_extra"])
def test_audit_entry_rejects_incomplete_or_mismatched_training_evidence(tmp_path, monkeypatch, mutation):
    training, config, counts, reference, backend, calls = fixture(tmp_path, monkeypatch)
    manifest_path = integration.directory(training.SCRATCH, 1) / "manifest.json"
    if mutation == "drop_manifest":
        manifest_path.unlink()
    elif mutation == "wrong_reference":
        config["anchor_retention"] = {**config["anchor_retention"], "weight": 1.}
    elif mutation == "omit_decision":
        manifest = probe.read(manifest_path)
        manifest["records"].pop()
        probe.write_json(manifest_path, manifest)
        row = json.loads((training.OUTPUT / "training.jsonl").read_text())
        row["retention_manifest_sha256"] = probe.sha256(manifest_path)
        (training.OUTPUT / "training.jsonl").write_text(json.dumps(row) + "\n")
    elif mutation == "record_hash":
        manifest = probe.read(manifest_path)
        manifest["complete"] = False
        probe.write_json(manifest_path, manifest)
    elif mutation == "configuration_steps":
        config["batches"].append(deepcopy(config["batches"][0]))
    else:
        integration.directory(training.SCRATCH, 2).mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        integration.RetentionReplay(training, config, counts, reference, None, backend)
    assert not calls


@pytest.mark.parametrize("mutation", ["eligibility", "source", "mask", "prediction", "repeat", "omit", "trace"])
def test_runtime_audit_never_omits_or_repairs_bad_reference_evidence(tmp_path, monkeypatch, mutation):
    training, config, counts, reference, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, reference, None, backend)
    path = Path(next(iter(state.tasks)))
    payload = torch.load(path, weights_only=False)
    if mutation == "eligibility":
        state.tasks[str(path)][1]["eligible"] = False
    elif mutation in ("source", "mask"):
        if mutation == "source":
            payload["episode"]["source"][0, 0, 1] += 1
        else:
            payload["episode"]["mask"][0, 0, 0] = False
    elif mutation == "prediction":
        monkeypatch.setattr(integration.retention, "streamed_point_predictions",
                            lambda *args, **kwargs: torch.zeros_like(payload["teacher"]))
    elif mutation == "repeat":
        state.check(path, payload)
    elif mutation == "omit":
        with pytest.raises(ValueError, match="complete"):
            state.finish()
        return
    else:
        replay_all(state)
        (training.OUTPUT / "training.jsonl").write_text("changed")
        with pytest.raises(ValueError):
            state.finish()
        return
    with pytest.raises(ValueError):
        state.check(path, payload)
    assert not state.summary_path.exists()


@pytest.mark.parametrize("mutation", ["unbound", "wrong_count", "omitted_row", "wrong_reference", "ensemble", "gradient_claim"])
def test_qualification_cannot_bypass_or_weaken_added_reference_audit(tmp_path, monkeypatch, mutation):
    training, config, counts, reference, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, reference, None, backend)
    replay_all(state)
    hashes = state.finish()
    audit = {"artifact_sha256": hashes, "admission_attempts": counts}
    report = probe.read(state.summary_path)
    if mutation == "unbound":
        hashes[str(state.summary_path)] = "wrong"
    elif mutation == "omitted_row":
        rows = state.rows_path.read_text().splitlines()
        state.rows_path.write_text("\n".join(rows[:-1]) + "\n")
        hashes[str(state.rows_path)] = probe.sha256(state.rows_path)
    else:
        key, value = {"wrong_count": ("reference_grids", 1), "wrong_reference": ("reference", {}),
                      "ensemble": ("no_inference_ensemble", False), "gradient_claim": ("historical_full_autograd_replayed", True)}[mutation]
        report[key] = value
        probe.write_json(state.summary_path, report)
        hashes[str(state.summary_path)] = probe.sha256(state.summary_path)
    with pytest.raises(ValueError):
        integration.require_report(training, audit)
