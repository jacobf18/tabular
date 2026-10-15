from copy import deepcopy
from types import SimpleNamespace

import pytest

import preflight_single_model_legacy_strata as preflight


def test_new_preflight_preserves_architecture_priors_teachers_and_held_out_scope():
    assert preflight.PARENT == preflight.template.PARENT
    assert preflight.PARENT_SHA256 == preflight.template.PARENT_SHA256
    assert preflight.PREFIXES == preflight.template.PREFIXES
    assert preflight.RATE == preflight.template.RATE == 5e-6
    assert preflight.teacher_policy is preflight.template.teacher_policy
    assert preflight.paired is not preflight.template.paired
    actual, previous = preflight.batch(), preflight.template.batch()
    for task, original in zip(actual["tasks"], previous["tasks"]):
        normalized = deepcopy(task)
        normalized["gradient_seed"] = original["gradient_seed"]
        normalized["specification"].get("structured", normalized["specification"])["seed"] = (
            original["specification"].get("structured", original["specification"])["seed"])
        assert normalized == original
    seeds = preflight.training.alignment.fixed.seeds_in(actual)
    assert len(seeds) == 56 and not seeds.intersection(preflight.training.alignment.fixed.seeds_in(previous))
    assert "paired.protected_update(" in preflight.ADAPTED_SOURCE
    assert "sampler.assert_consumed()" in preflight.ADAPTED_SOURCE
    assert "sampler = admitted_sampler" in preflight.ADAPTED_SOURCE
    assert "admission_evidence = audit_admission(output, config)" in preflight.ADAPTED_SOURCE
    assert '"selection_qualified": False' in preflight.ADAPTED_SOURCE
    assert preflight.run.__globals__["paired"] is preflight.paired
    assert preflight.audit_admitted.__globals__["paired"] is preflight.paired


def rejected_setup(tmp_path, monkeypatch):
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    config = {"base": {}, "batch": {}}
    record = {"eligible": False, "task_count": 28, "structured_draws": 24,
              "deferred_task_indices": [0, 13, 14, 27], "rejected_task_indices": [23]}
    consumed = []
    checker = SimpleNamespace(eligible=False, record=record, assert_consumed=lambda: consumed.append(True))
    monkeypatch.setattr(preflight, "StructuredSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(preflight.admission, "TrainingBatchAdmission", lambda *args: checker)
    monkeypatch.setattr(preflight, "verify", lambda *args: config)
    monkeypatch.setattr(preflight, "require_space", lambda: None)
    monkeypatch.setattr(preflight, "SCRATCH", scratch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(preflight.torch.cuda, "is_initialized", lambda: False)
    for name in ("config.json", "registration.json"):
        preflight.write_json(output / name, {})
    preflight.status(output, "registered")
    return output, config, consumed


def test_rejected_run_returns_before_gpu_wait_model_initialization_or_optimizer(tmp_path, monkeypatch):
    output, config, consumed = rejected_setup(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("A rejected batch must not wait for a GPU, initialize models or propose an optimizer step")

    monkeypatch.setattr(preflight, "wait_for_gpu", forbidden)
    monkeypatch.setattr(preflight.training.alignment.mixture.backend, "setup", forbidden)
    monkeypatch.setattr(preflight.paired, "protected_update", forbidden)
    preflight.run(output)
    assert consumed == [True]
    result = preflight.read(output / "result.json")
    assert result["data_admission_only"] and result["accepted_updates"] == 0
    assert "record" not in result and "checkpoint_sha256" not in result
    assert preflight.read(output / "status.json")["state"] == "awaiting_data_admission_audit"


def test_data_only_audit_has_no_fabricated_gradients_metrics_or_checkpoint(tmp_path, monkeypatch):
    output, config, consumed = rejected_setup(tmp_path, monkeypatch)
    preflight.run(output)
    result = preflight.read(output / "result.json")
    result["training_pid"] += 1
    preflight.write_json(output / "result.json", result)
    preflight.replay_targets(output)
    preflight.audit(output)
    audited = preflight.read(output / "audit.json")
    assert audited["complete"] and audited["data_admission_only"]
    assert audited["training_tasks"] == audited["prediction_grids"] == audited["accepted_updates"] == 0
    assert audited["data_admission"]["structured_draws"] == 24
    assert "raw_endpoint" not in audited and not (output / "preflight_checkpoint.pth").exists()
    gate = preflight.read(output / "gate.json")
    assert not gate["passed"] and gate["reason"] == "data_admission_rejected"
    assert preflight.read(output / "status.json")["state"] == "complete"
    assert consumed == [True, True]
    with pytest.raises(ValueError, match="once"):
        preflight.audit(output)


@pytest.mark.parametrize("mutation", ["hash", "same_process", "teacher", "checkpoint", "scratch", "cuda"])
def test_data_noop_audit_rejects_inconsistent_or_model_evidence(tmp_path, monkeypatch, mutation):
    output, config, consumed = rejected_setup(tmp_path, monkeypatch)
    preflight.run(output)
    result = preflight.read(output / "result.json")
    if mutation != "same_process":
        result["training_pid"] += 1
    if mutation == "hash":
        result["admission_sha256"] = "invalid"
    elif mutation == "teacher":
        (output / "teacher_replay").mkdir()
    elif mutation == "checkpoint":
        (output / "preflight_checkpoint.pth").touch()
    elif mutation == "scratch":
        (preflight.SCRATCH / "unexpected.pt").touch()
    elif mutation == "cuda":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    preflight.write_json(output / "result.json", result)
    with pytest.raises(ValueError):
        preflight.audit(output)
