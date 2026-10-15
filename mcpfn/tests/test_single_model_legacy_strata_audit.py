from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import audit_single_model_legacy_strata as audit


def replay_fixture(tmp_path, monkeypatch, eligible):
    batch = {"step": 1, "tasks": [{"index": 0, "specification": {"domain": "imputation"}}]}
    saved = {"step": 1, "eligible": eligible, "task_count": 1, "structured_draws": 1,
             "tasks": [{"task": task} for task in batch["tasks"]]}
    path = tmp_path / "data_admission/step_001.json"
    audit.write_json(path, saved)
    kind = "optimizer_attempt" if eligible else "data_admission_rejected"
    record = {"step": 1, "accepted": False, "accepted_updates": 0, "attempt_kind": kind, "reason": kind,
              "admission_manifest": str(path), "admission_sha256": audit.sha256(path), "seconds": 1., "peak_gib": 0.}
    consumed = []
    checker = SimpleNamespace(eligible=eligible, record=deepcopy(saved), assert_consumed=lambda: consumed.append(True))
    monkeypatch.setattr(audit.admission, "TrainingBatchAdmission", lambda *args: checker)
    active = {"original_sampler": object(), "sampler": object(), "admission_steps": [], "structured_draws": 0}
    return batch, record, checker, active, consumed


@pytest.mark.parametrize("eligible", [False, True])
def test_each_scheduled_admission_is_replayed_including_data_rejection(tmp_path, monkeypatch, eligible):
    batch, record, checker, active, consumed = replay_fixture(tmp_path, monkeypatch, eligible)
    token = audit.REPLAY.set(active)
    hashes = {}
    try:
        assert audit.replay_admission(tmp_path, batch, record, 0, hashes) is checker
        assert active["sampler"] is checker and active["admission_steps"] == [1]
        assert active["structured_draws"] == 1
        assert len(hashes) == 1 and consumed == ([] if eligible else [True])
    finally:
        audit.REPLAY.reset(token)


@pytest.mark.parametrize("mutation", [None, "terminal_as_actual", "model", "record"])
def test_terminal_optimizer_audit_uses_last_real_attempt_and_exact_final_state(tmp_path, monkeypatch, mutation):
    batches = [{"step": step, "tasks": []} for step in range(1, 5)]
    manifest_path = tmp_path / "training_batches/step_003/manifest.json"
    entries = [{"task": "actual", "kind": "composite", "index": 0}]
    audit.write_json(manifest_path, {"entries": entries})
    record = {"step": 3, "accepted_updates": 1, "task_manifest": str(manifest_path)}
    rows = [{"step": step} if step != 3 else record for step in range(1, 5)]
    config = {"batches": batches, "base": {"weight_decay": .01}}
    audit.write_json(tmp_path / "config.json", config)
    state = {"weight": torch.tensor([1.])}
    payload = {"config_sha256": audit.sha256(tmp_path / "config.json"), "batch": batches[2], "record": deepcopy(record),
               "entries": entries, "task_manifest": str(manifest_path), "snapshot": {"after": state["weight"].clone()}}
    if mutation == "record":
        payload["record"]["step"] = 4
    torch.save(payload, tmp_path / "raw_final_update.pt")
    descriptor = {"terminal_step": 4, "optimizer_attempt_step": 3, "accepted_updates": 1,
                  "raw_endpoint_sha256": audit.sha256(tmp_path / "raw_final_update.pt"),
                  "config_sha256": audit.sha256(tmp_path / "config.json")}
    if mutation == "terminal_as_actual":
        descriptor["optimizer_attempt_step"] = 4
    elif mutation == "model":
        state["weight"] += 1
    audit.write_json(tmp_path / "endpoint.json", descriptor)
    counts = {"scheduled_attempts": 4, "last_optimizer_attempt": 3, "accepted_updates": 1}
    checked = []

    def endpoint_check(payload, observed, sizes, decay, rollback_rate, optimizer):
        assert observed is record and sizes == [1] and decay == .01 and rollback_rate == 5e-6
        checked.append(True)
        return {"accepted_updates": 1}

    monkeypatch.setattr(audit.training, "audit_endpoint", lambda *args: {})
    monkeypatch.setattr(audit.checks, "check_endpoint", endpoint_check)
    arguments = (tmp_path, config, rows, counts, {"model": state, "optimizer": {}}, {},
                 [["weight", [1], 1]], {3: 5e-6}, {})
    if mutation is not None:
        with pytest.raises(ValueError):
            audit.audit_final_endpoint(*arguments)
    else:
        result = audit.audit_final_endpoint(*arguments)
        assert result == {"accepted_updates": 1, "optimizer_attempt_step": 3, "terminal_step": 4}
        assert checked == [True]


@pytest.mark.parametrize("mutation", ["mask", "hash", "path", "fabricated_task", "fabricated_metric"])
def test_data_noop_audit_rejects_altered_or_fabricated_evidence(tmp_path, monkeypatch, mutation):
    batch, record, checker, active, consumed = replay_fixture(tmp_path, monkeypatch, False)
    if mutation == "mask":
        checker.record["eligible"] = True
    elif mutation == "hash":
        record["admission_sha256"] = "wrong"
    elif mutation == "path":
        record["admission_manifest"] = "wrong"
    elif mutation == "fabricated_task":
        (tmp_path / "training_batches/step_001").mkdir(parents=True)
    else:
        record["point_after"] = {}
    token = audit.REPLAY.set(active)
    try:
        with pytest.raises(ValueError):
            audit.replay_admission(tmp_path, batch, record, 0, {})
    finally:
        audit.REPLAY.reset(token)
