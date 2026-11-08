import pytest
import torch

import relocate_single_model_balanced_native_long_capacity as capacity
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def payload(kind):
    updates = capacity.EXPECTED[kind]["gradient_updates_performed"]
    return {**capacity.EXPECTED[kind], "step": 64, "epoch": 64, "attempted_batches": 64,
        "parent_training_updates": 98, "parent_checkpoint_sha256": capacity.SELECTED_SHA256,
        "independent_confirmation_passed": False, "preflight_only": False, "full_network_training": True,
        "model": {"matrix": torch.ones(3)},
        "optimizer": {"state": {0: {"step": torch.tensor(float(updates)), "exp_avg": torch.zeros(3),
            "exp_avg_sq": torch.ones(3)}}, "param_groups": [{"params": [0]}]},
        "rng": {"cpu": torch.zeros(5, dtype=torch.uint8), "cuda": [torch.ones(5, dtype=torch.uint8)]}}


def validate(current, kind):
    selected = {"matrix": torch.ones(3) if kind == "v55_tail" else torch.zeros(3)}
    capacity.validate_payload(current, kind, model_tensors=1, optimizer_tensors=1, selected_model=selected)


@pytest.mark.parametrize("kind", ["v54_tail", "v55_tail"])
@pytest.mark.parametrize("field,value", [(None, None), ("best_step", 0), ("step", 256), ("epoch", 50),
    ("attempted_batches", 256), ("gradient_updates_performed", 1), ("parent_training_updates", 0),
    ("parent_checkpoint_sha256", "wrong"), ("config_sha256", "wrong"),
    ("independent_confirmation_passed", True), ("preflight_only", True), ("full_network_training", False)])
def test_only_exact_original_unqualified_tails_are_admitted(kind, field, value):
    current = payload(kind)
    if field is None:
        validate(current, kind)
    else:
        current[field] = value
        with pytest.raises(ValueError):
            validate(current, kind)


@pytest.mark.parametrize("kind", ["v54_tail", "v55_tail"])
def test_selected_tail_relationship_is_run_specific(kind):
    current = payload(kind)
    selected = {"matrix": torch.zeros(3) if kind == "v55_tail" else torch.ones(3)}
    with pytest.raises(ValueError, match="relationship"):
        capacity.validate_payload(current, kind, model_tensors=1, optimizer_tensors=1, selected_model=selected)


@pytest.mark.parametrize("mutation", ["model", "optimizer", "moment", "nan", "step", "parameter", "rng", "cuda"])
def test_incomplete_or_nonfinite_payloads_are_refused(mutation):
    current = payload("v55_tail")
    if mutation == "model":
        current["model"]["matrix"][0] = float("nan")
    elif mutation == "optimizer":
        current["optimizer"]["state"].clear()
    elif mutation == "moment":
        del current["optimizer"]["state"][0]["exp_avg_sq"]
    elif mutation == "nan":
        current["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    elif mutation == "step":
        current["optimizer"]["state"][0]["step"] += 1
    elif mutation == "parameter":
        current["optimizer"]["param_groups"][0]["params"] = [1]
    elif mutation == "rng":
        current["rng"]["cpu"] = torch.zeros(5)
    else:
        current["rng"]["cuda"] = []
    with pytest.raises(ValueError):
        validate(current, "v55_tail")


def test_exact_scope_and_protected_paths():
    assert capacity.TARGETS == {"standardized_teacher_huber_v54/latest_checkpoint.pth": "v54_tail",
        "balanced_native_quarter_v55/latest_checkpoint.pth": "v55_tail"}
    assert capacity.EXPECTED_BYTES == 1872586582
    assert capacity.TRAINING_ADMISSION_BYTES == 22 * 1024**3
    assert len(capacity.PROTECTED) == 3
    assert all(path.endswith("selected_checkpoint.pth") for path in capacity.PROTECTED)
    assert capacity.SELECTED.parent.name == "legacy_strata_v29"


@pytest.mark.parametrize("pid,live", [(1234, False), (1234, True), (None, False), (True, False),
    (0, False), (-1, False), ("1234", False)])
def test_stopped_owners_are_required(monkeypatch, tmp_path, pid, live):
    monkeypatch.setattr(capacity, "ROOT", tmp_path)
    monkeypatch.setattr(capacity, "read", lambda path: {"state": capacity.STATES[path.parent.name], "pid": pid})
    monkeypatch.setattr(capacity.diagnosis.original, "process_identity", lambda value: {} if live else None)
    if type(pid) is int and pid > 0 and not live:
        capacity.require_stopped()
    else:
        with pytest.raises(ValueError):
            capacity.require_stopped()


def test_owner_locks_are_checked(monkeypatch, tmp_path):
    monkeypatch.setattr(capacity, "ROOT", tmp_path)
    monkeypatch.setattr(capacity, "STATES", {"owner": "complete"})
    directory = tmp_path / "owner"
    directory.mkdir()
    (directory / "run.lock").touch()
    monkeypatch.setattr(capacity, "read", lambda path: {"state": "complete", "pid": 1234})
    monkeypatch.setattr(capacity.diagnosis.original, "process_identity", lambda value: None)
    with (directory / "run.lock").open("rb") as stream:
        capacity.fcntl.flock(stream, capacity.fcntl.LOCK_EX | capacity.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            capacity.require_stopped()
    capacity.require_stopped()


def test_copy_preserves_exact_bytes_and_refuses_overwrite(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"complete original tail")
    row = {"source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
        "sha256": capacity.sha256(source), "kind": "v55_tail"}
    checked = []
    monkeypatch.setattr(capacity, "validate_file", lambda path, kind: checked.append((path, kind)))
    capacity.copy_and_verify(row)
    assert destination.read_bytes() == source.read_bytes()
    assert checked == [(destination, "v55_tail")]
    with pytest.raises(FileExistsError):
        capacity.copy_and_verify(row)
    row.update(destination=str(tmp_path / "bad"), sha256="wrong")
    with pytest.raises(ValueError, match="inexact"):
        capacity.copy_and_verify(row)


def test_negative_closures_precede_evidence_collection(monkeypatch):
    calls = []
    monkeypatch.setattr(capacity.previous_diagnosis, "require_closed_failure", lambda: calls.append("v54"))
    def reject():
        calls.append("v55")
        raise ValueError("unclosed")
    monkeypatch.setattr(capacity.diagnosis, "require_closed_failure", reject)
    with pytest.raises(ValueError, match="unclosed"):
        capacity.evidence()
    assert calls == ["v54", "v55"]


def test_helpers_bind_only_the_new_scope():
    for name in ("read", "status", "inventory", "copy_and_verify", "recheck", "preserve", "require_completed"):
        function = getattr(capacity, name)
        assert function.__globals__ is vars(capacity) and function.__closure__ is None
        assert not unresolved_globals(function)
    assert "test_single_model_balanced_native_long_capacity.py" in capacity.PRESERVE_SOURCE
    assert "canonical_artifacts_checked': 2" in capacity.RECHECK_SOURCE
