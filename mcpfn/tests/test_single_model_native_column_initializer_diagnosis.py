from copy import deepcopy
import inspect

import pytest
import torch

import diagnose_single_model_native_column_initializer as diagnosis
import recover_single_model_stronger_mean_cuda_entry as globals_check


def evidence():
    gate = {"passed": False, "cases": 284, "selected_step": 10,
        "checkpoint_sha256": diagnosis.SELECTED_SHA256, "real_benchmark_success": False,
        "confirmation_evaluated": True, "training_audit_sha256": diagnosis.TRAINING_AUDIT_SHA256,
        "replay_audit_sha256": diagnosis.REPLAY_AUDIT_SHA256, "audit_sha256": diagnosis.CONFIRMATION_AUDIT_SHA256}
    audit = {"complete": True, "prediction_grids": 1136, "cases": 284, "cuda_initialized": False}
    closure = {"complete": True, "gate": gate, "campaign": None, "review": None,
        "no_inference_ensemble": True, "goal_success_not_assessed": True,
        "training_audit_sha256": diagnosis.TRAINING_AUDIT_SHA256}
    training = {"complete": True, "complete_training_audit": True, "through_step": 64,
        "attempted_batches": 64, "accepted_updates": 56, "training_episodes": 1792,
        "full_network_training": True, "full_optimizer_trajectory_replayed": False}
    replay_gate = {"endpoint_reproduced": True, "selected_validation_passed": True,
        "selected_step": 10, "audit_sha256": diagnosis.REPLAY_AUDIT_SHA256}
    replay_audit = {"complete": True, "endpoint_grids": 56, "selected_grids": 152, "cuda_initialized": False}
    return gate, audit, closure, training, replay_gate, replay_audit


def cases():
    return [{"index": index, "seed": 1000 + index, "rows": 128, "columns": 96,
        "targets": 300, "domain": "imputation", "family": "mcar", "structured": True} for index in range(284)]


def test_reused_science_is_unchanged_and_bound_to_v48():
    for name in ("require_storage", "verify", "require_full_case_identity", "require_frozen_state", "audit"):
        function = getattr(diagnosis, name)
        assert function.__code__ is getattr(diagnosis.blueprint, name).__code__
        assert function.__globals__ is vars(diagnosis)
    assert diagnosis.run.__wrapped__.__code__ is diagnosis.blueprint.run.__wrapped__.__code__
    assert diagnosis.run.__wrapped__.__globals__ is vars(diagnosis)
    assert diagnosis.main.__code__ is diagnosis.predecessor.main.__code__
    assert diagnosis.training.OUTPUT.name == "native_column_mean_v48"
    assert diagnosis.original.CONFIRMATION.name == "native_column_mean_v48_confirmation"
    assert diagnosis.WRITE_BUDGETS == diagnosis.predecessor.WRITE_BUDGETS


def test_every_cloned_global_is_resolved():
    for value in vars(diagnosis).values():
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(diagnosis):
            assert globals_check.unresolved_globals(value) == []


def test_complete_negative_evidence_is_accepted():
    diagnosis.validate_closed_evidence(*evidence())


@pytest.mark.parametrize("report,field,value", [
    (0, "passed", True), (0, "cases", 283), (0, "selected_step", 64),
    (0, "checkpoint_sha256", "changed"), (0, "real_benchmark_success", True),
    (0, "confirmation_evaluated", False), (0, "audit_sha256", "changed"),
    (1, "prediction_grids", 1135), (1, "cuda_initialized", True),
    (2, "campaign", {}), (2, "review", {}), (2, "no_inference_ensemble", False),
    (2, "training_audit_sha256", "changed"), (2, "goal_success_not_assessed", False),
    (3, "through_step", 50), (3, "attempted_batches", 63), (3, "accepted_updates", 55),
    (3, "training_episodes", 1791), (3, "full_network_training", False),
    (3, "full_optimizer_trajectory_replayed", True),
    (4, "endpoint_reproduced", False), (4, "selected_step", 25),
    (5, "endpoint_grids", 55), (5, "selected_grids", 151), (5, "cuda_initialized", True),
])
def test_changed_or_partial_evidence_is_rejected(report, field, value):
    reports = evidence()
    reports[report][field] = value
    with pytest.raises(ValueError, match="negative closure"):
        diagnosis.validate_closed_evidence(*reports)


@pytest.mark.parametrize("mutation", [None, "drop", "duplicate", "seed", "shape", "queries"])
def test_full_original_case_identity_is_required(mutation):
    reference, current = cases(), cases()
    if mutation == "drop":
        current.pop()
    elif mutation == "duplicate":
        current[20] = deepcopy(current[19])
    elif mutation in {"seed", "shape", "queries"}:
        field = {"seed": "seed", "shape": "columns", "queries": "targets"}[mutation]
        current[20][field] += 1
    if mutation is None:
        diagnosis.require_full_case_identity(current, reference)
    else:
        with pytest.raises(ValueError):
            diagnosis.require_full_case_identity(current, reference)


def test_frozen_buffers_and_parameters_cannot_change():
    model = torch.nn.Linear(2, 3)
    model.register_buffer("fixed", torch.ones(1))
    state = deepcopy(model.state_dict())
    diagnosis.require_frozen_state(model, state)
    model.fixed.add_(1)
    with pytest.raises(ValueError, match="bitwise unchanged"):
        diagnosis.require_frozen_state(model, state)


def test_inference_requires_verified_registration_and_inference_mode(monkeypatch):
    def stop():
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        raise RuntimeError("verified diagnostic entry")

    monkeypatch.setattr(diagnosis, "verify", stop)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="verified diagnostic entry"):
        diagnosis.run()


def test_all32_comparisons_preserve_both_regression_directions(monkeypatch):
    groups = ["imputation"] * 4 + ["recsys"] * 2 + ["structured_overall"] * 4 + [f"family_{index}" for index in range(12)]
    monkeypatch.setattr(diagnosis.fixed, "retention_values", lambda metrics, base:
        [(group, f"metric_{index}", metrics + index) for index, group in enumerate(groups)])
    monkeypatch.setattr(diagnosis.original, "read", lambda path: {"general": 1.5})
    adverse = diagnosis.comparisons(2., 3., 1., {})
    improved = diagnosis.comparisons(2., 1., 1.5, {})
    assert len(adverse) == len(improved) == 32
    assert sum(row["reference"] == "V8" for row in adverse) == 22
    assert sum(row["reference"] == "general" for row in adverse) == 10
    assert all(row["V48_minus_V29"] == 1. and row["candidate_failed"] for row in adverse)
    assert all(row["V48_minus_V29"] == -1. and not row["candidate_failed"] for row in improved)


@pytest.fixture
def closed(tmp_path, monkeypatch):
    for name in ("OUTPUT", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        path = tmp_path / name.lower()
        monkeypatch.setattr(diagnosis.original, name, path)
        if name in {"OUTPUT", "CONFIRMATION"}:
            path.mkdir()
    training = tmp_path / "training"
    training.mkdir()
    replay = tmp_path / "replay"
    replay.mkdir()
    monkeypatch.setattr(diagnosis.training, "OUTPUT", training)
    monkeypatch.setattr(diagnosis, "REPLAY", replay)
    monkeypatch.setattr(diagnosis, "CLOSURE", diagnosis.original.OUTPUT / "result.json")
    monkeypatch.setattr(diagnosis, "TRAINING_AUDIT", tmp_path / "training_audit.json")
    monkeypatch.setattr(diagnosis.original, "verify", lambda output: None)
    monkeypatch.setattr(diagnosis.original, "process_identity", lambda pid: None)
    monkeypatch.setattr(diagnosis, "verify_hashes", lambda mapping: None)
    reports = evidence()
    paths = (diagnosis.original.CONFIRMATION / "gate.json", diagnosis.original.CONFIRMATION / "audit.json",
        diagnosis.CLOSURE, diagnosis.TRAINING_AUDIT, replay / "gate.json", replay / "audit.json")
    for path, report in zip(paths, reports):
        diagnosis.write_json(path, report)
    for directory in (diagnosis.original.OUTPUT, diagnosis.original.CONFIRMATION, training, replay):
        diagnosis.write_json(directory / "status.json", {"state": "training_complete" if directory == training else "complete",
            "pid": 101, "confirmed": False, "passed": False})
    return tmp_path


def test_closed_verification_is_cuda_hidden_cpu_only(closed, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("No launcher check in CPU verification"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA initialization in verification"))
    diagnosis.require_closed_failure()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", ["live", "unfinished", "confirmed", "publication", "campaign", "changed_gate"])
def test_original_producers_must_be_stopped_and_unpublished(closed, monkeypatch, mutation):
    if mutation == "live":
        monkeypatch.setattr(diagnosis.original, "process_identity", lambda pid: {"pid": pid})
    elif mutation == "publication":
        (diagnosis.training.OUTPUT / "best_validation_checkpoint.pth").touch()
    elif mutation == "campaign":
        diagnosis.original.CAMPAIGN.mkdir()
    else:
        path = diagnosis.original.OUTPUT / "status.json"
        if mutation == "changed_gate":
            path = diagnosis.original.CONFIRMATION / "gate.json"
        report = diagnosis.original.read(path)
        field, value = {"unfinished": ("state", "confirm"), "confirmed": ("confirmed", True),
            "changed_gate": ("passed", True)}[mutation]
        report[field] = value
        diagnosis.write_json(path, report)
    with pytest.raises(ValueError):
        diagnosis.require_closed_failure()
