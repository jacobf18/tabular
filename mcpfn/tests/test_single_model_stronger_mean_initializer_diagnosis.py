from copy import deepcopy
import inspect

import pytest
import torch

import diagnose_single_model_stronger_mean_initializer as diagnosis
import recover_single_model_stronger_mean_cuda_entry as globals_check


def cases():
    return [{"index": index, "seed": 1000 + index, "rows": 128, "columns": 96,
             "targets": 300, "domain": "imputation", "family": "mcar", "structured": True} for index in range(284)]


def test_reused_science_is_unchanged_and_bound_to_actual_v46():
    for name in ("require_storage", "verify", "require_full_case_identity", "require_frozen_state", "audit"):
        function = getattr(diagnosis, name)
        assert function.__code__ is getattr(diagnosis.blueprint, name).__code__
        assert function.__globals__ is vars(diagnosis)
    assert diagnosis.run.__wrapped__.__code__ is diagnosis.blueprint.run.__wrapped__.__code__
    assert diagnosis.run.__wrapped__.__globals__ is vars(diagnosis)
    assert diagnosis.training.OUTPUT.name == "stronger_mean_distillation_v46"
    assert diagnosis.original.CONFIRMATION.name == "stronger_mean_distillation_v46_confirmation"
    assert diagnosis.PARENT_SHA256 == diagnosis.blueprint.PARENT_SHA256
    assert diagnosis.WRITE_BUDGETS == diagnosis.blueprint.WRITE_BUDGETS


def test_every_actual_cloned_global_is_resolved():
    for value in vars(diagnosis).values():
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(diagnosis):
            assert globals_check.unresolved_globals(value) == []


@pytest.mark.parametrize("mutation", [None, "drop", "duplicate", "seed", "shape", "query_population"])
def test_all_original_contexts_and_query_populations_are_required(mutation):
    reference, current = cases(), cases()
    if mutation == "drop":
        current.pop()
    elif mutation == "duplicate":
        current[20] = deepcopy(current[19])
    elif mutation == "seed":
        current[20]["seed"] += 1
    elif mutation == "shape":
        current[20]["columns"] += 1
    elif mutation == "query_population":
        current[20]["targets"] += 1
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


def test_initializer_entry_requires_registration_and_inference_mode(monkeypatch):
    def stop():
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        raise RuntimeError("verified diagnostic entry")

    monkeypatch.setattr(diagnosis, "verify", stop)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="verified diagnostic entry"):
        diagnosis.run()


def test_all32_comparisons_include_general_reference_and_adverse_directions(monkeypatch):
    groups = (["imputation"] * 4 + ["recsys"] * 2 + ["structured_overall"] * 4
              + [f"family_{index}" for index in range(12)])
    monkeypatch.setattr(diagnosis.fixed, "retention_values", lambda metrics, base:
                        [(group, f"metric_{index}", metrics + index) for index, group in enumerate(groups)])
    monkeypatch.setattr(diagnosis.original, "read", lambda path: {"general": 1.5})
    rows = diagnosis.comparisons(2., 3., 1., {})
    assert len(rows) == 32
    assert sum(row["reference"] == "V8" for row in rows) == 22
    assert sum(row["reference"] == "general" for row in rows) == 10
    assert all(row["V46_minus_V29"] == 1. for row in rows)
    assert all(row["initializer_failed"] and row["candidate_failed"] for row in rows)
    assert all(row["V46_minus_reference"] == (2. if row["reference"] == "V8" else 1.5) for row in rows)


def test_incomplete_retention_population_is_rejected(monkeypatch):
    monkeypatch.setattr(diagnosis.fixed, "retention_values", lambda metrics, base: [("mcar", "common_nrmse", 1.)])
    monkeypatch.setattr(diagnosis.original, "read", lambda path: {"general": {}})
    with pytest.raises(ValueError, match="original22"):
        diagnosis.comparisons({}, {}, {}, {})


@pytest.fixture
def closed(tmp_path, monkeypatch):
    for name in ("OUTPUT", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        path = tmp_path / name.lower()
        monkeypatch.setattr(diagnosis.original, name, path)
        if name in {"OUTPUT", "CONFIRMATION"}:
            path.mkdir()
    training = tmp_path / "training"
    training.mkdir()
    monkeypatch.setattr(diagnosis.training, "OUTPUT", training)
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    monkeypatch.setattr(diagnosis, "RECOVERIES", (recovery,))
    monkeypatch.setattr(diagnosis, "CLOSURE", recovery / "result.json")
    monkeypatch.setattr(diagnosis.original, "verify", lambda output: None)
    monkeypatch.setattr(diagnosis.original, "process_identity", lambda pid: None)
    monkeypatch.setattr(diagnosis, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(diagnosis, "sha256", lambda path: "digest")
    diagnosis.write_json(recovery / "config.json", {"source_sha256": {}, "input_sha256": {}})
    diagnosis.write_json(recovery / "registration.json", {"config_sha256": "digest"})
    gate = {"passed": False, "cases": 284, "selected_step": 10}
    diagnosis.write_json(diagnosis.original.CONFIRMATION / "gate.json", gate)
    diagnosis.write_json(diagnosis.original.CONFIRMATION / "audit.json", {
        "complete": True, "prediction_grids": 1136, "cases": 284, "cuda_initialized": False})
    diagnosis.write_json(diagnosis.CLOSURE, {"complete": True, "gate": gate, "training_repeated": False,
                        "scientific_stages_repeated": False, "cuda_initialized": False})
    for directory in (recovery, diagnosis.original.CONFIRMATION, training):
        diagnosis.write_json(directory / "status.json", {"state": "training_complete" if directory == training else "complete", "pid": 101})
    return tmp_path


def test_closed_receipt_verification_works_with_cuda_hidden_without_initialization(closed, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("No launcher hardware check in CPU audit"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("Never initialize CUDA while verifying receipts"))
    diagnosis.require_closed_failure()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", ["positive_gate", "partial", "repeated", "live", "publication", "registration"])
def test_only_complete_unchanged_negative_closure_is_admitted(closed, monkeypatch, mutation):
    if mutation == "live":
        monkeypatch.setattr(diagnosis.original, "process_identity", lambda pid: {"pid": pid})
    elif mutation == "publication":
        (diagnosis.training.OUTPUT / "best_validation_checkpoint.pth").touch()
    elif mutation == "registration":
        diagnosis.write_json(diagnosis.CLOSURE.parent / "registration.json", {"config_sha256": "changed"})
    else:
        path = diagnosis.original.CONFIRMATION / ("gate.json" if mutation == "positive_gate" else "audit.json")
        if mutation == "repeated":
            path = diagnosis.CLOSURE
        report = diagnosis.original.read(path)
        field, value = {"positive_gate": ("passed", True), "partial": ("prediction_grids", 1135),
                        "repeated": ("scientific_stages_repeated", True)}[mutation]
        report[field] = value
        diagnosis.write_json(path, report)
    with pytest.raises(ValueError):
        diagnosis.require_closed_failure()
