from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import audit_single_model_four_draw as audit
import single_model_four_draw_transfer_audit as transfers
import test_single_model_legacy_strata_audit as admission_tests
import test_single_model_legacy_blend_teacher_audit as component_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from recsys_evaluation_common import write_json, sha256


@pytest.mark.parametrize("eligible", [False, True])
def test_original_admission_replay(monkeypatch, tmp_path, eligible):
    monkeypatch.setattr(admission_tests, "audit", audit)
    admission_tests.test_each_scheduled_admission_is_replayed_including_data_rejection(tmp_path, monkeypatch, eligible)


@pytest.mark.parametrize("failure", [None, "check", "finish"])
def test_teacher_components_are_added_without_bypassing_original_checks(monkeypatch, failure):
    monkeypatch.setattr(component_tests, "audit", audit)
    component_tests.test_component_audit_is_additional_to_original_task_and_reference_replay(monkeypatch, failure)


def test_reference_and_component_layers_use_new_namespace(monkeypatch):
    calls = []
    payload = {"identity": "original"}
    monkeypatch.setattr(audit, "original_load_regenerated_task", lambda path: calls.append("original") or payload)
    monkeypatch.setattr(audit, "original_regeneration_evidence", lambda step: calls.append("original_finish") or {"original": "hash"})

    def layer(name):
        return SimpleNamespace(check=lambda path, current: calls.append(name) if current is payload else pytest.fail("payload changed"),
            finish=lambda: calls.append(name + "_finish") or {name: "hash"})

    token = audit.REPLAY.set({"anchor_retention": layer("retention"), "legacy_blend": layer("blend")})
    try:
        assert audit.load_regenerated_task("task") is payload
        assert audit.regeneration_evidence(128) == {"original": "hash", "retention": "hash", "blend": "hash"}
    finally:
        audit.REPLAY.reset(token)
    assert calls == ["original", "retention", "blend", "original_finish", "retention_finish", "blend_finish"]
    assert audit.REPLAY is not audit.blueprint.REPLAY


def test_all_full_audit_counts_and_callbacks_are_bound():
    assert audit.training.STEPS == 128
    assert audit.checks is audit.training.paired is audit.fixed_trial.protection
    assert audit.retention_training is audit.training.retention_training
    assert audit.blend_replay is audit.training.blend_replay
    assert audit.retention_training.expected_component_counts(128) == {"v8": 6144, "v29": 6656}
    assert 'len(manifest["tasks"]) != 56' in audit.ADAPTED_SOURCE
    assert '48 * through_step' in audit.REGENERATION_SOURCE and 'range(56)' in audit.REGENERATION_SOURCE
    assert 'transfer_audit.audit(training, config, rows)' in audit.ADAPTED_SOURCE
    assert '"gradient_transfers": transfers' in audit.ADAPTED_SOURCE
    assert 'len(decision["comparisons"]) != 22' in audit.ADAPTED_SOURCE
    for name, function in vars(audit).items():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(audit):
            assert not unresolved_globals(inspect.unwrap(function)), name


@pytest.mark.parametrize("mutation", [None, "omitted_task", "omitted_draw", "fabricated_task"])
def test_regeneration_requires_all56_tasks_and48_draws_per_scheduled_attempt(tmp_path, monkeypatch, mutation):
    directory = tmp_path / "training_batches/step_001"
    directory.mkdir(parents=True)
    expected = set()
    for index in range(56):
        path = directory / f"task_{index:02d}.pt"
        path.touch()
        expected.add(str(path))
    monkeypatch.setattr(audit.training, "OUTPUT", tmp_path)
    monkeypatch.setattr(audit.training, "read", lambda path: {})
    monkeypatch.setattr(audit, "read_prefix", lambda *args: ([], ""))
    monkeypatch.setattr(audit, "admission_inventory", lambda *args: {"admitted_steps": [1], "rating_contexts": 4})
    monkeypatch.setattr(audit, "sha256", lambda path: "digest")
    active = {"seen": expected, "rating_contexts": 4, "admission_steps": [1, 2], "structured_draws": 96, "output": tmp_path}
    if mutation == "omitted_task":
        active["seen"].pop()
    elif mutation == "omitted_draw":
        active["structured_draws"] = 48
    elif mutation == "fabricated_task":
        (directory / "extra.pt").touch()
    token = audit.REPLAY.set(active)
    try:
        if mutation:
            with pytest.raises(ValueError):
                audit.original_regeneration_evidence(2)
        else:
            assert len(audit.original_regeneration_evidence(2)) == 2
    finally:
        audit.REPLAY.reset(token)


@pytest.fixture
def transfer_fixture(tmp_path):
    parameters, rows = 7, 167
    training = SimpleNamespace(OUTPUT=tmp_path, TRAINABLE_PARAMETERS=parameters, read=audit.training.read,
        preflight=SimpleNamespace(HOST_WORKING_BYTES=176 * 1024**3))
    path = tmp_path / "gradient_transfers/step_001.json"
    write_json(path, transfers.expected_transfer(rows, parameters))
    record = {"step": 1, "attempt_kind": "optimizer_attempt", "gradient_transfer": str(path),
        "gradient_transfer_sha256": sha256(path), "protected_tasks": rows,
        "peak_gib": 88.75, "peak_reserved_gib": 125.8, "peak_host_rss_bytes": 152 * 1024**3,
        "capture_peak_allocated_bytes": 40 * 1024**3, "capture_peak_reserved_bytes": 80 * 1024**3,
        "cuda_total_bytes": 140 * 1024**3, "allocator_fraction": .9}
    return training, {"gradient_staging_policy": transfers.staging.POLICY}, [record, {"attempt_kind": "data_admission_rejected"}]


def test_exact_receipts_cover_only_actual_defined_prefixes(transfer_fixture):
    training, config, records = transfer_fixture
    report = transfers.audit(training, config, records)
    assert report["complete"] and report["optimizer_attempts"] == 1 and report["defined_gradient_rows"] == 167
    assert report["independent_neural_gradient_replay"] is False
    assert len(report["artifact_sha256"]) == 1


def test_entirely_data_rejected_run_has_no_fabricated_transfer(tmp_path):
    training = SimpleNamespace(OUTPUT=tmp_path)
    config = {"gradient_staging_policy": transfers.staging.POLICY}
    report = transfers.audit(training, config, [{"attempt_kind": "data_admission_rejected"} for step in range(128)])
    assert report["optimizer_attempts"] == report["defined_gradient_rows"] == 0
    assert report["artifact_sha256"] == {} and not (tmp_path / "gradient_transfers").exists()


@pytest.mark.parametrize("mutation", ["row_count", "path", "hash", "extra", "duplicate", "rejected_receipt", "policy", "capacity", "prefix"])
def test_receipt_inventory_cannot_be_fabricated_or_truncated(transfer_fixture, mutation):
    training, config, records = transfer_fixture
    path = Path(records[0]["gradient_transfer"])
    if mutation == "row_count":
        records[0]["protected_tasks"] -= 1
    elif mutation in ("path", "hash"):
        records[0]["gradient_transfer" if mutation == "path" else "gradient_transfer_sha256"] = "wrong"
    elif mutation == "extra":
        write_json(path.with_name("extra.json"), {})
    elif mutation == "duplicate":
        records.append(deepcopy(records[0]))
    elif mutation == "rejected_receipt":
        records[1]["gradient_transfer"] = str(path)
    elif mutation == "policy":
        config["gradient_staging_policy"] = "unregistered"
    else:
        receipt = training.read(path)
        receipt["capacity_rows" if mutation == "capacity" else "verified_elements"] -= 1
        write_json(path, receipt)
        records[0]["gradient_transfer_sha256"] = sha256(path)
    with pytest.raises(ValueError):
        transfers.audit(training, config, records)


@pytest.mark.parametrize("field,value", [("peak_gib", float("nan")), ("peak_reserved_gib", 127),
    ("peak_host_rss_bytes", 177 * 1024**3), ("capture_peak_allocated_bytes", 90 * 1024**3),
    ("capture_peak_reserved_bytes", 126 * 1024**3), ("cuda_total_bytes", True), ("allocator_fraction", .95)])
def test_original_memory_allowances_remain_enforced(transfer_fixture, field, value):
    training, config, records = transfer_fixture
    records[0][field] = value
    with pytest.raises(ValueError):
        transfers.audit(training, config, records)
