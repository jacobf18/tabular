from copy import deepcopy

import pytest

import archive_single_model_continuation_evidence as evidence
import relocate_single_model_continuation_archives as relocation


def reports():
    return [{"state": "complete", "pid": 999999998},
        {"complete": True, "training_tasks": 28, "prediction_grids": 56,
         "raw_endpoint": {"gradient_vectors": 84, "accepted": False},
         "teacher_replay": {"complete": True, "full_grids": 28}},
        {"passed": False, "preflight_only": True, "audit_sha256": "expected"},
        {"state": "complete", "pid": 999999999},
        {"complete": True, "gate": {"passed": False, "confirmation_evaluated": True}, "campaign": None, "review": None},
        {"complete": True, "prediction_grids": 1136}, {"passed": False}]


def patch_reports(monkeypatch, current):
    iterator = iter(deepcopy(current))
    monkeypatch.setattr(evidence, "read", lambda path: next(iterator))
    monkeypatch.setattr(evidence, "sha256", lambda path: "expected")


def test_terminal_negative_evidence_only(monkeypatch):
    patch_reports(monkeypatch, reports())
    paths, audit, confirmation = evidence.completed_sources()
    assert len(paths) == 7 and audit["raw_endpoint"]["accepted"] is False
    assert confirmation["prediction_grids"] == 1136
    assert evidence.MINIMUM_FREE == evidence.capacity.MINIMUM_FREE == 18 * 1024**3
    assert 'len(actual) != 2' in evidence.ADAPTED_SOURCE


@pytest.mark.parametrize("index,field,value", [(0, "state", "running"), (0, "pid", 1),
    (1, "complete", False), (1, "prediction_grids", 55),
    (1, "raw_endpoint", {"gradient_vectors": 84, "accepted": True}),
    (1, "teacher_replay", {"complete": True, "full_grids": 27}), (2, "passed", True),
    (2, "audit_sha256", "changed"), (3, "pid", 1), (4, "campaign", "real"),
    (5, "prediction_grids", 1135), (6, "passed", True)])
def test_changed_or_incomplete_sources_rejected(monkeypatch, index, field, value):
    current = reports()
    current[index][field] = value
    patch_reports(monkeypatch, current)
    with pytest.raises(ValueError, match="complete rejected"):
        evidence.completed_sources()


def test_relocation_does_not_relax_actual_capacity_reserve():
    assert relocation.NAMES == ("paired_endpoint_archive_v1", "structured_teacher_endpoint_archive_v1")
    assert relocation.RESERVE == 6 * 1024**3
    assert 'shutil.disk_usage(OUTPUT.parent).free - sum(sizes) < RESERVE' in relocation.ADAPTED_SOURCE
    assert relocation.CAPACITY.name == "continuation_snapshot_capacity_v1"
    assert len(relocation.template.NAMES) == 3


@pytest.mark.parametrize("count,valid", [(18, True), (17, False)])
def test_capacity_requires_full_inventory_even_without_target(monkeypatch, count, valid):
    monkeypatch.setattr(relocation, "terminal", lambda path: "status")
    monkeypatch.setattr(relocation, "read", lambda path: {"complete": True, "capacity_target_reached": False,
        "archived_checkpoints": count, "candidate_checkpoints": 18, "all_original_bytes_and_paths_preserved": True,
        "config_sha256": "expected"})
    monkeypatch.setattr(relocation, "sha256", lambda path: "expected")
    if valid:
        assert len(relocation.capacity_ready()) == 3
    else:
        with pytest.raises(ValueError, match="completed successful"):
            relocation.capacity_ready()
