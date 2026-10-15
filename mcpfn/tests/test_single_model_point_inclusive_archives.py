from copy import deepcopy

import pytest

import archive_single_model_point_inclusive_tasks as tasks
import archive_single_model_point_inclusive_endpoint as endpoint


def evidence():
    gate = {"passed": False, "confirmation_evaluated": False, "selected_step": 0}
    return [
        {"state": "training_complete", "pid": 999999998},
        {"state": "complete", "pid": 999999999},
        {"complete": True, "gate": deepcopy(gate), "campaign": None, "review": None},
        {"complete_training_audit": True, "through_step": 64, "accepted_updates": 63, "best": {"step": 0}, "artifact_sha256": {}},
        {"complete": True, "endpoint_grids": 56, "artifact_sha256": {}},
        gate,
        {"complete": True, "full_grids": 1792, "full_grid_bitwise_equal": True,
         "role_counts": {"initialization": 128, "general": 1536, "specialist": 128}, "artifact_sha256": {}},
        {"complete": True, "original_failure_preserved": True, "training_restarted": False,
         "scientific_configuration_changed": False, "teacher_predictions_repeated": False},
        {"state": "failed", "pid": 999999997},
    ]


def install_evidence(monkeypatch, records):
    current = iter(deepcopy(records))
    monkeypatch.setattr(tasks, "read", lambda path: next(current))
    checked = []
    monkeypatch.setattr(tasks, "verify_hashes", checked.append)
    return checked


def test_requires_complete_nonselecting_recovery(monkeypatch):
    checked = install_evidence(monkeypatch, evidence())
    paths, audit = tasks.terminal_evidence()
    assert len(paths) == 9 and len(checked) == 3
    assert audit["best"]["step"] == 0


@pytest.mark.parametrize("index,field,value", [
    (0, "state", "training"), (0, "pid", 1), (1, "state", "confirm"), (1, "pid", 1),
    (2, "campaign", "published"), (3, "through_step", 50), (3, "accepted_updates", 64),
    (3, "best", {"step": 64}), (4, "endpoint_grids", 55), (5, "confirmation_evaluated", True),
    (5, "passed", True), (6, "full_grids", 1791), (6, "full_grid_bitwise_equal", False),
    (6, "role_counts", {"general": 1792}), (7, "original_failure_preserved", False),
    (7, "training_restarted", True), (7, "scientific_configuration_changed", True),
    (7, "teacher_predictions_repeated", True), (8, "state", "complete"), (8, "pid", 1),
])
def test_rejects_incomplete_live_or_changed_history(monkeypatch, index, field, value):
    records = evidence()
    records[index][field] = value
    checked = install_evidence(monkeypatch, records)
    with pytest.raises(ValueError, match="terminal fully audited"):
        tasks.terminal_evidence()
    assert not checked


def test_preserves_inventory_and_write_reserve():
    assert tasks.RUNS == (("point_inclusive_v21", 64, 1216, "point_inclusive_v21_pipeline_entry_recovery_v1"),)
    assert tasks.MAXIMUM_GROUP == 384 * 1024**2
    assert tasks.MINIMUM_FREE == endpoint.idle.MINIMUM_FREE == 18 * 1024**3
    assert 'steps * 28' in tasks.ADAPTED_SOURCE and 'steps * 29' in tasks.ADAPTED_SOURCE
    assert endpoint.SOURCE.parent.name == "point_inclusive_v21"
    assert endpoint.template.SOURCE.parent.name == "decoder_repair_v14"
