from copy import deepcopy

import pytest

import archive_single_model_structured_teacher_tasks as tasks
import archive_single_model_structured_teacher_endpoint as endpoint


def evidence():
    return [
        {"state": "training_complete", "pid": 999999998},
        {"state": "complete", "pid": 999999999},
        {"complete": True, "gate": {"passed": False, "confirmation_evaluated": True}, "campaign": None, "review": None},
        {"complete_training_audit": True, "through_step": 64, "artifact_sha256": {}},
        {"complete": True, "endpoint_grids": 56, "artifact_sha256": {}},
        {"complete": True, "prediction_grids": 1136, "artifact_sha256": {}},
        {"passed": False},
        {"complete": True, "full_grids": 1792, "full_grid_bitwise_equal": True,
         "role_counts": {"initialization": 128, "general": 1536, "specialist": 128}, "artifact_sha256": {}},
    ]


def install_evidence(monkeypatch, records):
    current = iter(deepcopy(records))
    monkeypatch.setattr(tasks, "read", lambda path: next(current))
    checked = []
    monkeypatch.setattr(tasks, "verify_hashes", checked.append)
    return checked


def test_complete_audited_evidence_required(monkeypatch):
    checked = install_evidence(monkeypatch, evidence())
    paths, audit = tasks.terminal_evidence()
    assert len(paths) == 8 and len(checked) == 4
    assert audit["through_step"] == 64


@pytest.mark.parametrize("index,field,value", [
    (0, "state", "training"), (0, "pid", 1), (1, "state", "confirm"), (1, "pid", 1),
    (2, "campaign", "published"), (3, "through_step", 50), (4, "endpoint_grids", 55),
    (5, "prediction_grids", 1135), (6, "passed", True), (7, "full_grids", 1791),
    (7, "full_grid_bitwise_equal", False), (7, "role_counts", {"general": 1792}),
])
def test_reject_incomplete_or_live_evidence(monkeypatch, index, field, value):
    records = evidence()
    records[index][field] = value
    checked = install_evidence(monkeypatch, records)
    with pytest.raises(ValueError, match="terminal fully audited"):
        tasks.terminal_evidence()
    assert not checked


def test_original_templates_unchanged_and_reserve_preserved():
    assert tasks.RUNS == (("structured_teacher_v20", 64, 1216, "structured_teacher_v20_pipeline"),)
    assert tasks.MAXIMUM_GROUP == 384 * 1024**2
    assert tasks.MINIMUM_FREE == endpoint.idle.MINIMUM_FREE == 18 * 1024**3
    assert len(tasks.template.RUNS) == 5
    assert 'steps * 28' in tasks.ADAPTED_SOURCE and 'steps * 29' in tasks.ADAPTED_SOURCE
    assert endpoint.SOURCE.parent.name == "structured_teacher_v20"
    assert endpoint.template.SOURCE.parent.name == "decoder_repair_v14"
