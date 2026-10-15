from copy import deepcopy

import pytest

import archive_single_model_paired_endpoint as archive


def records():
    return [{"state": "training_complete", "pid": 999999999},
            {"state": "complete", "pid": 999999999},
            {"complete": True, "gate": {"passed": False, "confirmation_evaluated": True},
             "campaign": None, "review": None},
            {"complete_training_audit": True, "through_step": 64},
            {"complete": True, "endpoint_grids": 56},
            {"complete": True, "prediction_grids": 1136}, {"passed": False}]


@pytest.mark.parametrize("index,key,value", [
    (0, "state", "training"), (1, "pid", 1), (2, "complete", False),
    (3, "through_step", 50), (4, "endpoint_grids", 28), (5, "prediction_grids", 1135),
    (6, "passed", True)])
def test_only_terminal_fully_rejected_evidence_can_move(monkeypatch, index, key, value):
    values = deepcopy(records())
    values[index][key] = value
    iterator = iter(values)
    monkeypatch.setattr(archive, "read", lambda path: next(iterator))
    with pytest.raises(ValueError, match="terminal fully audited"):
        archive.terminal_evidence()


def test_valid_terminal_records_and_unchanged_reserve(monkeypatch):
    iterator = iter(records())
    monkeypatch.setattr(archive, "read", lambda path: next(iterator))
    paths, report = archive.terminal_evidence()
    assert len(paths) == 7 and report["through_step"] == 64
    assert archive.archive.MINIMUM_FREE == 21 * 1024**3
    assert archive.idle.MINIMUM_FREE == 18 * 1024**3
    assert archive.SOURCE.parent.name == "paired_decoder_v19"
