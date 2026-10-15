import pytest

import archive_single_model_teacher_trial_capacity as capacity


def test_archive_schema_and_floor_are_scoped():
    compressor = capacity.capacity_compressor()
    assert compressor.__globals__["MINIMUM_FREE"] == 18 * 1024**3
    assert capacity.archive.MINIMUM_FREE == 21 * 1024**3
    assert capacity.ARTIFACT_COUNT == 39
    assert len(capacity.BANK_RUNS) == len(capacity.CONFIRMATION_RUNS) == 3
    assert "(dict, list)" not in __import__("inspect").getsource(capacity.archive.compress_group)


@pytest.mark.parametrize("failure", ["live", "state"])
def test_reject_live_or_unfinished_sources(monkeypatch, failure):
    monkeypatch.setattr(capacity, "read", lambda path: {"state": "training_complete" if failure == "live" else "training",
                                                       "pid": 1 if failure == "live" else 999999999})
    with pytest.raises(ValueError, match="terminal idle"):
        capacity.terminal_sources()


def test_inventory_does_not_relabel_original_failure(monkeypatch):
    def read(path):
        name = path.parent.name
        state = "complete"
        if name in capacity.BANK_RUNS:
            state = "training_complete"
        if name == "family_projection_preflight_v1":
            state = "failed"
        return {"state": state, "pid": 999999999}

    monkeypatch.setattr(capacity, "read", read)
    assert len(capacity.terminal_sources()) == 9
