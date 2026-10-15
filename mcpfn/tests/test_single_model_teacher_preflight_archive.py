import pytest

import archive_single_model_teacher_preflight as archive


@pytest.mark.parametrize("failure", [None, "live", "unaudited", "incomplete_targets", "rejected"])
def test_require_complete_successful_idle_evidence(monkeypatch, failure):
    terminal = {"state": "complete", "pid": 1 if failure == "live" else 999999999}
    audit = {"complete": failure != "unaudited", "training_tasks": 28, "prediction_grids": 56,
             "raw_endpoint": {"gradient_vectors": 84}, "teacher_replay": {"complete": True, "full_grids": 28}}
    gate = {"passed": failure != "rejected", "preflight_only": True, "audit_sha256": "audit"}
    replay = {"complete": True, "rows": [None] * (27 if failure == "incomplete_targets" else 28)}
    iterator = iter((terminal, audit, gate, replay))
    monkeypatch.setattr(archive, "read", lambda path: next(iterator))
    monkeypatch.setattr(archive, "sha256", lambda path: "audit")
    if failure is not None:
        with pytest.raises(ValueError, match="terminal successful"):
            archive.evidence()
    else:
        paths, report = archive.evidence()
        assert len(paths) == 4 and report == audit
        assert archive.archive.MINIMUM_FREE == 21 * 1024**3
        assert archive.idle.MINIMUM_FREE == 18 * 1024**3
