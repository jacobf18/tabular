from contextlib import contextmanager
import inspect

import pytest

import recover_single_model_four_draw_audit_arithmetic as recovery
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize("stage", ["register", "pipeline", "audit"])
def test_every_recovery_entry_restores_original_arithmetic_before_callbacks(monkeypatch, stage):
    calls = []

    @contextmanager
    def arithmetic():
        calls.append("entered_original_arithmetic")
        yield
        calls.append("restored_previous_arithmetic")

    def main():
        assert calls == ["entered_original_arithmetic"]
        calls.append(stage)

    monkeypatch.setattr(recovery.arithmetic, "original_arithmetic", arithmetic)
    monkeypatch.setattr(recovery, "namespace", lambda: {"main": main})
    monkeypatch.setattr(recovery.previous.sys, "argv", ["recovery", "--stage", stage])
    recovery.main()
    assert calls == ["entered_original_arithmetic", stage, "restored_previous_arithmetic"]


def test_callbacks_bind_new_paths_but_original_corrected_math():
    namespace = recovery.namespace()
    assert namespace["OUTPUT"] == recovery.OUTPUT
    assert namespace["__file__"] == recovery.__file__
    assert namespace["require_closed"] is recovery.require_closed
    assert namespace["corrected_source"]() == recovery.previous.corrected_source()
    assert namespace["previous"] is recovery.previous.previous
    for name in ("register", "verify", "status", "audit", "pipeline", "main"):
        assert namespace[name].__globals__ is namespace
        assert not unresolved_globals(namespace[name]), name
    for function in vars(recovery).values():
        if inspect.isfunction(function) and function.__globals__ is vars(recovery):
            assert not unresolved_globals(function), function.__name__


def test_no_training_or_neural_replay_stage():
    source = inspect.getsource(recovery.main)
    assert 'choices=("diagnose", "register", "pipeline", "audit")' in source
    assert "original_arithmetic" in source


def test_diagnosis_never_overwrites_prior_evidence(monkeypatch, tmp_path):
    path = tmp_path / "diagnosis.json"
    path.write_text("preserved")
    monkeypatch.setattr(recovery, "DIAGNOSIS", path)
    with pytest.raises(FileExistsError):
        recovery.diagnose()
    assert path.read_text() == "preserved"
