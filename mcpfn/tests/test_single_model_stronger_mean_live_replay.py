import inspect

import pytest

import continue_single_model_stronger_mean_live_replay as recovery


def test_unchanged_observation_and_run_code_with_no_signal_release():
    before = dict(vars(recovery.stopped))
    namespace = recovery.execution_namespace()
    for name in ("run", "wait_for_original_replay"):
        assert namespace[name].__code__ is getattr(recovery.stopped, name).__code__
        assert namespace[name].__globals__ is namespace
        assert recovery.stopped.previous.unresolved_globals(namespace[name]) == []
    assert namespace["release_parent"] is recovery.confirm_release
    assert namespace["verify"] is recovery.verify
    assert namespace["STAGES"] == (("audit-replay", True), ("confirm", False))
    assert all(vars(recovery.stopped)[name] is value for name, value in before.items())


def test_actual_new_function_globals_resolve():
    for value in vars(recovery).values():
        if inspect.isfunction(value) and value.__globals__ is vars(recovery):
            assert recovery.stopped.previous.unresolved_globals(value) == []


def test_confirmed_release_records_observation_without_signaling(tmp_path, monkeypatch):
    config = {"child": {"pid": 101}}
    monkeypatch.setattr(recovery, "OUTPUT", tmp_path)
    monkeypatch.setattr(recovery, "require_surviving_chain", lambda: config)
    monkeypatch.setattr(recovery.stopped.signal, "pidfd_send_signal", lambda *args: pytest.fail("No second signal"))
    recovery.confirm_release(config)
    result = recovery.stopped.original.read(tmp_path / "release_observation.json")
    assert result == {"parent_already_stopped": True, "child": config["child"],
                      "signal_sent": False, "scientific_stage_repeated": False}


def test_continuation_observes_same_worker_then_only_remaining_stages(monkeypatch):
    namespace = recovery.execution_namespace()
    calls = []
    namespace["verify"] = lambda: {"child": {"pid": 101}}
    namespace["release_parent"] = lambda config: calls.append("no-signal observation")

    def wait(config):
        calls.append(("same original worker", config["child"]["pid"]))
        raise RuntimeError("waiting sentinel")

    namespace["wait_for_original_replay"] = wait
    with pytest.raises(RuntimeError, match="waiting sentinel"):
        namespace["run"]()
    assert calls == ["no-signal observation", ("same original worker", 101)]
