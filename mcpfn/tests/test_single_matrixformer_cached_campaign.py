from types import SimpleNamespace

import pytest

from recsys_evaluation_common import write_json
import run_single_matrixformer_cached_benchmarks as launcher


def fixture(root):
    arguments = SimpleNamespace(run=root / "run", confirmation=root / "confirmation", source=root / "source")
    write_json(arguments.run / "config.json", {"frozen": True})
    write_json(arguments.source / "manifest.json", {"source": True})
    write_json(arguments.source / "audit.json", {"complete": True})
    for stage in launcher.campaign.STAGES:
        write_json(root / f"{stage}_protocol_v1/protocol.json", {"stage": stage})
    output = root / "benchmarks"
    output.mkdir()
    return arguments, output


def test_waits_then_copies_only_controls_before_original_campaign(tmp_path, monkeypatch):
    arguments, output = fixture(tmp_path)
    gates = iter([None, {"passed": True}])
    events = []
    monkeypatch.setattr(launcher.campaign, "confirmation_state", lambda path: next(gates))
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: events.append("wait"))

    def prepare(source, protocol, checkpoint, confirmation, destination):
        assert source == arguments.source and confirmation == arguments.confirmation
        assert checkpoint == arguments.run / "best_validation_checkpoint.pth"
        assert destination == output / "recsys"
        events.append("copy_parent")
        return {"controls": [1, 2, 3, 4], "student_predictions_copied": False, "specialist_parity_copied": False}

    monkeypatch.setattr(launcher.cache, "prepare", prepare)
    monkeypatch.setattr(launcher, "handoff", lambda *args: events.append("original_campaign"))
    launcher.execute(arguments, output)
    assert events == ["wait", "copy_parent", "original_campaign"]
    assert launcher.cache.read(output / "status.json")["parent_controls_preseeded"]


def test_failed_synthetic_gate_never_reads_or_imports_baseline_predictions(tmp_path, monkeypatch):
    arguments, output = fixture(tmp_path)
    monkeypatch.setattr(launcher.campaign, "confirmation_state", lambda path: {"passed": False})
    monkeypatch.setattr(launcher.cache, "prepare", lambda *args: pytest.fail("No cache import after a failed gate"))
    handed_off = []
    monkeypatch.setattr(launcher, "handoff", lambda *args: handed_off.append(True))
    launcher.execute(arguments, output)
    assert handed_off == [True] and not launcher.cache.read(output / "status.json")["parent_controls_preseeded"]


def test_changed_configuration_while_waiting_is_rejected(tmp_path, monkeypatch):
    arguments, output = fixture(tmp_path)
    gates = iter([None, {"passed": True}])
    monkeypatch.setattr(launcher.campaign, "confirmation_state", lambda path: next(gates))
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: write_json(arguments.run / "config.json", {"changed": True}))
    monkeypatch.setattr(launcher.cache, "prepare", lambda *args: pytest.fail("Changed input must fail before cache import"))
    with pytest.raises(ValueError):
        launcher.execute(arguments, output)


def test_handoff_keeps_all_original_suites_and_uses_resume_for_seeded_directory(tmp_path, monkeypatch):
    arguments, output = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(launcher.os, "execv", lambda executable, command: calls.append(command))
    launcher.handoff(arguments, output)
    assert len(calls) == 1 and calls[0][-1] == "--resume"
    assert calls[0][2] == str(launcher.Path(launcher.campaign.__file__).resolve())
    assert "--stage" not in calls[0] and "--source" not in calls[0]
