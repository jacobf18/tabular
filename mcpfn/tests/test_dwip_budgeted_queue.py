import json
from types import SimpleNamespace

import pytest

import run_dwip_budgeted_queue_v1 as dispatcher
from run_dwip_anchor_mix import build_commands as control_commands
from run_dwip_anchor_mix_horizon import build_commands
from run_rowcol_output_remedies import replace_arg
from run_spectral_query_experiments import checkpoint_schedule


def test_horizon_preserves_everything_except_duration_and_snapshot(tmp_path):
    trials, controls, prior, anchor = control_commands(tmp_path)
    new_trials, commands, new_prior, new_anchor = build_commands(tmp_path)
    assert (trials, prior, anchor) == (new_trials, new_prior, new_anchor)
    command = commands["dwip1e5"]
    assert checkpoint_schedule(command) == [4000, 12000, 24000, 36000, 48000, 64000]
    phases = json.loads(dispatcher.argument(command, "--phases-json"))
    assert phases[-1]["end_step"] == 64000
    assert dispatcher.argument(command, "--decay-steps") == "63400"
    phases[-1]["end_step"] = 48000
    replace_arg(command, "--phases-json", json.dumps(phases))
    replace_arg(command, "--decay-steps", 47400)
    command.remove("48000")
    assert command == controls["dwip1e5"]


@pytest.fixture
def budget(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "checkpoint_48000.pth").write_bytes(b"model")
    (reference / "latest_checkpoint.pth").write_bytes(b"optimizer-state")
    _, commands, _, _ = build_commands(tmp_path)
    return dispatcher.artifact_budget(commands["dwip1e5"], reference)


def test_budget_covers_all_checkpoint_branches_and_atomic_temporary(budget):
    assert len(budget["model_only_files"]) == 9
    assert "checkpoint_phase1_64000.pth" in budget["model_only_files"]
    assert "best_checkpoint.pth" in budget["model_only_files"]
    assert "best_validation_checkpoint.pth" in budget["model_only_files"]
    assert budget["model_only_files"].count("checkpoint_64000.pth") == 1
    assert budget["model_file_ceiling_bytes"] > budget["reference_model_bytes"] * 1.25
    assert budget["latest_file_ceiling_bytes"] >= 3 * budget["model_file_ceiling_bytes"]
    persistent = 9 * budget["model_file_ceiling_bytes"] + budget["latest_file_ceiling_bytes"]
    assert budget["persistent_checkpoint_bytes"] == persistent
    assert budget["peak_additional_bytes"] == persistent + budget["atomic_temporary_bytes"] + 2 * dispatcher.GIB
    assert budget["required_free_bytes_at_launch"] == budget["peak_additional_bytes"] + 20 * dispatcher.GIB


def test_storage_launch_threshold_and_runtime_headroom(budget):
    required = budget["required_free_bytes_at_launch"]
    assert dispatcher.check_storage(budget, required, launch=True)["free_bytes"] == required
    with pytest.raises(RuntimeError, match="no files deleted"):
        dispatcher.check_storage(budget, required - 1, launch=True)
    allocated = budget["peak_additional_bytes"]
    required = budget["reserve_bytes"] + budget["atomic_temporary_bytes"]
    dispatcher.check_storage(budget, required, allocated)
    with pytest.raises(RuntimeError, match="storage guard"):
        dispatcher.check_storage(budget, required - 1, allocated)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        dispatcher.check_storage(budget, 100 * dispatcher.GIB, allocated + 1)


@pytest.mark.parametrize("flag,value", [("--init-model-checkpoint", "/tmp/other.pth"),
    ("--precision-mode", "full_bfloat16"), ("--checkpoint-every", "0"), ("--validation-every", "0")])
def test_dispatcher_rejects_unreviewed_modes(tmp_path, flag, value):
    _, commands, _, _ = build_commands(tmp_path)
    command = commands["dwip1e5"]
    replace_arg(command, flag, value)
    with pytest.raises(ValueError):
        dispatcher.validate_command(command)


@pytest.mark.parametrize("flag", ["--resume", "--retain-init-optimizer-state", "--allow-partial-init"])
def test_dispatcher_rejects_restarts(tmp_path, flag):
    _, commands, _, _ = build_commands(tmp_path)
    with pytest.raises(ValueError, match="fresh"):
        dispatcher.validate_command(commands["dwip1e5"] + [flag])


def test_dispatcher_rejects_changed_architecture(tmp_path):
    _, commands, _, _ = build_commands(tmp_path)
    command = commands["dwip1e5"]
    replace_arg(command, "--model-overrides-json", json.dumps({"emsize": 9999}))
    with pytest.raises(ValueError, match="unchanged DWIP architecture"):
        dispatcher.validate_command(command)


def test_regular_bytes_does_not_charge_linked_baselines(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "weights").write_bytes(b"original")
    output = tmp_path / "output"
    output.mkdir()
    (output / "baseline").symlink_to(original, target_is_directory=True)
    (output / "checkpoint").symlink_to(original / "weights")
    (output / "metrics").write_bytes(b"new")
    assert dispatcher.regular_bytes(output) == 3
    assert dispatcher.regular_bytes(output / "baseline") == 0
    assert dispatcher.regular_bytes(tmp_path / "missing") == 0


def test_registration_never_replaces_changed_artifacts(tmp_path):
    path = tmp_path / "manifest.json"
    dispatcher.immutable_json(path, {"seed": 1201})
    dispatcher.immutable_json(path, {"seed": 1201})
    with pytest.raises(ValueError, match="changed"):
        dispatcher.immutable_json(path, {"seed": 1202})
    assert json.loads(path.read_text()) == {"seed": 1201}


def test_guard_rejection_does_not_start_subprocess(tmp_path, monkeypatch):
    def reject():
        raise RuntimeError("storage guard")

    def forbidden(*args, **kwargs):
        pytest.fail("Must not spawn after a failed guard")

    monkeypatch.setattr(dispatcher.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="storage guard"):
        dispatcher.run_guarded(["unused"], tmp_path / "log", reject)


def test_runtime_guard_stops_only_owned_child(tmp_path, monkeypatch):
    checks = []
    calls = []

    def guard():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError("storage guard")

    def wait(timeout):
        calls.append(("wait", timeout))
        if len(calls) == 1:
            raise dispatcher.subprocess.TimeoutExpired("owned", timeout)
        return 0

    process = SimpleNamespace(wait=wait, poll=lambda: None, terminate=lambda: calls.append("terminate"))
    monkeypatch.setattr(dispatcher.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(RuntimeError, match="storage guard"):
        dispatcher.run_guarded(["owned"], tmp_path / "log", guard)
    assert calls == [("wait", 15), "terminate", ("wait", 120)]


def test_missing_reference_artifacts_fail_closed(tmp_path):
    _, commands, _, _ = build_commands(tmp_path)
    with pytest.raises(ValueError, match="completed matched-run"):
        dispatcher.artifact_budget(commands["dwip1e5"], tmp_path)


def test_guarded_success_sanitizes_environment_and_checks_after_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHSELECT_TABIMPUTE_CHECKPOINT", "unrelated")
    observations = []
    guards = []

    def spawn(command, **kwargs):
        observations.append(kwargs)
        return SimpleNamespace(wait=lambda timeout: 0, poll=lambda: 0)

    monkeypatch.setattr(dispatcher.subprocess, "Popen", spawn)
    dispatcher.run_guarded(["owned"], tmp_path / "log", lambda: guards.append(True))
    assert len(guards) == 2
    assert not any(key.startswith("BENCHSELECT_") for key in observations[0]["env"])
    assert observations[0]["env"]["WANDB_MODE"] == "disabled"


def test_guarded_nonzero_exit_fails_without_restarting(tmp_path, monkeypatch):
    launches = []

    def spawn(command, **kwargs):
        launches.append(command)
        return SimpleNamespace(wait=lambda timeout: 7, poll=lambda: 7)

    monkeypatch.setattr(dispatcher.subprocess, "Popen", spawn)
    with pytest.raises(dispatcher.subprocess.CalledProcessError) as error:
        dispatcher.run_guarded(["owned"], tmp_path / "log", lambda: None)
    assert error.value.returncode == 7
    assert launches == [["owned"]]
