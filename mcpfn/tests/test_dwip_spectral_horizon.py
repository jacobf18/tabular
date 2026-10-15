import json
from pathlib import Path

import pytest

from run_dwip_spectral_geometry import build_commands as control_commands
from run_dwip_spectral_horizon import build_commands, link_baseline, CACHED_BASELINE_FILES
from run_spectral_query_experiments import checkpoint_schedule


def test_horizon_changes_only_schedule_and_adds_prespecified_snapshots():
    root = Path("/tmp/dwip_horizon_test")
    _, controls, prior, anchor = control_commands(root)
    _, commands, new_prior, new_anchor = build_commands(root)
    assert prior == new_prior and anchor == new_anchor
    command, control = commands["dwip1e5"], controls["dwip1e5"]
    assert checkpoint_schedule(control) == [4000, 12000, 24000]
    assert checkpoint_schedule(command) == [4000, 12000, 24000, 36000, 48000]
    for flag in ("--phases-json", "--decay-steps"):
        command[command.index(flag) + 1] = control[control.index(flag) + 1]
    for snapshot in ("24000", "36000"):
        command.remove(snapshot)
    assert command == control
    assert "--retain-init-optimizer-state" not in command


@pytest.mark.parametrize("steps", [0, 200001])
def test_horizon_rejects_out_of_budget_training(steps):
    command = ["--phases-json", json.dumps([{"end_step": steps}])]
    with pytest.raises(ValueError, match="horizon"):
        checkpoint_schedule(command)


def test_horizon_rejects_snapshots_after_completion():
    command = ["--phases-json", '[{"end_step": 100}]', "--snapshot-steps", "200"]
    with pytest.raises(ValueError, match="Snapshot"):
        checkpoint_schedule(command)


def test_baseline_linking_is_idempotent_and_does_not_overwrite(tmp_path):
    root, control = tmp_path / "new", tmp_path / "control"
    root.mkdir()
    control.mkdir()
    for name in CACHED_BASELINE_FILES:
        (control / name).write_text("cached")
    link_baseline(root, control)
    link_baseline(root, control)
    assert all((root / name).is_symlink() for name in CACHED_BASELINE_FILES)
    assert all((root / name).resolve() == (control / name) for name in CACHED_BASELINE_FILES)
    collision = tmp_path / "collision"
    collision.mkdir()
    (collision / CACHED_BASELINE_FILES[0]).write_text("preserve")
    with pytest.raises(ValueError, match="Refusing to replace"):
        link_baseline(collision, control)
    assert (collision / CACHED_BASELINE_FILES[0]).read_text() == "preserve"
