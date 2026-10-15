from pathlib import Path

from run_dwip_point_objective import build_commands
from run_dwip_spectral_geometry import build_commands as control_commands
from run_spectral_query_experiments import checkpoint_schedule


def test_point_objective_changes_exactly_one_training_factor():
    root = Path("/tmp/dwip_point_objective_test")
    trials, controls, prior, anchor = control_commands(root)
    new_trials, commands, new_prior, new_anchor = build_commands(root)
    assert (trials, prior, anchor) == (new_trials, new_prior, new_anchor)
    command, control = commands["dwip1e5"], controls["dwip1e5"]
    position = command.index("--point-loss-weight") + 1
    assert float(control[position]) == .2
    assert float(command[position]) == 2.
    command[position] = control[position]
    assert command == control
    assert checkpoint_schedule(command) == [4000, 12000, 24000]
    assert "--retain-init-optimizer-state" not in command
