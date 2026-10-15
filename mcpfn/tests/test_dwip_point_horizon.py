from pathlib import Path

from analyze_dwip_horizon_comparison import normalized_command
from run_dwip_point_horizon import build_commands
from run_dwip_point_objective import build_commands as short_point_commands
from run_dwip_spectral_horizon import build_commands as control_commands
from run_spectral_query_experiments import checkpoint_schedule


def test_fourth_factorial_arm_matches_both_single_factor_controls():
    root = Path("/tmp/dwip_point_horizon_test")
    trials, controls, prior, anchor = control_commands(root)
    new_trials, commands, new_prior, new_anchor = build_commands(root)
    assert (trials, prior, anchor) == (new_trials, new_prior, new_anchor)
    command, control = commands["dwip1e5"], controls["dwip1e5"]
    _, short, short_prior, short_anchor = short_point_commands(root)
    assert (prior, anchor) == (short_prior, short_anchor)
    assert normalized_command(command) == normalized_command(short["dwip1e5"])
    position = command.index("--point-loss-weight") + 1
    assert float(control[position]) == .2 and float(command[position]) == 2.
    command[position] = control[position]
    assert command == control
    assert int(command[command.index("--data-seed") + 1]) == 1150924
    assert checkpoint_schedule(command) == [4000, 12000, 24000, 36000, 48000]
