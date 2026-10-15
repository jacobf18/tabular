import json
from pathlib import Path

from analyze_dwip_horizon_comparison import normalized_command
from run_dwip_continuous_horizon import build_commands
from run_dwip_continuous_spectrum import build_commands as short_commands
from run_dwip_spectral_horizon import build_commands as control_commands
from run_spectral_query_experiments import checkpoint_schedule


def test_continuous_horizon_matches_both_single_factor_controls():
    root = Path("/tmp/dwip_continuous_horizon_test")
    trials, controls, prior, anchor = control_commands(root)
    candidate_trials, commands, candidate_prior, candidate_anchor = build_commands(root)
    assert (trials, anchor) == (candidate_trials, candidate_anchor)
    _, short, short_prior, short_anchor = short_commands(root)
    assert (candidate_prior, candidate_anchor) == (short_prior, short_anchor)
    command, control = commands["dwip1e5"], controls["dwip1e5"]
    assert normalized_command(command) == normalized_command(short["dwip1e5"])
    position = command.index("--prior-overrides-json") + 1
    assert json.loads(command[position]) == candidate_prior
    assert candidate_prior["validation_spectral_specs"] == prior["validation_spectral_specs"]
    command[position] = control[position]
    assert command == control
    assert int(command[command.index("--data-seed") + 1]) == 1150924
    assert int(command[command.index("--seed") + 1]) == 1201
    assert float(command[command.index("--point-loss-weight") + 1]) == .2
    assert checkpoint_schedule(command) == [4000, 12000, 24000, 36000, 48000]
