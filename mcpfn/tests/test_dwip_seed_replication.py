from pathlib import Path

from run_dwip_seed_replication import build_commands, DATA_SEED
from run_dwip_spectral_horizon import build_commands as control_commands
from run_spectral_query_experiments import checkpoint_schedule


def test_replication_changes_only_one_fixed_training_seed():
    root = Path("/tmp/dwip_seed_replication_test")
    trials, controls, prior, anchor = control_commands(root)
    new_trials, commands, new_prior, new_anchor = build_commands(root)
    assert (trials, prior, anchor) == (new_trials, new_prior, new_anchor)
    command, control = commands["dwip1e5"], controls["dwip1e5"]
    position = command.index("--data-seed") + 1
    assert int(control[position]) == 1150924
    assert int(command[position]) == DATA_SEED == 1150925
    command[position] = control[position]
    assert command == control
    assert checkpoint_schedule(command) == [4000, 12000, 24000, 36000, 48000]
    assert "--retain-init-optimizer-state" not in command
