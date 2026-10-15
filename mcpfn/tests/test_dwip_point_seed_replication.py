from pathlib import Path

from run_dwip_point_horizon import build_commands as seed1_commands
from run_dwip_point_seed_replication import build_commands
from run_dwip_seed_replication import build_commands as seed2_commands
from run_spectral_query_experiments import checkpoint_schedule


def test_point_seed_replication_matches_both_single_factor_controls():
    root = Path("/tmp/dwip_point_seed_replication_test")
    trials, controls, prior, anchor = seed2_commands(root)
    candidate_trials, commands, candidate_prior, candidate_anchor = build_commands(root)
    assert (trials, prior, anchor) == (candidate_trials, candidate_prior, candidate_anchor)
    command = commands["dwip1e5"]
    restored = list(command)
    position = restored.index("--point-loss-weight") + 1
    assert float(command[position]) == 2.
    restored[position] = controls["dwip1e5"][position]
    assert restored == controls["dwip1e5"]
    _, seed1, seed1_prior, seed1_anchor = seed1_commands(root)
    assert (prior, anchor) == (seed1_prior, seed1_anchor)
    restored = list(command)
    position = restored.index("--data-seed") + 1
    assert int(command[position]) == 1150925
    restored[position] = seed1["dwip1e5"][position]
    assert restored == seed1["dwip1e5"]
    assert int(command[command.index("--seed") + 1]) == 1201
    assert checkpoint_schedule(command) == [4000, 12000, 24000, 36000, 48000]
