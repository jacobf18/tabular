import json

import pytest

from run_dwip_anchor_mix_horizon import build_commands as control_commands
from run_dwip_anchor_mix_point_objective import build_commands, require_complete_control, verify_point_only_change
from run_dwip_budgeted_queue_v1 import artifact_budget, argument
from run_rowcol_controlled_experiments import write_json
from run_rowcol_output_remedies import replace_arg
from run_spectral_query_experiments import checkpoint_schedule


def test_point_factor_changes_exactly_one_command_value(tmp_path):
    trials, controls, prior, anchor = control_commands(tmp_path)
    new_trials, commands, new_prior, new_anchor = build_commands(tmp_path)
    assert (trials, prior, anchor) == (new_trials, new_prior, new_anchor)
    before, after = controls["dwip1e5"], commands["dwip1e5"]
    verify_point_only_change(before, after)
    assert len(before) == len(after)
    assert [index for index, values in enumerate(zip(before, after)) if values[0] != values[1]] == [after.index("--point-loss-weight") + 1]
    assert argument(after, "--point-loss-weight") == "2.0"
    assert argument(after, "--nll-loss-weight") == "1"
    assert argument(after, "--cdf-loss-weight") == "0.05"
    assert argument(after, "--data-seed") == "1150924"
    assert argument(after, "--seed") == "1201"
    assert checkpoint_schedule(after) == [4000, 12000, 24000, 36000, 48000, 64000]
    components = json.loads(argument(after, "--prior-overrides-json"))["mixed_priors"]
    assert sum(component["weight"] for component in components if component["kind"] == "latent_factor") / sum(component["weight"] for component in components) == .25


@pytest.mark.parametrize("flag,value", [("--data-seed", "1150925"), ("--seed", "1202"),
    ("--lr", "2e-5"), ("--decay-steps", "95400"), ("--point-loss-prediction", "median"),
    ("--support-variance-df", "4"), ("--point-loss-weight", "3")])
def test_point_factor_rejects_undeclared_changes(tmp_path, flag, value):
    _, controls, _, _ = control_commands(tmp_path)
    _, commands, _, _ = build_commands(tmp_path)
    replace_arg(commands["dwip1e5"], flag, value)
    with pytest.raises(ValueError):
        verify_point_only_change(controls["dwip1e5"], commands["dwip1e5"])


def test_point_factor_rejects_duplicate_coefficient(tmp_path):
    _, controls, _, _ = control_commands(tmp_path)
    _, commands, _, _ = build_commands(tmp_path)
    with pytest.raises(ValueError, match="one explicit"):
        verify_point_only_change(controls["dwip1e5"], commands["dwip1e5"] + ["--point-loss-weight", "2"])


def test_point_factor_preserves_reviewed_artifact_budget(tmp_path):
    (tmp_path / "checkpoint_64000.pth").write_bytes(b"model")
    (tmp_path / "latest_checkpoint.pth").write_bytes(b"optimizer state")
    _, controls, _, _ = control_commands(tmp_path)
    _, commands, _, _ = build_commands(tmp_path)
    assert artifact_budget(controls["dwip1e5"], tmp_path) == artifact_budget(commands["dwip1e5"], tmp_path)


@pytest.mark.parametrize("pending", ["training", "transfer", "retention", "runtime"])
def test_point_factor_requires_all_control_stages_complete(tmp_path, pending):
    folders = {"training": tmp_path}
    for name in ("transfer", "retention", "runtime"):
        folders[name] = tmp_path / f"{name}_evaluation"
        folders[name].mkdir()
    for folder in folders.values():
        write_json(folder / "status.json", {"state": "complete"})
        write_json(folder / "results.json", {})
    require_complete_control(tmp_path)
    write_json(folders[pending] / "status.json", {"state": "waiting"})
    with pytest.raises(RuntimeError, match="all four"):
        require_complete_control(tmp_path)


def test_point_factor_rejects_completion_without_results(tmp_path):
    write_json(tmp_path / "status.json", {"state": "complete"})
    with pytest.raises(RuntimeError, match="all four"):
        require_complete_control(tmp_path)
