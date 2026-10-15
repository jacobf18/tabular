from copy import deepcopy
import json

import pytest

from run_dwip_anchor_mix_huber_continuous import build_commands, continuous_rehearsal_prior, verify_continuous_only_change
from run_dwip_anchor_mix_huber_objective import build_commands as control_commands
from run_rowcol_output_remedies import replace_arg


def test_only_four_alpha_values_change_with_latent_rehearsal_preserved(tmp_path):
    control = control_commands(tmp_path)
    candidate = build_commands(tmp_path)
    verify_continuous_only_change(control[1]["dwip1e5"], candidate[1]["dwip1e5"])
    assert control[0] == candidate[0] and control[3] == candidate[3]
    assert control[2]["mixed_priors"][:2] == candidate[2]["mixed_priors"][:2]
    assert control[2]["mixed_priors"][6:] == candidate[2]["mixed_priors"][6:]
    assert control[2]["validation_spectral_specs"] == candidate[2]["validation_spectral_specs"]
    changed = []
    for before, after in zip(control[2]["mixed_priors"], candidate[2]["mixed_priors"]):
        for key in before["config"]:
            if before["config"][key] != after["config"][key]:
                changed.append(key)
        if before["name"].startswith(("slow_tail", "steep_tail")):
            assert after["config"]["spectrum_alpha_low"] == 0.
            assert after["config"]["spectrum_alpha_high"] == 2.5
    assert sorted(changed) == ["spectrum_alpha_high"] * 2 + ["spectrum_alpha_low"] * 2


def test_prior_transformation_is_pure_and_rejects_already_continuous(tmp_path):
    original = control_commands(tmp_path)[2]
    saved = deepcopy(original)
    updated = continuous_rehearsal_prior(original)
    assert original == saved
    with pytest.raises(ValueError):
        continuous_rehearsal_prior(updated)


@pytest.mark.parametrize("flag,value", [("--point-loss-weight", "4"), ("--huber-delta", "2"),
    ("--data-seed", "7"), ("--decay-steps", "127400"), ("--grad-clip-norm", "1"),
    ("--support-variance-df", "2"), ("--point-loss-kind", "mse")])
def test_continuous_control_rejects_other_command_changes(tmp_path, flag, value):
    control = control_commands(tmp_path)[1]["dwip1e5"]
    candidate = build_commands(tmp_path)[1]["dwip1e5"]
    replace_arg(candidate, flag, value)
    with pytest.raises(ValueError):
        verify_continuous_only_change(control, candidate)


@pytest.mark.parametrize("mutation", ["mass", "latent_shape", "low_rank", "validation", "alpha"])
def test_continuous_control_rejects_other_prior_changes(tmp_path, mutation):
    control = control_commands(tmp_path)[1]["dwip1e5"]
    _, commands, prior, _ = build_commands(tmp_path)
    candidate = commands["dwip1e5"]
    if mutation == "mass":
        prior["mixed_priors"][6]["weight"] = 2.
    elif mutation == "latent_shape":
        prior["mixed_priors"][6]["shape_override"]["num_rows_high"] = 60
    elif mutation == "low_rank":
        prior["mixed_priors"][0]["config"]["spectrum_rank_high"] = 15
    elif mutation == "validation":
        prior["validation_spectral_specs"] = {}
    else:
        prior["mixed_priors"][2]["config"]["spectrum_alpha_high"] = 3.
    replace_arg(candidate, "--prior-overrides-json", json.dumps(prior))
    with pytest.raises(ValueError):
        verify_continuous_only_change(control, candidate)
