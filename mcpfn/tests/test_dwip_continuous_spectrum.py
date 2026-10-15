from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import run_dwip_continuous_spectrum as runner
from run_dwip_continuous_spectrum import build_commands, case_digest, continuous_prior
from run_dwip_spectral_geometry import build_commands as control_commands


def test_continuous_prior_changes_only_high_rank_exponent_bounds():
    root = Path("/tmp/dwip_continuous_spectrum_test")
    trials, controls, prior, anchor = control_commands(root)
    untouched = deepcopy(prior)
    updated_trials, commands, updated, updated_anchor = build_commands(root)
    assert prior == untouched
    assert (trials, anchor) == (updated_trials, updated_anchor)
    assert updated["validation_spectral_specs"] == prior["validation_spectral_specs"]
    for before, after in zip(prior["mixed_priors"], updated["mixed_priors"]):
        restored = deepcopy(after)
        if before["config"]["spectrum_rank_high"] > 16:
            assert (after["config"]["spectrum_alpha_low"], after["config"]["spectrum_alpha_high"]) == (0., 2.5)
            for key in ("spectrum_alpha_low", "spectrum_alpha_high"):
                restored["config"][key] = before["config"][key]
        assert restored == before
    command, control = commands["dwip1e5"], controls["dwip1e5"]
    position = command.index("--prior-overrides-json") + 1
    assert json.loads(command[position]) == updated
    command[position] = control[position]
    assert command == control


def test_continuous_prior_refuses_unexpected_component_layout():
    with pytest.raises(ValueError, match="four high-rank"):
        continuous_prior({"mixed_priors": []})


def test_case_digest_covers_values_masks_identity_and_metadata():
    cases = [{"id": "generic_0", "source": torch.tensor([1., float("nan")]),
              "mask": torch.tensor([False, True]), "stats": {"support": .5}}]
    original = case_digest(cases)
    assert case_digest(deepcopy(cases)) == original
    for key, value in (("id", "generic_1"), ("source", torch.tensor([2., float("nan")])),
                       ("mask", torch.tensor([True, False])), ("stats", {"support": .6})):
        changed = deepcopy(cases)
        changed[0][key] = value
        assert case_digest(changed) != original


def test_cache_preparation_accepts_only_identical_cases(tmp_path, monkeypatch):
    root, control = tmp_path / "candidate", tmp_path / "control"
    root.mkdir()
    control.mkdir()
    prior = {"validation_spectral_specs": {"generic": {"spectrum_alpha_low": .6}}}
    cases = [{"id": "generic_0", "source": torch.tensor([float("nan"), 1.]),
              "target": torch.tensor([2., 1.]), "mask": torch.tensor([True, False])}]
    cache = {"manifest": {"prior": prior, "seed": 42, "cases_per_group": 1}, "cases": cases}
    torch.save(cache, control / "validation.pth")
    monkeypatch.setattr(runner, "CACHED_BASELINE_FILES", ("validation.pth",))

    def regenerate(destination, actual_prior, seed, cases_per_group):
        assert actual_prior == prior and seed == 42 and cases_per_group == 1
        torch.save({"cases": cases}, destination)
        return cases

    monkeypatch.setattr(runner, "load_or_build_validation", regenerate)
    evidence = runner.prepare_caches(root, control, prior)
    assert evidence["validation.pth"]["identical_case_sha256"] == case_digest(cases)
    cases[0]["target"][0] = 3.
    with pytest.raises(ValueError, match="cases differ"):
        runner.prepare_caches(root, control, prior)
    with pytest.raises(ValueError, match="distribution changed"):
        runner.prepare_caches(root, control, {"validation_spectral_specs": {}})
