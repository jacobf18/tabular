import tempfile
from pathlib import Path

import numpy as np
import torch

from rowcol_fixed_validation import load_or_build_validation, preserved_rng
from run_spectral_query_experiments import spectral_config, build_commands
from tabimpute.prior.spectrum_training_set_generation import DecayingSpectrumGenerator
from train_direct_pure_rabel_curriculum import CurriculumPhase, MixedMatrixPrior


def test_general_spectra_preserve_full_clean_target():
    prior = spectral_config()
    assert len(prior["mixed_priors"]) == 6
    with preserved_rng(22):
        for spec in prior["mixed_priors"]:
            assert spec["kind"] == "spectrum_decay"
            cfg = dict(spec["config"], num_rows_low=30, num_rows_high=30,
                       num_cols_low=20, num_cols_high=20)
            bundle = DecayingSpectrumGenerator(cfg).generate_matrix_bundle()
            assert bundle["rank"] == bundle["target_rank"]
            torch.testing.assert_close(bundle["target_matrix"], bundle["clean_signal"], atol=0, rtol=0)
            assert np.isfinite(bundle["spectrum"]).all()


def test_commands_are_matched_and_remove_old_prior():
    trials, commands, prior, anchor = build_commands(Path("/tmp/spectral_queue_test"))
    assert len(trials) == 4
    assert anchor.name == "checkpoint_12000.pth"
    for command in commands.values():
        assert command[command.index("--prior-kind") + 1] == "mixed_matrix"
        assert "priormix_v1" not in " ".join(command)
        assert command.count("--snapshot-steps") == 1
        assert "24000" in command[command.index("--phases-json") + 1]
    assert "--freeze-param-substrings" in commands["frozen_low"]
    assert "--param-lr-multipliers-json" in commands["axis_low"]


def test_lr_followup_changes_only_lr_and_run_name():
    from run_spectral_lr_followups import build_commands as lr_commands
    root = Path("/tmp/spectral_lr_test")
    _, original, prior, anchor = build_commands(root)
    trials, commands, new_prior, new_anchor = lr_commands(root)
    assert prior == new_prior and anchor == new_anchor
    for trial in trials:
        command = list(commands[trial["tag"]])
        assert float(command[command.index("--lr") + 1]) == trial["lr"]
        for flag in ("--lr", "--run-name"):
            command[command.index(flag) + 1] = original["joint_low"][original["joint_low"].index(flag) + 1]
        assert command == original["joint_low"]


def test_every_spectral_missingness_child_produces_finite_targets():
    phase = CurriculumPhase(name="test", end_step=6, rows_low=20, rows_high=20,
                            cols_low=16, cols_high=16, batch_size=2)
    with preserved_rng(82):
        prior = MixedMatrixPrior(phase=phase, device="cpu", missingness_type="mcar",
                                 mixed_config=spectral_config())
        for child in prior.children:
            (source, target, _, _, _), _ = child.get_batch()
            assert source.shape == target.shape == (2, 20, 16)
            assert torch.isfinite(target).all()
            assert torch.isfinite(source).any() and torch.isnan(source).any()


def test_spectral_validation_cache_is_reproducible():
    config = spectral_config()
    config["validation_spectral_specs"] = {
        "spectral_test": next(iter(config["validation_spectral_specs"].values()))}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cases.pth"
        first = load_or_build_validation(path, config, seed=991, cases_per_group=1)
        second = load_or_build_validation(path, config, seed=991, cases_per_group=1)
        assert len(first) == 6
        assert sum(c["monitor_only"] for c in first) == 2
        for a, b in zip(first, second):
            torch.testing.assert_close(a["source"], b["source"], equal_nan=True)
            assert a["mask"].any()
            assert torch.isnan(a["source"])[a["mask"]].all()
            assert torch.isfinite(a["target"]).all()
