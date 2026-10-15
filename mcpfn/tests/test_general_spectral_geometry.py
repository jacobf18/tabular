import numpy as np
import pytest
import torch
from pathlib import Path

from general_spectral_geometry import GeometrySpectrumGenerator
from rowcol_fixed_validation import preserved_rng
from run_spectral_query_experiments import spectral_config
from tabimpute.prior.spectrum_training_set_generation import DecayingSpectrumGenerator


def config(kind):
    return dict(spectral_config()["mixed_priors"][0]["config"], spectrum_basis_kind=kind,
                spectrum_rank_low=8, spectrum_rank_high=8,
                num_rows_low=50, num_rows_high=50, num_cols_low=30, num_cols_high=30)


def test_haar_default_is_exactly_legacy():
    with preserved_rng(22):
        old = DecayingSpectrumGenerator(config("haar")).generate_matrix_bundle()
    with preserved_rng(22):
        new = GeometrySpectrumGenerator(config("haar")).generate_matrix_bundle()
    torch.testing.assert_close(old["full_matrix"], new["full_matrix"], rtol=0, atol=0)
    torch.testing.assert_close(old["target_matrix"], new["target_matrix"], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["haar", "coherent", "smooth", "offset", "mixed"])
def test_geometry_preserves_spectrum_and_clean_target(kind):
    with preserved_rng(41):
        generator = GeometrySpectrumGenerator(config(kind))
        basis = generator._sample_orthonormal_matrix(50, 8)
        np.testing.assert_allclose(basis.T @ basis, np.eye(8), atol=1e-12)
        sample = generator.generate_matrix_bundle()
        singular = torch.linalg.svdvals(sample["clean_signal"].double()).numpy()
        np.testing.assert_allclose(singular[:8], sample["spectrum"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(sample["clean_signal"], sample["target_matrix"], rtol=0, atol=0)


def test_nonhaar_geometries_actually_change_basis_statistics():
    leverage, constant_alignment = {}, {}
    for kind in ("haar", "coherent", "offset"):
        with preserved_rng(72):
            bases = [GeometrySpectrumGenerator(config(kind))._sample_orthonormal_matrix(100, 8)
                     for _ in range(32)]
        leverage[kind] = np.mean([np.max((q * q).sum(1)) for q in bases])
        constant_alignment[kind] = np.mean([abs(q[:, 0].sum()) / 10 for q in bases])
    assert leverage["coherent"] > 2 * leverage["haar"]
    assert constant_alignment["offset"] > .99
    assert constant_alignment["haar"] < .3


def test_queue_changes_training_geometry_not_validation():
    from run_spectral_combination_experiments import build_commands as old
    from run_spectral_geometry_experiments import build_commands as new
    import json
    root = Path("/tmp/geometry_test_queue")
    _, old_commands, old_prior, anchor = old(root)
    trials, commands, new_prior, new_anchor = new(root)
    assert anchor == new_anchor
    assert old_prior["validation_spectral_specs"] == new_prior["validation_spectral_specs"]
    for spec in new_prior["mixed_priors"]:
        assert spec["kind"] == "spectrum_decay" and spec["config"]["spectrum_basis_kind"] == "mixed"
    for trial, tag in zip(trials, ("df1_lr6e6", "df1_lr1e5")):
        command = list(commands[trial["tag"]])
        command[1] = old_commands[tag][1]
        for flag in ("--run-name", "--prior-overrides-json"):
            command[command.index(flag) + 1] = old_commands[tag][old_commands[tag].index(flag) + 1]
        assert command == old_commands[tag]


def test_continuations_match_except_learning_rate_and_name():
    from run_spectral_geometry_continuations import build_commands
    trials, commands, prior, anchor = build_commands(Path("/tmp/geometry_continue_test"))
    assert anchor.name == "best_validation_checkpoint.pth"
    assert "geometry_lr1e5" in str(anchor)
    a, b = [list(commands[t["tag"]]) for t in trials]
    assert a[a.index("--data-seed") + 1] == "1150924"
    for flag in ("--run-name", "--lr"):
        b[b.index(flag) + 1] = a[a.index(flag) + 1]
    assert a == b
    assert "--resume" not in a
