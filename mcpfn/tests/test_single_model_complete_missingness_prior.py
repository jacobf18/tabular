from copy import deepcopy

import pytest
import torch

import single_model_complete_missingness_prior as prior
from rowcol_fixed_validation import preserved_rng


def test_fixed_preflight_covers_all_families_components_and_regimes():
    cases = prior.preflight_cases()
    assert len(cases) == 192 and len({case["seed"] for case in cases}) == 192
    for family in prior.coverage.FAMILIES:
        selected = [case for case in cases if case["family"] == family]
        assert len(selected) == 16 and {case["component"] for case in selected} == set(prior.COMPONENTS)
        assert {case["regime"] for case in selected} == {"legacy", "area_scaled"}
        shapes = prior.COMPACT_SHAPES if family in prior.NEURAL_FAMILIES else prior.SPECTRAL_SHAPES
        assert {tuple(case["shape"]) for case in selected} == set(shapes)


def test_scaled_generator_preserves_noisy_clean_distinction_and_noise_variance():
    config = {"num_rows_low": 12, "num_rows_high": 12, "num_cols_low": 8, "num_cols_high": 8,
              "spectrum_noise_std_low": .3, "spectrum_noise_std_high": .3, "spectrum_target_rank": 10000}
    with preserved_rng(19):
        legacy = prior.ScaledCoverageGenerator(config, 1.).generate_matrix_bundle()
    with preserved_rng(19):
        scaled = prior.ScaledCoverageGenerator(config, 4.).generate_matrix_bundle()
    torch.testing.assert_close(scaled["target_matrix"], 4 * legacy["target_matrix"])
    torch.testing.assert_close(scaled["full_matrix"] - scaled["target_matrix"],
                               legacy["full_matrix"] - legacy["target_matrix"], rtol=1e-4, atol=1e-4)


def test_configurations_preserve_anchor_shapes_and_cannot_be_mutated_by_callers():
    sampler = prior.CompleteMissingnessSampler()
    config = sampler.configuration("anchor_latent_mcar", (1024, 256))
    assert config["num_rows_high"] < 1024 and config["num_cols_high"] < 256
    original = deepcopy(config)
    config["num_rows_high"] = 10000
    assert sampler.configuration("anchor_latent_mcar", (1024, 256)) == original


@pytest.mark.parametrize("family", prior.coverage.FAMILIES)
def test_sampler_covers_all_masks_without_changing_global_rng(family):
    sampler = prior.CompleteMissingnessSampler()
    shape = (24, 12) if family in prior.NEURAL_FAMILIES else (240, 80)
    specification = dict(family=family, component="anchor_latent_mcar", shape=shape, regime="legacy",
                         requested_missingness=.6, seed=2331)
    with preserved_rng(29):
        expected = torch.rand(4)
    with preserved_rng(29):
        first = sampler.sample(**specification)
        assert torch.equal(torch.rand(4), expected)
    second = sampler.sample(**specification)
    torch.testing.assert_close(first.source, second.source, equal_nan=True)
    assert torch.equal(first.target, second.target) and torch.equal(first.mask, second.mask)
    assert first.metadata["gradient_data"] == "synthetic"
    assert torch.equal(torch.isnan(first.source), first.mask) and torch.isfinite(first.target).all()
    assert first.metadata["signal_gain"] == 1. and first.metadata["rows"] < 240


def test_neural_context_cannot_be_silently_truncated_from_a_large_request():
    sampler = prior.CompleteMissingnessSampler()
    with pytest.raises(ValueError, match="compact"):
        sampler.sample(family="mar_neural", component="low_rank_mcar", shape=(1024, 256), regime="legacy",
                       requested_missingness=.6, seed=1)


def test_actual_large_spectral_dimensions_and_gain_are_reported():
    sampler = prior.CompleteMissingnessSampler()
    episode, arrays = sampler.draw(family="mnar_polarization", component="low_rank_mcar", shape=(1024, 256),
                                   regime="area_scaled", requested_missingness=.6, seed=45)
    assert episode.source.shape == (1, 1024, 256)
    assert episode.metadata["signal_gain"] == prior.signal_gain(1024, 256)
    assert not torch.equal(arrays["raw_source"], arrays["raw_target"])
    torch.testing.assert_close(episode.target[0] * arrays["observed_scale"] + arrays["observed_mean"],
                               arrays["raw_target"], atol=1e-5, rtol=1e-5)
