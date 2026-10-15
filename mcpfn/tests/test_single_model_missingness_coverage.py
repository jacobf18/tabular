from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import diagnose_single_model_missingness_coverage as probe
from rowcol_fixed_validation import preserved_rng


def test_probability_audit_includes_all_absent_families():
    prior = SimpleNamespace(names=["mcar", "profile"], weights=[1., 1.], children=[
        SimpleNamespace(missingness_inducer=probe.MCARPattern({}), config={}, generator=object()),
        SimpleNamespace(missingness_inducer=probe.ProfileMixedPattern({}), config={}, generator=object()),
    ])
    result = probe.audit_prior(prior)
    assert result["probabilities"]["mcar"] == pytest.approx(.65)
    assert result["probabilities"]["mar"] == pytest.approx(.1)
    assert len(result["absent_families"]) == 5 and len(result["probabilities"]) == 12
    assert sum(result["probabilities"].values()) == pytest.approx(1.)
    prior.children[0].missingness_inducer = object()
    with pytest.raises(ValueError, match="Unexpected"):
        probe.audit_prior(prior)


@pytest.mark.parametrize("family", probe.FAMILIES)
def test_all_families_preserve_inputs_and_reproduce_without_rng_leakage(family):
    source = torch.linspace(-3., 3., 144).reshape(12, 12)
    original = source.clone()
    config = probe.pattern_configuration({}, family, .6)
    saved_config = deepcopy(config)
    with preserved_rng(100):
        expected = torch.rand(4)
    with preserved_rng(100):
        first = probe.apply_family(source, config, family, 17)
        assert torch.equal(torch.rand(4), expected)
    second = probe.apply_family(source, config, family, 17)
    assert torch.equal(first, second) and first.dtype == torch.bool
    assert torch.equal(source, original) and config == saved_config


def test_normalization_uses_only_observed_source_and_preserves_clean_target():
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    target = source + 3.
    query = torch.tensor([[True, False, True], [False, False, True], [False, True, True], [False, False, True]])
    normalized, truth, mean, scale = probe.prepare_input(source, target, query)
    changed = source.clone()
    changed[query] = 10000.
    alternative, alternative_truth, alternative_mean, alternative_scale = probe.prepare_input(changed, target, query)
    torch.testing.assert_close(normalized, alternative, equal_nan=True)
    assert torch.equal(truth, alternative_truth) and torch.equal(mean, alternative_mean) and torch.equal(scale, alternative_scale)
    torch.testing.assert_close(truth * scale + mean, target)
    assert not torch.equal(truth * scale + mean, source)


def test_spectral_generation_retains_noisy_clean_distinction_and_seed():
    specification = {"seed": 61, "kind": "spectrum_decay", "generator_config": {
        "num_rows_low": 12, "num_rows_high": 12, "num_cols_low": 8, "num_cols_high": 8,
        "spectrum_noise_std_low": .5, "spectrum_noise_std_high": .5, "spectrum_target_rank": 10000,
    }}
    first = probe.generate_pair(specification)
    second = probe.generate_pair(specification)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert not torch.equal(first[0], first[1])


def test_polarization_diagnostic_recognizes_hidden_center_and_retained_tails():
    source = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    mask = probe.apply_family(source, {}, "mnar_polarization", 1)
    result = probe.describe_mask(source, source, mask)
    assert result["central_missing_fraction"] == 1. and result["tail_missing_fraction"] == 0.
    assert 0. < result["query_to_observed_target_std"] < 1.


@pytest.mark.parametrize("missing", [True, False])
def test_empty_and_constant_columns_are_retained_as_undefined(missing):
    source = torch.ones(4, 3)
    mask = torch.full(source.shape, missing)
    result = probe.describe_mask(source, source, mask)
    assert result["query_to_observed_target_std"] is None and result["std_ratio_columns"] == 0
    assert result["queries"] == 12 * int(missing)
    normalized, truth, mean, scale = probe.prepare_input(source, source, mask)
    assert torch.isfinite(truth).all() and torch.equal(torch.isnan(normalized), mask)


def test_undefined_summary_is_not_a_fabricated_zero():
    assert probe.average_defined([None, None]) is None
    assert probe.average_defined([1., None, 3.]) == 2.
    with pytest.raises(ValueError, match="explicit nulls"):
        probe.average_defined([np.nan])


def test_no_overwriting_registered_probe(tmp_path):
    with pytest.raises(FileExistsError):
        probe.register(tmp_path)
