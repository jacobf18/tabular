import numpy as np
import pytest

from diagnose_dwip_noise_sensitivity import perturb_observed, summarize


def test_noise_preserves_missingness_sources_and_constant_columns():
    source = np.array([[1., 4., np.nan], [2., 4., 5.], [3., 4., np.nan]])
    original = source.copy()
    clean, _ = perturb_observed(source, 0., 1., 42)
    corrupted, stats = perturb_observed(source, 1., 1., 42)
    assert np.array_equal(source, original, equal_nan=True)
    assert np.array_equal(clean, source, equal_nan=True)
    assert np.array_equal(np.isnan(corrupted), np.isnan(source))
    np.testing.assert_array_equal(corrupted[:, 1], source[:, 1])
    assert corrupted[1, 2] == 5.
    assert stats["eligible_observed_fraction"] == pytest.approx(3 / 7)


def test_noise_severities_share_draws_and_sparse_dense_match_expected_rms():
    source = np.tile(np.linspace(-1., 1., 1000)[:, None], (1, 100))
    low, _ = perturb_observed(source, .25, 1., 12)
    high, _ = perturb_observed(source, 1., 1., 12)
    np.testing.assert_allclose(high - source, 4 * (low - source), atol=1e-15)
    _, dense = perturb_observed(source, 1., 1., 12)
    _, sparse = perturb_observed(source, 1., .01, 12)
    assert dense["realized_noise_rms_in_original_observed_std"] == pytest.approx(1., abs=.02)
    assert sparse["realized_noise_rms_in_original_observed_std"] == pytest.approx(1., abs=.08)
    with pytest.raises(ValueError, match="configuration"):
        perturb_observed(source, 1., 0., 12)


def test_summary_uses_fixed_targets_and_counts_for_noise_error_ratios():
    scenarios = {"clean": {"n": 2, "sse": 2., "prediction_change_sse": 0.},
                 "noisy": {"n": 2, "sse": 8., "prediction_change_sse": 2.}}
    result = summarize([{"population": "test", "group": "group", "scenarios": scenarios}])
    assert result["test/group"]["noisy"]["rmse_relative_to_clean"] == 2
    assert result["test/group"]["noisy"]["prediction_change_rms"] == 1
