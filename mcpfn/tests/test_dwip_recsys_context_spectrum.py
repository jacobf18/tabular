import numpy as np
import pytest

from diagnose_dwip_recsys_context_spectrum import shuffle_observed_rows, spectral_statistics


def test_row_shuffle_preserves_support_and_each_users_observed_values():
    matrix = np.array([[1., np.nan, 2., 3.], [np.nan, 4., 5., 6.], [np.nan] * 4])
    shuffled = shuffle_observed_rows(matrix, 42)
    np.testing.assert_array_equal(np.isnan(shuffled), np.isnan(matrix))
    for original, changed in zip(matrix, shuffled):
        np.testing.assert_array_equal(np.sort(original[np.isfinite(original)]), np.sort(changed[np.isfinite(changed)]))
    np.testing.assert_array_equal(shuffled, shuffle_observed_rows(matrix, 42))
    assert not np.shares_memory(matrix, shuffled)


def test_spectrum_matches_known_rank_and_is_scale_invariant():
    matrix = np.diag([3., 2., 1., 0.])
    actual = spectral_statistics(matrix)
    assert actual["energy_top_1"] == pytest.approx(9 / 14)
    assert actual["participation_rank"] == pytest.approx(14 ** 2 / (81 + 16 + 1))
    assert actual["energy_top_4"] == pytest.approx(1.)
    scaled = spectral_statistics(matrix * 5)
    assert actual == pytest.approx(scaled)


def test_missing_context_is_zero_filled_without_creating_signal():
    empty = spectral_statistics(np.full((3, 4), np.nan))
    assert all(value == 0 for value in empty.values())
    missing = spectral_statistics(np.array([[1., np.nan], [np.nan, 1.]]))
    assert missing["participation_rank"] == pytest.approx(2.)


@pytest.mark.parametrize("matrix", [np.ones(3), np.array([[np.inf]])])
def test_invalid_spectral_inputs_rejected(matrix):
    with pytest.raises(ValueError, match="two-dimensional context"):
        spectral_statistics(matrix)
