import numpy as np
import pytest

from diagnose_dwip_recsys_user_scale import describe_scales, split_half_variances


def test_split_half_variances_preserve_missing_counts_and_constant_profiles():
    source = np.array([[2., 2., np.nan, 2., 2.], [np.nan] * 5, [1., np.nan, np.nan, np.nan, np.nan]])
    counts, variances = split_half_variances(source, 17)
    np.testing.assert_array_equal(counts, [4, 0, 1])
    np.testing.assert_array_equal(variances[0], [0., 0.])
    assert np.isnan(variances[1:]).all()


def test_split_half_variances_match_unbiased_variance_of_partition():
    source = np.arange(1., 13.).reshape(1, -1)
    order = np.random.default_rng(23).permutation(source.size)
    _, variances = split_half_variances(source, 23)
    expected = [np.var(source[0, order[half::2]], ddof=1) for half in (0, 1)]
    np.testing.assert_allclose(variances[0], expected)


def test_split_half_variances_are_translation_and_scale_equivariant():
    source = np.random.default_rng(8).normal(size=(10, 80))
    counts, variances = split_half_variances(source, 9)
    changed_counts, changed = split_half_variances(3 * source + np.arange(10)[:, None], 9)
    np.testing.assert_array_equal(counts, changed_counts)
    np.testing.assert_allclose(changed, 9 * variances)


def test_split_half_variances_do_not_consume_global_random_state():
    source = np.arange(1., 81.).reshape(2, -1)
    np.random.seed(31)
    expected = np.random.random(5)
    np.random.seed(31)
    _, first = split_half_variances(source, 3)
    _, second = split_half_variances(source, 3)
    np.testing.assert_array_equal(np.random.random(5), expected)
    np.testing.assert_array_equal(first, second)


def test_scale_description_stratifies_context_counts():
    counts = np.array([10, 20, 50, 100])
    result = describe_scales(counts, np.array([[4., 4.], [1., 1.], [2., 2.], [3., 3.]]))
    assert result["20_49"]["block_user_profiles"] == 1
    assert result["20_1000000"]["block_user_profiles"] == 3
    assert result["20_1000000"]["split_half_variance_correlation"] == pytest.approx(1)
    assert result["20_1000000"]["mean_split_half_variance_gap"] == 0
    assert result["20_49"]["split_half_variance_correlation"] is None


@pytest.mark.parametrize("source", [np.zeros(2), np.array([[np.inf, 1.]])])
def test_invalid_scale_context_fails(source):
    with pytest.raises(ValueError):
        split_half_variances(source, 0)


def test_scale_description_rejects_wrong_pair_shape():
    with pytest.raises(ValueError):
        describe_scales(np.array([30, 40]), np.array([1., 2.]))
