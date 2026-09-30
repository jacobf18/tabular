import numpy as np
import pytest

from evaluate_matrixformer_recsys_ratings import block_parameters, build_observed, normalization_parameters


def test_wider_user_blocks_preserve_item_width_and_chunk_count():
    assert block_parameters(2000, 1000) == {"max_num_rows": 2000, "max_num_cols": 1000, "max_num_chunks": 1}
    assert block_parameters(1000, 1000)["max_num_rows"] == 1000


@pytest.mark.parametrize("rows, columns", [(0, 1000), (2000, 0), (-1, 1000)])
def test_block_dimensions_must_be_positive(rows, columns):
    with pytest.raises(ValueError, match="positive"):
        block_parameters(rows, columns)


def test_global_normalization_shares_observed_only_statistics():
    observed = np.array([[1.0, 5.0], [3.0, np.nan]])
    means, stds = normalization_parameters(observed, "global")
    assert np.all(means == 3.0)
    assert np.all(stds == np.std([1.0, 5.0, 3.0]))
    column_means, _ = normalization_parameters(observed, "column")
    assert np.array_equal(column_means, [2.0, 5.0])


def test_global_normalization_handles_constants_and_rejects_empty_data():
    means, stds = normalization_parameters(np.array([[5.0, np.nan]]), "global")
    assert np.array_equal(means, [5.0, 5.0])
    assert np.array_equal(stds, [1.0, 1.0])
    with pytest.raises(ValueError, match="No observed"):
        normalization_parameters(np.full((2, 2), np.nan), "global")


def test_heldout_values_cannot_affect_observations_or_normalization():
    split = {"shape": (2, 2), "train_users": np.array([0, 1]), "train_items": np.array([0, 1]),
             "train_ratings": np.array([1.0, 5.0]), "test_users": np.array([0]),
             "test_items": np.array([1]), "test_ratings": np.array([4.0])}
    original = build_observed(split)
    split["test_ratings"][:] = -1000
    changed = build_observed(split)
    assert np.array_equal(original, changed, equal_nan=True)
    for mode in ("global", "column"):
        for first, second in zip(normalization_parameters(original, mode), normalization_parameters(changed, mode)):
            assert np.array_equal(first, second)
    split["test_items"][:] = 0
    with pytest.raises(ValueError, match="overlap"):
        build_observed(split)
