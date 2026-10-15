import numpy as np
import pytest

from compare_recsys_column_context import load_training, select_columns, sparse_training


def test_width_selection_uses_capacity_not_heldout_scores():
    probes = [dict(columns=2000, status='ok', peak_gpu_allocated_bytes=40, total_gpu_bytes=100, RMSE=0.1),
              dict(columns=3000, status='ok', peak_gpu_allocated_bytes=70, total_gpu_bytes=100, RMSE=100),
              dict(columns=4000, status='oom')]
    assert select_columns(probes, [2000, 3000, 4000], 0.8) == 3000


@pytest.mark.parametrize('probes,candidates', [
    ([dict(columns=2000, status='oom')], [2000]),
    ([dict(columns=2000, status='error')], [2000]),
    ([dict(columns=2000, status='ok', peak_gpu_allocated_bytes=81, total_gpu_bytes=100)], [2000]),
    ([dict(columns=2000, status='oom')], [2000, 3000]),
])
def test_width_selection_rejects_incomplete_or_invalid_probes(probes, candidates):
    with pytest.raises(ValueError):
        select_columns(probes, candidates, 0.8)


def test_capacity_inputs_exclude_test_ratings(tmp_path):
    path = tmp_path / 'split.npz'
    np.savez(path, shape=[3, 2], train_users=[0, 2], train_items=[0, 1], train_ratings=[1., 5.],
             test_ratings=np.array([object()], dtype=object))
    train = load_training(path)
    assert set(train) == {'shape', 'train_users', 'train_items', 'train_ratings'}
    np.testing.assert_array_equal(sparse_training(train).toarray(), [[1., 0.], [0., 0.], [0., 5.]])


def test_duplicate_training_entries_are_rejected():
    with pytest.raises(ValueError, match='unique'):
        sparse_training(dict(shape=[2, 2], train_users=[0, 0], train_items=[0, 0], train_ratings=np.array([1., 2.])))
