import numpy as np
import pandas as pd
from scipy.stats import t

from diagnose_synthetic_embedding_geometry import geometry, partition_scatter, summarize


def test_pure_mask_signal_and_group_signal_are_distinguished():
    labels = np.repeat(np.arange(4), 4)
    observed = np.tile([False, True], 8)
    result = geometry(observed[:, None].astype(float), labels, observed)
    np.testing.assert_allclose([result['mask_eta2'], result['group_eta2'], result['joint_eta2']], [1., 0., 1.])
    result = geometry(np.eye(4)[labels], labels, observed)
    np.testing.assert_allclose([result['mask_eta2'], result['group_eta2'], result['group_after_mask_eta2']], [0., 1., 1.])


def test_scatter_is_translation_and_rotation_invariant():
    generator = np.random.default_rng(13)
    features = generator.normal(size=(64, 7))
    labels = np.repeat(np.arange(4), 16)
    rotation, _ = np.linalg.qr(generator.normal(size=(7, 7)))
    expected = partition_scatter(features, labels)
    np.testing.assert_allclose(partition_scatter(features @ rotation + 100, labels), expected)


def test_joint_partition_contains_both_signals_without_double_counting():
    labels = np.repeat(np.arange(4), 4)
    observed = np.tile([False, True], 8)
    features = np.column_stack([np.eye(4)[labels], observed * 5])
    result = geometry(features, labels, observed)
    assert 0 < result['group_eta2'] < result['mask_eta2'] < 1
    np.testing.assert_allclose(result['group_after_mask_eta2'], result['group_eta2'])
    np.testing.assert_allclose(result['joint_eta2'], 1.)


def test_constant_features_do_not_invent_explained_variance():
    result = geometry(np.ones((8, 3)), np.repeat(np.arange(4), 2), np.tile([True, False], 4))
    assert all(np.isnan(value) for value in result.values())


def test_summary_uses_matrix_level_student_t_intervals():
    frame = pd.DataFrame(dict(noise=[.25] * 10, axis=['row'] * 10, score=np.arange(10) / 10))
    summary = summarize(frame, ['noise', 'axis'], ['score']).iloc[0]
    assert summary.matrices == 10
    np.testing.assert_allclose(summary.score, .45)
    np.testing.assert_allclose(summary.score_ci95_half, t.ppf(.975, 9) * frame.score.std(ddof=1) / np.sqrt(10))
