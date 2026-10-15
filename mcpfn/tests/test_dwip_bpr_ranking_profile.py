import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dwip_bpr_ranking_profile import ranking_profile
from dwip_recsys_ranking_audit import independent_ranking_metrics


def test_profile_matches_frozen_ranking_and_does_not_mutate_predictions():
    train = csr_matrix(([5., 2., 1., 4.], ([0, 0, 1, 1], [0, 1, 2, 3])), shape=(2, 16))
    test = csr_matrix(([4., 5.], ([0, 1], [4, 5])), shape=(2, 16))
    prediction = np.tile(np.linspace(1, 5, 16), (2, 1))
    prediction[0, 0] = 5
    original = prediction.copy()
    profile = ranking_profile(prediction, train, test, np.arange(16))
    metrics, _ = independent_ranking_metrics(prediction, train, test)
    assert profile["all"]["recall10"] == metrics["Recall@10"]
    assert profile["all"]["users"] == 2
    assert profile["all"]["top10_first1000_fraction"] == 1
    assert np.array_equal(prediction, original)


def test_profile_preserves_observed_low_candidates_and_excludes_observed_positives():
    train = csr_matrix(([2., 5.], ([0, 0], [0, 1])), shape=(1, 12))
    test = csr_matrix(([4.], ([0], [2])), shape=(1, 12))
    prediction = np.array([[3., 5., 4., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    profile = ranking_profile(prediction, train, test, np.arange(12))["all"]
    assert profile["top10_observed_low_fraction"] == pytest.approx(.1)
    assert profile["top10_score_ceiling_fraction"] == 0
    assert profile["recall10"] == 1
