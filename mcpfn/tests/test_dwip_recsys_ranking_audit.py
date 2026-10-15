import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dwip_recsys_experiment import cornac_split
from dwip_recsys_ranking_audit import independent_ranking_metrics, RANKING_METRICS
from evaluate_dwip_recsys import FrozenRatings, metrics
from evaluate_dwip_recsys_candidate import remap_prediction


@pytest.mark.parametrize("tied", [False, True])
def test_independent_ranking_matches_cornac_per_user_with_ties_and_mapping(tied):
    users, items = np.indices((3, 32))
    ratings = 1 + (users * 3 + items) % 5
    held_out = (users + items) % 4 == 0
    ratings[(users == 2) & held_out] = 2
    context = tuple(values[~held_out] for values in (users, items, ratings))
    queries = tuple(values[held_out] for values in (users, items, ratings))
    split = cornac_split(context, queries)
    prediction = np.random.default_rng(42).uniform(1, 5, (3, 32))
    if tied:
        prediction = prediction.round()
    prediction[context[0], context[1]] = context[2]
    mapped, _ = remap_prediction(prediction, split.train_set)
    result, _ = split.evaluate(model=FrozenRatings(prediction=mapped), metrics=metrics(), user_based=True)
    actual, per_user = independent_ranking_metrics(mapped, split.train_set.csr_matrix, split.test_set.csr_matrix)
    for metric in RANKING_METRICS:
        assert actual[metric] == pytest.approx(result.metric_avg_results[metric], abs=1e-12)
        assert set(per_user) == set(result.metric_user_results[metric])
        for user, scores in per_user.items():
            assert scores[metric] == pytest.approx(result.metric_user_results[metric][user], abs=1e-12)
    assert len(per_user) == 2


def test_auc_ties_are_strict_and_ap_uses_maximum_tied_rank():
    prediction = np.full((1, 12), 3.)
    context = csr_matrix(([5.], ([0], [11])), shape=(1, 12))
    queries = csr_matrix(([5.], ([0], [0])), shape=(1, 12))
    actual, _ = independent_ranking_metrics(prediction, context, queries)
    assert actual["AUC"] == 0
    assert actual["MAP"] == pytest.approx(1 / 11)


def test_context_low_ratings_remain_candidates():
    prediction = np.arange(12, dtype=float).reshape(1, 12)
    context = csr_matrix(([2.], ([0], [11])), shape=(1, 12))
    queries = csr_matrix(([5.], ([0], [10])), shape=(1, 12))
    actual, _ = independent_ranking_metrics(prediction, context, queries)
    assert actual["AUC"] == pytest.approx(10 / 11)
    assert actual["MAP"] == pytest.approx(.5)


def test_ranking_context_query_overlap_is_rejected():
    context = csr_matrix(([2.], ([0], [0])), shape=(1, 12))
    queries = csr_matrix(([5.], ([0], [0])), shape=(1, 12))
    with pytest.raises(ValueError, match="overlaps observed"):
        independent_ranking_metrics(np.ones((1, 12)), context, queries)
