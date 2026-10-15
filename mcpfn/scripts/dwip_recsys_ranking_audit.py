"""Independent NumPy ranking metrics for the fixed known-item Cornac protocol."""
import numpy as np


RANKING_METRICS = ("AUC", "MAP", "NDCG@10", "Precision@10", "Recall@10")


def independent_ranking_metrics(prediction, train_matrix, test_matrix):
    prediction = np.asarray(prediction)
    if prediction.shape != train_matrix.shape or not np.isfinite(prediction).all():
        raise ValueError("Require finite predictions aligned with the training mapping")
    per_user = {}
    discounts = 1 / np.log2(np.arange(2, 12))
    for user in np.unique(test_matrix.nonzero()[0]):
        test_row = test_matrix.getrow(user)
        positive_items = test_row.indices[test_row.data >= 4]
        if not len(positive_items):
            continue
        if user >= prediction.shape[0] or np.any(positive_items >= prediction.shape[1]):
            raise ValueError("Ranking queries must use known users and items")
        train_row = train_matrix.getrow(user)
        excluded_items = train_row.indices[train_row.data >= 4]
        if np.intersect1d(positive_items, train_row.indices).size:
            raise ValueError("Ranking query overlaps observed context")
        eligible = np.ones(prediction.shape[1], dtype=bool)
        eligible[excluded_items] = False
        positive = np.zeros(prediction.shape[1], dtype=bool)
        positive[positive_items] = True
        items = np.flatnonzero(eligible)
        scores = prediction[user, items]
        relevant = positive[items]
        positive_scores = scores[relevant]
        negative_scores = scores[~relevant]
        if len(scores) < 10 or not len(negative_scores):
            raise ValueError("Require ten candidates and nonempty negative items")
        ordered_scores = np.sort(scores)
        ordered_positive = np.sort(positive_scores)
        ordered_negative = np.sort(negative_scores)
        strict_wins = np.searchsorted(ordered_negative, positive_scores, side="left").sum()
        all_ranks = len(scores) - np.searchsorted(ordered_scores, positive_scores, side="left")
        positive_ranks = len(positive_scores) - np.searchsorted(ordered_positive, positive_scores, side="left")
        top = np.argpartition(scores, -10)[-10:]
        top = top[np.argsort(scores[top])[::-1]]
        hits = relevant[top]
        per_user[int(user)] = {
            "AUC": float(strict_wins / (len(positive_scores) * len(negative_scores))),
            "MAP": float(np.mean(positive_ranks / all_ranks)),
            "NDCG@10": float(np.sum(hits * discounts) / np.sum(discounts[:min(len(positive_items), 10)])),
            "Precision@10": float(hits.sum() / 10),
            "Recall@10": float(hits.sum() / len(positive_items)),
        }
    if not per_user:
        raise ValueError("No users with relevant ranking queries")
    return {metric: float(np.mean([values[metric] for values in per_user.values()])) for metric in RANKING_METRICS}, per_user
