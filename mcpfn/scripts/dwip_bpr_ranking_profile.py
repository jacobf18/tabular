"""Read-only ranking diagnostics under the frozen Cornac candidate protocol."""
import numpy as np


def ranking_profile(prediction, train_matrix, test_matrix, original_items):
    if prediction.shape != train_matrix.shape or original_items.shape != (prediction.shape[1],):
        raise ValueError("Ranking profile shape/mapping mismatch")
    item_support = np.asarray(train_matrix.getnnz(axis=0))
    records = []
    for user in np.unique(test_matrix.nonzero()[0]):
        test = test_matrix.getrow(user)
        positive_items = test.indices[test.data >= 4]
        if not len(positive_items):
            continue
        context = train_matrix.getrow(user)
        eligible = np.ones(prediction.shape[1], dtype=bool)
        eligible[context.indices[context.data >= 4]] = False
        items = np.flatnonzero(eligible)
        scores = prediction[user, items]
        top = np.argpartition(scores, -10)[-10:]
        top = top[np.argsort(scores[top])[::-1]]
        recommended = items[top]
        low_observed = context.indices[context.data < 4]
        records.append({
            "context_count": len(context.indices),
            "top10_mean_item_support": float(item_support[recommended].mean()),
            "positive_mean_item_support": float(item_support[positive_items].mean()),
            "top10_support_le10_fraction": float(np.mean(item_support[recommended] <= 10)),
            "positive_support_le10_fraction": float(np.mean(item_support[positive_items] <= 10)),
            "top10_observed_low_fraction": float(np.isin(recommended, low_observed).mean()),
            "top10_score_ceiling_fraction": float(np.mean(scores[top] >= 5)),
            "candidate_score_ceiling_fraction": float(np.mean(scores >= 5)),
            "positive_score_ceiling_fraction": float(np.mean(prediction[user, positive_items] >= 5)),
            "top10_first1000_fraction": float(np.mean(original_items[recommended] < 1000)),
            "positive_first1000_fraction": float(np.mean(original_items[positive_items] < 1000)),
            "recall10": float(np.isin(recommended, positive_items).sum() / len(positive_items)),
        })
    if not records:
        raise ValueError("Require ranking users")

    def summarize(values):
        return {"users": len(values), **{key: float(np.mean([row[key] for row in values]))
                for key in values[0] if key != "context_count"}}

    strata = {}
    for low, high in ((1, 10), (10, 30), (30, 100), (100, 1000000)):
        selected = [row for row in records if low <= row["context_count"] < high]
        if selected:
            strata[f"{low}_to_{high}"] = summarize(selected)
    return {"all": summarize(records), "by_context_count": strata,
            "protocol": "Read-only; context positives excluded, context low ratings retained; top-10 ties match Cornac"}
