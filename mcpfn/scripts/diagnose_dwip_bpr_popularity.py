"""Inner-only popularity controls; never neural training targets or goal baselines."""
import json

import numpy as np
import torch

from dwip_bpr_evaluation import CAMPAIGN, split_hashes
from dwip_recsys_experiment import load_split, cornac_split
from dwip_recsys_ranking_audit import independent_ranking_metrics
from evaluate_dwip_recsys import DATASETS, FrozenRatings, metrics, write_json


def main():
    torch.set_num_threads(2)
    report = {"partition": "validation", "purpose": "Diagnostic evaluation only; no neural gradients or teacher targets",
              "replaces_goal_baseline": False, "split_sha256": split_hashes(), "datasets": {}}
    for spec in DATASETS.values():
        _, context, queries = load_split(spec["folder"], "validation")
        split = cornac_split(context, queries)
        matrix = split.train_set.csr_matrix
        scores = {"interaction_popularity": np.asarray(matrix.getnnz(axis=0), dtype=np.float64),
                  "positive_popularity": np.asarray((matrix >= 4).sum(axis=0)).ravel().astype(np.float64)}
        results = {}
        for name, values in scores.items():
            prediction = np.broadcast_to(values, matrix.shape)
            independent, _ = independent_ranking_metrics(prediction, matrix, split.test_set.csr_matrix)
            result, _ = split.evaluate(model=FrozenRatings(prediction=prediction), metrics=metrics()[2:], user_based=True)
            if any(not np.isclose(score, result.metric_avg_results[metric], rtol=1e-12, atol=1e-12)
                   for metric, score in independent.items()):
                raise ValueError("Popularity control ranking audit mismatch")
            results[name] = independent
        report["datasets"][spec["folder"]] = results
        print(json.dumps({spec["folder"]: results}), flush=True)
    write_json(CAMPAIGN / "inner_popularity_diagnostic.json", report)


if __name__ == "__main__":
    main()
