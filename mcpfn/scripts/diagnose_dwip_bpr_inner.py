"""Inner-only ranking-gap diagnostics from saved reference factors and DWIP scores."""
import csv
import json
from pathlib import Path

import cornac.models.bpr.recom_bpr as bpr_module
import numpy as np

from dwip_bpr_evaluation import CAMPAIGN, INITIAL, TARGET_METRICS
from dwip_recsys_experiment import load_split, cornac_split
from dwip_recsys_ranking_audit import independent_ranking_metrics
from evaluate_dwip_recsys import DATASETS, sha256, write_json


def main():
    references = json.loads((CAMPAIGN / "bpr_baseline.json").read_text())["validation"]
    report = {"partition": "validation", "purpose": "Evaluation diagnostics only; not gradient data or teachers", "datasets": {}}
    for spec in DATASETS.values():
        dataset = spec["folder"]
        observed, context, queries = load_split(dataset, "validation")
        split = cornac_split(context, queries)
        path = CAMPAIGN / "bpr_reference/validation" / dataset / "factors.npz"
        factors = np.load(path)
        model = bpr_module.BPR(k=10, trainable=False)
        model.u_factors = factors["user_factors"]
        model.i_factors = factors["item_factors"]
        model.i_biases = factors["item_biases"]
        model.num_users, model.num_items = split.train_set.num_users, split.train_set.num_items
        prediction = np.stack([model.score(user) for user in range(model.num_users)])
        values, per_user = independent_ranking_metrics(prediction, split.train_set.csr_matrix, split.test_set.csr_matrix)
        if any(not np.isclose(values[metric], references[dataset][metric], rtol=1e-12, atol=1e-12) for metric in values):
            raise ValueError("Saved BPR factors do not reproduce frozen reference")
        source = INITIAL.parent / "context_2000x1000_validation" / dataset / "per_user_metrics.csv"
        dwip = {int(row["user"]): row for row in csv.DictReader(source.open())}
        raw_users = np.array([int(raw) for raw, index in sorted(split.train_set.uid_map.items(), key=lambda entry: entry[1])])
        support = np.isfinite(observed).sum(1)
        strata = []
        for low, high in ((1, 10), (10, 30), (30, 100), (100, 100000)):
            users = [user for user in per_user if low <= support[raw_users[user]] < high]
            if not users:
                continue
            strata.append({"context_range": [low, high], "users": len(users),
                           "BPR": {metric: float(np.mean([per_user[user][metric] for user in users])) for metric in TARGET_METRICS},
                           "DWIP": {metric: float(np.mean([float(dwip[raw_users[user]][metric]) for user in users])) for metric in TARGET_METRICS},
                           "BPR_zero_recall_fraction": float(np.mean([per_user[user]["Recall@10"] == 0 for user in users])),
                           "DWIP_zero_recall_fraction": float(np.mean([float(dwip[raw_users[user]]["Recall@10"]) == 0 for user in users]))})
        report["datasets"][dataset] = {"strata": strata, "factors_sha256": sha256(path), "dwip_scores_sha256": sha256(source)}
        print(json.dumps({dataset: strata}), flush=True)
    write_json(CAMPAIGN / "inner_ranking_gap_by_user_support.json", report)


if __name__ == "__main__":
    main()
