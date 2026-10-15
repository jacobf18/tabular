"""Synthetic-only audit of cross-catalog candidate diversity."""
import json

import numpy as np
import torch

from dwip_bpr_evaluation import CAMPAIGN
from evaluate_dwip_recsys import write_json
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def main():
    torch.set_num_threads(2)
    results = {}
    for ordering in ("first_observation", "first_observation_catalog"):
        records = []
        for seed in range(101, 141):
            episode = sample_rating_episode(seed, rows=64, cols=96, parent_rows=512, parent_cols=1024,
                                             density_range=(.004, .08), normalization="two_way_user_scale",
                                             query_limit=512, point_target="expected_rating", ordinal_targets=True,
                                             preference_family="mixed", context_ordering=ordering,
                                             item_popularity_sigma_range=(1.4, 2.8), query_sampling="user_balanced",
                                             exclude_unknown_queries=True, ranking_query_mode="user_matched",
                                             negatives_per_user=96, ranking_max_users=16)
            if episode.metadata["gradient_data"] != "synthetic":
                raise ValueError("Prior audit cannot load real data")
            support = episode.global_col_support.float()
            known = support > 0
            records.append({"seed": seed, "log_item_support_std": float(support[known].log1p().std()),
                            "known_item_support_median": float(support[known].median()),
                            "known_item_support_max": float(support[known].max()),
                            "rating_queries": episode.metadata["rating_queries"],
                            "negative_queries": episode.metadata["negative_queries"]})
        results[ordering] = {"episodes": records, "mean_log_item_support_std": float(np.mean([row["log_item_support_std"] for row in records]))}
    report = {"synthetic_only": True, "real_data_loaded": False, "results": results,
              "interpretation": "Measures candidate diversity only; does not establish recommendation-quality improvement"}
    write_json(CAMPAIGN / "catalog_prior_contrast_audit.json", report)
    print(json.dumps({name: values["mean_log_item_support_std"] for name, values in results.items()}), flush=True)


if __name__ == "__main__":
    main()
