"""Audit synthetic finite-query macro geometry against inner evaluation counts."""
import argparse
from pathlib import Path

import numpy as np
import torch

from dwip_recsys_experiment import load_split
from evaluate_dwip_recsys import DATASETS, sha256, write_json
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def summarize_counts(counts):
    counts = np.asarray(counts)
    counts = counts[counts > 0]
    return {"users": len(counts), "mean_queries": float(counts.mean()),
            "singleton_fraction": float((counts == 1).mean()),
            "two_query_fraction": float((counts == 2).mean()),
            "quantiles": np.quantile(counts, [.1, .25, .5, .75, .9]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    report = {"purpose": "Diagnostic only; no neural training or outer labels",
              "seed_base": 98000000, "episodes": args.episodes, "synthetic": {}, "inner_evaluation": {}}
    for query_limit in (1024, 4096):
        counts = []
        for index in range(args.episodes):
            episode = sample_rating_episode(98000000 + index, 1000, 1000, query_limit=query_limit,
                                             parent_rows=2048, parent_cols=2048, normalization="two_way",
                                             point_target="expected_rating", density_range=(.002, .08),
                                             item_popularity_sigma_range=(1.4, 2.8),
                                             context_ordering="first_observation", query_sampling="user_balanced")
            counts.extend(episode.rating_query[0].sum(1).tolist())
        report["synthetic"][str(query_limit)] = summarize_counts(counts)
    for spec in DATASETS.values():
        _, _, queries = load_split(spec["folder"], "validation")
        report["inner_evaluation"][spec["folder"]] = summarize_counts(np.bincount(queries[0]))
    report["source_sha256"] = {str(path): sha256(path) for path in (
        Path(__file__), Path(__file__).parents[1] / "src/tabimpute/prior/recsys_rating_episode.py")}
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
