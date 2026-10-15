"""Audit synthetic parent-context support without using outer evaluation labels."""
import argparse
from pathlib import Path

import numpy as np
import torch

from diagnose_dwip_recsys_prior_fit import describe, query_context_support, user_weights
from dwip_recsys_experiment import load_split
from evaluate_dwip_recsys import DATASETS, sha256, write_json
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def summarize(supports, weights):
    values = [np.concatenate([entry[index] for entry in supports]) for index in range(4)]
    weights = np.concatenate(weights)
    return {"local": describe(values[0], values[1], weights),
            "global": describe(values[2], values[3], weights),
            "mean_global_to_local_row_ratio": float(np.average(values[2] / np.maximum(values[0], 1), weights=weights)),
            "mean_global_to_local_col_ratio": float(np.average(values[3] / np.maximum(values[1], 1), weights=weights))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    report = {"purpose": "Inner context counts only; no outer labels or neural fitting",
              "real_inner": {}, "synthetic": {}, "episodes": args.episodes, "seed_base": 97000000}
    for spec in DATASETS.values():
        observed, _, queries = load_split(spec["folder"], "validation")
        rows, cols = query_context_support(observed, queries[0], queries[1])
        support = np.isfinite(observed)
        global_rows = support.sum(1)[queries[0]]
        global_cols = support.sum(0)[queries[1]]
        report["real_inner"][spec["folder"]] = summarize([(rows, cols, global_rows, global_cols)], [user_weights(queries[0])])
    recipes = ((2048, 2048, (.002, .08)), (8192, 4096, (.002, .04)), (8192, 4096, (.002, .08)))
    if args.extended:
        recipes = ((8192, 4096, (.002, .08)), (12288, 4096, (.004, .08)), (16384, 4096, (.004, .08)))
    for parent_rows, parent_cols, density in recipes:
        supports, weights = [], []
        for index in range(args.episodes):
            episode = sample_rating_episode(97000000 + index, 1000, 1000, query_limit=4096,
                                             parent_rows=parent_rows, parent_cols=parent_cols,
                                             density_range=density, normalization="two_way",
                                             item_popularity_sigma_range=(1.4, 2.8), context_ordering="first_observation",
                                             query_sampling="user_balanced", exclude_unknown_queries=True)
            query = episode.rating_query[0].nonzero().numpy()
            if not len(query):
                continue
            rows, cols = query_context_support(episode.source[0].numpy(), query[:, 0], query[:, 1])
            supports.append((rows, cols, episode.global_row_support.numpy()[query[:, 0]], episode.global_col_support.numpy()[query[:, 1]]))
            weights.append(user_weights(query[:, 0]))
        label = f"parent{parent_rows}x{parent_cols}_density{density[0]}_{density[1]}"
        report["synthetic"][label] = summarize(supports, weights)
        print(label, report["synthetic"][label], flush=True)
    report["source_sha256"] = {str(path): sha256(path) for path in (
        Path(__file__), Path(__file__).parents[1] / "src/tabimpute/prior/recsys_rating_episode.py")}
    write_json(args.output, report)


if __name__ == "__main__":
    main()
