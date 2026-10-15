"""Screen ordered synthetic crops against inner-context support statistics."""
import argparse
from pathlib import Path

import numpy as np
import torch

from diagnose_dwip_recsys_prior_fit import describe, query_context_support, user_weights
from dwip_recsys_experiment import load_split
from evaluate_dwip_recsys import DATASETS, sha256, write_json
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    report = {"real_inner": {}, "synthetic": {}, "seed_base": 96000000, "episodes": args.episodes}
    for spec in DATASETS.values():
        observed, _, queries = load_split(spec["folder"], "validation")
        rows, cols = query_context_support(observed, queries[0], queries[1])
        report["real_inner"][spec["folder"]] = describe(rows, cols, user_weights(queries[0]))
    recipes = [("iid", 1024, (.02, .20)), ("first_observation", 1024, (.02, .20)),
               ("first_observation", 2048, (.005, .08)), ("first_observation", 2048, (.01, .12))]
    for ordering, parent_size, density in recipes:
        supports, weights, empty = [], [], 0
        for repeat in range(args.episodes):
            rows, cols = (512, 320) if repeat % 2 else (320, 512)
            episode = sample_rating_episode(96000000 + repeat, rows, cols, parent_rows=parent_size,
                                            parent_cols=parent_size, normalization="two_way", density_range=density,
                                            item_popularity_sigma_range=(1.4, 2.8), context_ordering=ordering)
            query = episode.rating_query[0].nonzero().numpy()
            if not len(query):
                empty += 1
                continue
            supports.append(query_context_support(episode.source[0].numpy(), query[:, 0], query[:, 1]))
            weights.append(user_weights(query[:, 0]))
        label = f"{ordering}_parent{parent_size}_density{density[0]}_{density[1]}"
        rows = np.concatenate([value[0] for value in supports])
        cols = np.concatenate([value[1] for value in supports])
        report["synthetic"][label] = {**describe(rows, cols, np.concatenate(weights)), "empty_episodes": empty}
        print(label, report["synthetic"][label], flush=True)
    sources = (Path(__file__), Path(__file__).parents[1] / "src/tabimpute/prior/recsys_rating_episode.py")
    report["source_sha256"] = {str(path): sha256(path) for path in sources}
    write_json(args.output, report)


if __name__ == "__main__":
    main()
