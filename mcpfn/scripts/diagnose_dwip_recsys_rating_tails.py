"""Compare synthetic ordinal residual tails with the fixed inner evaluation."""
import argparse
from pathlib import Path

import numpy as np
import torch

from diagnose_dwip_recsys_prior_fit import user_weights, weighted_quantiles
from dwip_recsys_experiment import load_split, inference_components
from evaluate_dwip_recsys import DATASETS, sha256, write_json
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def describe_tails(residuals, ratings, weights):
    residuals, ratings, weights = (np.asarray(values, dtype=np.float64) for values in (residuals, ratings, weights))
    weights = weights / weights.sum()
    mean = float(np.sum(weights * residuals))
    variance = float(np.sum(weights * (residuals - mean) ** 2))
    standardized = (residuals - mean) / max(variance ** .5, 1e-12)
    return {"residual_mean": mean, "residual_std": variance ** .5,
            "skewness": float(np.sum(weights * standardized ** 3)),
            "excess_kurtosis": float(np.sum(weights * standardized ** 4) - 3),
            "below_minus_two": float(weights[residuals < -2].sum()),
            "above_plus_two": float(weights[residuals > 2].sum()),
            "residual_quantiles": weighted_quantiles(residuals, weights),
            "rating_probabilities": {str(rating): float(weights[np.rint(ratings) == rating].sum()) for rating in range(1, 6)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--uniform-noise-high", type=float, default=0.0)
    parser.add_argument("--user-noise-log-std-low", type=float, default=0.0)
    parser.add_argument("--user-noise-log-std-high", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    report = {"purpose": "Inner evaluation/prior diagnostic only; no outer labels, neural fitting or real-trained teachers",
              "real_inner": {}, "synthetic": {}, "seed_base": 95000000, "episodes": args.episodes,
              "uniform_noise_range": [0.0, args.uniform_noise_high],
              "user_noise_log_std_range": [args.user_noise_log_std_low, args.user_noise_log_std_high]}
    for spec in DATASETS.values():
        observed, _, queries = load_split(spec["folder"], "validation")
        means, stds, offsets = inference_components(observed, "two_way")
        users, items, truth = queries
        residual = (truth - means[items] - offsets[users]) / stds[items]
        report["real_inner"][spec["folder"]] = describe_tails(residual, truth, user_weights(users))
    for parent_rows, parent_cols, density in ((2048, 2048, (.002, .08)), (12288, 4096, (.004, .08))):
        residuals, ratings, weights = [], [], []
        for index in range(args.episodes):
            episode = sample_rating_episode(95000000 + index, 1000, 1000, query_limit=4096,
                                             parent_rows=parent_rows, parent_cols=parent_cols, density_range=density,
                                             normalization="two_way", point_target="expected_rating",
                                             item_popularity_sigma_range=(1.4, 2.8), context_ordering="first_observation",
                                             query_sampling="user_balanced", preference_family="mixed", exclude_unknown_queries=True,
                                             uniform_noise_range=(0.0, args.uniform_noise_high),
                                             user_noise_log_std_range=(args.user_noise_log_std_low, args.user_noise_log_std_high))
            coordinates = episode.rating_query[0].nonzero()
            if not len(coordinates):
                continue
            users, items = coordinates[:, 0], coordinates[:, 1]
            residual = episode.target[episode.rating_query]
            truth = residual * episode.stds[items] + episode.means[items] + episode.row_offsets[users]
            residuals.extend(residual.tolist())
            ratings.extend(truth.tolist())
            weights.extend(user_weights(users.numpy()).tolist())
        label = f"parent{parent_rows}x{parent_cols}"
        report["synthetic"][label] = describe_tails(residuals, ratings, weights)
        print(label, report["synthetic"][label], flush=True)
    report["source_sha256"] = {str(path): sha256(path) for path in (
        Path(__file__), Path(__file__).parents[1] / "src/tabimpute/prior/recsys_rating_episode.py")}
    write_json(args.output, report)


if __name__ == "__main__":
    main()
