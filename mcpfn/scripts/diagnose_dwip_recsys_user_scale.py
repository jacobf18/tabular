"""Context-only split-half residual variability diagnostic for synthetic priors."""
import argparse
from pathlib import Path

import numpy as np
import torch

from dwip_recsys_experiment import block_spans, inference_components, load_split
from evaluate_dwip_recsys import DATASETS, sha256, write_json
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def split_half_variances(source, seed):
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2 or np.isinf(source).any():
        raise ValueError("Require a finite-or-missing matrix")
    rows, cols = np.where(np.isfinite(source))
    permutation = np.random.default_rng(seed).permutation(len(rows))
    order = permutation[np.argsort(rows[permutation], kind="stable")]
    rows, values = rows[order], source[rows[order], cols[order]]
    counts = np.bincount(rows, minlength=source.shape[0])
    starts = np.cumsum(counts) - counts
    halves = (np.arange(len(rows)) - starts[rows]) % 2
    groups = rows * 2 + halves
    half_counts = np.bincount(groups, minlength=2 * len(counts)).reshape(-1, 2)
    sums = np.bincount(groups, weights=values, minlength=2 * len(counts)).reshape(-1, 2)
    squares = np.bincount(groups, weights=values ** 2, minlength=2 * len(counts)).reshape(-1, 2)
    variances = np.maximum(squares - sums ** 2 / half_counts.clip(1), 0) / (half_counts - 1).clip(1)
    variances[half_counts < 2] = np.nan
    return counts, variances


def describe_scales(counts, variances):
    counts, variances = np.asarray(counts), np.asarray(variances)
    if variances.shape != (len(counts), 2):
        raise ValueError("Require two split-half variances per context profile")
    result = {}
    for lower, upper in ((20, 49), (50, 99), (100, 1000000), (20, 1000000)):
        selected = (counts >= lower) & (counts <= upper) & np.isfinite(variances).all(1)
        paired = variances[selected]
        if not len(paired):
            continue
        deviations = np.sqrt(paired.mean(1))
        correlation = None
        if len(paired) > 1 and (paired.std(0) > 1e-12).all():
            correlation = float(np.corrcoef(paired.T)[0, 1])
        result[f"{lower}_{upper}"] = {
            "block_user_profiles": int(selected.sum()),
            "mean_context_count": float(counts[selected].mean()),
            "split_half_variance_correlation": correlation,
            "std_quantiles": np.quantile(deviations, [.1, .25, .5, .75, .9]).tolist(),
            "std_coefficient_of_variation": float(deviations.std() / max(deviations.mean(), 1e-12)),
            "mean_split_half_variance_gap": float(np.abs(paired[:, 0] - paired[:, 1]).mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--user-noise-log-std-low", type=float, default=0.)
    parser.add_argument("--user-noise-log-std-high", type=float, default=0.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes < 1:
        raise ValueError("Require at least one synthetic episode")
    torch.set_num_threads(4)
    report = {"purpose": "Inner context only; no query labels, neural fitting or outer evaluation",
              "scope": "Equal block-user profiles with >=20 observations; residual variation includes signal and noise, not an identified noise estimate",
              "block_shape": [1000, 1000], "seed_base": 94000000, "episodes": args.episodes,
              "user_noise_log_std_range": [args.user_noise_log_std_low, args.user_noise_log_std_high],
              "real_inner_context": {}, "synthetic": {}}
    for spec in DATASETS.values():
        observed, _, _ = load_split(spec["folder"], "validation")
        means, stds, offsets = inference_components(observed, "two_way")
        source = (observed - offsets[:, None] - means) / stds
        counts, variances = [], []
        for row_start, row_end in block_spans(source.shape[0]):
            for col_start, col_end in block_spans(source.shape[1]):
                count, variance = split_half_variances(source[row_start:row_end, col_start:col_end], 94000000)
                counts.append(count)
                variances.append(variance)
        report["real_inner_context"][spec["folder"]] = describe_scales(np.concatenate(counts), np.concatenate(variances))
    for uniform_high in (0., .15):
        counts, variances = [], []
        for index in range(args.episodes):
            episode = sample_rating_episode(94000000 + index, 1000, 1000, query_limit=4096,
                                             parent_rows=12288, parent_cols=4096, density_range=(.004, .08),
                                             normalization="two_way", point_target="expected_rating",
                                             item_popularity_sigma_range=(1.4, 2.8), context_ordering="first_observation",
                                             query_sampling="user_balanced", preference_family="mixed", exclude_unknown_queries=True,
                                             uniform_noise_range=(0., uniform_high),
                                             user_noise_log_std_range=(args.user_noise_log_std_low, args.user_noise_log_std_high))
            count, variance = split_half_variances(episode.source[0].numpy(), 94000000 + index)
            counts.append(count)
            variances.append(variance)
        report["synthetic"][f"uniform_high_{uniform_high}"] = describe_scales(np.concatenate(counts), np.concatenate(variances))
        print(uniform_high, report["synthetic"][f"uniform_high_{uniform_high}"], flush=True)
    report["source_sha256"] = {str(path): sha256(path) for path in (
        Path(__file__), Path(__file__).parents[1] / "src/tabimpute/prior/recsys_rating_episode.py")}
    write_json(args.output, report)


if __name__ == "__main__":
    main()
