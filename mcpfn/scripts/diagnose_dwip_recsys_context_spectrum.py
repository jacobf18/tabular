"""Context-only residual spectra relative to row-shuffled null matrices."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svdvals
import torch

from dwip_recsys_experiment import DEFAULT_ROOT, block_spans, inference_components, inference_row_scales, load_split
from evaluate_dwip_recsys import DATASETS, OUTPUT, sha256, write_json
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def shuffle_observed_rows(matrix, seed):
    random = np.random.default_rng(seed)
    shuffled = np.asarray(matrix).copy()
    for row in shuffled:
        observed = np.isfinite(row)
        row[observed] = random.permutation(row[observed])
    return shuffled


def spectral_statistics(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or np.isinf(matrix).any():
        raise ValueError("Require a finite-or-missing two-dimensional context")
    power = svdvals(np.nan_to_num(matrix), check_finite=False) ** 2
    total = power.sum()
    if total <= 0:
        return {"participation_rank": 0., **{f"energy_top_{count}": 0. for count in (1, 4, 16, 64)}}
    return {"participation_rank": float(total ** 2 / (power ** 2).sum()),
            **{f"energy_top_{count}": float(power[:count].sum() / total) for count in (1, 4, 16, 64)}}


def describe_context(matrix, seed):
    observed = np.isfinite(matrix)
    actual = spectral_statistics(matrix)
    nulls = [spectral_statistics(shuffle_observed_rows(matrix, seed + repeat)) for repeat in range(3)]
    null_mean = {metric: float(np.mean([values[metric] for values in nulls])) for metric in actual}
    return {"shape": list(matrix.shape), "density": float(observed.mean()),
            "row_support_median": float(np.median(observed.sum(1))),
            "column_support_median": float(np.median(observed.sum(0))),
            "actual": actual, "row_shuffled_null_mean": null_mean,
            "difference_from_null": {metric: actual[metric] - null_mean[metric] for metric in actual}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "context_residual_spectrum.json")
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes < 1:
        raise ValueError("Require at least one synthetic episode")
    torch.set_num_threads(4)
    report = {"purpose": "Context-only prior diagnostic; no neural fitting, MF teacher, or held-out rating values",
              "normalization": "two_way_user_scale", "real_inner_context": {}, "synthetic": [],
              "interpretation": "Sparse residual spectra, not identified latent ranks; row-shuffling retains exposure masks and each local user's residual distribution",
              "weighting": "Equal sampled blocks/episodes, not user-macro evaluation weighting"}
    for name, spec in DATASETS.items():
        observed, _, _ = load_split(spec["folder"], "validation")
        means, stds, offsets = inference_components(observed, "two_way_user_scale")
        scales = inference_row_scales(observed, "two_way_user_scale", means, stds, offsets)
        normalized = (observed - offsets[:, None] - means) / stds / scales[:, None]
        blocks = [(row_start, row_end, col_start, col_end)
                  for row_start, row_end in block_spans(observed.shape[0])
                  for col_start, col_end in block_spans(observed.shape[1])]
        indices = np.unique(np.linspace(0, len(blocks) - 1, min(8, len(blocks)), dtype=int))
        descriptions = []
        for index in indices:
            row_start, row_end, col_start, col_end = blocks[index]
            description = describe_context(normalized[row_start:row_end, col_start:col_end], 81000 + int(index))
            description["block"] = blocks[index]
            descriptions.append(description)
        report["real_inner_context"][name] = descriptions
    for repeat in range(args.episodes):
        episode = sample_rating_episode(82000 + repeat, 1000, 1000, parent_rows=12288, parent_cols=4096,
                                         density_range=(.004, .08), query_limit=4096, normalization="two_way_user_scale",
                                         point_target="expected_rating", preference_family="gaussian" if repeat % 2 == 0 else "clustered",
                                         item_popularity_sigma_range=(1.4, 2.8), context_ordering="first_observation",
                                         query_sampling="user_balanced", exclude_unknown_queries=True,
                                         uniform_noise_range=(0., .15), user_noise_log_std_range=(.15, .6))
        description = describe_context(episode.source[0].numpy(), 83000 + repeat)
        description["metadata"] = episode.metadata
        report["synthetic"].append(description)
    report["source_sha256"] = {str(path): sha256(path) for path in
        (Path(__file__), Path(__file__).with_name("dwip_recsys_experiment.py"),
         Path(__file__).parents[1] / "src/tabimpute/prior/recsys_rating_episode.py")}
    report["split_sha256"] = {spec["folder"]: sha256(OUTPUT / spec["folder"] / "split.npz") for spec in DATASETS.values()}
    write_json(args.output, report)
    groups = {**report["real_inner_context"], "synthetic": report["synthetic"]}
    print(json.dumps({name: {metric: float(np.mean([block["difference_from_null"][metric] for block in blocks]))
                             for metric in blocks[0]["actual"]} for name, blocks in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
