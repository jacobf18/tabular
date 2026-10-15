"""Compare synthetic and real query support without reading outer-test labels."""
import argparse
from pathlib import Path

import numpy as np
import torch

from dwip_recsys_experiment import DEFAULT_ROOT, load_split, block_spans
from diagnose_dwip_recsys_training import quantiles
from evaluate_dwip_recsys import DATASETS, write_json, sha256
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def query_context_support(observed, users, items, block=1000):
    support = np.isfinite(observed)
    row_support = np.zeros(len(users))
    col_support = np.zeros(len(users))
    coverage = np.zeros(len(users))
    for row_start, row_end in block_spans(observed.shape[0], block):
        for col_start, col_end in block_spans(observed.shape[1], block):
            selected = (users >= row_start) & (users < row_end) & (items >= col_start) & (items < col_end)
            local = support[row_start:row_end, col_start:col_end]
            row_support[selected] += local.sum(1)[users[selected] - row_start]
            col_support[selected] += local.sum(0)[items[selected] - col_start]
            coverage[selected] += 1
    if not (coverage > 0).all():
        raise ValueError("A query is outside the context geometry")
    return row_support / coverage, col_support / coverage


def weighted_quantiles(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    probabilities = (0, .1, .25, .5, .75, .9, 1)
    indices = np.searchsorted(cumulative, probabilities).clip(max=len(values) - 1)
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"), values[order][indices].tolist()))


def user_weights(users):
    unique, inverse = np.unique(users, return_inverse=True)
    return 1 / (np.bincount(inverse)[inverse] * len(unique))


def describe(rows, cols, weights=None):
    if weights is None:
        return {"query_row_support": quantiles(rows), "query_col_support": quantiles(cols),
                "fraction_query_row_support_le2": float(np.mean(rows <= 2)),
                "fraction_query_col_support_le2": float(np.mean(cols <= 2))}
    return {"query_row_support": weighted_quantiles(rows, weights), "query_col_support": weighted_quantiles(cols, weights),
            "fraction_query_row_support_le2": float(np.sum(weights * (rows <= 2)) / np.sum(weights)),
            "fraction_query_col_support_le2": float(np.sum(weights * (cols <= 2)) / np.sum(weights)),
            "quantile_method": "inverted weighted empirical CDF"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, type=Path)
    parser.add_argument("--calibrate-popularity", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"real_inner": {}, "synthetic": {}}
    for spec in DATASETS.values():
        observed, _, queries = load_split(spec["folder"], "validation")
        rows, cols = query_context_support(observed, queries[0], queries[1])
        report["real_inner"][spec["folder"]] = describe(rows, cols)
        report["real_inner"][spec["folder"]]["user_macro_weighted"] = describe(rows, cols, user_weights(queries[0]))
    for label, shape in (("pilot_320x160", (320, 160)), ("large_512x320", (512, 320))):
        all_rows, all_cols, all_weights = [], [], []
        for repeat in range(32):
            rows, cols = shape if repeat % 2 else shape[::-1]
            episode = sample_rating_episode(73000000 + repeat, rows, cols, normalization="shrunk_column")
            observed = episode.source[0].numpy()
            query = episode.rating_query[0].nonzero().numpy()
            row_support, col_support = query_context_support(observed, query[:, 0], query[:, 1])
            all_rows.extend(row_support.tolist())
            all_cols.extend(col_support.tolist())
            all_weights.extend(user_weights(query[:, 0]).tolist())
        report["synthetic"][label] = describe(np.asarray(all_rows), np.asarray(all_cols))
        report["synthetic"][label]["user_macro_weighted"] = describe(np.asarray(all_rows), np.asarray(all_cols), np.asarray(all_weights))
    report["source_sha256"] = sha256(Path(__file__))
    report["prior_sha256"] = sha256(Path(__file__).parents[1] / "src/tabimpute/prior/recsys_rating_episode.py")
    if args.calibrate_popularity:
        report["popularity_candidates"] = {}
        for popularity in ((.7, 1.9), (1.4, 2.8), (2.0, 3.2)):
            for density in ((.01, .12), (.02, .20)):
                all_rows, all_cols, all_weights = [], [], []
                for repeat in range(16):
                    rows, cols = (512, 320) if repeat % 2 else (320, 512)
                    episode = sample_rating_episode(75000000 + repeat, rows, cols, normalization="two_way",
                                                    density_range=density, item_popularity_sigma_range=popularity)
                    observed = episode.source[0].numpy()
                    query = episode.rating_query[0].nonzero().numpy()
                    row_support, col_support = query_context_support(observed, query[:, 0], query[:, 1])
                    all_rows.extend(row_support.tolist())
                    all_cols.extend(col_support.tolist())
                    all_weights.extend(user_weights(query[:, 0]).tolist())
                key = f"popularity_{popularity[0]}_{popularity[1]}_density_{density[0]}_{density[1]}"
                report["popularity_candidates"][key] = describe(np.asarray(all_rows), np.asarray(all_cols), np.asarray(all_weights))
    write_json(args.root / "prior_support_fit.json", report)
    print(report)


if __name__ == "__main__":
    main()
