"""Measure rating-prior targets exclusively from outer-training inner splits."""
import argparse

import numpy as np

from dwip_recsys_experiment import DEFAULT_ROOT, load_split
from evaluate_dwip_recsys import DATASETS, write_json, user_rating_metrics


def quantiles(values):
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                    np.quantile(values, (0, .1, .25, .5, .75, .9, 1)).tolist()))


def diagnose():
    results = {}
    for spec in DATASETS.values():
        observed, context, queries = load_split(spec["folder"], "validation")
        users, items, ratings = context
        row_counts = np.bincount(users, minlength=observed.shape[0])
        col_counts = np.bincount(items, minlength=observed.shape[1])
        global_mean = ratings.mean()
        user_bias = np.zeros(observed.shape[0])
        item_bias = np.zeros(observed.shape[1])
        for _ in range(20):
            user_bias = np.bincount(users, weights=ratings - global_mean - item_bias[items], minlength=len(user_bias)) / (row_counts + 10)
            item_bias = np.bincount(items, weights=ratings - global_mean - user_bias[users], minlength=len(item_bias)) / (col_counts + 10)
        residuals = ratings - global_mean - user_bias[users] - item_bias[items]
        prediction = np.clip(global_mean + user_bias[queries[0]] + item_bias[queries[1]], 1, 5)
        results[spec["folder"]] = {
            "shape": list(observed.shape), "density": len(ratings) / observed.size,
            "mean": float(ratings.mean()), "std": float(ratings.std()), "row_support": quantiles(row_counts),
            "column_support": quantiles(col_counts), "bias_residual_std": float(residuals.std()),
            "user_bias_std": float(user_bias.std()), "item_bias_std": float(item_bias.std()),
            "popularity_quality_correlation": float(np.corrcoef(np.log1p(col_counts), item_bias)[0, 1]),
            "inner_bias_only": user_rating_metrics(queries[0], queries[2], prediction),
            "purpose": "Prior calibration/diagnostic only; no fitted bias model is deployed or used in neural predictions",
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_ROOT / "training_diagnostics.json", type=type(DEFAULT_ROOT))
    args = parser.parse_args()
    write_json(args.output, diagnose())


if __name__ == "__main__":
    main()
