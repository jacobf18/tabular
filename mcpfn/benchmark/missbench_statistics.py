"""Dataset-level uncertainty and paired comparisons for MissBench."""

import numpy as np
import pandas as pd
from scipy import stats
from seaborn.algorithms import bootstrap


def bootstrap_mean_interval(series):
    values = np.asarray(series.dropna(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap scores must be finite")
    if len(values) < 2:
        return (float("nan"), float("nan"))
    draws = bootstrap(values, func=np.mean, n_boot=1000, seed=0)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return float(lower), float(upper)


def paired_dataset_wilcoxon(frame, first="TabImpute", second="MissForest"):
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2 or not frame.index.is_unique:
        raise ValueError("Expected unique (dataset, pattern) rows")
    scores = frame[[first, second]]
    valid = np.isfinite(scores.to_numpy(dtype=np.float64)).all(axis=1)
    matched = scores.loc[valid]
    if matched.empty:
        raise ValueError("No finite paired scores")
    paired = matched.groupby(level=0).mean()
    paired["patterns_used"] = matched.groupby(level=0).size()
    differences = (paired[first] - paired[second]).to_numpy(dtype=np.float64)
    if np.all(differences == 0):
        statistic, one_sided, two_sided = 0.0, 1.0, 1.0
    else:
        test = stats.wilcoxon(differences, alternative="less", zero_method="pratt", method="auto")
        statistic, one_sided = float(test.statistic), float(test.pvalue)
        two_sided = float(stats.wilcoxon(differences, alternative="two-sided", zero_method="pratt", method="auto").pvalue)
    result = {"first_method": first, "second_method": second, "metric": "NRMSE",
              "pairing_unit": "dataset", "aggregation": "Mean over the same finite paired patterns within each dataset",
              "dataset_pairs": len(paired), "paired_dataset_pattern_rows": len(matched),
              "omitted_dataset_pattern_rows": int((~valid).sum()),
              "first_mean": float(paired[first].mean()), "second_mean": float(paired[second].mean()),
              "first_dataset_wins": int((differences < 0).sum()),
              "second_dataset_wins": int((differences > 0).sum()), "dataset_ties": int((differences == 0).sum()),
              "mean_paired_difference": float(differences.mean()),
              "median_paired_difference": float(np.median(differences)),
              "statistic": statistic, "alternative": "less", "zero_method": "pratt", "method": "auto",
              "one_sided_pvalue": one_sided, "two_sided_pvalue": two_sided}
    return paired, result
