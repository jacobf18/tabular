from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from seaborn.algorithms import bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from missbench_statistics import bootstrap_mean_interval, paired_dataset_wilcoxon


def test_wilcoxon_uses_42_paired_dataset_means():
    index = pd.MultiIndex.from_product([range(42), range(11)])
    values = np.random.default_rng(0).normal(size=(462, 2))
    frame = pd.DataFrame(values, index=index, columns=["TabImpute", "MissForest"])
    paired, result = paired_dataset_wilcoxon(frame)
    expected = values.reshape(42, 11, 2).mean(axis=1)
    np.testing.assert_allclose(paired.iloc[:, :2], expected)
    difference = expected[:, 0] - expected[:, 1]
    assert result["dataset_pairs"] == 42
    assert result["paired_dataset_pattern_rows"] == 462
    assert result["one_sided_pvalue"] == stats.wilcoxon(difference, alternative="less", zero_method="pratt").pvalue


def test_pair_before_averaging_missing_patterns():
    index = pd.MultiIndex.from_tuples([("first", "MCAR"), ("first", "MAR"), ("second", "MCAR"), ("second", "MAR")])
    frame = pd.DataFrame({"TabImpute": [1., 100., 2., np.inf], "MissForest": [2., np.nan, 4., 0.]}, index=index)
    paired, result = paired_dataset_wilcoxon(frame)
    np.testing.assert_array_equal(paired.to_numpy(), [[1, 2, 1], [2, 4, 1]])
    assert result["omitted_dataset_pattern_rows"] == 2


def test_repeated_patterns_do_not_inflate_sample_size_or_significance():
    frame = pd.DataFrame({"TabImpute": np.arange(1., 9.), "MissForest": np.arange(2., 10.)},
                         index=pd.MultiIndex.from_product([range(8), ["MCAR"]]))
    repeated = pd.concat([frame.reset_index(level=1, drop=True).assign(pattern=str(pattern)) for pattern in range(11)])
    repeated = repeated.set_index("pattern", append=True)
    _, original = paired_dataset_wilcoxon(frame)
    _, duplicated = paired_dataset_wilcoxon(repeated)
    assert original["dataset_pairs"] == duplicated["dataset_pairs"] == 8
    assert original["one_sided_pvalue"] == duplicated["one_sided_pvalue"]
    assert original["one_sided_pvalue"] < original["two_sided_pvalue"]


def test_all_ties_are_not_significant_and_duplicates_are_rejected():
    frame = pd.DataFrame({"TabImpute": [1., 2.], "MissForest": [1., 2.]},
                         index=pd.MultiIndex.from_product([range(2), ["MCAR"]]))
    _, result = paired_dataset_wilcoxon(frame)
    assert result["dataset_ties"] == 2 and result["one_sided_pvalue"] == 1
    with pytest.raises(ValueError, match="unique"):
        paired_dataset_wilcoxon(pd.concat([frame, frame]))


def test_bootstrap_matches_barplot_percentile_interval_and_drops_nan():
    values = np.array([1., 2., 5., 9., 20.])
    interval = bootstrap_mean_interval(pd.Series(np.append(values, np.nan)))
    expected = np.percentile(bootstrap(values, func=np.mean, n_boot=1000, seed=0), [2.5, 97.5])
    np.testing.assert_array_equal(interval, expected)
    assert np.isnan(bootstrap_mean_interval(pd.Series([1.]))).all()
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_interval(pd.Series([1., np.inf]))
