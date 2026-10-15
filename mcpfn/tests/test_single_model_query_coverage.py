import numpy as np
import pytest

from probe_single_model_query_coverage import column_statistics, summarize


def test_tiny_query_std_does_not_imply_tiny_full_column_signal():
    target = np.array([[2.], [2.0006], [-3.], [4.]], dtype=np.float32)
    mask = np.array([[True], [True], [False], [False]])
    source = np.where(mask, np.nan, target)
    row = column_statistics(source, target, mask)[0]
    assert row["uci_only_nonconstant"] and not row["common_nonconstant"]
    assert row["queries"] == row["observations"] == 2
    assert row["query_band"] == "2"
    assert row["full_target_std"] > 1 and row["observed_input_std"] > 1


def test_constant_singleton_and_empty_query_columns_keep_their_meaning():
    target = np.arange(12, dtype=np.float32).reshape(4, 3)
    mask = np.zeros_like(target, dtype=bool)
    mask[0, 0], mask[:2, 1] = True, True
    rows = column_statistics(np.where(mask, np.nan, target), target, mask)
    assert len(rows) == 2 and rows[0]["query_std"] == 0
    assert rows[0]["query_band"] == "1" and not rows[0]["uci_only_nonconstant"]
    assert rows[1]["common_nonconstant"]


def test_summary_counts_distinct_cases_and_rare_query_counts():
    target = np.array([[2.], [2.0006], [-3.], [4.]], dtype=np.float32)
    mask = np.array([[True], [True], [False], [False]])
    rows = [{"index": index, **row} for index in range(2)
            for row in column_statistics(np.where(mask, np.nan, target), target, mask)]
    summary = summarize(rows, [{"index": index, "query_fraction": .5} for index in range(2)])
    assert summary["cases_with_uci_only_columns"] == summary["uci_only_columns"] == 2
    assert summary["uci_only_query_counts"] == {2: 2}
    assert summary["uci_only_full_target_std_at_least_point_one"] == 2


def test_probe_rejects_visible_queries_and_duplicate_case_indices():
    target = np.ones((3, 2), dtype=np.float32)
    mask = np.ones_like(target, dtype=bool)
    with pytest.raises(ValueError):
        column_statistics(target, target, mask)
    with pytest.raises(ValueError):
        summarize([{"index": 0}], [{"index": 0}, {"index": 0}])
