from copy import deepcopy

import numpy as np
import pytest
import torch

import single_model_native_parent_alignment as metrics
import single_model_native_parent_retention as retention


def arrays():
    return {role: np.full((2, 3), value, dtype=np.float32) for role, value in
        (("general", 9.), ("teacher", 7.), ("v8", 1.), ("actual_v29", 3.), ("student", 2.))}


def test_fixed_mean_preserves_original_fp32_order_without_overflow_or_parent_aliasing():
    values = arrays()
    values["v8"][0, 0] = values["actual_v29"][0, 0] = np.finfo(np.float32).max
    values["v8"][0, 1] = np.nextafter(np.float32(0), np.float32(1))
    before = deepcopy(values)
    actual = metrics.structured_mean(values)
    expected = retention.combine_predictions("structured_mean", {
        "v8": torch.from_numpy(values["v8"]), "v29": torch.from_numpy(values["actual_v29"]),
    }).numpy()
    assert np.array_equal(actual.view(np.uint8), expected.view(np.uint8))
    assert np.isfinite(actual).all()
    assert all(np.array_equal(value, before[role]) for role, value in values.items())
    assert not any(np.shares_memory(actual, value) for value in values.values())


@pytest.mark.parametrize("mutation", ["missing_v29", "missing_v8", "dtype", "shape", "dimension"])
def test_never_substitute_or_cast_parent_predictions(mutation):
    values = arrays()
    if mutation.startswith("missing"):
        values.pop("actual_v29" if mutation.endswith("v29") else "v8")
    elif mutation == "dtype":
        values["actual_v29"] = values["actual_v29"].astype(np.float64)
    elif mutation == "shape":
        values["actual_v29"] = values["actual_v29"][:, :2]
    else:
        values["v8"] = values["v8"][None]
        values["actual_v29"] = values["actual_v29"][None]
    with pytest.raises(ValueError):
        metrics.structured_mean(values)


def test_added_distances_keep_every_old_metric_and_observed_query_overlap():
    source = np.array([[0., np.nan, 1.], [2., np.nan, np.nan]], dtype=np.float32)
    truth = np.zeros_like(source)
    query = np.array([[True, True, False], [False, True, False]])
    values = arrays()
    old = metrics.previous.score_case(source, truth, query, values)
    rows = metrics.score_case(source, truth, query, values)
    assert len(rows) == len(old) == 10
    for new, previous in zip(rows, old):
        assert {key: new[key] for key in previous} == previous
    for row in rows:
        if row["role"] == "actual_v29":
            assert row["actual_v29_mse"] == row["actual_v29_shrunk_mse"] == 0.
        if row["role"] == "student":
            assert row["actual_v29_mse"] == 1.
            assert row["structured_mean_mse"] == row["structured_mean_shrunk_mse"] == 0.
    assert int((query & np.isfinite(source)).sum()) == 1


def test_equal_case_summary_retains_every_added_distance_and_all_original_columns():
    first = {"bank": "bank", "cohort": "legacy", "split": "query", "role": "student",
        "family": "native", "index": 0, "selected_cells": 1, **{key: 2. for key in metrics.METRICS}}
    second = {**first, "index": 1, "selected_cells": 100, **{key: 4. for key in metrics.METRICS}}
    summaries = metrics.summarize([first, second])
    assert len(summaries) == 2
    assert all(row["cases"] == 2 and row["cells"] == 101 for row in summaries)
    assert all(row[key] == 3. for row in summaries for key in metrics.METRICS)
    with pytest.raises(ValueError):
        metrics.summarize([first, first])
