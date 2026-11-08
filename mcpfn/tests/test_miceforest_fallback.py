import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def load_runner(failed_runs):
    calls = []

    class FakeKernel:
        def __init__(self, data, **kwargs):
            self.data = data.copy()
            self.run_index = len(calls)
            calls.append({"constructor": kwargs, "completed_datasets": []})

        def mice(self, **kwargs):
            calls[self.run_index]["fit"] = kwargs

        def complete_data(self, dataset):
            calls[self.run_index]["completed_datasets"].append(dataset)
            result = self.data.fillna(float(dataset + 1))
            if self.run_index in failed_runs and dataset == 2:
                result.iloc[0, 0] = np.nan
            return result

    source = Path(__file__).resolve().parents[1] / "benchmark/openml/get_openml_errors.py"
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
        "fill_all_nan_columns", "fill_all_nan_rows", "run_hyperimpute",
    }]
    namespace = {"np": np, "pd": pd, "mf": SimpleNamespace(ImputationKernel=FakeKernel)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["run_hyperimpute"], calls


@pytest.mark.parametrize("failed_runs,expected_runs", [(set(), 1), ({0}, 2)])
def test_miceforest_preserves_defaults_and_retries_only_nonfinite_results(failed_runs, expected_runs):
    runner, calls = load_runner(failed_runs)
    missing = np.array([[np.nan, 2.], [3., np.nan], [4., 5.]])
    original = missing.copy()
    result = runner("mice", missing, random_state=7)
    assert len(calls) == expected_runs
    np.testing.assert_array_equal(missing, original)
    np.testing.assert_array_equal(result, [[3., 2.], [3., 3.], [4., 5.]])
    for call in calls:
        assert call["constructor"] == {"num_datasets": 5, "mean_match_candidates": 0, "random_state": 7}
        assert call["completed_datasets"] == list(range(5))
    assert calls[0]["fit"] == {"iterations": 5, "verbose": False}
    if expected_runs == 2:
        assert calls[1]["fit"] == {
            "iterations": 5, "verbose": False,
            "bagging_fraction": 1.0, "bagging_freq": 0, "feature_fraction": 0.632,
        }


def test_miceforest_rejects_results_when_both_attempts_are_nonfinite():
    runner, calls = load_runner({0, 1})
    with pytest.raises(ValueError, match="still produced nonfinite"):
        runner("mice", np.array([[np.nan, 2.], [3., 4.]]))
    assert len(calls) == 2
