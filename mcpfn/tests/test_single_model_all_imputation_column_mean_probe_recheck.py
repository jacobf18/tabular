import inspect
import math

import pytest

import recheck_single_model_all_imputation_column_mean_probe as recheck


def test_recheck_binds_complete_probe_and_every_imputation_term():
    assert recheck.ROOT == recheck.probe.OUTPUT
    source = inspect.getsource(recheck.main)
    assert "669d4b633a362e5e577771ed48d7838253a3ced0ea7e496d76087bc7c93f6d11" in source
    assert "mean['imputation_reweighted_tasks'] == 26" in source
    assert "{'True': 26, 'False': 2}" in source
    assert "(registered['cohort'] != 'recsys')" in source
    assert "len(columns) == mean['columns']" in source
    assert "sum(row['change'] < 0 for row in individual)" in source
    assert "len(adverse_mean) == mean['adverse_teacher_mean_tasks']" in source
    assert "len(individual) == 56" in source
    assert "len(grouped) == 30" in source
    assert "len(penalties) == 26" in source
    assert "training_repeated': False" in source
    assert "real_benchmarks_evaluated': False" in source
    assert "autograd replay" in source


@pytest.mark.parametrize("actual, expected", [(math.nan, 1.), (1., math.inf), (1.01, 1.)])
def test_saved_scalar_recheck_rejects_nonfinite_and_inexact_values(actual, expected):
    with pytest.raises(ValueError):
        recheck.close(actual, expected)


def test_csv_export_retains_adverse_values_and_empty_columns(tmp_path):
    rows = [{"task": "native", "change": .02, "queries": 0},
        {"task": "structured", "change": -.01, "queries": 2}]
    path = tmp_path / "diagnostics.csv"
    path.write_text(recheck.csv_text(rows))
    actual = recheck.read_csv(path)
    assert float(actual[0]["change"]) == .02
    assert int(actual[0]["queries"]) == 0
    assert float(actual[1]["change"]) == -.01
