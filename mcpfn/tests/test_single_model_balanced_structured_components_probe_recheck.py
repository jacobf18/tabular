import math

import pytest

import recheck_single_model_balanced_structured_components_probe as recheck


def test_recheck_binds_balanced_batch_and_complete_saved_scope():
    assert recheck.ROOT == recheck.probe.OUTPUT
    source = recheck.RECHECK_SOURCE
    for text in ('a5d077f51bffbaa44f1f577b19f61f0f7a53864b581d29e8993a3c84fcfa26ed',
            "config['structured_component_balance'] == probe.components.SPECIFICATION",
            "config['batch'] == probe.batch()", "mean['imputation_reweighted_tasks'] == 26",
            "{'True': 26, 'False': 2}", "len(columns) == mean['columns']", 'len(individual) == 56',
            'len(grouped) == 30', 'len(penalties) == 26', "training_repeated': False",
            "real_benchmarks_evaluated': False", 'autograd replay', 'TEMPLATE_SHA256'):
        assert text in source


@pytest.mark.parametrize('actual, expected', [(math.nan, 1.), (1., math.inf), (1.01, 1.)])
def test_saved_scalar_recheck_rejects_nonfinite_and_inexact_values(actual, expected):
    with pytest.raises(ValueError):
        recheck.close(actual, expected)


def test_complete_csv_keeps_adverse_and_empty_query_rows(tmp_path):
    rows = [{'task': 'native', 'change': .02, 'queries': 0}, {'task': 'structured', 'change': -.01, 'queries': 2}]
    path = tmp_path / 'diagnostics.csv'
    path.write_text(recheck.csv_text(rows))
    actual = recheck.read_csv(path)
    assert len(actual) == 2 and float(actual[0]['change']) == .02 and int(actual[0]['queries']) == 0
