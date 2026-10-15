from copy import deepcopy

import pytest

import diagnose_single_model_point_alignment as diagnostic
from single_model_point_metric_gradients import point_metric_terms
from test_single_model_point_metric_gradients import imputation_fixture, rating_fixture


def test_prospective_cases_retain_all_families_and_fresh_seeds():
    cases = diagnostic.specifications()
    original = diagnostic.preflight.specifications()
    assert len(cases) == 14 and cases != original
    assert [case["name"] for case in cases] == ["legacy", *diagnostic.mixture.FAMILIES, "recsys"]
    assert sum(case["weight"] for case in cases) == pytest.approx(1.)
    assert len(diagnostic.fixed.seeds_in(cases)) == 28
    assert not diagnostic.fixed.seeds_in(cases).intersection(diagnostic.fixed.seeds_in(original))
    for case, previous in zip(cases, original):
        case["gradient_seed"] = previous["gradient_seed"]
        source = case["specification"].get("structured", case["specification"])
        old = previous["specification"].get("structured", previous["specification"])
        source["seed"] = old["seed"]
    assert cases == original


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
def test_original_scorer_summary_and_cpu_comparison(domain):
    prediction, episode = imputation_fixture() if domain == "imputation" else rating_fixture()
    prediction = prediction.float()
    records = point_metric_terms(prediction, episode)
    summary = diagnostic.metric_summary(prediction, episode, records)
    diagnostic.compare_metric_records(summary, deepcopy(summary))
    assert all("gradient" not in row and "query_mask" not in row for row in summary.values())
    altered = deepcopy(summary)
    altered["rmse"]["value"] += 1e-6
    with pytest.raises(AssertionError):
        diagnostic.compare_metric_records(summary, altered)
    altered = deepcopy(summary)
    altered["rmse"]["queries"] += 1
    with pytest.raises(ValueError, match="coverage"):
        diagnostic.compare_metric_records(summary, altered)


def test_metric_record_preserves_nulls_and_string_definitions():
    diagnostic.compare_metric_records({"common_nrmse": None}, {"common_nrmse": None})
    with pytest.raises(ValueError, match="definition"):
        diagnostic.compare_metric_records({"metric": "pooled"}, {"metric": "equal_user"})
    with pytest.raises(ValueError, match="keys"):
        diagnostic.compare_metric_records({}, {"common_nrmse": None})
