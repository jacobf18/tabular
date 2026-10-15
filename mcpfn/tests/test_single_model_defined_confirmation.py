from copy import deepcopy
import os

import pytest

import confirm_single_matrixformer_defined_columns as confirmation
from recsys_evaluation_common import write_json
from test_single_model_query_registration import fixture_config


def fixture_reports():
    cases = fixture_config()["confirmation_cases"]
    records = [{"index": index, "domain": case["domain"], "student": {
                "common_nrmse": None if index == 6 else 1.,
                "column_coverage": {"common_nonconstant_columns": 0 if index == 6 else 1}}}
               for index, case in enumerate(cases)]
    metrics = {"imputation": {"common_nrmse_cases": 59, "canonical_metric_cases": 60}}
    single = {"cases": records, "metrics": metrics}
    reference = {"cases": [{**record, "general": record["student"], "teacher": record["student"]} for record in records],
                 "general": metrics, "teacher": metrics}
    return deepcopy(reference), deepcopy(single), deepcopy(single)


@pytest.mark.parametrize("change", [None, "missing", "model_coverage", "fake_zero", "different_count"])
def test_every_role_uses_same_truth_defined_subset_without_dropping_canonical_cases(change):
    reference, initial, student = fixture_reports()
    if change == "missing":
        student["cases"].pop()
    elif change == "model_coverage":
        student["cases"][7]["student"]["column_coverage"]["common_nonconstant_columns"] = 0
    elif change == "fake_zero":
        student["cases"][6]["student"]["common_nrmse"] = 0.
    elif change == "different_count":
        student["metrics"]["imputation"]["common_nrmse_cases"] = 60
    if change:
        with pytest.raises(ValueError):
            confirmation.role_coverage(reference, initial, student)
    else:
        result = confirmation.role_coverage(reference, initial, student)
        assert result["coverage_equal_across_all_roles"] and result["canonical_imputation_cases_per_role"] == 60
        assert result["common_guard_cases_per_role"] == 59


def test_wait_for_average_terminal_never_races_live_worker(tmp_path):
    write_json(tmp_path / "status.json", {"state": "evaluating", "pid": os.getpid()})
    assert not confirmation.campaign_terminal(tmp_path)
    write_json(tmp_path / "status.json", {"state": "complete", "pid": os.getpid()})
    assert not confirmation.campaign_terminal(tmp_path)
    write_json(tmp_path / "status.json", {"state": "not_promoted"})
    assert confirmation.campaign_terminal(tmp_path)
