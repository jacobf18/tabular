from copy import deepcopy

import numpy as np
import pytest

import analyze_single_model_general_native_development as analysis
from test_single_model_development_columns import records


def paired_reports():
    original = records()
    for record in original:
        record.update(prior="example", transposed=False, support_transposed=1.)
    return original, [{"step": step, "cases": deepcopy(original)} for step in analysis.BOUNDARIES]


def test_all_boundaries_and_case_metadata_preserved():
    original, reports = paired_reports()
    analysis.validate_pairing(original, reports)
    with pytest.raises(ValueError, match="five"):
        analysis.validate_pairing(original, reports[1:])


@pytest.mark.parametrize("field,value", [("transposed", True), ("support_transposed", 0.), ("prior", "different"),
    ("component", "changed"), ("seed", -1), ("family", "mar"), ("structured", True)])
def test_changed_case_metadata_rejected(field, value):
    original, reports = paired_reports()
    reports[-1]["cases"][0][field] = value
    with pytest.raises(ValueError):
        analysis.validate_pairing(original, reports)


def test_metadata_excludes_only_prediction_specific_fields():
    original, reports = paired_reports()
    original[0].update(teacher={}, general={}, routing={})
    reports[0]["cases"][0].update(student={}, prediction_file="different", prediction_sha256="different")
    analysis.validate_pairing(original, reports)


def test_absent_orientation_is_not_false():
    record = {"index": 0, "seed": 1, "rows": 3, "columns": 2, "domain": "imputation", "structured": True, "family": "mar"}
    result = analysis.details(record, 0)
    assert result["transposed"] == result["support_transposed"] == "not_recorded"
    record.update(transposed=False, support_transposed=1.)
    result = analysis.details(record, 0)
    assert result["transposed"] == "False"
    assert result["support_transposed"] == "1.0"


def column_fixture():
    truth = np.array([[0., 3., 8.], [1., 3., 9.], [2., 3., 10.]], dtype=np.float32)
    mask = np.array([[True, True, False], [True, True, False], [True, False, False]])
    source = np.where(mask, np.nan, truth)
    arrays = {role: truth.copy() for role in analysis.ROLES}
    arrays["initialization"][:, 0] += 1.
    arrays["candidate"][:, 0] += .5
    arrays["candidate"][:, 1] += 1.
    return truth, mask, source, arrays


def test_empty_and_constant_columns_and_role_coverage_preserved():
    truth, mask, source, arrays = column_fixture()
    rows = analysis.complete_columns(truth, mask, source, arrays)
    assert [row["column"] for row in rows] == [0, 1, 2]
    assert rows[2]["queries"] == 0
    assert rows[2]["candidate_common_nrmse"] is None
    assert rows[2]["candidate_common_contribution"] == 0.
    assert rows[1]["initialization_missbench_nrmse"] == 0.
    assert rows[1]["candidate_missbench_nrmse"] is None
    assert rows[0]["common_direction"] == "better"
    assert rows[1]["mse_direction"] == "worse"
    for role in analysis.ROLES:
        error = arrays[role][mask].astype(float) - truth[mask].astype(float)
        assert sum(row[f"{role}_mse_contribution"] for row in rows) == pytest.approx(np.mean(error ** 2))


def test_column_grouping_keeps_full_cohort_denominator():
    truth, mask, source, arrays = column_fixture()
    record = {"index": 0, "seed": 1, "rows": 3, "columns": 3, "domain": "imputation", "prior": "example"}
    rows = [{**analysis.details(record, 10), **row} for row in analysis.complete_columns(truth, mask, source, arrays)]
    groups = analysis.grouped_columns(rows)
    baseline = next(row for row in groups if row["grouping"] == "all" and row["metric"] == "common")
    assert baseline["candidate"] == pytest.approx(rows[0]["candidate_common_nrmse"] / 40)
    assert baseline["columns"] == 3 and baseline["queried_columns"] == 2
    for grouping in (*analysis.GROUPINGS, "scale_band", "support_band", "common_direction", "mse_direction"):
        selected = [row for row in groups if row["grouping"] == grouping and row["metric"] == "common"]
        assert sum(row["candidate"] for row in selected) == pytest.approx(baseline["candidate"])


def test_case_groups_do_not_conceal_adverse_cases():
    record = {"index": 0, "seed": 1, "rows": 3, "columns": 3, "domain": "recsys"}
    rows = []
    for index, delta in enumerate((-.2, .1, 0.)):
        rows.append({**analysis.details({**record, "index": index}, 64), "metric": "rmse", "general": 2.,
            "initialization": 1., "candidate": 1. + delta, "delta_vs_actual_V29": delta, "direction": analysis.direction(delta)})
    result = analysis.grouped_cases(rows)
    overall = next(row for row in result if row["grouping"] == "all")
    assert overall["better_cases"] == overall["worse_cases"] == overall["unchanged_cases"] == 1
    assert overall["cohort_delta_contribution"] == pytest.approx(-.1 / 16)
    assert {row["group"] for row in result if row["grouping"] == "direction"} == {"better", "worse", "unchanged"}


@pytest.mark.parametrize("delta,label", [(-1., "better"), (0., "unchanged"), (1., "worse")])
def test_direction(delta, label):
    assert analysis.direction(delta) == label
