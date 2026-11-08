from copy import deepcopy

import numpy as np
import pytest

import analyze_single_model_standardized_huber_confirmation_columns as analysis


def reports():
    cases = [{"index": index, "seed": 1000 + index, "rows": 8, "columns": 3,
        "gradient_data": "synthetic", "domain": "recsys" if 60 <= index < 92 else "imputation",
        "structured": index >= 92} for index in range(284)]
    for case in cases[92:]:
        case["family"] = analysis.FAMILIES[(case["index"] - 92) % len(analysis.FAMILIES)]
    return {name: {"cases": deepcopy(cases)} for name in ("reference", "initialization", "parent", "student")}


def test_all_reports_keep_the_entire_original_confirmation():
    paired = analysis.paired_records(reports())
    assert len(paired) == 284
    assert {cohort: sum(name == cohort for name, _ in paired) for cohort in analysis.COHORT_COUNTS} == analysis.COHORT_COUNTS


@pytest.mark.parametrize("mutation", ["drop", "order", "seed", "shape", "real", "queries", "cohort", "missing_role"])
def test_no_case_or_role_can_be_omitted_or_relabelled(mutation):
    original = reports()
    cases = original["student"]["cases"]
    if mutation == "drop":
        cases.pop()
    elif mutation == "order":
        cases[0], cases[1] = cases[1], cases[0]
    elif mutation == "missing_role":
        original.pop("parent")
    else:
        field, value = {"seed": ("seed", -1), "shape": ("rows", 9), "real": ("gradient_data", "real"),
            "queries": ("targets", 1), "cohort": ("structured", True)}[mutation]
        cases[0][field] = value
    with pytest.raises(ValueError):
        analysis.paired_records(original)


def column_fixture():
    truth = np.array([[0., 4.], [2., 4.]], dtype=np.float32)
    offsets = {"general": 1., "teacher": 2., "baseline": 0., "parent": .5, "candidate": .25}
    arrays = {role: truth + offset for role, offset in offsets.items()}
    expected = {role: {"rmse": offset, "missbench_nrmse": offset, "uci_nrmse": offset,
        "common_nrmse": offset, "column_coverage": {"missbench_scored_columns": 2 if offset == 0 else 1,
            "uci_scored_columns": 2 if offset == 0 else 1, "common_nonconstant_columns": 1}}
        for role, offset in offsets.items()}
    return truth, np.ones_like(truth, dtype=bool), np.full_like(truth, np.nan), arrays, expected


def test_five_roles_keep_canonical_coverage_and_actual_parent_drift():
    rows, scores = analysis.checked_columns(*column_fixture())
    assert set(scores) == set(analysis.ROLES)
    assert len(rows) == 2 and rows[1]["common_eligible"] is False
    assert rows[1]["baseline_missbench_nrmse"] == 0.
    assert rows[1]["candidate_missbench_nrmse"] is None
    assert rows[0]["parent_drift_rmse"] == 0.
    assert rows[0]["baseline_drift_rmse"] == .5
    assert rows[0]["teacher_drift_rmse"] == 1.5
    assert rows[0]["candidate_drift_rmse"] == .25
    assert scores["candidate"]["common_nrmse"] == .25


@pytest.mark.parametrize("mutation", ["missing_role", "shape", "nonfinite", "score", "coverage"])
def test_saved_scores_and_full_grids_are_not_silently_changed(mutation):
    truth, mask, source, arrays, expected = column_fixture()
    if mutation == "missing_role":
        arrays.pop("teacher")
    elif mutation == "shape":
        arrays["teacher"] = arrays["teacher"][:1]
    elif mutation == "nonfinite":
        arrays["teacher"][0, 0] = np.nan
    elif mutation == "score":
        expected["parent"]["common_nrmse"] = 10.
    else:
        expected["baseline"]["column_coverage"]["missbench_scored_columns"] = 1
    with pytest.raises(ValueError):
        analysis.checked_columns(truth, mask, source, arrays, expected)


def test_contributions_preserve_case_weights_and_all_five_roles():
    rows = []
    for index, count in ((0, 1), (1, 3)):
        for column in range(count):
            row = {"index": index, "cohort": "native", "prior": "anchor" if index == 0 else "spectral",
                "pattern": "native", "scale_band": "small" if column == 0 else "large", "support_band": "few",
                "query_band": "two", "shape": str(count)}
            for suite in analysis.development.METRICS:
                for value, role in enumerate(analysis.ROLES, 1):
                    row[f"{role}_{suite}_contribution"] = value / count
            rows.append(row)
    summaries = analysis.grouped_contributions(rows, {"native": 2})
    for suite in analysis.development.METRICS:
        total = next(row for row in summaries if row["grouping"] == "all" and row["metric"] == suite)
        assert total["candidate"] == 5. and total["candidate_minus_parent"] == 1.
        for grouping in ("prior", "pattern", "scale_band", "support_band", "query_band", "shape"):
            assert sum(row["candidate"] for row in summaries if row["grouping"] == grouping and row["metric"] == suite) == pytest.approx(5.)


def test_no_narrow_attribution_can_replace_all32_checks():
    with pytest.raises(ValueError, match="all32"):
        analysis.verify_attribution([], [], [])


def attribution_fixture():
    cases, ratings = [], []
    roles = {role: float(index) for index, role in enumerate(analysis.ROLES, 1)}
    for index, (cohort, group) in enumerate(analysis.paired_records(reports())):
        if cohort == "ratings":
            ratings.extend({"index": index, "metric": metric, **roles} for metric in ("rmse", "mae"))
        else:
            for role, value in roles.items():
                cases.append({"index": index, "cohort": cohort, "role": role,
                    "pattern": group["reference"].get("family", "native"),
                    **{metric: value for metric in ("missbench_nrmse", "uci_nrmse", "common_nrmse", "rmse")}})
    comparisons = [{"reference": reference, "group": group, "metric": metric,
        "V29_initializer": roles["parent"], "V54_step50": roles["candidate"],
        "reference_value": roles["baseline"] if reference == "V8" else roles["general"]}
        for reference, group, metric in sorted(analysis.attribution_keys())]
    return cases, ratings, comparisons


def test_every_original_guard_is_reconstructed_with_full_case_counts():
    analysis.verify_attribution(*attribution_fixture())


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_reference", "omitted_case", "omitted_rating", "changed_value"])
def test_duplicates_and_partial_aggregates_cannot_pass(mutation):
    cases, ratings, comparisons = attribution_fixture()
    if mutation == "duplicate":
        comparisons[0] = deepcopy(comparisons[1])
    elif mutation == "wrong_reference":
        comparisons[0]["reference"] = "teacher"
    elif mutation == "omitted_case":
        cases = [row for row in cases if row["index"] != 0]
    elif mutation == "omitted_rating":
        ratings.pop()
    else:
        comparisons[0]["V29_initializer"] += .1
    with pytest.raises(ValueError):
        analysis.verify_attribution(cases, ratings, comparisons)


def test_all_roles_cannot_together_relabel_a_structured_family():
    original = reports()
    for report in original.values():
        report["cases"][92]["family"] = "unknown"
    with pytest.raises(ValueError, match="twelve"):
        analysis.paired_records(original)
