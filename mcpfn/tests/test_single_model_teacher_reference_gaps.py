from copy import deepcopy

import pytest

from report_single_model_teacher_reference_gaps import summarize


def fixture():
    metadata = [{"index": index, "seed": 100 + index, "domain": domain, "rows": 20, "columns": 5}
                for index, domain in enumerate(("imputation", "recsys"))]
    reference = [{**row, "general": {"rmse": .9}, "teacher": {"rmse": .8}, "routing": {"weight": .5},
                  "prediction_file": "reference", "prediction_sha256": "old"} for row in metadata]
    initial = [{**row, "student": {"rmse": 1.}, "prediction_file": "initial"} for row in metadata]
    current = [{**row, "student": {"rmse": .95}, "prediction_file": "candidate"} for row in metadata]
    comparisons = [{"group": group, "metric": "rmse", "initialization": 1., "candidate": .95, "retained": True}
                   for group in ("imputation", "recsys")]
    return reference, initial, current, comparisons


def test_actual_training_teacher_is_not_confused_with_the_reference_imputation_blend():
    arguments = fixture()
    saved = deepcopy(arguments)
    result = summarize(*arguments)
    assert arguments == saved
    imputation, ratings = result["guards"]
    assert imputation["actual_training_teacher"] == 1.
    assert imputation["reference_teacher"] == .8
    assert ratings["actual_training_teacher"] == ratings["reference_teacher"] == .8
    assert imputation["original_general_change_vs_v8_percent"] == pytest.approx(-10.)
    assert imputation["candidate_change_vs_v8_percent"] == pytest.approx(-5.)
    assert result["case_identities"] == [(0, 100), (1, 101)]


def test_truth_defined_nulls_stay_explicit_without_changing_any_role_denominator():
    reference, initial, current, comparisons = fixture()
    reference[1]["domain"] = initial[1]["domain"] = current[1]["domain"] = "imputation"
    for index in range(2):
        for row, roles in ((reference[index], ("general", "teacher")), (initial[index], ("student",)),
                           (current[index], ("student",))):
            for role in roles:
                row[role]["common_nrmse"] = row[role]["rmse"] if index == 0 else None
    comparison = {**comparisons[0], "metric": "common_nrmse"}
    result = summarize(reference, initial, current, [comparison])
    assert result["guards"][0]["cases"] == 2
    assert result["guards"][0]["defined_cases"] == result["guards"][0]["undefined_cases"] == 1
    assert result["cases"][1]["v8"] is None
    reference[1]["teacher"]["common_nrmse"] = .1
    with pytest.raises(ValueError, match="coverage"):
        summarize(reference, initial, current, [comparison])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "metadata", "aggregate", "guard", "decision", "nonfinite"])
def test_changed_coverage_metadata_or_audited_decisions_are_rejected(mutation):
    reference, initial, current, comparisons = fixture()
    if mutation == "missing":
        current.pop()
    elif mutation == "duplicate":
        for rows in (reference, initial, current):
            rows[1]["index"], rows[1]["seed"] = rows[0]["index"], rows[0]["seed"]
    elif mutation == "metadata":
        reference[0]["columns"] = 6
    elif mutation == "aggregate":
        comparisons[0]["candidate"] = .8
    elif mutation == "guard":
        comparisons.append(deepcopy(comparisons[0]))
    elif mutation == "decision":
        comparisons[0]["retained"] = False
    else:
        reference[0]["general"]["rmse"] = float("nan")
    with pytest.raises(ValueError):
        summarize(reference, initial, current, comparisons)
