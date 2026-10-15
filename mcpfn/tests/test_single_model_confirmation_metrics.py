from copy import deepcopy

import pytest

import diagnose_single_model_confirmation_metrics as diagnostic


def cases():
    result = []
    for index, (pooled, columnwise) in enumerate(((.8, 1.2), (1.4, .9))):
        scores = {role: {name: 1. for name in diagnostic.metric.METRICS} for role in diagnostic.ROLES}
        scores["candidate"] = {"pooled_rmse": pooled, "missbench_nrmse": columnwise, "uci_nrmse": columnwise}
        result.append({"index": index, "seed": 100 + index, "gradient_data": "synthetic", "domain": "imputation",
                       "prior": f"family_{index}", "scores": scores})
    return result


def test_candidate_is_compared_with_its_initialization_and_all_families_retained():
    data = cases()
    original = deepcopy(data)
    rows = diagnostic.summarize(data)
    assert data == original and len(rows) == 12
    candidate = next(row for row in rows if row["family"] == "ALL" and row["role"] == "candidate")
    assert candidate["cases"] == 2
    assert candidate["pooled_rmse"] == pytest.approx(1.1)
    assert candidate["missbench_nrmse"] == pytest.approx(1.05)
    assert candidate["missbench_opposite_direction_cases"] == candidate["uci_opposite_direction_cases"] == 2
    assert candidate["pooled_rmse_wins_vs_initialization"] == candidate["pooled_rmse_losses_vs_initialization"] == 1


def test_initialization_is_the_comparator_not_general():
    data = cases()
    for case in data:
        for metric in diagnostic.metric.METRICS:
            case["scores"]["general"][metric] = 100.
    candidate = next(row for row in diagnostic.summarize(data) if row["family"] == "ALL" and row["role"] == "candidate")
    assert candidate["pooled_rmse_wins_vs_initialization"] == 1
    assert candidate["missbench_opposite_direction_cases"] == 2


@pytest.mark.parametrize("failure", ["empty", "duplicate", "real", "domain", "missing_role", "nan", "negative"])
def test_summary_fails_closed_for_invalid_comparisons(failure):
    data = cases()
    if failure == "empty":
        data = []
    elif failure == "duplicate":
        data[1]["seed"] = data[0]["seed"]
    elif failure == "real":
        data[0]["gradient_data"] = "real"
    elif failure == "domain":
        data[0]["domain"] = "recsys"
    elif failure == "missing_role":
        del data[0]["scores"]["candidate"]
    else:
        data[0]["scores"]["candidate"]["pooled_rmse"] = float("nan") if failure == "nan" else -1.
    with pytest.raises(ValueError):
        diagnostic.summarize(data)
