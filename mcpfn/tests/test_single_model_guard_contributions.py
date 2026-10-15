from copy import deepcopy

import pytest

from report_single_model_guard_contributions import decompose, matches


def fixture():
    initial = [{"index": index, "seed": 11 + index, "rows": 20, "columns": 5, "domain": "imputation",
                "prior": "anchor" if index == 0 else "spectral", "student": {"common_nrmse": value}}
               for index, value in enumerate((1., 2., 3.))]
    current = deepcopy(initial)
    for row, value in zip(current, (2., 1., 2.)):
        row["student"]["common_nrmse"] = value
    guards = [{"group": "imputation", "metric": "common_nrmse", "initialization": 2., "candidate": 5 / 3, "retained": True}]
    return initial, current, guards


def test_component_contributions_add_to_the_guard_change():
    initial, current, guards = fixture()
    result = decompose(initial, current, guards)
    assert sum(row["guard_delta_contribution"] for row in result["components"]) == pytest.approx(-1 / 3)
    assert result["guards"][0]["worsened_cases"] == 1
    assert result["components"][0]["guard_delta_contribution"] == pytest.approx(1 / 3)
    assert result["components"][1]["guard_delta_contribution"] == pytest.approx(-2 / 3)


def test_undefined_cases_remain_explicit_and_have_no_fabricated_score():
    initial, current, guards = fixture()
    initial[-1]["student"]["common_nrmse"] = None
    current[-1]["student"]["common_nrmse"] = None
    guards[0].update(initialization=1.5, candidate=1.5)
    result = decompose(initial, current, guards)
    assert result["guards"][0]["cases"] == 3
    assert result["guards"][0]["defined_cases"] == 2
    assert result["guards"][0]["undefined_cases"] == 1
    assert len(result["cases"]) == 2
    current[-1]["student"]["common_nrmse"] = 1.
    with pytest.raises(ValueError, match="coverage"):
        decompose(initial, current, guards)


@pytest.mark.parametrize("mutation", ["drop", "reorder", "metadata", "aggregate", "decision", "duplicate"])
def test_case_or_guard_changes_are_rejected(mutation):
    initial, current, guards = fixture()
    if mutation == "drop":
        current.pop()
    elif mutation == "reorder":
        current.reverse()
    elif mutation == "metadata":
        current[0]["prior"] = "changed"
    elif mutation == "aggregate":
        guards[0]["candidate"] += .01
    elif mutation == "decision":
        guards[0]["retained"] = False
    else:
        guards.append(deepcopy(guards[0]))
    with pytest.raises(ValueError):
        decompose(initial, current, guards)


def test_structured_and_legacy_imputation_are_distinct():
    row = {"domain": "imputation", "structured": True, "family": "mnar_skip_logic"}
    assert not matches(row, "imputation")
    assert matches(row, "structured_overall")
    assert matches(row, "mnar_skip_logic")
    assert not matches(row, "mcar")
    assert not matches(row, "recsys")
