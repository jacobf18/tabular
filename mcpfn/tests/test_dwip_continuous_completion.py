from copy import deepcopy

import numpy as np
import pytest

from audit_dwip_continuous_completion import fixed_selection, full_panel


def cache():
    return {"benchmark_names": np.asarray(["first", "second"]),
            **{f"{method}_{metric}": np.ones((10, 15)) for method in ("entropy", "mi", "random")
               for metric in ("r2", "rmse")}}


def test_continuous_panel_includes_nonstandard_budgets_and_all_folds():
    before, after = cache(), cache()
    after["mi_r2"][:, 0] += .2
    after["entropy_r2"][:, 13] -= .3
    panel, rows = full_panel(before, after)
    assert len(rows) == 45
    assert panel["mean_setting_counts"] == {"improved": 1, "regressed": 1, "tied": 43}
    assert {row["budget"] for row in rows} == set(range(1, 16))
    assert {row["fold_count"] for row in rows} == {10}
    assert len(panel["fold_arrays"]) == 6
    assert panel["standard_budgets"]["mi_r2_5"]["delta"] == 0


@pytest.mark.parametrize("change", ("names", "shape", "nonfinite_r2", "nonfinite_rmse"))
def test_continuous_panel_rejects_changed_or_incomplete_populations(change):
    before, after = cache(), cache()
    if change == "names":
        after["benchmark_names"] = np.asarray(["changed", "second"])
    elif change == "shape":
        after["random_r2"] = np.ones((9, 15))
    else:
        after[f"mi_{change.removeprefix('nonfinite_')}"][9, 14] = np.nan
    with pytest.raises(ValueError):
        full_panel(before, after)


def test_continuous_selection_uses_complete_synthetic_schedule_only():
    validation = [{"step": 0, "macro_rmse": 1.1}, {"step": 2000, "macro_rmse": .9}, {"step": 4000, "macro_rmse": 1.0}]
    selected = deepcopy(validation[1])
    assert fixed_selection(validation, selected, final_step=4000) == selected
    with pytest.raises(ValueError, match="not the fixed-validation minimum"):
        fixed_selection(validation, validation[2], final_step=4000)
    with pytest.raises(ValueError, match="complete ordered"):
        fixed_selection(validation[:-1], selected, final_step=4000)
    validation[-1]["macro_rmse"] = float("nan")
    with pytest.raises(ValueError, match="Invalid fixed"):
        fixed_selection(validation, selected, final_step=4000)
