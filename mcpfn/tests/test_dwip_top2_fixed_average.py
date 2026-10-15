import pytest

from evaluate_dwip_top2_fixed_average import average_budget, strict_gate
from run_dwip_budgeted_queue_v1 import check_storage


def members():
    return [{"fixed_macro_rmse": 1.}, {"fixed_macro_rmse": 1.1}]


def selection(average=.9, accepted=True):
    return {"macro_rmse": {"seed1": 1., "seed2": 1.1, "average": average}, "average_accepted": accepted,
            "selected_variant": "average" if accepted else "seed1"}


def test_top2_average_gate_ignores_benchmarks_and_rejects_ties():
    candidate = selection()
    candidate["mteb"] = -1000.
    assert strict_gate(candidate, members())
    assert not strict_gate(selection(average=1., accepted=False), members())
    assert not strict_gate(selection(average=1.2, accepted=False), members())


@pytest.mark.parametrize("change", ("missing", "nan", "negative", "raw_drift", "decision", "variant"))
def test_top2_average_gate_rejects_invalid_or_drifting_scores(change):
    candidate = selection()
    if change == "missing":
        del candidate["macro_rmse"]["seed2"]
    elif change == "decision":
        candidate["average_accepted"] = False
    elif change == "variant":
        candidate["selected_variant"] = "seed2"
    else:
        candidate["macro_rmse"]["seed1"] = {"nan": float("nan"), "negative": -1., "raw_drift": 1.001}[change]
    with pytest.raises(ValueError):
        strict_gate(candidate, members())


def test_top2_average_cannot_exploit_tiny_upward_raw_replay_drift():
    candidate = selection(average=1. + 1e-11)
    candidate["macro_rmse"]["seed1"] = 1. + 2e-11
    with pytest.raises(ValueError, match="registered fixed winner"):
        strict_gate(candidate, members())


def test_top2_average_budget_covers_atomic_replacement_and_preserves_reserve():
    budget = average_budget([312107295, 312109000])
    assert budget["reserve_bytes"] == 20 * 1024**3
    assert budget["atomic_temporary_bytes"] == budget["persistent_checkpoint_bytes"]
    assert budget["model_file_ceiling_bytes"] >= 1.25 * 312109000
    assert budget["peak_additional_bytes"] == 2 * budget["model_file_ceiling_bytes"] + 2 * 1024**3
    assert check_storage(budget, budget["required_free_bytes_at_launch"], launch=True)
    with pytest.raises(RuntimeError):
        check_storage(budget, budget["required_free_bytes_at_launch"] - 1, launch=True)
    with pytest.raises(ValueError):
        average_budget([0, 1])
