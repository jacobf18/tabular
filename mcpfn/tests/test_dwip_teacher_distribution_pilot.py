from copy import deepcopy

import pytest

from pilot_dwip_teacher_distribution import ARMS, EVALUATION_STEPS, METRICS, pilot_decision, summarize_cases


def arms():
    return {arm: [{"step": step, "equal_batch_mean": dict.fromkeys(METRICS, 1.)} for step in EVALUATION_STEPS] for arm in ARMS}


@pytest.mark.parametrize("teacher,truth,expected", ((.9, .9, True), (.9, 1., False), (1., .9, False), (1.1, .9, False)))
def test_pilot_gate_requires_both_final_mse_improvements_without_selecting_weights(teacher, truth, expected):
    values = arms()
    values[ARMS[1]][-1]["equal_batch_mean"].update(teacher_mse=teacher, true_mse=truth)
    decision = pilot_decision(values)
    assert decision["supports_full_trial_admission_review"] is expected
    assert not decision["training_ready"] and not decision["checkpoint_created_or_selected"]


def test_intermediate_pilot_winner_cannot_replace_fixed_final_endpoint():
    values = arms()
    values[ARMS[1]][1]["equal_batch_mean"].update(teacher_mse=0., true_mse=0.)
    assert not pilot_decision(values)["supports_full_trial_admission_review"]


@pytest.mark.parametrize("change", ("missing_arm", "missing_step", "unordered", "nonfinite"))
def test_incomplete_or_changed_pilot_scope_is_rejected(change):
    values = arms()
    if change == "missing_arm":
        values.pop(ARMS[0])
    elif change == "missing_step":
        values[ARMS[1]].pop(1)
    elif change == "unordered":
        values[ARMS[1]].reverse()
    else:
        values[ARMS[1]][-1]["equal_batch_mean"]["true_nll"] = float("nan")
    with pytest.raises(ValueError):
        pilot_decision(values)


def test_reference_summary_uses_equal_batch_weights_and_retains_components():
    records = [{"id": "first", "component": "one", **dict.fromkeys(METRICS, 1.)},
               {"id": "second", "component": "two", **dict.fromkeys(METRICS, 3.)}]
    result = summarize_cases(records)
    assert result["equal_batch_mean"] == dict.fromkeys(METRICS, 2.)
    assert result["components"]["one"] == dict.fromkeys(METRICS, 1.)
    assert result["components"]["two"] == dict.fromkeys(METRICS, 3.)


@pytest.mark.parametrize("change", ("empty", "duplicate", "nonfinite"))
def test_reference_summary_fails_closed(change):
    records = [{"id": "first", "component": "one", **dict.fromkeys(METRICS, 1.)}]
    if change == "empty":
        records = []
    elif change == "duplicate":
        records.append(deepcopy(records[0]))
    else:
        records[0]["teacher_mse"] = float("inf")
    with pytest.raises(ValueError):
        summarize_cases(records)
