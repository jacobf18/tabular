import pytest

from preflight_dwip_teacher_mean_horizon import runtime_projection, verify_teacher_pair


def test_teacher_identity_uses_frozen_hashes_not_different_reporting_fields():
    proposed = [{"queue": name, "sha256": name, "step": 54000} for name in ("first", "second")]
    previous = [{"queue": name, "sha256": name, "weight": .5} for name in ("first", "second")]
    verify_teacher_pair(proposed, previous)
    with pytest.raises(ValueError):
        verify_teacher_pair(list(reversed(proposed)), previous)
    previous[0]["weight"] = .7
    with pytest.raises(ValueError):
        verify_teacher_pair(proposed, previous)


def test_conservative_projection_includes_validation_saves_and_full_followups():
    projection = runtime_projection(.5, 10., 1000.)
    assert projection["training_seconds"] == 64000.
    assert projection["fixed_validation_seconds"] == 65 * 10 * 1.25
    assert projection["checkpoint_seconds"] == 975.
    assert projection["followup_seconds"] == 3600.
    assert projection["total_seconds"] < 24 * 3600
    assert projection["overhead_seconds"] < 4 * 3600


@pytest.mark.parametrize("inputs", ((.6, 10., 1000.), (.5, 200., 1000.), (.5, 10., 7200.),
                                    (float("nan"), 10., 1000.), (.5, 0., 1000.), (.5, 10., -1.)))
def test_unmeasured_nonfinite_or_infeasible_runtime_is_rejected(inputs):
    with pytest.raises(ValueError):
        runtime_projection(*inputs)


def test_larger_observed_followups_are_not_hidden_by_floor():
    assert runtime_projection(.4, 10., 3000.)["followup_seconds"] == 6000.
