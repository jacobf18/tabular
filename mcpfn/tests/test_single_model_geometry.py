import math

import pytest
import torch

from analyze_single_model_geometry import compare_states


def test_projection_distinguishes_interpolation_from_shared_ancestor_task_delta():
    shared = {"decoder.weight": torch.tensor([0., 0.])}
    general = {"decoder.weight": torch.tensor([1., 0.])}
    specialist = {"decoder.weight": torch.tensor([1., 2.])}
    student = {"decoder.weight": torch.tensor([1., 1.])}
    result = compare_states(student, general, specialist, shared)
    assert result["all"]["student_update_l2"] == 1.
    assert result["all"]["interpolation"]["projection_coefficient"] == .5
    assert result["all"]["interpolation"]["cosine_with_update"] == 1.
    assert result["all"]["recsys_task"]["projection_coefficient"] == pytest.approx(.4)
    assert result["all"]["recsys_task"]["cosine_with_update"] == pytest.approx(2 / math.sqrt(5))
    assert result["all"]["large_matrix_task"]["cosine_with_update"] == 0.
    assert result["all"] == result["decoder"]


def test_zero_update_and_unchanged_direction_have_no_cosine():
    state = {"decoder.weight": torch.ones(2)}
    result = compare_states(state, state, state, state)["all"]
    assert result["student_update_l2"] == 0.
    assert result["interpolation"]["projection_coefficient"] is None
    assert result["interpolation"]["cosine_with_update"] is None


def test_incompatible_or_nonfinite_checkpoints_are_rejected():
    state = {"decoder.weight": torch.ones(2)}
    for bad in ({"other": torch.ones(2)}, {"decoder.weight": torch.ones(3)}, {"decoder.weight": torch.tensor([1., float("nan")])}):
        with pytest.raises(ValueError):
            compare_states(state, state, bad, state)
