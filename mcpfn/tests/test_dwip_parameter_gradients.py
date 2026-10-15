import pytest
import torch

from diagnose_dwip_parameter_gradients import gradient_alignment, select_cases


def test_case_selection_is_sorted_balanced_and_excludes_monitors():
    cases = [{"id": name, "group": group, "monitor_only": monitor} for name, group, monitor in
             (("b1", "b", False), ("a2", "a", False), ("a0", "a", True), ("a1", "a", False))]
    assert [case["id"] for case in select_cases(cases, 1)] == ["a1", "b1"]
    with pytest.raises(ValueError):
        select_cases(cases, 0)


def test_gradient_norms_cosine_and_groups_match_concatenated_reference():
    names = ["feature_encoder.weight", "transformer_blocks.0.weight", "transformer_blocks.1.weight"]
    nll = [torch.tensor([3., 4.]), torch.tensor([1., 0.]), None]
    point = [torch.tensor([.6, .8]), torch.tensor([0., 2.]), torch.tensor([1.])]
    result = gradient_alignment(names, nll, point)
    overall = result["groups"]["all"]
    assert overall["nll_norm"] == pytest.approx(26 ** .5)
    assert overall["weighted_point_norm"] == pytest.approx(6 ** .5)
    assert overall["point_to_nll_ratio"] == pytest.approx((6 / 26) ** .5)
    assert overall["cosine"] == pytest.approx(5 / (26 * 6) ** .5)
    assert result["groups"]["feature_encoder"]["cosine"] == pytest.approx(1.)
    assert result["groups"]["transformer_blocks.0"]["cosine"] == 0.
    assert result["groups"]["transformer_blocks.1"]["cosine"] is None
    assert result["unused_parameters"] == {"nll": 1, "point": 0}


def test_gradient_alignment_rejects_nonfinite_values_and_mismatched_lengths():
    with pytest.raises(FloatingPointError):
        gradient_alignment(["weight"], [torch.tensor(float("nan"))], [torch.ones(())])
    with pytest.raises(ValueError):
        gradient_alignment(["weight"], [], [])


def test_tenfold_weight_changes_norm_but_not_direction():
    nll = [torch.tensor([1., 2.])]
    point = [torch.tensor([.3, -.1])]
    current = gradient_alignment(["weight"], nll, point)["groups"]["all"]
    scaled = gradient_alignment(["weight"], nll, [10 * point[0]])["groups"]["all"]
    assert scaled["point_to_nll_ratio"] == pytest.approx(10 * current["point_to_nll_ratio"])
    assert scaled["cosine"] == pytest.approx(current["cosine"])
