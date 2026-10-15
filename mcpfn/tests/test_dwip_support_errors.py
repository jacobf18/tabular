import pytest
import torch

from analyze_dwip_support_errors import compare_cases, support_slices


def support_case():
    source = torch.tensor([[[float("nan"), float("nan")], [1., float("nan")], [1., 2.]]])
    mask = torch.isnan(source)
    case = {"id": "case", "group": "test_group", "monitor_only": False,
            "source": source, "target": torch.zeros_like(source), "mask": mask}
    cached = {"target": case["target"][mask], "flat_indices": mask.flatten().nonzero().flatten(), "mean": torch.ones(3)}
    return case, cached


def test_support_slices_partition_each_axis_and_keep_locations():
    case, cached = support_case()
    result = support_slices(case, cached, cached)
    assert result["row_0"]["n"] == 2
    assert result["row_1"]["n"] == 1
    assert result["column_1"]["n"] == 2
    assert result["column_2_4"]["n"] == 1
    with pytest.raises(ValueError, match="locations"):
        support_slices(case, cached, {**cached, "flat_indices": cached["flat_indices"] + 1})


def test_support_comparison_excludes_monitor_groups_without_dropping_evidence():
    case, cached = support_case()
    after = {**cached, "mean": cached["mean"] * 2}
    result = compare_cases([case], {"case": cached}, {"case": after})
    assert result["summaries"]["row_0"]["mean_relative_group_rmse"] == 2
    assert result["summaries"]["row_0"]["groups_improved"] == 0
    result = compare_cases([{**case, "monitor_only": True}], {"case": cached}, {"case": after})
    assert not result["summaries"]
    assert result["groups"]["row_0"]["test_group"]["monitor_only"]
