import pytest

from compare_imputation_nrmse import MODELS, aggregate, markdown_table
from evaluate_imputation_baseline import model_spec


def test_summary_uses_equal_dataset_weights():
    rows = []
    for dataset, values in (("first", [1.0, 3.0]), ("second", [10.0])):
        for value in values:
            rows.append({"suite": "missbench", "dataset": dataset, **dict.fromkeys(MODELS, value)})
    summaries = aggregate(rows, ("suite",))
    assert len(summaries) == 1
    assert summaries[0]["cases"] == 3
    assert summaries[0]["datasets"] == 2
    assert all(summaries[0][model] == pytest.approx(6.0) for model in MODELS)


def test_lowest_nrmse_is_highlighted():
    row = {"suite": "uci", "cases": 1800, "new": 2.0, "tabimpute_v2": 1.0, "dwip_v2_2": 0.5}
    rendered = markdown_table([row], ("suite",))
    assert "**0.500000**" in rendered
    assert "**2.000000**" not in rendered


def test_baseline_identity_and_native_prediction_statistics():
    dwip = model_spec("dwip_v2_2")
    paper = model_spec("tabimpute_v2")
    assert dwip["checkpoint"].endswith("dwip_v2_t003_checkpoint_17000.pth")
    assert dwip["prediction_stat"] == "mean"
    assert paper["checkpoint"].endswith("tabimputev2.ckpt")
    assert paper["prediction_stat"] == "median"
