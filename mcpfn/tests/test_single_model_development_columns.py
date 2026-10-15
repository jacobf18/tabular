from copy import deepcopy

import numpy as np
import pytest

import analyze_single_model_development_columns as analysis


def records():
    result = []
    for index in range(152):
        result.append({"index": index, "seed": 1000 + index, "rows": 8, "columns": 3,
            "gradient_data": "synthetic", "domain": "recsys" if 40 <= index < 56 else "imputation",
            "structured": index >= 56})
    return result


def test_all_roles_and_every_case_must_match():
    original = records()
    assert len(analysis.paired_records(original, deepcopy(original), deepcopy(original))) == 152


def test_bank_is_explicit_even_when_other_registered_banks_exist(tmp_path):
    bank = tmp_path / "development/synthetic_bank.pt"
    other = tmp_path / "earlier/synthetic_bank.pt"
    config = {"input_sha256": {str(bank): "bank", str(bank.parent / "reference.json"): "reference", str(other): "earlier"}}
    assert analysis.registered_bank(config, bank) == ("bank", "reference")
    with pytest.raises(ValueError, match="Explicitly"):
        analysis.registered_bank(config, other)
    with pytest.raises(ValueError, match="Explicitly"):
        analysis.registered_bank(config, tmp_path / "unregistered/synthetic_bank.pt")


@pytest.mark.parametrize("mutation", ["omitted", "seed", "order", "shape", "cohort", "real_data"])
def test_missing_or_relabelled_cases_cannot_be_hidden(mutation):
    original = records()
    candidate = deepcopy(original)
    if mutation == "omitted":
        candidate.pop()
    elif mutation == "order":
        candidate[0], candidate[1] = candidate[1], candidate[0]
    else:
        field, value = {"seed": ("seed", -1), "shape": ("rows", 9), "cohort": ("structured", True),
                        "real_data": ("gradient_data", "real")}[mutation]
        candidate[0][field] = value
    with pytest.raises(ValueError):
        analysis.paired_records(original, deepcopy(original), candidate)


def test_column_contributions_keep_original_case_weights():
    rows = []
    for index, count in ((0, 1), (1, 3)):
        for column in range(count):
            row = {"index": index, "cohort": "legacy", "prior": "anchor" if index == 0 else "spectral",
                   "scale_band": "small" if column == 0 else "large", "support_band": "few", "pattern": "mcar"}
            for suite in analysis.METRICS:
                for role, value in (("general", 1.), ("initialization", 2.), ("candidate", 3.)):
                    row[f"{role}_{suite}_contribution"] = value / count
            rows.append(row)
    result = analysis.grouped_contributions(rows, {"legacy": 2})
    for suite in analysis.METRICS:
        total = next(row for row in result if row["grouping"] == "all" and row["metric"] == suite)
        assert total["candidate"] == 3. and total["initialization"] == 2.
        for grouping in ("prior", "scale_band", "support_band", "pattern"):
            assert sum(row["candidate"] for row in result if row["grouping"] == grouping and row["metric"] == suite) == pytest.approx(3.)


def test_reconstructed_metrics_and_role_specific_coverage_must_match():
    truth = np.array([[0., 2.], [1., 2.]], dtype=np.float32)
    mask = np.ones_like(truth, dtype=bool)
    arrays = {role: truth.copy() for role in analysis.columns.ROLES}
    rows = analysis.columns.column_rows(truth, mask, truth, arrays)
    metrics = {"rmse": 0., "missbench_nrmse": 0., "uci_nrmse": 0., "common_nrmse": 0.,
        "column_coverage": {"missbench_scored_columns": 2, "uci_scored_columns": 2, "common_nonconstant_columns": 1}}
    expected = {role: deepcopy(metrics) for role in arrays}
    analysis.verify_metrics(rows, truth, mask, arrays, expected)
    expected["candidate"]["common_nrmse"] = .1
    with pytest.raises(ValueError, match="stored metric"):
        analysis.verify_metrics(rows, truth, mask, arrays, expected)
    expected["candidate"]["common_nrmse"] = 0.
    expected["candidate"]["column_coverage"]["common_nonconstant_columns"] = 2
    with pytest.raises(ValueError, match="coverage"):
        analysis.verify_metrics(rows, truth, mask, arrays, expected)
