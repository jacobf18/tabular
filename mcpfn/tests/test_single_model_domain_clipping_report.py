from copy import deepcopy

import pytest

import report_single_model_domain_clipping as report
from test_single_model_domain_clipping_recipe import log_row
from test_single_model_structured_report import example


def test_report_labels_change_without_changing_any_scores_or_callback():
    summary, rows = report.original.summarize(*example())
    unchanged = deepcopy(rows)
    previous = report.original.markdown
    with report.report_label():
        text = report.original.markdown(summary, rows)
        assert text.startswith("# V10 per-domain-clipping training outcome")
        assert "# V9 structured" not in text
        assert "no bitwise-input or real-benchmark parity" in text
    assert rows == unchanged
    assert report.original.markdown is previous


def test_report_callback_restores_on_error():
    previous = report.original.markdown
    with pytest.raises(RuntimeError):
        with report.report_label():
            raise RuntimeError("test")
    assert report.original.markdown is previous


def test_clipping_report_requires_the_full_original_audit():
    rows = []
    for step in range(1, 401):
        row = log_row()
        row["step"] = step
        rows.append(row)
    audit = {"complete": True, "through_step": 400, "logged_domain_gradients_checked": 800,
             "raw_prediction_audit_sha256": "fixed", "research_selection_tolerance": 0,
             "optimizer_trajectory_replayed": False, "checked_rows": rows}
    report.validate_clipping_audit(audit, "fixed")
    with pytest.raises(ValueError):
        report.validate_clipping_audit(audit, "changed")
    audit["checked_rows"].pop(5)
    with pytest.raises(ValueError, match="every registered optimizer update"):
        report.validate_clipping_audit(audit, "fixed")
