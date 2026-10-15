from copy import deepcopy

import pytest

import report_single_model_family_rehearsal as report
from test_single_model_structured_report import example


def test_report_keeps_every_guard_and_boundary_under_the_correct_experiment_label():
    summary, rows = report.original.summarize(*example())
    unchanged = deepcopy(rows)
    previous = report.original.markdown
    with report.report_label():
        text = report.original.markdown(summary, rows)
        assert text.startswith("# V11 family-rehearsal training outcome")
        assert "all22 original retention guards and all nine validation boundaries" in text
        assert "No real-benchmark parity" in text
        assert "not a blinded comparison" in text
        assert "# V9" not in text and "# V10" not in text
    assert rows == unchanged
    assert report.original.markdown is previous


def test_report_callback_restores_after_an_error():
    previous = report.original.markdown
    with pytest.raises(RuntimeError):
        with report.report_label():
            raise RuntimeError("test")
    assert report.original.markdown is previous


def test_unexpected_template_cannot_silently_mislabel_results(monkeypatch):
    callback = lambda summary, rows: "Different report format"
    monkeypatch.setattr(report.original, "markdown", callback)
    with pytest.raises(ValueError, match="do not mislabel"):
        with report.report_label():
            report.original.markdown({}, [])
    assert report.original.markdown is callback
