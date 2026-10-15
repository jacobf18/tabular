from copy import deepcopy

import pytest

import recover_single_model_decoder_finite_audit as recovery


def test_nested_report_retains_every_list_record_and_exact_decision():
    report = {"changes": [{"role": "decoder", "metrics": {"rmse": [1., .9]}, "retained": True},
                          {"role": "output_layer", "undefined": None}]}
    recovery.compare_nested(deepcopy(report), report)
    changed = deepcopy(report)
    changed["changes"][0]["retained"] = False
    with pytest.raises(ValueError):
        recovery.compare_nested(changed, report)
    changed = deepcopy(report)
    changed["changes"].pop()
    with pytest.raises(ValueError, match="coverage"):
        recovery.compare_nested(changed, report)


def test_nested_numeric_mismatch_keeps_the_original_tolerance():
    with pytest.raises(AssertionError):
        recovery.compare_nested({"changes": [{"value": 1.00001}]}, {"changes": [{"value": 1.}]})
    with pytest.raises(ValueError, match="keys"):
        recovery.compare_nested({"changes": [{"other": 1.}]}, {"changes": [{"value": 1.}]})


def test_original_comparator_is_restored_after_error():
    original = recovery.finite.probe.alignment.compare_metric_records
    with pytest.raises(RuntimeError):
        with recovery.nested_report_comparator():
            assert recovery.finite.probe.alignment.compare_metric_records is recovery.compare_nested
            raise RuntimeError("test")
    assert recovery.finite.probe.alignment.compare_metric_records is original
