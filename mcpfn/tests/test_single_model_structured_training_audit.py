from copy import deepcopy
import json
import os

import pytest

import audit_single_matrixformer_structured_training as audit
from test_single_model_structured_mixture import records


def reports():
    cases = records()
    baseline = audit.mixture.aggregate(cases, "student")
    reference = {"references": baseline}
    score, ratios, eligible = audit.mixture.selection_score(baseline, baseline, baseline)
    return [{"step": 0, "student": baseline, "cases": cases, "score": score, "ratios": ratios,
             "eligible": eligible, "best_step": 0}], reference


def test_initialization_selection_reproduces_exactly():
    values, reference = reports()
    result = audit.selection_trace(values, reference)
    assert result == [{"step": 0, "score": 1., "eligible": True, "best_step": 0}]


@pytest.mark.parametrize("field", ["score", "best_step", "eligible", "student", "cases", "ratios"])
def test_modified_selection_evidence_is_rejected(field):
    values, reference = reports()
    if field in ("score", "best_step"):
        values[0][field] += .00001
    elif field == "eligible":
        values[0][field] = False
    elif field == "student":
        values[0][field]["recsys"]["rmse"] += .1
    elif field == "cases":
        values[0][field][0]["student"]["rmse"] += .1
    else:
        values[0][field]["recsys_rmse"] += .1
    with pytest.raises(ValueError):
        audit.selection_trace(values, reference)


def test_numerical_reproduction_tolerance_never_changes_selection_tolerance():
    audit.close_values({"value": 1.00000001, "count": 3, "undefined": None}, {"value": 1., "count": 3, "undefined": None})
    with pytest.raises(ValueError):
        audit.close_values({"count": 4}, {"count": 3})
    values, reference = reports()
    second = deepcopy(values[0])
    second["step"] = 50
    second["cases"][0]["student"]["rmse"] += 1e-10
    second["student"] = audit.mixture.aggregate(second["cases"], "student")
    second["score"], second["ratios"], second["eligible"] = audit.mixture.selection_score(second["student"], values[0]["student"], reference["references"])
    assert not second["eligible"]
    assert audit.selection_trace(values + [second], reference)[-1]["best_step"] == 0


def test_wait_checks_a_specific_live_process_not_just_a_status_file(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "training", "pid": os.getpid()}))
    assert not audit.ready(tmp_path, 50)
    (tmp_path / "validation_50.json").write_text("{}")
    assert audit.ready(tmp_path, 50)


def test_terminal_without_requested_prefix_is_not_a_restart_signal(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed", "pid": os.getpid()}))
    with pytest.raises(ValueError, match="never restart"):
        audit.ready(tmp_path, 50)
