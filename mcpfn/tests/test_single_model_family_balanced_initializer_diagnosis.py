from copy import deepcopy

import pytest
import torch

import diagnose_single_model_family_balanced_initializer as diagnosis
import diagnose_single_model_legacy_blend_initializer as predecessor


def cases():
    return [{"index": index, "seed": 1000 + index, "rows": 128, "columns": 96,
             "targets": 300, "domain": "imputation", "family": "mcar", "structured": True} for index in range(284)]


def test_all_reused_functions_bind_new_original_run_and_frozen_bank_only():
    for name in ("require_storage", "verify", "require_full_case_identity", "require_frozen_state", "audit"):
        function = getattr(diagnosis, name)
        assert function.__globals__["original"] is diagnosis.original
        assert function.__globals__["training"].OUTPUT.name == "family_balanced_retention_v43"
        assert function.__globals__["OUTPUT"] == diagnosis.OUTPUT
        assert getattr(predecessor, name).__globals__["OUTPUT"] != diagnosis.OUTPUT
    assert diagnosis.run.__wrapped__.__globals__["require_closed_failure"] is diagnosis.require_closed_failure
    assert diagnosis.audit.__globals__["comparisons"] is diagnosis.comparisons


@pytest.mark.parametrize("mutation", [None, "drop", "duplicate", "seed", "shape", "mask_count"])
def test_all_original_contexts_and_query_populations_are_required(mutation):
    original, changed = cases(), cases()
    if mutation == "drop":
        changed.pop()
    elif mutation == "duplicate":
        changed[20] = deepcopy(changed[19])
    elif mutation == "seed":
        changed[20]["seed"] += 1
    elif mutation == "shape":
        changed[20]["rows"] += 1
    elif mutation == "mask_count":
        changed[20]["targets"] += 1
    if mutation is None:
        diagnosis.require_full_case_identity(changed, original)
    else:
        with pytest.raises(ValueError):
            diagnosis.require_full_case_identity(changed, original)


def test_frozen_state_includes_buffers():
    model = torch.nn.Linear(2, 3)
    model.register_buffer("fixed", torch.tensor([1.]))
    state = deepcopy(model.state_dict())
    diagnosis.require_frozen_state(model, state)
    model.fixed.add_(1)
    with pytest.raises(ValueError):
        diagnosis.require_frozen_state(model, state)


def test_inference_wrapper_uses_the_new_verified_registration_without_gradients(monkeypatch):
    def sentinel():
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        raise RuntimeError("new verified entry")

    monkeypatch.setattr(diagnosis, "verify", sentinel)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="new verified entry"):
        diagnosis.run()


def test_drift_table_keeps_all_22_metrics_and_both_reference_directions(monkeypatch):
    monkeypatch.setattr(diagnosis.fixed, "retention_values", lambda metrics, base:
        [(f"group_{index}", "metric", metrics + index) for index in range(22)])
    rows = diagnosis.comparisons(2., 3., 1., {})
    assert len(rows) == 22
    assert all(row["V43_minus_V29"] == 1. and row["V29_minus_V8"] == 1. and row["V43_minus_V8"] == 2. for row in rows)
    assert all("V41_step50" not in row for row in rows)


def test_incomplete_metric_population_is_rejected(monkeypatch):
    monkeypatch.setattr(diagnosis.fixed, "retention_values", lambda metrics, base: [("mcar", "common_nrmse", 1.)])
    with pytest.raises(ValueError, match="original22"):
        diagnosis.comparisons({}, {}, {}, {})


def test_consumed_bank_does_not_become_fresh_confirmation():
    assert diagnosis.original.CONFIRMATION.name == "family_balanced_retention_v43_confirmation"
    assert diagnosis.PARENT_SHA256 == predecessor.PARENT_SHA256
    assert "cannot count as fresh qualification" in diagnosis.POLICY
    assert diagnosis.WRITE_BUDGETS == predecessor.WRITE_BUDGETS
    assert diagnosis.OUTPUT != predecessor.OUTPUT and diagnosis.SCRATCH != predecessor.SCRATCH
