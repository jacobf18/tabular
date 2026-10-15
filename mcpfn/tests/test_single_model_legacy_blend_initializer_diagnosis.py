from copy import deepcopy

import pytest
import torch

import diagnose_single_model_legacy_blend_initializer as diagnosis


def cases():
    return [{"index": index, "seed": 1000 + index, "rows": 128, "columns": 96,
        "targets": 300, "domain": "imputation", "prior": "spectral", "component": "continuous",
        "family": "mcar", "structured": True} for index in range(284)]


def test_full_identity_accepts_only_the_same_284_case_population():
    original = cases()
    diagnosis.require_full_case_identity(deepcopy(original), original)


@pytest.mark.parametrize("field", ["index", "seed", "rows", "columns", "targets", "domain", "prior", "component", "family", "structured"])
def test_context_metadata_changes_are_rejected(field):
    original = cases()
    changed = deepcopy(original)
    changed[20][field] = "different"
    with pytest.raises(ValueError, match="identity"):
        diagnosis.require_full_case_identity(changed, original)


def test_no_dropped_reordered_or_duplicated_cases():
    original = cases()
    for changed in (original[:-1], list(reversed(original)), original[:20] + original[19:-1]):
        with pytest.raises(ValueError):
            diagnosis.require_full_case_identity(changed, original)


def test_parameters_and_buffers_must_remain_bitwise_unchanged():
    model = torch.nn.Linear(2, 3)
    model.register_buffer("fixed", torch.tensor([1.]))
    state = deepcopy(model.state_dict())
    diagnosis.require_frozen_state(model, state)
    with torch.no_grad():
        model.fixed.add_(1)
    with pytest.raises(ValueError, match="bitwise"):
        diagnosis.require_frozen_state(model, state)
    model.load_state_dict(state)
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(ValueError, match="bitwise"):
        diagnosis.require_frozen_state(model, state)


def test_inference_mode_wraps_the_only_new_model_evaluation(monkeypatch):
    def stop():
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        raise RuntimeError("entry sentinel")

    monkeypatch.setattr(diagnosis, "verify", stop)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="entry sentinel"):
        diagnosis.run()
    assert not torch.is_inference_mode_enabled()


def test_comparisons_keep_all_metrics_and_both_drift_directions(monkeypatch):
    monkeypatch.setattr(diagnosis.fixed, "retention_values", lambda metrics, base:
        [(f"group_{index}", "metric", metrics + index) for index in range(22)])
    rows = diagnosis.comparisons(2., 3., 1., {})
    assert len(rows) == 22
    assert all(row["V41_minus_V29"] == 1. and row["V29_minus_V8"] == 1. and row["V41_minus_V8"] == 2. for row in rows)


def test_missing_comparison_population_is_rejected(monkeypatch):
    monkeypatch.setattr(diagnosis.fixed, "retention_values", lambda metrics, base: [("mcar", "common_nrmse", 1.)])
    with pytest.raises(ValueError, match="original22"):
        diagnosis.comparisons({}, {}, {}, {})


def test_frozen_scope_cannot_be_mistaken_for_fresh_confirmation():
    assert diagnosis.PARENT_SHA256 == "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
    assert diagnosis.WRITE_BUDGETS["scratch"] == 1024**3
    assert "Post-hoc synthetic diagnosis only" in diagnosis.POLICY
    assert "cannot count as fresh qualification" in diagnosis.POLICY
