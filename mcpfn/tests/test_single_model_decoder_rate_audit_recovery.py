from copy import deepcopy
import inspect

import pytest

import recover_single_model_decoder_rate_audit as recovery


def decisions(change):
    before = {"0": {"rmse": 1., "mae": 1., "constant": None}, "1": {"rmse": 1., "mae": 1., "constant": None}}
    candidate = {repeat: {"rmse": 1. + change, "mae": 1.1, "constant": None} for repeat in before}
    return recovery.probe.repeat_decision(before, candidate)


def test_cpu_reduction_rounding_does_not_change_exact_decisions():
    original = decisions(-.01)
    measured = deepcopy(original)
    measured["repetitions"]["0"]["changes"]["rmse"] += 9e-15
    recovery.compare_repeated_decisions(measured, original)
    recovery.compare_repeated_decisions(None, None)


@pytest.mark.parametrize("before,after", [(-1e-14, 1e-14), (-1e-14, 0.), (0., 1e-14), (1e-14, -1e-14)])
def test_even_subtolerance_sign_changes_fail_when_aggregate_acceptance_agrees(before, after):
    original = decisions(-.01)
    measured = deepcopy(original)
    original["repetitions"]["0"]["changes"]["rmse"] = before
    measured["repetitions"]["0"]["changes"]["rmse"] = after
    with pytest.raises(ValueError, match="exact improvement and retention signs"):
        recovery.compare_repeated_decisions(measured, original)


def test_changed_coverage_or_acceptance_remains_an_error():
    original = decisions(-.01)
    measured = deepcopy(original)
    measured["finite_feasible"] = not original["finite_feasible"]
    with pytest.raises(ValueError, match="exact coverage"):
        recovery.compare_repeated_decisions(measured, original)
    measured = deepcopy(original)
    measured["repetitions"]["1"]["changes"].pop("constant")
    with pytest.raises(ValueError, match="keys"):
        recovery.compare_repeated_decisions(measured, original)


def test_existing_scalar_tolerance_is_not_relaxed():
    original = decisions(-.01)
    measured = deepcopy(original)
    measured["repetitions"]["0"]["changes"]["rmse"] += 1e-8
    with pytest.raises(AssertionError):
        recovery.compare_repeated_decisions(measured, original)


def test_recovery_leaves_original_source_and_scientific_checks_unchanged():
    original = inspect.getsource(recovery.probe.audit)
    adapted = recovery.adapt_audit_source(original)
    assert inspect.getsource(recovery.probe.audit) == original
    assert 'compare_nested(before, result["before"])' in adapted
    assert 'compare_nested(candidate["0"], record["point_after"])' in adapted
    assert 'checks.check_endpoint(' in adapted
    assert 'np.testing.assert_allclose(products, record["candidate_products"], rtol=1e-9, atol=1e-12)' in adapted
    assert 'write_json(output / "audit.json"' not in adapted
    assert 'status(output, "complete")' not in adapted
    assert recovery.recovered_auditor().__name__ == "recovered_audit"
    with pytest.raises(ValueError, match="exact original"):
        recovery.adapt_audit_source(original.replace('def audit(output):', 'def altered(output):'))
