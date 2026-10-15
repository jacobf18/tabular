from copy import deepcopy

import pytest

import recover_single_model_top_block_rate_audit as recovery


def decision():
    return {'repetitions': {'0': {'changes': {'cohort/rmse': -1e-15}, 'accepted': True,
                                 'individual_draw_diagnostics': {'changes': {'cohort_draw_0/rmse': -1e-15,
                                                                           'cohort_draw_1/rmse': None}}}},
            'finite_feasible': True}


def test_scalar_roundoff_is_not_exact_log_identity_but_signs_remain_mandatory():
    expected = decision()
    actual = deepcopy(expected)
    actual['repetitions']['0']['changes']['cohort/rmse'] -= 1e-16
    recovery.compare_repeated_decisions(actual, expected)
    recovery.compare_repeated_decisions(None, None)


@pytest.mark.parametrize('scope', ['paired', 'individual'])
@pytest.mark.parametrize('replacement', [0., 1e-16])
def test_tiny_sign_change_cannot_hide_behind_existing_scalar_tolerance(scope, replacement):
    expected, actual = decision(), decision()
    target = actual['repetitions']['0']
    if scope == 'individual':
        target = target['individual_draw_diagnostics']
    target['changes']['cohort/rmse' if scope == 'paired' else 'cohort_draw_0/rmse'] = replacement
    with pytest.raises(ValueError, match='exact paired and individual'):
        recovery.compare_repeated_decisions(actual, expected)


@pytest.mark.parametrize('mutation', ['decision', 'undefined', 'missing_repeat'])
def test_original_decisions_and_coverage_cannot_change(mutation):
    expected, actual = decision(), decision()
    if mutation == 'decision':
        actual['finite_feasible'] = False
    elif mutation == 'undefined':
        actual['repetitions']['0']['individual_draw_diagnostics']['changes']['cohort_draw_1/rmse'] = 0.
    else:
        actual['repetitions'].clear()
    with pytest.raises((ValueError, AssertionError)):
        recovery.compare_repeated_decisions(actual, expected)


def test_additive_audit_preserves_full_registered_checks_and_never_writes_source_status():
    original = recovery.probe.ADAPTED_SOURCE
    actual = recovery.adapted_audit_source()
    assert 'paired.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], result["before"]["0"], 84)' in actual
    assert 'for measured in before.values():' in actual
    assert 'compare_nested(measured, original["record"]["point_before"])' in actual
    assert 'compare_repeated_decisions(repeated, row["repeated"])' in actual
    assert 'paired.check_endpoint(' in actual
    assert 'check_reference(original, record, snapshot, directory / "candidate_parameters.pt")' in actual
    assert 'audit_endpoint_storage(row["endpoint"])' in actual
    assert 'grids += 56' in actual and 'grids, endpoints = 56, []' in actual
    assert 'write_json(recovery / "audit.json",' in actual
    assert 'status(output, "complete")' not in actual
    assert 'write_json(output / "audit.json",' not in actual
    assert recovery.probe.ADAPTED_SOURCE == original
    compile(actual, '<test-audit-adaptation>', 'exec')
