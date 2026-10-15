import inspect
import math

import numpy as np
import pytest
import torch

import audit_single_model_standardized_teacher_huber_probe as audit
import test_single_model_column_anchor_retention as fixtures
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize('imputation', [False, True])
def test_every_column_and_endpoint_matches_independent_formula(imputation):
    target, mask = fixtures.column_example()
    teacher = torch.nan_to_num(target) - .25
    predictions = {'before': (torch.nan_to_num(target) + .3).numpy(),
        'candidate': (torch.nan_to_num(target) + .4).numpy()}
    rows, values = audit.describe(target, mask, teacher, predictions, native=imputation)
    assert len(rows) == target.shape[-1]
    assert sum(row['queries'] for row in rows) == int(mask.sum())
    assert any(row['queries'] == 0 for row in rows)
    assert any(row['single_query'] for row in rows)
    assert any(row['constant_multi_query'] for row in rows)
    for role, prediction in predictions.items():
        if imputation:
            expected = audit.probe.retention.independent_teacher_mean(target, mask, torch.from_numpy(prediction), teacher)
        else:
            residual = prediction[mask.numpy()].astype(np.float64) - teacher[mask].double().numpy()
            expected = 20. * float(np.mean(np.where(np.abs(residual) <= 1., .5 * residual**2, np.abs(residual) - .5)))
            assert values[role + '_applied_mean20'] == values[role + '_uniform_mean20']
        assert values[role + '_applied_mean20'] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert values['applied_change'] > 0


def test_column_mass_does_not_apply_variance_twice():
    target, mask = fixtures.column_example()
    teacher = torch.nan_to_num(target)
    prediction = (teacher + .5).numpy()
    rows, values = audit.describe(target, mask, teacher, {'before': prediction, 'candidate': prediction}, native=True)
    for row in rows:
        if row['queries']:
            expected = int(mask.sum()) / (values['queried_columns'] * row['queries'])
            assert row['applied_coefficient'] == expected
            assert row['residual_standardization_scale'] == math.sqrt(row['shrunk_variance'])
    assert values['applied_change'] == 0


def test_compiled_callback_sees_all_queries_and_scales(monkeypatch):
    target, mask = fixtures.column_example()
    teacher = torch.nan_to_num(target)
    prediction = (teacher + .5).numpy()
    original = audit.probe.retention.reweight_teacher_mean
    calls = []
    def capture(base, point, reference, weights, scales):
        calls.append((point.numel(), weights.numel(), scales.numel()))
        return original(base, point, reference, weights, scales)
    monkeypatch.setattr(audit.probe.retention, 'reweight_teacher_mean', capture)
    audit.describe(target, mask, teacher, {'before': prediction, 'candidate': prediction}, native=True)
    assert calls == [(int(mask.sum()),) * 3] * 2


@pytest.mark.parametrize('bad', ['', '0' * 63, 'x' * 64, None, 1])
def test_cli_requires_exact_original_configuration_sha(bad):
    with pytest.raises(ValueError, match='configuration SHA256'):
        audit.audit(bad)


def test_complete_probe_and_correct_runtime_bindings():
    assert audit.OUTPUT.name == 'standardized_teacher_huber_preflight_v1_mean_audit'
    assert 'test_single_model_standardized_teacher_huber_probe_audit.py' in audit.AUDIT_SOURCE
    assert '"imputation_reweighted_tasks": 26' in audit.AUDIT_SOURCE
    assert '"component_role_counts"] != {"v8": 24, "v29": 26}' in audit.AUDIT_SOURCE
    for name in ('describe', 'imputation_task', '_audit', 'audit', 'main'):
        function = inspect.unwrap(getattr(audit, name))
        assert function.__globals__ is vars(audit) and not unresolved_globals(function)
