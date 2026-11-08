import numpy as np
import pytest
import torch

import audit_single_model_four_draw_probe as audit
import audit_single_model_standardized_teacher_huber_probe as original
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_independent_mean_audit_binds_full_four_draw_scope():
    assert audit.probe.OUTPUT.name == "four_draw_preflight_v1"
    assert audit.OUTPUT.name == "four_draw_preflight_v1_mean_audit"
    for token in ('"tasks": 56', '"imputation_reweighted_tasks": 52', '"native_reweighted_tasks": 4',
            '"structured_reweighted_tasks": 48', '"unchanged_rating_tasks": 4', "{True: 52, False: 4}"):
        assert token in audit.AUDIT_SOURCE
    for function in (audit.describe, audit.imputation_task, audit._audit, audit.audit, audit.main):
        assert not unresolved_globals(function)
        assert function.__globals__ is vars(audit)
    assert '"tasks": 28' in original.AUDIT_SOURCE
    assert '"tasks": 28' not in audit.AUDIT_SOURCE
    assert '{"v8": 48, "v29": 52}' in audit.AUDIT_SOURCE
    assert '["reference_grids"] != 52' in audit.AUDIT_SOURCE


@pytest.mark.parametrize("imputation", [False, True])
def test_complete_column_arithmetic_remains_identical(imputation):
    target = torch.tensor([[[1., 2., 0.], [1., 5., 4.], [2., 3., 6.]]])
    mask = torch.tensor([[[True, True, False], [True, False, False], [False, True, False]]])
    teacher = torch.zeros_like(target)
    predictions = {"before": np.ones(target.shape, dtype=np.float32), "candidate": np.full(target.shape, .5, dtype=np.float32)}
    assert audit.describe(target, mask, teacher, predictions, native=imputation) == original.describe(
        target, mask, teacher, predictions, native=imputation)
