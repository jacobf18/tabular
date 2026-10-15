import pytest
import torch

import preflight_single_model_complete_missingness_training as smoke


def test_smoke_test_covers_all_masks_large_contexts_and_original_rating_domain():
    cases = smoke.cases()
    assert len(cases) == 21 and len({case["seed"] for case in cases}) == 21
    assert {case["family"] for case in cases if case["domain"] == "imputation"} == set(smoke.coverage.FAMILIES)
    assert sum(case["shape"] == [1024, 256] for case in cases) == 8
    assert cases[-1]["domain"] == "recsys" and cases[-1]["shape"] == [1000, 1000]


def test_training_finiteness_rejects_missing_and_nonfinite_gradients():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(FloatingPointError):
        smoke.finite_training_state(model, gradients=True)
    model(torch.ones(1, 2)).sum().backward()
    assert smoke.finite_training_state(model, gradients=True) == 2
    model.weight.grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        smoke.finite_training_state(model, gradients=True)


def test_training_finiteness_rejects_nonfinite_parameters():
    model = torch.nn.Linear(2, 1)
    assert smoke.finite_training_state(model, gradients=False) == 2
    with torch.no_grad():
        model.bias.fill_(float("inf"))
    with pytest.raises(FloatingPointError):
        smoke.finite_training_state(model, gradients=False)
