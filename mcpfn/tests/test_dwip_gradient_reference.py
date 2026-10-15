import copy

import pytest
import torch

from diagnose_dwip_gradient_reference import GradientMoments, check_reproduction, reference_ids


def test_reference_population_preserves_old_reference_and_excludes_probes():
    identities = reference_ids()
    assert len(identities) == len(set(identities)) == 1024
    assert identities[:64] == list(range(64001, 64065))
    assert identities[64:] == list(range(64129, 65089))
    assert not set(identities).intersection(range(64065, 64129))


def test_gradient_reference_moments_match_explicit_sample_covariance():
    values = torch.tensor([[1., 2.], [3., -2.], [-1., 1.], [5., 3.]])
    moments = GradientMoments()
    for value in values:
        moments.update(value)
    statistics = moments.statistics()
    expected_mean = values.mean(0)
    variance = float(values.var(0, unbiased=True).sum())
    assert torch.equal(moments.mean(), expected_mean)
    assert statistics["sample_variance_trace"] == pytest.approx(variance)
    assert statistics["estimated_standard_error_l2"] == pytest.approx((variance / len(values)) ** .5)
    assert statistics["unbiased_squared_mean_norm_estimate"] == pytest.approx(float(expected_mean.square().sum()) - variance / len(values))


def test_negative_unbiased_squared_mean_estimate_is_retained():
    moments = GradientMoments()
    moments.update(torch.tensor([1.]))
    moments.update(torch.tensor([-1.]))
    statistics = moments.statistics()
    assert statistics["unbiased_squared_mean_norm_estimate"] == -1
    assert statistics["standard_error_to_mean_norm"] is None


def test_empty_singleton_and_immutable_inputs():
    moments = GradientMoments()
    with pytest.raises(ValueError):
        moments.statistics()
    value = torch.tensor([2.])
    moments.update(value)
    assert moments.statistics()["sample_variance_trace"] is None
    assert torch.equal(value, torch.tensor([2.]))
    with pytest.raises(ValueError):
        moments.update(torch.ones(2))
    with pytest.raises(ValueError):
        moments.update(torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError):
        moments.update(torch.tensor([float("nan")]))


def test_reproduction_rejects_changed_cases_and_losses():
    original = {"id": 64001, "cohort": "reference", "family": "spectral", "shape": [2, 80, 40],
                "support": {"support_episode": 1.}, "losses": {"mse": 1., "total": 2.}, "preclip_norm": 3.}
    check_reproduction(original, original)
    for key, value in (("cohort", "probe"), ("id", 64129), ("preclip_norm", 4.)):
        changed = copy.deepcopy(original)
        changed[key] = value
        with pytest.raises(ValueError):
            check_reproduction(changed, original)
    changed = copy.deepcopy(original)
    changed["losses"]["mse"] = 1.01
    with pytest.raises(ValueError):
        check_reproduction(changed, original)
