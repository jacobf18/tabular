import copy

import pytest
import torch

from diagnose_dwip_training_objective import adam_descent, clipping_coefficient, combined_report, direction_report, prepare_training_batch, validate_training_batch


def test_batch_preparation_transfers_before_support_mask(monkeypatch):
    import diagnose_dwip_training_objective as diagnostic
    calls = []

    def support(source, target, mode, seed, variance_df):
        calls.append((source.dtype, target.dtype, source.device.type, mode, seed, variance_df))
        mask = torch.isnan(source)
        return source, target, mask, {"support_episode": 0.}

    monkeypatch.setattr(diagnostic, "apply_shrunk_support_mask", support)
    source = torch.tensor([[[float("nan"), 1.]]], dtype=torch.float64)
    prepare_training_batch(source, torch.ones_like(source), {"support_mode": "broad_support", "support_variance_df": 1.}, 42, device="cpu")
    assert calls == [(torch.float32, torch.float32, "cpu", "broad_support", 42, 1.)]


def test_training_decode_mask_may_be_subset_of_missing_cells():
    source = torch.tensor([[[float("nan"), 1., float("nan")]]])
    target = torch.ones_like(source)
    mask = torch.tensor([[[True, False, False]]])
    validate_training_batch(source, target, mask)
    source[0, 0, 1] = float("inf")
    with pytest.raises(FloatingPointError):
        validate_training_batch(source, target, mask)


def test_training_batch_rejects_observed_decode_targets_and_nonfinite_targets():
    source = torch.ones(1, 2, 2)
    target = torch.ones_like(source)
    mask = torch.ones_like(source, dtype=torch.bool)
    with pytest.raises(FloatingPointError):
        validate_training_batch(source, target, mask)
    source.fill_(float("nan"))
    target[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        validate_training_batch(source, target, mask)


def test_clipping_matches_pytorch_and_preserves_input_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    gradients = {"nll": (torch.tensor([3., 0.]),), "point": (torch.tensor([0., 4.]),),
                 "cdf": (torch.zeros(2),), "total": (torch.tensor([3., 4.]),)}
    report = combined_report([parameter], gradients, .5)
    assert report["preclip_norm"] == pytest.approx(5.)
    assert report["postclip_norm"] == pytest.approx(.5, abs=1e-6)
    assert report["coefficient"] == pytest.approx(.5 / (5 + 1e-6))
    assert report["separate_vs_combined_relative_error"] == 0
    assert torch.equal(gradients["total"][0], torch.tensor([3., 4.]))
    assert parameter.grad is None


@pytest.mark.parametrize("norm,maximum", [(-1, .5), (float("nan"), .5), (1, 0), (1, float("inf"))])
def test_clipping_rejects_invalid_inputs(norm, maximum):
    with pytest.raises(ValueError):
        clipping_coefficient(norm, maximum)


def test_analytical_adam_matches_reference_without_mutating_inputs():
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=.01, betas=(.9, .999), weight_decay=.02, eps=1e-8, fused=False)
    parameter.grad = torch.tensor([.4, -.6], dtype=torch.float64)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    before = parameter.detach().clone()
    gradient = torch.tensor([-.2, .7], dtype=torch.float64)
    expected = adam_descent(before, gradient, state["state"][0], state["param_groups"][0])
    assert torch.equal(before, parameter.detach())
    assert torch.equal(state["state"][0]["exp_avg"], optimizer.state[parameter]["exp_avg"])
    parameter.grad = gradient.clone()
    optimizer.step()
    assert torch.allclose(before - parameter, expected, atol=1e-14, rtol=1e-10)


def test_direction_projection_and_unused_parameters():
    parameters = [torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([2.]))]
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0).state_dict()
    gradients = {"total": (torch.tensor([2.]), None), "nll": (torch.tensor([-1.]), None)}
    result = direction_report(parameters, gradients, optimizer, 1.)
    assert result["objectives"]["total"]["descent_cosine"] == pytest.approx(1.)
    assert result["objectives"]["nll"]["descent_cosine"] == pytest.approx(-1.)
    assert result["objectives"]["nll"]["linearized_loss_delta"] > 0
    assert not optimizer["state"]
    optimizer["param_groups"][0]["params"].reverse()
    with pytest.raises(ValueError):
        direction_report(parameters, gradients, optimizer, 1.)


def test_adam_rejects_nonfinite_inputs_and_unsupported_modes():
    parameter = torch.ones(1)
    group = torch.optim.AdamW([torch.nn.Parameter(parameter)]).state_dict()["param_groups"][0]
    with pytest.raises(FloatingPointError):
        adam_descent(parameter, torch.tensor([float("nan")]), {}, group)
    group["amsgrad"] = True
    with pytest.raises(ValueError):
        adam_descent(parameter, parameter, {}, group)
