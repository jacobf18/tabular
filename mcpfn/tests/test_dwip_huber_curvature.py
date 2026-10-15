import pytest
import torch

from run_dwip_anchor_mix_huber_curvature import build_commands, verify_curvature_only_change
from run_dwip_anchor_mix_huber_objective import build_commands as control_commands
from run_rowcol_output_remedies import replace_arg


def test_only_declared_huber_coefficient_changes(tmp_path):
    control = control_commands(tmp_path)
    candidate = build_commands(tmp_path)
    assert control[0] == candidate[0]
    assert control[2:] == candidate[2:]
    verify_curvature_only_change(control[1]["dwip1e5"], candidate[1]["dwip1e5"])


@pytest.mark.parametrize("flag,value", [("--point-loss-weight", "3"), ("--huber-delta", "2"),
    ("--point-loss-kind", "mse"), ("--data-seed", "8"), ("--decay-steps", "95400"),
    ("--support-variance-df", "2"), ("--grad-clip-norm", "1"), ("--nll-loss-weight", "2")])
def test_curvature_control_rejects_any_other_factor(tmp_path, flag, value):
    control = control_commands(tmp_path)[1]["dwip1e5"]
    candidate = build_commands(tmp_path)[1]["dwip1e5"]
    replace_arg(candidate, flag, value)
    with pytest.raises(ValueError):
        verify_curvature_only_change(control, candidate)


def test_curvature_matched_huber_equals_mse2_locally_and_bounds_tails():
    errors = torch.tensor([-.9, -.25, .25, .9], requires_grad=True)
    robust = 4 * torch.nn.functional.huber_loss(errors, torch.zeros_like(errors), delta=1., reduction="sum")
    mse = 2 * errors.square().sum()
    assert torch.equal(robust, mse)
    assert torch.equal(torch.autograd.grad(robust, errors)[0], torch.autograd.grad(mse, errors)[0])
    tails = torch.tensor([-10., 10.], requires_grad=True)
    tail_loss = 4 * torch.nn.functional.huber_loss(tails, torch.zeros_like(tails), delta=1., reduction="sum")
    assert torch.equal(torch.autograd.grad(tail_loss, tails)[0], torch.tensor([-4., 4.]))
