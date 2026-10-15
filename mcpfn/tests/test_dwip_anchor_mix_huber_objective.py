import pytest
import torch

from run_dwip_anchor_mix_huber_objective import build_commands, verify_huber_only_change
from run_dwip_anchor_mix_point_objective import build_commands as control_commands
from run_rowcol_output_remedies import replace_arg


def test_huber_changes_only_the_declared_point_function(tmp_path):
    original = control_commands(tmp_path)
    changed = build_commands(tmp_path)
    assert original[0] == changed[0]
    assert original[2:] == changed[2:]
    verify_huber_only_change(original[1]["dwip1e5"], changed[1]["dwip1e5"])


@pytest.mark.parametrize("flag,value", [("--point-loss-weight", "4"), ("--huber-delta", ".5"),
    ("--data-seed", "1"), ("--decay-steps", "999"), ("--support-variance-df", "4"), ("--grad-clip-norm", "1")])
def test_huber_rejects_additional_or_unregistered_changes(tmp_path, flag, value):
    original = control_commands(tmp_path)[1]["dwip1e5"]
    changed = build_commands(tmp_path)[1]["dwip1e5"]
    replace_arg(changed, flag, value)
    with pytest.raises(ValueError):
        verify_huber_only_change(original, changed)


def test_huber_delta1_halves_local_mse_curvature_and_bounds_tail_derivative():
    errors = torch.tensor([.5, 2.], requires_grad=True)
    loss = torch.nn.functional.huber_loss(errors, torch.zeros_like(errors), delta=1., reduction="sum")
    gradient, = torch.autograd.grad(loss, errors)
    assert torch.equal(gradient, torch.tensor([.5, 1.]))
    assert torch.allclose(loss.detach(), torch.tensor(1.625))
