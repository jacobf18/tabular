import copy

import pytest
import torch

from diagnose_dwip_huber_training_alignment import matched_commands, point_losses, statistics, validate_cases
from run_dwip_anchor_mix_huber_objective import build_commands


def test_matched_huber_commands_allow_only_output_paths_and_registered_function(tmp_path):
    from run_dwip_anchor_mix_point_objective import build_commands as control_commands
    control = control_commands(tmp_path / "control")[1]["dwip1e5"]
    candidate = build_commands(tmp_path / "candidate")[1]["dwip1e5"]
    before = copy.deepcopy((control, candidate))
    matched_commands(control, candidate)
    assert (control, candidate) == before
    candidate[candidate.index("--data-seed") + 1] = "17"
    with pytest.raises(ValueError):
        matched_commands(control, candidate)


def test_exact_point_objectives_preserve_fp32_means_and_differ_from_rounded_proxy():
    means = torch.tensor([1.003, 3.25, -.125], requires_grad=True)
    target = torch.zeros_like(means)
    losses = point_losses(means, target, torch.bfloat16)
    assert losses["mse"].dtype == torch.float32
    assert losses["mse"] != losses["rounded_mse_proxy"]
    assert torch.allclose(torch.autograd.grad(losses["mse"], means, retain_graph=True)[0], 2 * means / 3)
    assert torch.allclose(torch.autograd.grad(losses["huber"], means)[0], means.clamp(-1, 1) / 3)


@pytest.mark.parametrize("means,target", [(torch.empty(0), torch.empty(0)), (torch.ones(2), torch.ones(1)),
    (torch.tensor([float("nan")]), torch.zeros(1)), (torch.zeros(1), torch.tensor([float("inf")]))])
def test_point_objectives_reject_invalid_inputs(means, target):
    with pytest.raises((ValueError, FloatingPointError)):
        point_losses(means, target, torch.bfloat16)


def frozen_cases():
    return [{"id": index, "family": f"family{index // 4}", "cohort": "frozen32",
        "source": torch.full((2, 1, 1), float("nan")), "target": torch.ones(2, 1, 1),
        "mask": torch.ones(2, 1, 1, dtype=torch.bool)} for index in range(32)]


def test_case_validation_requires_exact_frozen_cohorts_order_and_masks():
    cases = frozen_cases()
    cases.extend({**copy.deepcopy(cases[0]), "id": step, "cohort": "targeted_training_tail"} for step in (2250, 61275))
    validate_cases(cases)
    with pytest.raises(ValueError):
        validate_cases(list(reversed(cases)))
    with pytest.raises(ValueError):
        validate_cases(cases + [cases[0]])
    cases[0]["source"][0, 0, 0] = 1.
    with pytest.raises(FloatingPointError):
        validate_cases(cases)


def test_statistics_rejects_nonfinite_or_empty_values():
    assert statistics([-1., 1.])["negative_count"] == 1
    for values in ([], [float("nan")], [float("inf")]):
        with pytest.raises(FloatingPointError):
            statistics(values)
