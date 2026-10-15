from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dwip_teacher_mean_training import FrozenMeanTeachers, control_recipe, equal_teacher_mean, objective_terms
from rowcol_fixed_validation import preserved_rng
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from train_direct_pure_rabel_curriculum import cdf_alignment_loss


def distribution():
    return FullSupportBarDistribution(torch.linspace(-4., 4., 9))


def test_control_retains_cdf_and_original_recipe():
    args = control_recipe()
    assert args.cdf_loss_weight == .05
    assert args.point_loss_weight == 2.
    assert args.warmup_steps == 600 and args.decay_steps == 63400
    assert args.seed == 1201 and args.data_seed == 1150924


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_baseline_loss_and_logit_gradients_match_existing_expression(dtype):
    with preserved_rng(19):
        logits = torch.randn(17, 8, dtype=dtype, requires_grad=True)
        target = torch.randn(17)
    criterion = distribution()
    terms = objective_terms(SimpleNamespace(), criterion, logits, target)
    nll = criterion(logits.float(), target).mean()
    point = torch.nn.functional.huber_loss(criterion.mean(logits.float()), target, delta=1.)
    cdf = cdf_alignment_loss(logits, target, criterion, chunk_size=2048)
    zero = logits.new_zeros((), dtype=torch.float32)
    expected = 1. * nll + 2. * point + 0. * zero + .05 * cdf + 0. * zero + 0. * zero + 0. * zero + 0. * zero + 0. * zero
    assert torch.equal(terms["loss"], expected)
    actual_gradient = torch.autograd.grad(terms["loss"], logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    assert torch.equal(actual_gradient, expected_gradient)


def test_teacher_target_changes_only_point_loss_and_is_detached():
    criterion = distribution()
    logits = torch.zeros(3, 8, requires_grad=True)
    truth = torch.tensor([0., 1., 2.])
    teacher = torch.tensor([2., 3., 4.], requires_grad=True)
    before = objective_terms(SimpleNamespace(), criterion, logits, truth)
    after = objective_terms(SimpleNamespace(), criterion, logits, truth, teacher)
    assert torch.equal(before["nll"], after["nll"])
    assert torch.equal(before["cdf"], after["cdf"])
    assert torch.equal(before["true_rmse"], after["true_rmse"])
    assert not torch.equal(before["point"], after["point"])
    after["loss"].backward()
    assert teacher.grad is None
    assert torch.isfinite(logits.grad).all()


def test_equal_mean_is_fp64_then_fp32_and_has_no_teacher_gradients():
    first = torch.tensor([3e38, 1.], requires_grad=True)
    second = torch.tensor([3e38, -1.], requires_grad=True)
    result = equal_teacher_mean([first, second])
    assert torch.equal(result, ((first.double() + second.double())/2).float())
    assert torch.isfinite(result).all() and not result.requires_grad


@pytest.mark.parametrize("bad", ([], [torch.ones(2)], [torch.ones(2), torch.ones(3)],
                                  [torch.ones(2), torch.full((2,), float("nan"))],
                                  [torch.ones(2), torch.ones(2, dtype=torch.float64)]))
def test_invalid_teacher_means_are_rejected(bad):
    with pytest.raises(ValueError):
        equal_teacher_mean(bad)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(8).float())
        self.calls = 0

    def forward(self, source, decode_mask):
        self.calls += 1
        torch.rand(1)
        return self.logits.expand(int(decode_mask.sum()), -1)


def test_external_teachers_preserve_rng_and_student_state():
    first, second, student = TinyModel(), TinyModel(), TinyModel()
    original = deepcopy(student.state_dict())
    teachers = FrozenMeanTeachers([first, second], distribution())
    source = torch.tensor([[[1., float("nan")], [float("nan"), 2.]]])
    mask = torch.isnan(source)
    rng = torch.get_rng_state()
    target = teachers.target(source, mask)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not target.requires_grad and target.shape == (2,)
    assert all(not model.training and all(not parameter.requires_grad for parameter in model.parameters()) for model in teachers.models)
    assert student.training and student.logits.requires_grad
    assert student.state_dict().keys() == original.keys()
    assert all(torch.equal(student.state_dict()[key], value) for key, value in original.items())
    assert student.calls == 0 and first.calls == second.calls == 1


def test_observed_target_or_mutated_teacher_mode_is_rejected():
    teachers = FrozenMeanTeachers([TinyModel(), TinyModel()], distribution())
    source = torch.ones(1, 2, 2)
    with pytest.raises(ValueError, match="unobserved"):
        teachers.target(source, torch.ones_like(source, dtype=torch.bool))
    teachers.models[0].train()
    source.fill_(float("nan"))
    with pytest.raises(ValueError, match="eval-only"):
        teachers.target(source, torch.ones_like(source, dtype=torch.bool))


def test_distinct_external_teachers_are_required():
    model = TinyModel()
    with pytest.raises(ValueError, match="distinct"):
        FrozenMeanTeachers([model, model], distribution())


@pytest.mark.parametrize("target", (torch.ones(2), torch.full((3,), float("inf")), torch.ones(3, dtype=torch.float64)))
def test_bad_point_targets_fail_closed(target):
    with pytest.raises(ValueError, match="point target"):
        objective_terms(SimpleNamespace(), distribution(), torch.zeros(3, 8), torch.zeros(3), target)
