from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional

import single_model_stronger_mean_distillation as stronger


class Criterion:
    bucket_means = torch.tensor([-1., 0., 1.])

    def mean(self, logits):
        return (logits.softmax(-1) * self.bucket_means.to(logits)).sum(-1)

    def __call__(self, logits, truth):
        return (self.mean(logits) - truth).square() + logits.square().mean(-1)


def fixture():
    logits = torch.tensor([[.2, -.3, .8], [.1, .7, -.2]], requires_grad=True)
    truth = torch.tensor([.1, -.4])
    teacher = torch.tensor([.6, -2.], requires_grad=True)
    return Criterion(), logits, truth, teacher


@pytest.mark.parametrize('domain', ['imputation', 'recsys'])
@pytest.mark.parametrize('imputation_scale', [.1, 1.])
def test_only_uniform_huber_coefficient_changes(domain, imputation_scale):
    criterion, logits, truth, teacher = fixture()
    scales = {'imputation': imputation_scale, 'recsys': 1.}
    before = stronger.BASE_LOSS(criterion, logits, truth, teacher, domain=domain, probability_scales=scales)
    after = stronger.balanced_loss(criterion, logits, truth, teacher, domain=domain, probability_scales=scales)
    huber = functional.huber_loss(criterion.mean(logits), teacher.detach(), delta=1., reduction='none')
    torch.testing.assert_close(after, before + 18. * huber, rtol=0., atol=0.)
    old_gradient = torch.autograd.grad(before.sum(), logits, retain_graph=True)[0]
    extra_gradient = torch.autograd.grad((18. * huber).sum(), logits, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(after.sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(new_gradient, old_gradient + extra_gradient)
    assert torch.autograd.grad(after.sum(), teacher, allow_unused=True)[0] is None


@pytest.mark.parametrize('domain', ['imputation', 'recsys'])
def test_true_target_probability_terms_are_unchanged(domain):
    criterion, logits, truth, teacher = fixture()
    scales = dict(stronger.original.PROBABILITY_SCALES)
    arguments = {'domain': domain, 'probability_scales': scales}
    other_truth = truth + .3
    old_change = stronger.BASE_LOSS(criterion, logits, other_truth, teacher, **arguments) - stronger.BASE_LOSS(criterion, logits, truth, teacher, **arguments)
    new_change = stronger.balanced_loss(criterion, logits, other_truth, teacher, **arguments) - stronger.balanced_loss(criterion, logits, truth, teacher, **arguments)
    torch.testing.assert_close(new_change, old_change, rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize('raise_inside', [False, True])
def test_training_scope_restores_original_callback(raise_inside):
    initial = stronger.original.balanced_loss
    with pytest.raises(RuntimeError) if raise_inside else nullcontext():
        with stronger.training_objective():
            assert stronger.original.balanced_loss is stronger.balanced_loss
            if raise_inside:
                raise RuntimeError('original training error')
    assert stronger.original.balanced_loss is initial is stronger.BASE_LOSS


def test_nested_or_foreign_objective_is_not_overwritten(monkeypatch):
    with stronger.training_objective():
        with pytest.raises(ValueError, match='overlap'):
            with stronger.training_objective():
                pytest.fail('Nested override must be rejected')
        assert stronger.original.balanced_loss is stronger.balanced_loss
    foreign = object()
    monkeypatch.setattr(stronger.original, 'balanced_loss', foreign)
    with pytest.raises(ValueError, match='overlap'):
        with stronger.training_objective():
            pytest.fail('Foreign callback must be retained')
    assert stronger.original.balanced_loss is foreign


def test_original_invalid_domain_and_probability_validation_remain():
    criterion, logits, truth, teacher = fixture()
    with pytest.raises(ValueError):
        stronger.balanced_loss(criterion, logits, truth, teacher, domain='unknown',
                              probability_scales=dict(stronger.original.PROBABILITY_SCALES))
    with pytest.raises(ValueError):
        stronger.balanced_loss(criterion, logits, truth, teacher, domain='imputation',
                              probability_scales={'imputation': 0., 'recsys': 1.})
