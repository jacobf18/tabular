"""Prospective training-only strengthening of the original teacher-mean Huber term."""

from contextlib import contextmanager

from torch.nn import functional

import train_single_matrixformer_probability_balance as original


ORIGINAL_MEAN_WEIGHT = 2.
MEAN_WEIGHT = 20.
BASE_LOSS = original.balanced_loss
SPECIFICATION = {
    'original_teacher_mean_huber_weight': ORIGINAL_MEAN_WEIGHT,
    'teacher_mean_huber_weight': MEAN_WEIGHT,
    'huber_delta': 1.,
    'domains': ['imputation', 'recsys'],
    'teacher_detached': True,
    'probability_losses_unchanged': True,
    'retention_and_point_objectives_unchanged': True,
    'architecture_and_inference_unchanged': True,
}
POLICY = (
    'Prospective fixed-strength mean-distillation hypothesis after V45 full negative closure and '
    'complete saved-output synthetic alignment diagnosis. Change only the original teacher-mean '
    'Huber coefficient from2 to20, uniformly in imputation and rating composite callbacks. This '
    'makes the quadratic-region coefficient10, nominally matching the existing weight10 squared '
    'V8-retention coefficient before their different cell/column/task weighting; it does not '
    'equalize actual gradients or guarantee teacher fidelity. Keep the original delta1, detached '
    'observed-blend imputation and specialist rating targets, NLL/CDF probability scales, query '
    'majorizer, half-composite/half-point proposal, uniform V8 retention, all priors, task masses '
    'and optimizer protections. Stronger imitation can worsen synthetic truth accuracy because '
    'the imputation blend is not uniformly better than the original general teacher. Keep all '
    'existing development, fresh-confirmation and real-benchmark requirements; no weight sweep, '
    'target selection by truth, new inference operation, ensemble or gate relaxation. Output-space '
    'distances are descriptive and do not establish parameter-gradient conflict or causality. '
    'This standalone implementation authorizes no scientific probe, seed, training or promotion.'
)


def balanced_loss(criterion, logits, truth, teacher, *, domain, probability_scales):
    base = BASE_LOSS(criterion, logits, truth, teacher, domain=domain, probability_scales=probability_scales)
    mean = criterion.mean(logits.float())
    point = functional.huber_loss(mean, teacher.detach().float(), delta=1., reduction='none')
    return base + (MEAN_WEIGHT - ORIGINAL_MEAN_WEIGHT) * point


@contextmanager
def training_objective():
    if original.balanced_loss is not BASE_LOSS:
        raise ValueError('Do not overlap or replace an existing training-loss override')
    original.balanced_loss = balanced_loss
    try:
        yield
    finally:
        original.balanced_loss = BASE_LOSS
