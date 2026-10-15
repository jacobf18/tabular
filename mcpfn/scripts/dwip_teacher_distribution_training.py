"""External frozen categorical teachers and a single auxiliary KL substitution."""
import torch
import torch.nn.functional as functional

from dwip_teacher_mean_training import (
    FrozenMeanTeachers, base, compute_context, equal_teacher_mean,
)
from rowcol_fixed_validation import preserved_rng


def validate_logits(logits):
    if logits.ndim != 2 or not logits.numel() or logits.shape[1] < 2 or not logits.is_floating_point():
        raise ValueError("Require nonempty floating missing-cell categorical logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Require finite categorical logits")


def validate_probabilities(probabilities):
    if probabilities.ndim != 2 or not probabilities.numel() or probabilities.dtype != torch.float32:
        raise ValueError("Require nonempty FP32 categorical probabilities")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Require finite nonnegative probabilities bounded by one")
    sums = probabilities.sum(-1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=0., atol=2e-6):
        raise ValueError("Categorical probabilities must sum to one")


def shared_bar_means(model, criterion, logits):
    validate_logits(logits)
    if getattr(model, "last_prediction_location", None) is not None:
        raise ValueError("Translated distributions are not supported by the frozen shared-bar KL")
    means = getattr(criterion, "_full_bucket_means", None)
    if means is None or means.ndim != 1 or means.numel() != logits.shape[1] or not torch.isfinite(means).all():
        raise ValueError("Require matching finite full-support bar means")
    return means.to(device=logits.device, dtype=torch.float32)


def equal_probability_mixture(logits):
    if len(logits) != 2:
        raise ValueError("Require exactly two frozen teacher logits")
    for value in logits:
        validate_logits(value)
        if value.shape != logits[0].shape or value.device != logits[0].device:
            raise ValueError("Teacher logits must share shape and device")
    probabilities = [value.detach().float().softmax(-1) for value in logits]
    mixture = ((probabilities[0].double() + probabilities[1].double()) / 2).float()
    validate_probabilities(mixture)
    return mixture


def categorical_kl(logits, probabilities, chunk_size=2048):
    validate_logits(logits)
    validate_probabilities(probabilities)
    if probabilities.shape != logits.shape or probabilities.device != logits.device:
        raise ValueError("Teacher probabilities must match student logits")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("Require a positive integer KL chunk size")
    target = probabilities.detach()
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, logits.shape[0], chunk_size):
        stop = start + chunk_size
        total = total + functional.kl_div(
            logits[start:stop].float().log_softmax(-1), target[start:stop], reduction="sum")
    loss = total / logits.shape[0]
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite categorical teacher KL")
    return loss


class FrozenDistributionTeachers(FrozenMeanTeachers):
    def distribution_target(self, source, decode_mask):
        if source.ndim != 3 or decode_mask.shape != source.shape or decode_mask.dtype != torch.bool:
            raise ValueError("Require a matching Boolean missing-cell decode mask")
        if not decode_mask.any() or not torch.isnan(source[decode_mask]).all():
            raise ValueError("Teachers may only target unobserved cells")
        if any(model.training or any(parameter.requires_grad for parameter in model.parameters()) for model in self.models):
            raise ValueError("Teachers must remain eval-only and frozen")
        teacher_logits, teacher_means = [], []
        with preserved_rng(), torch.no_grad():
            for model in self.models:
                with compute_context(source.device):
                    logits = model(source, decode_mask=decode_mask)
                means = shared_bar_means(model, self.criterion, logits)
                if logits.shape[0] != int(decode_mask.sum()):
                    raise ValueError("Teacher rows do not match scored missing cells")
                teacher_logits.append(logits)
                teacher_means.append(self.criterion.mean(logits.float()))
            probabilities = equal_probability_mixture(teacher_logits)
            reference_mean = equal_teacher_mean(teacher_means)
            mixture_mean = probabilities @ means
            if not torch.allclose(mixture_mean, reference_mean, rtol=2e-5, atol=2e-5):
                raise ValueError("Probability mixture does not preserve the equal teacher predictive mean")
        return {"probabilities": probabilities, "mean": reference_mean,
                "mean_identity_max_absolute_error": float((mixture_mean - reference_mean).abs().max()),
                "probability_sum_max_absolute_error": float((probabilities.sum(-1) - 1).abs().max())}


def objective_terms(model, criterion, logits, true_target, teacher_probabilities):
    means = shared_bar_means(model, criterion, logits)
    if true_target.ndim != 1 or true_target.shape[0] != logits.shape[0] or not torch.isfinite(true_target).all():
        raise ValueError("Require aligned finite true missing-cell targets")
    nll = criterion(logits.float(), true_target.float()).mean()
    kl = categorical_kl(logits, teacher_probabilities)
    cdf = base.cdf_alignment_loss(logits, true_target, criterion, chunk_size=2048)
    point_mean = criterion.mean(logits.float())
    teacher_mean = teacher_probabilities.detach() @ means
    loss = nll + 2. * kl + .05 * cdf
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite distribution-distillation objective")
    return {"loss": loss, "nll": nll, "kl": kl, "cdf": cdf,
            "true_rmse": (point_mean.detach() - true_target.float()).square().mean().sqrt(),
            "teacher_rmse": (point_mean.detach() - teacher_mean).square().mean().sqrt()}
