"""Apply query-budget majorization in training while leaving all evaluation callbacks alone."""

from contextlib import contextmanager
from functools import partial

import torch

from dwip_teacher_mean_training import compute_context
import single_model_column_objective as column
from single_model_query_budget import query_budget_majorizer
from streamed_large_matrix_training import TRAIN_OPTIONS
from tabimpute.streamed_training import streamed_training_loss


def backward_episode(student, criterion, episode, teacher, weight, *, query_seed_offset,
                     probability_scales=column.balanced.PROBABILITY_SCALES):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Query-budget supervision requires synthetic data")
    if episode.metadata["domain"] == "recsys":
        return column.BASE_BACKWARD(student, criterion, episode, teacher, weight,
                                    probability_scales=probability_scales)
    if (episode.metadata["domain"] != "imputation" or type(episode.metadata.get("seed")) is not int
            or type(query_seed_offset) is not int or query_seed_offset < 0):
        raise ValueError("Require a seeded imputation episode and a nonnegative query seed offset")
    coefficients, offset, diagnostics = query_budget_majorizer(
        episode.target, episode.mask, teacher, seed=episode.metadata["seed"] + query_seed_offset)
    episode.metadata["column_objective"] = diagnostics["full"]
    episode.metadata["query_budget_objective"] = diagnostics
    teacher_values, cell_weights = teacher.detach().reshape(-1), coefficients.reshape(-1)
    indices = torch.arange(episode.target.numel(), device=episode.target.device).reshape_as(episode.target)

    def loss_function(logits, truth, lookup):
        point = criterion.mean(logits.float())
        original = column.balanced.balanced_loss(criterion, logits, truth, teacher_values[lookup],
                                                domain="imputation", probability_scales=probability_scales)
        return original + column.COLUMN_OBJECTIVE["weight"] * cell_weights[lookup] * (point - truth.float()).square()

    with compute_context(episode.source.device):
        loss = streamed_training_loss(student, episode.source, episode.target, episode.mask, loss_function,
                                      auxiliary_targets=indices, **TRAIN_OPTIONS)
    loss = loss + column.COLUMN_OBJECTIVE["weight"] * offset
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite query-budget training loss")
    (weight * loss).backward()
    return float(loss.detach())


@contextmanager
def training_objective(query_seed_offset):
    if type(query_seed_offset) is not int or query_seed_offset < 0:
        raise ValueError("Require an explicit nonnegative query seed offset")
    original_training = column.continuation.backward_episode
    original_preflight = column.balanced.backward_episode
    callback = partial(backward_episode, query_seed_offset=query_seed_offset)
    column.continuation.backward_episode = column.balanced.backward_episode = callback
    try:
        yield
    finally:
        column.continuation.backward_episode = original_training
        column.balanced.backward_episode = original_preflight
