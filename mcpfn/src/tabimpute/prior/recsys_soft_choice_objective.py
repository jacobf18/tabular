"""Finite-gap synthetic frequency distillation on the model/teacher top-k union."""
import math

import torch
import torch.nn.functional as functional

from tabimpute.prior.recsys_ranking_objective import topk_union_candidate_weights


def topk_soft_choice_loss(prediction, frequencies, users, items, selected, shape, *,
                          frequency_smoothing, temperature=1.0, top_k=10):
    if not math.isfinite(temperature) or temperature <= 0 or top_k < 1:
        raise ValueError("Require positive finite temperature and positive top-k")
    if not math.isfinite(frequency_smoothing) or frequency_smoothing <= 0:
        raise ValueError("Require positive finite synthetic frequency smoothing")
    if prediction.ndim != 1 or any(value.shape != prediction.shape for value in (frequencies, users, items, selected)):
        raise ValueError("Require aligned one-dimensional synthetic queries")
    if frequencies.requires_grad or not torch.isfinite(frequencies).all() or torch.any((frequencies < 0) | (frequencies > 1)):
        raise ValueError("Require detached synthetic frequencies in [0, 1]")
    if not torch.isfinite(prediction).all():
        raise ValueError("Require finite query predictions")
    additional_counts = torch.bincount(users[~selected], minlength=shape[0])
    active = additional_counts > 0
    diagnostics = {
        "ranking_positive_users": int((torch.bincount(users[frequencies > 0], minlength=shape[0]) > 0).sum()),
        "ranking_paired_users": 0,
        "ranking_negative_queries": int((~selected).sum()),
        "ranking_top_k": top_k,
        "ranking_candidate_scope": "topk_union",
        "ranking_effective_candidates_mean": 0.0,
        "ranking_pair_target": "smoothed_frequency_choice",
        "ranking_frequency_smoothing": frequency_smoothing,
    }
    if not active.any():
        return prediction.sum() * 0, diagnostics
    dense_scores = prediction.new_full(shape, -torch.inf)
    dense_scores[users, items] = prediction
    dense_frequencies = prediction.new_zeros(shape)
    dense_frequencies[users, items] = frequencies
    ordering = dense_scores[active].argsort(dim=1, descending=True, stable=True)
    scores = dense_scores[active].gather(1, ordering)
    valid = torch.isfinite(scores)
    scores = torch.where(valid, scores, 0)
    rates = dense_frequencies[active].gather(1, ordering)
    positions = torch.arange(shape[1], device=prediction.device)
    discounts = torch.where(positions < top_k, 1 / (positions.to(prediction.dtype) + 2).log2(), 0)
    anchors = min(top_k, shape[1])
    pairs = ((positions[None, :] > positions[:anchors, None])[None]
             & valid[:, :anchors, None] & valid[:, None, :])
    differences = rates[:, :anchors, None] - rates[:, None, :]
    weights = torch.where(pairs, differences.abs() * (discounts[:anchors, None] - discounts[None, :]), 0)
    candidates = topk_union_candidate_weights(rates, valid, top_k)
    weights = weights * candidates[:, None, :]
    diagnostics["ranking_effective_candidates_mean"] = candidates.sum(1).mean().item()
    choice = ((rates[:, :anchors, None] + frequency_smoothing)
              / (rates[:, :anchors, None] + rates[:, None, :] + 2 * frequency_smoothing))
    gaps = (scores[:, :anchors, None] - scores[:, None, :]) / temperature
    pair_losses = choice * functional.softplus(-gaps) + (1 - choice) * functional.softplus(gaps)
    mass = weights.sum((1, 2))
    included = mass > 0
    diagnostics["ranking_paired_users"] = int(included.sum())
    if not included.any():
        return prediction.sum() * 0, diagnostics
    return ((weights * pair_losses).sum((1, 2))[included] / mass[included]).mean(), diagnostics
