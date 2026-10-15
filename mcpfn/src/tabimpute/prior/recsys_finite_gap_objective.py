"""Synthetic top-k ranking with no reward for unnecessary score separation."""

import math

import torch
import torch.nn.functional as functional

from tabimpute.prior.recsys_ranking_objective import topk_union_candidate_weights


def finite_gap_pair_loss(signed_gap, *, margin, temperature):
    """Match logistic slope at tied scores, stop rewarding gaps above margin."""
    if not math.isfinite(margin) or margin <= 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Require positive finite margin and temperature')
    violation = (margin - signed_gap).clamp_min(0)
    return functional.smooth_l1_loss(violation, torch.zeros_like(violation), reduction='none', beta=margin) / (2 * temperature)


def finite_gap_topk_loss(prediction, probabilities, users, items, selected, shape, *,
                         provenance, margin=.01, temperature=.1, top_k=10):
    if provenance != 'synthetic':
        raise ValueError('Only synthetic ranking targets are permitted')
    if not math.isfinite(margin) or margin <= 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Require positive finite margin and temperature')
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k * margin > 4:
        raise ValueError('Require feasible top-k score margins in the bounded rating range')
    if (prediction.ndim != 1 or probabilities.shape != prediction.shape or users.shape != prediction.shape
            or items.shape != prediction.shape or selected.shape != prediction.shape or selected.dtype != torch.bool):
        raise ValueError('Require aligned scores, probabilities, indices and Boolean rating selection')
    if probabilities.requires_grad or not torch.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Require detached synthetic probabilities in [0,1]')
    if not torch.isfinite(prediction).all():
        raise ValueError('Require finite synthetic scores')
    if (len(shape) != 2 or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape)
            or users.dtype != torch.long or items.dtype != torch.long):
        raise ValueError('Require positive user/item shape and int64 indices')
    if any(value.device != prediction.device for value in (probabilities, users, items, selected)):
        raise ValueError('Require aligned devices')
    if ((users < 0) | (users >= shape[0]) | (items < 0) | (items >= shape[1])).any():
        raise ValueError('Query indices outside synthetic shape')
    if (users * shape[1] + items).unique().numel() != prediction.numel():
        raise ValueError('Require unique synthetic query coordinates')
    additional_counts = torch.bincount(users[~selected], minlength=shape[0])
    active = additional_counts > 0
    diagnostics = {'ranking_positive_users': int((torch.bincount(users[probabilities > 0], minlength=shape[0]) > 0).sum()),
                   'ranking_paired_users': 0, 'ranking_negative_queries': int((~selected).sum()),
                   'ranking_top_k': top_k, 'ranking_candidate_scope': 'topk_union',
                   'ranking_effective_candidates_mean': 0., 'ranking_finite_gap_margin': margin,
                   'ranking_unsatisfied_weight_fraction': 0.}
    if not active.any():
        return prediction.sum() * 0, diagnostics
    dense_scores = prediction.new_full(shape, -torch.inf)
    dense_scores[users, items] = prediction
    dense_probabilities = prediction.new_zeros(shape)
    dense_probabilities[users, items] = probabilities
    scores = dense_scores[active]
    ordering = scores.argsort(dim=1, descending=True, stable=True)
    scores = scores.gather(1, ordering)
    valid = torch.isfinite(scores)
    scores = torch.where(valid, scores, 0)
    relevance = dense_probabilities[active].gather(1, ordering)
    positions = torch.arange(shape[1], device=prediction.device)
    discounts = torch.where(positions < top_k, 1 / (positions.to(prediction.dtype) + 2).log2(), 0)
    anchors = min(top_k, shape[1])
    pairs = ((positions[None, :] > positions[:anchors, None])[None]
             & valid[:, :anchors, None] & valid[:, None, :])
    differences = relevance[:, :anchors, None] - relevance[:, None, :]
    weights = torch.where(pairs, differences.abs() * (discounts[:anchors, None] - discounts[None, :]), 0)
    candidate_weights = topk_union_candidate_weights(relevance, valid, top_k)
    diagnostics['ranking_effective_candidates_mean'] = candidate_weights.sum(1).mean().item()
    weights = weights * candidate_weights[:, None, :]
    score_differences = scores[:, :anchors, None] - scores[:, None, :]
    signed_gaps = differences.sign() * score_differences
    losses = finite_gap_pair_loss(signed_gaps, margin=margin, temperature=temperature)
    mass = weights.sum((1, 2))
    included = mass > 0
    diagnostics['ranking_paired_users'] = int(included.sum())
    if not included.any():
        return prediction.sum() * 0, diagnostics
    unsatisfied = (weights * (signed_gaps < margin)).sum((1, 2))
    diagnostics['ranking_unsatisfied_weight_fraction'] = (unsatisfied[included] / mass[included]).mean().item()
    return ((weights * losses).sum((1, 2))[included] / mass[included]).mean(), diagnostics
