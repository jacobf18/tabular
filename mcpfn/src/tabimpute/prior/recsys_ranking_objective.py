"""User-macro ranking optimization on strictly synthetic rating episodes."""
import math

import torch

from tabimpute.prior.recsys_rating_episode import rating_objective
from tabimpute.prior.recsys_teacher_risk_budget import teacher_risk_budget_anchor


def raw_query_predictions(logits, episode, criterion):
    indices = episode.query.nonzero()
    users, items = indices[:, 1], indices[:, 2]
    prediction = criterion.mean(logits.float())
    truth = episode.target[episode.query]
    if episode.row_scales is not None:
        prediction = prediction * episode.row_scales[users]
        truth = truth * episode.row_scales[users]
    prediction = prediction * episode.stds[items] + episode.means[items]
    truth = truth * episode.stds[items] + episode.means[items]
    if episode.row_offsets is not None:
        prediction = prediction + episode.row_offsets[users]
        truth = truth + episode.row_offsets[users]
    return prediction, truth, users, items


def user_pairwise_loss(prediction, truth, users, items, selected, shape, hard_negatives=0, temperature=1.0, negative_only=False):
    if not math.isfinite(temperature) or temperature <= 0 or hard_negatives < 0:
        raise ValueError("Require positive finite temperature and nonnegative hard-negative count")
    positive = selected & (truth.round() >= 4)
    negative = ~selected
    negative_counts = torch.bincount(users[negative], minlength=shape[0])
    positive_counts = torch.bincount(users[positive], minlength=shape[0])
    eligible_users = (negative_counts > 0) & (positive_counts > 0)
    diagnostics = {"ranking_positive_users": int((positive_counts > 0).sum()),
                   "ranking_paired_users": int(eligible_users.sum()), "ranking_negative_queries": int(negative.sum())}
    if not eligible_users.any():
        return prediction.sum() * 0, diagnostics
    dense = prediction.new_full(shape, -torch.inf)
    dense[users[negative], items[negative]] = prediction[negative]
    count = int(negative_counts.max())
    count = min(count, hard_negatives) if hard_negatives else count
    negatives = dense.topk(count, dim=1).values
    positive = positive & eligible_users[users]
    positive_users = users[positive]
    positive_scores = prediction[positive, None].detach() if negative_only else prediction[positive, None]
    differences = (negatives[positive_users] - positive_scores) / temperature
    pair_losses = torch.nn.functional.softplus(differences)
    pair_counts = negative_counts[positive_users].clamp_max(count)
    positive_losses = pair_losses.sum(1) / pair_counts
    user_losses = prediction.new_zeros(shape[0]).scatter_add_(0, positive_users, positive_losses)
    return (user_losses[eligible_users] / positive_counts[eligible_users]).mean(), diagnostics


def expected_listwise_loss(prediction, probabilities, users, items, selected, shape, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Require positive finite ranking temperature")
    if probabilities.shape != prediction.shape or probabilities.requires_grad:
        raise ValueError("Require detached synthetic probabilities aligned to query scores")
    if not torch.isfinite(probabilities).all() or torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid synthetic relevance probability")
    additional_counts = torch.bincount(users[~selected], minlength=shape[0])
    mass = prediction.new_zeros(shape[0]).scatter_add_(0, users, probabilities)
    active = (additional_counts > 0) & (mass > 0)
    diagnostics = {"ranking_positive_users": int((mass > 0).sum()), "ranking_paired_users": int(active.sum()),
                   "ranking_negative_queries": int((~selected).sum())}
    if not active.any():
        return prediction.sum() * 0, diagnostics
    scores = prediction / temperature
    dense = prediction.new_full(shape, -torch.inf)
    dense[users, items] = scores
    normalizer = prediction.new_zeros(shape[0])
    normalizer[active] = torch.logsumexp(dense[active], dim=1)
    included = active[users]
    weights = probabilities[included] / mass[users[included]]
    log_probability = scores[included] - normalizer[users[included]]
    loss = -(weights * log_probability).sum() / active.sum()
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum() / active.sum()
    return loss, {**diagnostics, "ranking_target_entropy": entropy.item(), "ranking_kl": (loss - entropy).item()}


def bounded_ranking_scores(prediction, users, selected, num_users):
    additional_counts = torch.bincount(users[~selected], minlength=num_users)
    active = additional_counts > 0
    included = active[users]
    if not included.any():
        return prediction.clamp(1, 5), prediction.sum() * 0
    counts = torch.bincount(users[included], minlength=num_users)
    distance = prediction[included] - prediction[included].clamp(1.05, 4.95)
    penalty = (distance.square() / counts[users[included]]).sum() / active.sum()
    return prediction.clamp(1, 5), penalty


def topk_union_candidate_weights(relevance, valid, top_k):
    """Fractional teacher-cutoff ties unioned with the current model top-k."""
    cutoff = relevance.masked_fill(~valid, -torch.inf).topk(min(top_k, relevance.shape[1]), dim=1).values[:, -1:]
    positive = valid & (relevance > 0)
    above = positive & (relevance > cutoff)
    tied = positive & (relevance == cutoff)
    remaining = (top_k - above.sum(1, keepdim=True)).clamp_min(0)
    tie_weight = (remaining.to(relevance.dtype) / tied.sum(1, keepdim=True).clamp_min(1)).clamp_max(1)
    teacher = above.to(relevance.dtype) + tied * tie_weight
    model = valid & (torch.arange(relevance.shape[1], device=relevance.device)[None] < top_k)
    return torch.where(model, 1, teacher)


def topk_expected_pairwise_loss(prediction, probabilities, users, items, selected, shape, temperature=1.0, top_k=10,
                               negative_only=False, center_negative_only=False, candidate_scope="all"):
    if candidate_scope not in ("all", "topk_union"):
        raise ValueError("Unknown top-k candidate scope")
    if center_negative_only and not negative_only:
        raise ValueError("Gradient centering requires one-sided ranking")
    if not math.isfinite(temperature) or temperature <= 0 or top_k < 1:
        raise ValueError("Require positive finite temperature and positive top-k")
    if probabilities.shape != prediction.shape or probabilities.requires_grad:
        raise ValueError("Require detached synthetic probabilities aligned to query scores")
    if not torch.isfinite(probabilities).all() or torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid synthetic relevance probability")
    additional_counts = torch.bincount(users[~selected], minlength=shape[0])
    active = additional_counts > 0
    diagnostics = {"ranking_positive_users": int((torch.bincount(users[probabilities > 0], minlength=shape[0]) > 0).sum()),
                   "ranking_paired_users": 0, "ranking_negative_queries": int((~selected).sum()), "ranking_top_k": top_k}
    if candidate_scope == "topk_union":
        diagnostics.update({"ranking_candidate_scope": candidate_scope, "ranking_effective_candidates_mean": 0.0})
        if center_negative_only:
            diagnostics["ranking_gradient_center_scope"] = "all_valid_queries_per_user"
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
    if center_negative_only:
        center = scores.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        scores = scores - (center - center.detach())
    relevance = dense_probabilities[active].gather(1, ordering)
    positions = torch.arange(shape[1], device=prediction.device)
    discounts = torch.where(positions < top_k, 1 / (positions.to(prediction.dtype) + 2).log2(), 0)
    anchors = min(top_k, shape[1])
    pairs = ((positions[None, :] > positions[:anchors, None])[None]
             & valid[:, :anchors, None] & valid[:, None, :])
    differences = relevance[:, :anchors, None] - relevance[:, None, :]
    weights = torch.where(pairs, differences.abs() * (discounts[:anchors, None] - discounts[None, :]), 0)
    if candidate_scope == "topk_union":
        candidate_weights = topk_union_candidate_weights(relevance, valid, top_k)
        weights = weights * candidate_weights[:, None, :]
        diagnostics["ranking_effective_candidates_mean"] = candidate_weights.sum(1).mean().item()
    score_differences = scores[:, :anchors, None] - scores[:, None, :]
    if negative_only:
        score_differences = torch.where(differences > 0,
                                       scores[:, :anchors, None].detach() - scores[:, None, :],
                                       scores[:, :anchors, None] - scores[:, None, :].detach())
    pair_losses = torch.nn.functional.softplus(-differences.sign() * score_differences / temperature)
    mass = weights.sum((1, 2))
    included = mass > 0
    diagnostics["ranking_paired_users"] = int(included.sum())
    if not included.any():
        return prediction.sum() * 0, diagnostics
    return ((weights * pair_losses).sum((1, 2))[included] / mass[included]).mean(), diagnostics


def synthetic_ranking_objective(logits, episode, criterion, ranking_weight=1.0, rating_weight=1.0,
                                hard_negatives=0, temperature=1.0, nll_weight=.1, mae_weight=.1,
                                point_objective="expected_user_rmse", macro_query_scope="parent",
                                teacher_prediction=None, teacher_anchor_weight=0.0, ranking_objective="pairwise",
                                negative_only_ranking=False, return_components=False, ranking_boundary_weight=0.0,
                                ranking_top_k=10, center_negative_only_ranking=False, ranking_candidate_scope="all",
                                ranking_frequency_pseudocount=.5, teacher_anchor_mode='prediction_mse',
                                ranking_finite_gap_margin=.01):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Ranking optimization requires synthetic episodes exclusively")
    if teacher_anchor_mode not in ('prediction_mse', 'macro_risk_budget'):
        raise ValueError('Unknown synthetic-teacher anchor mode')
    if teacher_anchor_mode == 'macro_risk_budget' and (not math.isfinite(teacher_anchor_weight) or teacher_anchor_weight <= 0
                                                      or point_objective != 'expected_user_rmse'):
        raise ValueError('Risk-budget anchoring requires positive finite teacher weight and expected-user-RMSE rating risk')
    visible_teacher = episode.metadata.get("relevance_target") == "visible_item_frequency"
    if visible_teacher:
        expected = torch.isfinite(episode.source).float().mean(1, keepdim=True).expand_as(episode.source)
        if episode.relevance_probabilities is None or not torch.equal(episode.relevance_probabilities, expected):
            raise ValueError("Visible-frequency teachers must exactly match the encoded synthetic mask")
    if any(not math.isfinite(weight) or weight < 0 for weight in (ranking_weight, rating_weight, nll_weight, mae_weight, teacher_anchor_weight, ranking_boundary_weight)):
        raise ValueError("Require finite nonnegative objective weights")
    if ranking_objective not in ("pairwise", "expected_listwise", "topk_expected_pairwise", "topk_soft_choice", "topk_finite_gap"):
        raise ValueError("Unknown synthetic ranking objective")
    if ranking_candidate_scope not in ("all", "topk_union"):
        raise ValueError("Unknown top-k candidate scope")
    if ranking_candidate_scope != "all" and ranking_objective not in ("topk_expected_pairwise", "topk_soft_choice", "topk_finite_gap"):
        raise ValueError("Candidate restriction requires the top-k expected pairwise objective")
    if ranking_objective == "topk_finite_gap":
        if not visible_teacher or ranking_candidate_scope != "topk_union" or negative_only_ranking or center_negative_only_ranking:
            raise ValueError("Finite-gap ranking requires synthetic visible frequencies and ordinary two-sided top-k union")
        if (not math.isfinite(ranking_finite_gap_margin) or ranking_finite_gap_margin <= 0
                or isinstance(ranking_top_k, bool) or not isinstance(ranking_top_k, int)
                or ranking_top_k < 1 or ranking_top_k * ranking_finite_gap_margin > 4):
            raise ValueError("Require positive finite and bounded-rating-feasible top-k margin")
    if ranking_objective == "topk_soft_choice":
        if not visible_teacher or ranking_candidate_scope != "topk_union" or negative_only_ranking or center_negative_only_ranking:
            raise ValueError("Soft choice requires synthetic visible frequencies and ordinary two-sided top-k union")
        if not math.isfinite(ranking_frequency_pseudocount) or ranking_frequency_pseudocount <= 0:
            raise ValueError("Require positive finite synthetic frequency pseudo-count")
    if negative_only_ranking and ranking_objective not in ("pairwise", "topk_expected_pairwise"):
        raise ValueError("Negative-only gradients require a pairwise objective")
    if center_negative_only_ranking and (not negative_only_ranking or ranking_objective != "topk_expected_pairwise"):
        raise ValueError("Gradient centering requires one-sided top-k ranking")
    if ranking_objective == "pairwise" and episode.metadata.get("ranking_query_mode") == "user_catalog":
        raise ValueError("Do not label unselected catalog positives as pairwise negatives")
    rating, diagnostics = rating_objective(logits, episode, criterion, nll_weight=nll_weight,
                                           ranking_weight=0, mae_weight=mae_weight,
                                           point_objective=point_objective, macro_query_scope=macro_query_scope)
    prediction, truth, users, items = raw_query_predictions(logits, episode, criterion)
    ranking_prediction = prediction
    boundary = prediction.sum() * 0
    if ranking_boundary_weight:
        ranking_prediction, boundary = bounded_ranking_scores(prediction, users, episode.rating_query[episode.query],
                                                              episode.source.shape[1])
    if ranking_objective in ("expected_listwise", "topk_expected_pairwise", "topk_soft_choice", "topk_finite_gap"):
        if episode.relevance_probabilities is None or episode.metadata.get("relevance_targets") is not True:
            raise ValueError("Expected ranking requires synthetic latent-relevance targets")
        arguments = (ranking_prediction, episode.relevance_probabilities[episode.query], users, items,
                     episode.rating_query[episode.query], episode.source.shape[1:], temperature)
        if ranking_objective == "expected_listwise":
            ranking, coverage = expected_listwise_loss(*arguments)
        elif ranking_objective == "topk_finite_gap":
            from tabimpute.prior.recsys_finite_gap_objective import finite_gap_topk_loss

            ranking, coverage = finite_gap_topk_loss(*arguments[:-1], temperature=temperature, top_k=ranking_top_k,
                                                    margin=ranking_finite_gap_margin, provenance=episode.metadata['gradient_data'])
        elif ranking_objective == "topk_soft_choice":
            from tabimpute.prior.recsys_soft_choice_objective import topk_soft_choice_loss

            ranking, coverage = topk_soft_choice_loss(*arguments[:-1], temperature=temperature, top_k=ranking_top_k,
                                                      frequency_smoothing=ranking_frequency_pseudocount / episode.source.shape[1])
        else:
            ranking, coverage = topk_expected_pairwise_loss(*arguments, top_k=ranking_top_k, negative_only=negative_only_ranking,
                                                           center_negative_only=center_negative_only_ranking,
                                                           candidate_scope=ranking_candidate_scope)
    else:
        ranking, coverage = user_pairwise_loss(ranking_prediction, truth, users, items, episode.rating_query[episode.query],
                                               episode.source.shape[1:], hard_negatives, temperature, negative_only_ranking)
    anchor = prediction.sum() * 0
    anchor_diagnostics = {}
    if teacher_anchor_weight:
        selected = episode.rating_query[episode.query]
        if teacher_prediction is None or teacher_prediction.shape != prediction[selected].shape or teacher_prediction.requires_grad:
            raise ValueError("Require detached synthetic-teacher predictions aligned to rating queries")
        if not torch.isfinite(teacher_prediction).all():
            raise ValueError("Nonfinite synthetic-teacher prediction")
        if teacher_anchor_mode == 'macro_risk_budget':
            anchor, anchor_diagnostics = teacher_risk_budget_anchor(prediction, episode, teacher_prediction,
                                                                     mae_weight=mae_weight, macro_query_scope=macro_query_scope)
        else:
            counts = torch.bincount(users[selected], minlength=episode.source.shape[1])
            errors = (prediction[selected] - teacher_prediction).square()
            anchor = (errors / counts[users[selected]]).sum() / (counts > 0).sum()
    ranking_loss = ranking_weight * (ranking + ranking_boundary_weight * boundary)
    loss = rating_weight * rating + ranking_loss + teacher_anchor_weight * anchor
    diagnostics = {**diagnostics, **coverage, **anchor_diagnostics, "ranking": ranking.item(), "rating_total": rating.item(),
                  "ranking_weight": ranking_weight, "rating_weight": rating_weight,
                  "ranking_objective": ranking_objective,
                  "negative_only_ranking": negative_only_ranking,
                  "center_negative_only_ranking": center_negative_only_ranking,
                  "ranking_score_transform": "frozen_clip_1_5" if ranking_boundary_weight else "raw",
                  "ranking_boundary": boundary.item(), "ranking_boundary_weight": ranking_boundary_weight,
                  "synthetic_teacher_anchor": anchor.item(), "teacher_anchor_weight": teacher_anchor_weight}
    if visible_teacher:
        diagnostics["visible_mask_teacher_verified"] = True
    if return_components:
        return loss, diagnostics, {"protected": rating_weight * rating + teacher_anchor_weight * anchor,
                                   "ranking": ranking_loss}
    return loss, diagnostics
