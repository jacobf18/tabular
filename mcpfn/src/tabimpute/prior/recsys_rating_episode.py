"""Synthetic explicit-rating episodes with exposure-conditioned held-out queries."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from tabimpute.prior.recsys_clustered_factors import clustered_affinity


@dataclass
class RatingEpisode:
    source: torch.Tensor
    target: torch.Tensor
    query: torch.Tensor
    rating_query: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor
    metadata: dict
    point_target: torch.Tensor | None = None
    row_offsets: torch.Tensor | None = None
    ordinal_probabilities: torch.Tensor | None = None
    global_row_support: torch.Tensor | None = None
    global_col_support: torch.Tensor | None = None
    global_query_counts: torch.Tensor | None = None
    row_scales: torch.Tensor | None = None
    relevance_probabilities: torch.Tensor | None = None


def validate_noise_scale(noise_scale):
    valid = (torch.isfinite(noise_scale) & (noise_scale > 0)).all().item() if isinstance(noise_scale, torch.Tensor) else math.isfinite(noise_scale) and noise_scale > 0
    if not valid:
        raise ValueError("Require finite positive rating noise")


def expected_ordinal_rating(preference, noise_scale, uniform_probability=0.0):
    validate_noise_scale(noise_scale)
    if not 0 <= uniform_probability <= 1:
        raise ValueError("Uniform rating probability must lie in [0,1]")
    expectation = torch.ones_like(preference)
    for threshold in (1.5, 2.5, 3.5, 4.5):
        expectation = expectation + 0.5 * torch.erfc((threshold - preference) / (noise_scale * math.sqrt(2)))
    return (1 - uniform_probability) * expectation + 3 * uniform_probability if uniform_probability else expectation


def ordinal_rating_probabilities(preference, noise_scale, uniform_probability=0.0):
    validate_noise_scale(noise_scale)
    if not 0 <= uniform_probability <= 1:
        raise ValueError("Uniform rating probability must lie in [0,1]")
    survival = torch.stack([0.5 * torch.erfc((threshold - preference) / (noise_scale * math.sqrt(2)))
                            for threshold in (1.5, 2.5, 3.5, 4.5)], dim=-1)
    probabilities = torch.cat((1 - survival[..., :1], survival[..., :-1] - survival[..., 1:], survival[..., -1:]), dim=-1)
    return (1 - uniform_probability) * probabilities + uniform_probability / 5 if uniform_probability else probabilities


def observed_scaling(ratings, observed, mode="column"):
    if mode in ("global", "shrunk_column"):
        values = ratings[observed]
        if not values.numel():
            return ratings.new_zeros(ratings.shape[1]), ratings.new_ones(ratings.shape[1])
        std = values.std(correction=0)
        std = torch.where(std > 1e-4, std, 1)
        means = values.mean().expand(ratings.shape[1])
        if mode == "shrunk_column":
            counts = observed.sum(0)
            means = (torch.where(observed, ratings, 0).sum(0) + 10 * values.mean()) / (counts + 10)
        return means, std.expand(ratings.shape[1])
    if mode != "column":
        raise ValueError("Unknown rating normalization")
    counts = observed.sum(0)
    means = torch.where(observed, ratings, 0).sum(0) / counts.clamp_min(1)
    variance = torch.where(observed, (ratings - means).square(), 0).sum(0) / counts.clamp_min(1)
    stds = variance.sqrt()
    stds = torch.where(stds > 1e-4, stds, 1)
    return means, stds


def fit_bias_normalization(ratings, observed, regularization=3.0, iterations=30):
    row_indices, col_indices = observed.nonzero(as_tuple=True)
    values = ratings[observed]
    if not values.numel():
        return ratings.new_zeros(ratings.shape[1]), ratings.new_ones(ratings.shape[1]), ratings.new_zeros(ratings.shape[0])
    mean = values.mean()
    row_counts = torch.bincount(row_indices, minlength=ratings.shape[0])
    col_counts = torch.bincount(col_indices, minlength=ratings.shape[1])
    row_bias = ratings.new_zeros(ratings.shape[0])
    col_bias = ratings.new_zeros(ratings.shape[1])
    for _ in range(iterations):
        row_bias = ratings.new_zeros(ratings.shape[0]).scatter_add(
            0, row_indices, values - mean - col_bias[col_indices]) / (row_counts + regularization)
        col_bias = ratings.new_zeros(ratings.shape[1]).scatter_add(
            0, col_indices, values - mean - row_bias[row_indices]) / (col_counts + regularization)
    residual = values - mean - row_bias[row_indices] - col_bias[col_indices]
    std = residual.std(correction=0)
    std = torch.where(std > 1e-4, std, 1)
    return mean + col_bias, std.expand(ratings.shape[1]), row_bias


def observed_user_scales(ratings, observed, means, stds, row_offsets):
    row_indices, col_indices = observed.nonzero(as_tuple=True)
    residual = (ratings[observed] - means[col_indices] - row_offsets[row_indices]) / stds[col_indices]
    counts = torch.bincount(row_indices, minlength=ratings.shape[0]).to(ratings.dtype)
    sums = ratings.new_zeros(ratings.shape[0]).scatter_add(0, row_indices, residual)
    squares = ratings.new_zeros(ratings.shape[0]).scatter_add(0, row_indices, residual.square())
    centered = (squares - sums.square() / counts.clamp_min(1)).clamp_min(0)
    variance = (centered + 20) / ((counts - 1).clamp_min(0) + 20)
    return variance.sqrt().clamp(.5, 2.)


def context_crop_indices(observed, rows, cols, generator, ordering="iid", query_mask=None, paired_window_cols=0):
    if ordering not in ("iid", "first_observation", "first_observation_catalog", "first_observation_paired"):
        raise ValueError("Unknown context ordering")
    if rows > observed.shape[0] or cols > observed.shape[1]:
        raise ValueError("Crop exceeds parent geometry")
    if ordering == "first_observation_paired" and (not isinstance(paired_window_cols, int) or paired_window_cols < 1 or cols != 2 * paired_window_cols):
        raise ValueError("Paired contexts require exactly two equal-width item windows")
    if ordering == "iid":
        return torch.arange(rows, device=observed.device), torch.arange(cols, device=observed.device)
    arrival = torch.rand(observed.shape, generator=generator, device=observed.device).masked_fill(~observed, torch.inf)
    row_order = arrival.amin(1).argsort(stable=True)
    col_order = arrival.amin(0).argsort(stable=True)
    anchors = (None, None)
    if query_mask is not None:
        eligible = query_mask & observed.any(1)[:, None] & observed.any(0)[None, :]
        query_rows = eligible.any(1).nonzero().flatten()
        if query_rows.numel():
            anchor_row = query_rows[torch.randint(len(query_rows), (), generator=generator, device=observed.device)]
            query_cols = eligible[anchor_row].nonzero().flatten()
            anchor_col = query_cols[torch.randint(len(query_cols), (), generator=generator, device=observed.device)]
            anchors = ((row_order == anchor_row).nonzero().item(), (col_order == anchor_col).nonzero().item())
    selected = []
    for axis, (order, width, anchor) in enumerate(((row_order, rows, anchors[0]), (col_order, cols, anchors[1]))):
        if axis == 1 and ordering == "first_observation_paired":
            starts = list(range(0, len(order), paired_window_cols))
            starts[-1] = min(starts[-1], len(order) - paired_window_cols)
            primary_starts = [start for start in starts if anchor is None or start <= anchor < start + paired_window_cols]
            primary = primary_starts[torch.randint(len(primary_starts), (), generator=generator, device=observed.device).item()]
            partner_starts = [start for start in starts if abs(start - primary) >= paired_window_cols]
            partner = partner_starts[torch.randint(len(partner_starts), (), generator=generator, device=observed.device).item()]
            selected.append(torch.cat((order[primary:primary + paired_window_cols], order[partner:partner + paired_window_cols])))
            continue
        if axis == 1 and ordering == "first_observation_catalog":
            candidates = torch.randperm(len(order), generator=generator, device=observed.device)
            if anchor is not None:
                candidates = candidates[candidates != anchor]
                positions = torch.cat((candidates[:width - 1], candidates.new_tensor([anchor])))
            else:
                positions = candidates[:width]
            selected.append(order[positions.sort().values])
            continue
        starts = list(range(0, len(order), width))
        starts[-1] = min(starts[-1], len(order) - width)
        if anchor is not None:
            starts = [start for start in starts if start <= anchor < start + width]
        start = starts[torch.randint(len(starts), (), generator=generator, device=observed.device).item()]
        selected.append(order[start:start + width])
    return tuple(selected)


def cap_rating_queries(query, query_limit, generator, sampling="uniform_cells"):
    if sampling not in ("uniform_cells", "user_balanced"):
        raise ValueError("Unknown query sampling")
    indices = query.flatten().nonzero().flatten()
    if indices.numel() <= query_limit:
        return query
    shuffled = indices[torch.randperm(indices.numel(), generator=generator, device=query.device)]
    if sampling == "uniform_cells":
        chosen = shuffled[:query_limit]
    else:
        positions = torch.arange(shuffled.numel(), device=query.device)
        first = torch.full((query.shape[0],), shuffled.numel(), dtype=torch.long, device=query.device)
        first.scatter_reduce_(0, shuffled // query.shape[1], positions, reduce="amin")
        mandatory = first[first < shuffled.numel()]
        if mandatory.numel() > query_limit:
            mandatory = mandatory[torch.randperm(mandatory.numel(), generator=generator, device=query.device)[:query_limit]]
        remaining = torch.ones(shuffled.numel(), dtype=torch.bool, device=query.device)
        remaining[mandatory] = False
        chosen = torch.cat((shuffled[mandatory], shuffled[remaining][:query_limit - mandatory.numel()]))
    return torch.zeros_like(query).flatten().scatter_(0, chosen, True).reshape_as(query)


def matched_ranking_queries(rating_query, ratings, eligible, generator, negatives_per_user, max_users=0, user_mask=None):
    if negatives_per_user < 1 or max_users < 0:
        raise ValueError("Require at least one negative per positive user")
    if user_mask is not None:
        if not isinstance(user_mask, torch.Tensor) or user_mask.dtype != torch.bool or user_mask.shape != eligible.shape[:1] or user_mask.device != eligible.device:
            raise ValueError("Require an aligned boolean ranking-user mask on the candidate device")
        positive_users = user_mask & eligible.any(1)
    else:
        positive_users = (rating_query & (ratings >= 4)).any(1)
    if max_users and positive_users.sum() > max_users:
        users = positive_users.nonzero().flatten()
        selected_users = users[torch.randperm(users.numel(), generator=generator, device=users.device)[:max_users]]
        positive_users = torch.zeros_like(positive_users).scatter_(0, selected_users, True)
    eligible = eligible & positive_users[:, None]
    random_scores = torch.rand(eligible.shape, generator=generator, device=eligible.device)
    random_scores = random_scores.masked_fill(~eligible, -1)
    count = min(negatives_per_user, eligible.shape[1])
    chosen = random_scores.topk(count, dim=1).indices
    result = torch.zeros_like(eligible).scatter_(1, chosen, True)
    return result & eligible


def conditional_relevance_probability(exposure_probability, positive_probability, holdout_fraction):
    if not 0 < holdout_fraction < 1:
        raise ValueError("Require a holdout fraction strictly between zero and one")
    for probability in (exposure_probability, positive_probability):
        if not torch.isfinite(probability).all() or torch.any((probability < 0) | (probability > 1)):
            raise ValueError("Require finite probabilities in [0,1]")
    probability = exposure_probability * holdout_fraction * positive_probability / (1 - (1 - holdout_fraction) * exposure_probability)
    return probability.clamp(0, 1)


def item_marginal_positive_probability(exposure_probability, preference, noise_scale, uniform_probability=0.0):
    validate_noise_scale(noise_scale)
    if exposure_probability.shape != preference.shape or preference.ndim != 2 or not preference.shape[0]:
        raise ValueError("Require aligned user-by-item synthetic latent matrices")
    if not 0 <= uniform_probability <= 1:
        raise ValueError("Uniform rating probability must lie in [0,1]")
    if not torch.isfinite(preference).all() or not torch.isfinite(exposure_probability).all():
        raise ValueError("Require finite synthetic latents")
    if torch.any((exposure_probability < 0) | (exposure_probability > 1)):
        raise ValueError("Exposure probabilities must lie in [0,1]")
    positive = .5 * torch.erfc((3.5 - preference) / (noise_scale * math.sqrt(2)))
    positive = (1 - uniform_probability) * positive + .4 * uniform_probability
    return (exposure_probability * positive).mean(0).clamp(0, 1)


def sample_rating_episode(seed, rows=256, cols=256, device="cpu", query_limit=1024,
                          density_range=(0.01, 0.12), parent_rows=1024, normalization="column", point_target="observed",
                          parent_cols=None, item_popularity_sigma_range=(0.7, 1.9), context_ordering="iid",
                          query_sampling="uniform_cells", ordinal_targets=False, preference_family="gaussian",
                          exclude_unknown_queries=False, uniform_noise_range=(0.0, 0.0),
                          user_noise_log_std_range=(0.0, 0.0), ranking_query_mode="global", negatives_per_user=16,
                          ranking_max_users=0, relevance_targets=False, relevance_target="conditional",
                          ranking_user_scope="local_positive", paired_window_cols=0,
                          exposure_affinity_range=(0.0, 1.0), exposure_quality_range=(0.0, 1.5)):
    if rows < 2 or cols < 2 or query_limit < 2 or parent_rows < rows:
        raise ValueError("Require positive episode geometry, parent_rows >= rows, and at least two queries")
    if not 0 < density_range[0] <= density_range[1] < 1:
        raise ValueError("Require exposure density strictly between zero and one")
    if point_target not in ("observed", "expected_rating"):
        raise ValueError("Unknown point target")
    generated_cols = parent_cols if parent_cols is not None else max(1024, cols) if normalization in ("two_way", "two_way_user_scale") else cols
    if generated_cols < cols:
        raise ValueError("Require parent_cols >= cols")
    if not 0 <= item_popularity_sigma_range[0] <= item_popularity_sigma_range[1]:
        raise ValueError("Require nonnegative ordered item-popularity spread")
    if context_ordering not in ("iid", "first_observation", "first_observation_catalog", "first_observation_paired"):
        raise ValueError("Unknown context ordering")
    if query_sampling not in ("uniform_cells", "user_balanced"):
        raise ValueError("Unknown query sampling")
    if preference_family not in ("gaussian", "clustered", "mixed"):
        raise ValueError("Unknown synthetic preference family")
    if ranking_query_mode not in ("global", "user_matched", "user_catalog"):
        raise ValueError("Unknown synthetic ranking query mode")
    if ranking_query_mode == "user_catalog" and not relevance_targets:
        raise ValueError("Full unseen catalogs require synthetic expected-relevance targets")
    if ranking_user_scope not in ("local_positive", "parent_positive"):
        raise ValueError("Unknown synthetic ranking-user scope")
    if ranking_user_scope != "local_positive" and (ranking_query_mode != "user_catalog" or not relevance_targets):
        raise ValueError("Parent-positive ranking users require synthetic expected-relevance user_catalog queries")
    if relevance_target not in ("conditional", "item_marginal_positive", "visible_item_frequency"):
        raise ValueError("Unknown synthetic relevance target")
    if relevance_target != "conditional" and not relevance_targets:
        raise ValueError("Alternative ranking requires synthetic relevance targets")
    if negatives_per_user < 1 or ranking_max_users < 0:
        raise ValueError("Require positive negative-query count")
    if not 0 <= uniform_noise_range[0] <= uniform_noise_range[1] <= 1:
        raise ValueError("Uniform rating probability range must be ordered within [0,1]")
    if not 0 <= user_noise_log_std_range[0] <= user_noise_log_std_range[1] <= 1:
        raise ValueError("User noise log-std range must be ordered within [0,1]")
    if (len(exposure_affinity_range) != 2 or not all(math.isfinite(value) for value in exposure_affinity_range)
            or not 0 <= exposure_affinity_range[0] <= exposure_affinity_range[1]):
        raise ValueError("Exposure-affinity range must be finite, nonnegative and ordered")
    if (len(exposure_quality_range) != 2 or not all(math.isfinite(value) for value in exposure_quality_range)
            or not 0 <= exposure_quality_range[0] <= exposure_quality_range[1]):
        raise ValueError("Exposure-quality range must be finite, nonnegative and ordered")
    generator = torch.Generator(device=device).manual_seed(seed)

    def uniform(low, high):
        return low + (high - low) * torch.rand((), generator=generator, device=device).item()

    def normal(*shape):
        return torch.randn(*shape, generator=generator, device=device)

    rank = int(uniform(2, 17))
    density = math.exp(uniform(math.log(density_range[0]), math.log(density_range[1])))
    global_mean = uniform(3.1, 3.9)
    user_bias = normal(parent_rows, 1) * uniform(0.25, 0.65)
    item_bias = normal(1, generated_cols) * uniform(0.35, 0.75)
    if preference_family == "mixed":
        preference_family = "clustered" if uniform(0, 1) < .5 else "gaussian"
    factor_metadata = {}
    if preference_family == "clustered":
        affinity, factor_metadata = clustered_affinity(parent_rows, generated_cols, rank, generator, device)
    else:
        affinity = normal(parent_rows, rank) @ normal(rank, generated_cols) / math.sqrt(rank)
    signal_scale, noise_scale = uniform(0.2, 0.85), uniform(0.55, 1.0)
    preference = global_mean + user_bias + item_bias + signal_scale * affinity
    user_noise_log_std = uniform(*user_noise_log_std_range) if user_noise_log_std_range[1] else 0.0
    rating_noise = noise_scale
    if user_noise_log_std:
        rating_noise = noise_scale * torch.exp(user_noise_log_std * normal(parent_rows, 1) - user_noise_log_std ** 2)
    ratings = (preference + rating_noise * normal(parent_rows, generated_cols)).round().clamp(1, 5)
    uniform_probability = uniform(*uniform_noise_range) if uniform_noise_range[1] else 0.0
    if uniform_probability:
        contaminated = torch.rand(ratings.shape, generator=generator, device=device) < uniform_probability
        uniform_ratings = torch.randint(1, 6, ratings.shape, generator=generator, device=device, dtype=ratings.dtype)
        ratings = torch.where(contaminated, uniform_ratings, ratings)
    activity = normal(parent_rows, 1)
    activity_sigma = uniform(0.5, 1.5)
    popularity = normal(1, generated_cols)
    popularity_sigma = uniform(*item_popularity_sigma_range)
    exposure_logits = activity * activity_sigma + popularity * popularity_sigma
    exposure_affinity = uniform(*exposure_affinity_range)
    exposure_quality = uniform(*exposure_quality_range)
    exposure_logits += exposure_affinity * affinity + exposure_quality * item_bias
    if tuple(exposure_affinity_range) != (0.0, 1.0):
        factor_metadata.update(exposure_affinity_range=list(exposure_affinity_range),
                               exposure_affinity_coefficient=exposure_affinity)
    if tuple(exposure_quality_range) != (0.0, 1.5):
        factor_metadata.update(exposure_quality_range=list(exposure_quality_range),
                               exposure_quality_coefficient=exposure_quality)
    lower, upper = -20.0, 20.0
    for _ in range(24):
        middle = (lower + upper) / 2
        if (exposure_logits + middle).sigmoid().mean().item() < density:
            lower = middle
        else:
            upper = middle
    exposure_probability = (exposure_logits + middle).sigmoid()
    exposed = torch.rand(ratings.shape, generator=generator, device=device) < exposure_probability
    holdout_uniforms = torch.rand(ratings.shape, generator=generator, device=device)
    holdout_fraction = uniform(0.15, 0.3)
    holdout = exposed & (holdout_uniforms < holdout_fraction)
    observed = exposed & ~holdout
    known_context = observed.any(1)[:, None] & observed.any(0)[None, :]
    if exclude_unknown_queries:
        holdout = holdout & known_context
    row_offsets = None
    row_scales = None
    if normalization in ("two_way", "two_way_user_scale"):
        means, stds, row_offsets = fit_bias_normalization(ratings, observed)
        if normalization == "two_way_user_scale":
            row_scales = observed_user_scales(ratings, observed, means, stds, row_offsets)
    else:
        means, stds = observed_scaling(ratings, observed, normalization)
    row_indices, col_indices = context_crop_indices(observed, rows, cols, generator, context_ordering, holdout,
                                                   paired_window_cols=paired_window_cols)
    means, stds = means[col_indices], stds[col_indices]
    if row_offsets is not None:
        row_offsets = row_offsets[row_indices]
    if row_scales is not None:
        row_scales = row_scales[row_indices]
    local_ratings = ratings[row_indices[:, None], col_indices]
    local_observed = observed[row_indices[:, None], col_indices]
    rating_query = holdout[row_indices[:, None], col_indices].clone()
    rating_query = cap_rating_queries(rating_query, query_limit, generator, query_sampling)
    negative_query = torch.zeros_like(rating_query)
    negative_eligible = ~exposed[row_indices[:, None], col_indices]
    if exclude_unknown_queries:
        negative_eligible = negative_eligible & known_context[row_indices[:, None], col_indices]
    if ranking_query_mode in ("user_matched", "user_catalog"):
        ranking_eligible = negative_eligible
        if ranking_query_mode == "user_catalog":
            ranking_eligible = ~local_observed & known_context[row_indices[:, None], col_indices]
        ranking_user_mask = (holdout & known_context & (ratings >= 4)).any(1)[row_indices] if ranking_user_scope == "parent_positive" else None
        negative_query = matched_ranking_queries(rating_query, local_ratings, ranking_eligible, generator,
                                                negatives_per_user, ranking_max_users, user_mask=ranking_user_mask)
    else:
        candidates = negative_eligible.flatten().nonzero().flatten()
        chosen = candidates[torch.randperm(candidates.numel(), generator=generator, device=device)[:query_limit // 2]]
        negative_query.flatten()[chosen] = True
    location = means if row_offsets is None else means + row_offsets[:, None]
    normalized = (local_ratings - location) / stds
    if row_scales is not None:
        normalized = normalized / row_scales[:, None]
    source = normalized.masked_fill(~local_observed, torch.nan)
    local_noise = rating_noise[row_indices] if isinstance(rating_noise, torch.Tensor) else rating_noise
    probabilities = ordinal_rating_probabilities(preference[row_indices[:, None], col_indices], local_noise, uniform_probability) if ordinal_targets or relevance_targets else None
    relevance = conditional_relevance_probability(exposure_probability[row_indices[:, None], col_indices],
                                                  probabilities[..., 3:].sum(-1), holdout_fraction) if relevance_targets else None
    if relevance_targets and relevance_target == "item_marginal_positive":
        marginal = item_marginal_positive_probability(exposure_probability, preference, rating_noise, uniform_probability)
        relevance = marginal[col_indices][None].expand_as(local_ratings)
    elif relevance_targets and relevance_target == "visible_item_frequency":
        relevance = local_observed.float().mean(0)[None].expand_as(local_ratings)
    return RatingEpisode(
        source=source.unsqueeze(0), target=normalized.unsqueeze(0),
        query=(rating_query | negative_query).unsqueeze(0), rating_query=rating_query.unsqueeze(0),
        means=means, stds=stds,
        metadata={"gradient_data": "synthetic", "seed": seed, "density": density,
                  "support_density": local_observed.float().mean().item(),
                  "rank": rank, "signal_scale": signal_scale, "noise_scale": noise_scale,
                  "user_noise_log_std": user_noise_log_std,
                  "uniform_rating_probability": uniform_probability,
                  "rating_queries": rating_query.sum().item(), "parent_rows": parent_rows,
                  "normalization": normalization, "point_target": point_target, "parent_cols": generated_cols,
                  "activity_sigma": activity_sigma, "popularity_sigma": popularity_sigma,
                  "context_ordering": context_ordering, "query_sampling": query_sampling,
                  "paired_window_cols": paired_window_cols,
                  "rating_query_users": rating_query.any(1).sum().item(),
                  "ranking_query_mode": ranking_query_mode, "negative_queries": (negative_query & negative_eligible).sum().item(),
                  "additional_ranking_queries": (negative_query & ~rating_query).sum().item(),
                  "relevance_targets": relevance_targets, "relevance_target": relevance_target, "holdout_fraction": holdout_fraction,
                  "ranking_max_users": ranking_max_users,
                  "ranking_user_scope": ranking_user_scope,
                  "ranking_selected_users": negative_query.any(1).sum().item(),
                  "ranking_users_without_local_positive": (negative_query.any(1) & ~(holdout[row_indices[:, None], col_indices] & (local_ratings >= 4)).any(1)).sum().item(),
                  "ranking_users_without_sampled_positive": (negative_query.any(1) & ~(rating_query & (local_ratings >= 4)).any(1)).sum().item(),
                  "ranking_users_without_local_context": (negative_query.any(1) & ~local_observed.any(1)).sum().item(),
                  "preference_family": preference_family, **factor_metadata,
                  "exclude_unknown_queries": exclude_unknown_queries,
                  "global_unknown_rating_queries": (rating_query & ~known_context[row_indices[:, None], col_indices]).sum().item()},
        point_target=expected_ordinal_rating(preference[row_indices[:, None], col_indices], local_noise, uniform_probability).unsqueeze(0)
        if point_target == "expected_rating" else None,
        row_offsets=row_offsets,
        ordinal_probabilities=probabilities.unsqueeze(0) if ordinal_targets else None,
        global_row_support=observed.sum(1)[row_indices],
        global_col_support=observed.sum(0)[col_indices],
        global_query_counts=holdout.sum(1)[row_indices],
        row_scales=row_scales,
        relevance_probabilities=relevance.unsqueeze(0) if relevance_targets else None,
    )


def expected_macro_rmse(prediction, probabilities, user_indices, num_users, seed, samples=32, effective_counts=None):
    """Expected risk; optional counts replicate sampled query profiles as a population surrogate."""
    if samples < 2 or samples % 2:
        raise ValueError("Expected macro RMSE requires an even sample count of at least two")
    if probabilities.shape != (prediction.numel(), 5):
        raise ValueError("Require one five-rating probability distribution per prediction")
    counts = torch.bincount(user_indices, minlength=num_users)
    generator = torch.Generator(device=prediction.device).manual_seed(seed)
    if effective_counts is not None:
        if effective_counts.shape != counts.shape or effective_counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("Effective query counts must be an integer vector over users")
        if torch.any(effective_counts < counts):
            raise ValueError("Effective query counts cannot be smaller than decoded query counts")
        if torch.any(effective_counts[counts > 0] > counts[counts > 0]):
            order = torch.randperm(prediction.numel(), generator=generator, device=prediction.device)
            order = order[user_indices[order].argsort(stable=True)]
            ordered_users = user_indices[order]
            positions = torch.arange(prediction.numel(), device=prediction.device) - (counts.cumsum(0) - counts)[ordered_users]
            ordered_repeats = effective_counts[ordered_users] // counts[ordered_users]
            ordered_repeats = ordered_repeats + (positions < effective_counts[ordered_users] % counts[ordered_users])
            repeats = torch.zeros_like(user_indices).scatter_(0, order, ordered_repeats)
            prediction = prediction.repeat_interleave(repeats)
            probabilities = probabilities.repeat_interleave(repeats, dim=0)
            user_indices = user_indices.repeat_interleave(repeats)
            counts = torch.bincount(user_indices, minlength=num_users)
    singleton = counts[user_indices] == 1
    values = torch.arange(1, 6, device=prediction.device, dtype=prediction.dtype)
    singleton_loss = ((prediction[singleton, None] - values).abs() * probabilities[singleton]).sum()
    multiple = ~singleton
    if not multiple.any():
        return singleton_loss / (counts > 0).sum()
    uniforms = torch.rand(samples // 2, int(multiple.sum()), device=prediction.device, generator=generator)
    uniforms = torch.cat((uniforms, 1 - uniforms), dim=0)
    cumulative = probabilities[multiple].detach().cumsum(-1)[..., :-1]
    outcomes = 1 + (uniforms[..., None] > cumulative[None]).sum(-1)
    errors = (prediction[multiple][None] - outcomes).square()
    user_errors = prediction.new_zeros(samples, num_users).scatter_add(
        1, user_indices[multiple][None].expand(samples, -1), errors)
    multiple_loss = (user_errors[:, counts > 1] / counts[counts > 1] + 1e-6).sqrt().mean(0).sum()
    return (singleton_loss + multiple_loss) / (counts > 0).sum()


def rating_objective(logits, episode, criterion, nll_weight=0.1, ranking_weight=0.0, mae_weight=0.0,
                     point_objective="mse", ranking_negatives="unexposed", macro_query_scope="decoded",
                     mean_rating_weight=0.0):
    if mae_weight < 0:
        raise ValueError("Expected MAE weight must be nonnegative")
    if not math.isfinite(mean_rating_weight) or mean_rating_weight < 0:
        raise ValueError("Mean rating weight must be finite and nonnegative")
    if mean_rating_weight and (episode.metadata.get("gradient_data") != "synthetic"
                               or episode.metadata.get("point_target") != "expected_rating"
                               or episode.point_target is None):
        raise ValueError("Mean rating auxiliary requires synthetic expected-rating targets")
    if point_objective not in ("mse", "user_rmse", "expected_user_rmse"):
        raise ValueError("Unknown rating point objective")
    if ranking_negatives not in ("unexposed", "observed_low"):
        raise ValueError("Unknown ranking negative source")
    if macro_query_scope not in ("decoded", "parent") or (macro_query_scope == "parent" and point_objective != "expected_user_rmse"):
        raise ValueError("Parent query scope requires the expected user-RMSE objective")
    query_indices = episode.query.nonzero()
    row_indices, col_indices = query_indices[:, 1], query_indices[:, 2]
    prediction = criterion.mean(logits.float())
    truth = episode.target[episode.query]
    if episode.row_scales is not None:
        prediction = prediction * episode.row_scales[row_indices]
        truth = truth * episode.row_scales[row_indices]
    prediction = prediction * episode.stds[col_indices] + episode.means[col_indices]
    truth = truth * episode.stds[col_indices] + episode.means[col_indices]
    if episode.row_offsets is not None:
        prediction = prediction + episode.row_offsets[row_indices]
        truth = truth + episode.row_offsets[row_indices]
    selected = episode.rating_query[episode.query]
    if not selected.any():
        raise ValueError("Episode contains no exposed held-out rating queries")
    point_truth = truth if episode.point_target is None else episode.point_target[episode.query]
    errors = (prediction[selected] - point_truth[selected]).square()
    counts = torch.bincount(row_indices[selected], minlength=episode.source.shape[1])
    weights = 1 / counts[row_indices[selected]].float().clamp_min(1)
    point_mse = (errors * weights).sum() / (counts > 0).sum()
    point_loss = point_mse
    if point_objective == "user_rmse":
        user_errors = prediction.new_zeros(episode.source.shape[1]).scatter_add(0, row_indices[selected], errors)
        point_loss = (user_errors[counts > 0] / counts[counts > 0] + 1e-6).sqrt().mean()
    if point_objective == "expected_user_rmse":
        if episode.metadata.get("gradient_data") != "synthetic" or episode.ordinal_probabilities is None:
            raise ValueError("Expected user RMSE requires synthetic ordinal probabilities")
        effective_counts = episode.global_query_counts if macro_query_scope == "parent" else None
        if macro_query_scope == "parent" and effective_counts is None:
            raise ValueError("Parent query scope requires synthetic global query counts")
        point_loss = expected_macro_rmse(prediction[selected], episode.ordinal_probabilities[episode.query][selected],
                                         row_indices[selected], episode.source.shape[1], episode.metadata["seed"] + 7919,
                                         effective_counts=effective_counts)
    expected_mae = prediction.sum() * 0
    if mae_weight:
        if episode.ordinal_probabilities is None:
            raise ValueError("Expected MAE requires ordinal target probabilities")
        probabilities = episode.ordinal_probabilities[episode.query][selected]
        values = torch.arange(1, 6, device=prediction.device, dtype=prediction.dtype)
        absolute_errors = ((prediction[selected, None] - values).abs() * probabilities).sum(-1)
        expected_mae = (absolute_errors * weights).sum() / (counts > 0).sum()
    rating_mse = ((prediction[selected] - truth[selected]).square() * weights).sum() / (counts > 0).sum()
    nll = criterion(logits.float()[selected], episode.target[episode.query][selected]).mean()
    ranking = prediction.sum() * 0
    if ranking_weight:
        positive = selected & (truth.round() >= 4)
        negative = ~selected if ranking_negatives == "unexposed" else selected & (truth.round() <= 3)
        positive_rows = row_indices[positive]
        same_user = positive_rows[:, None] == row_indices[negative][None, :]
        pair_losses = torch.nn.functional.softplus(prediction[negative][None, :] - prediction[positive][:, None])
        row_sums = prediction.new_zeros(episode.source.shape[1]).scatter_add(
            0, positive_rows, torch.where(same_user, pair_losses, 0).sum(1))
        row_counts = prediction.new_zeros(episode.source.shape[1]).scatter_add(
            0, positive_rows, same_user.sum(1).float())
        ranking = (row_sums / row_counts.clamp_min(1)).sum() / (row_counts > 0).sum().clamp_min(1)
    loss = point_loss + nll_weight * nll + ranking_weight * ranking + mae_weight * expected_mae
    if mean_rating_weight:
        loss = loss + mean_rating_weight * point_mse
    return loss, {"mse": point_mse.item(), "point_loss": point_loss.item(), "point_objective": point_objective,
                  "macro_query_scope": macro_query_scope,
                  "macro_population_queries": int(episode.global_query_counts[counts > 0].sum()) if macro_query_scope == "parent" else int(selected.sum()),
                  "rating_mse": rating_mse.item(), "nll": nll.item(),
                  "ranking": ranking.item(), "expected_mae": expected_mae.item()}
