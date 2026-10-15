"""Synthetic mixed-prior episodes and frozen-expert targets for one student."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional

from dwip_teacher_mean_training import compute_context
from streamed_large_matrix_training import CurriculumSampler, STREAM_OPTIONS, parse_curriculum
from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from tabimpute.routing import fit_routing, observed_scales
from tabimpute.streamed_inference import streamed_point_predictions


@dataclass
class Episode:
    source: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    metadata: dict
    ratings: object = None


class MixedSampler:
    def __init__(self, config):
        self.config = config
        self.imputation = CurriculumSampler(
            parse_curriculum(config["phases"]), data_seed=config["data_seed"], spectral_replay=.2,
        )

    def sample(self, domain, step, *, seed=None, shape=None, component=None, transpose=None):
        seed = self.config["data_seed"] + 16 * step + (domain == "recsys") if seed is None else seed
        if domain == "imputation":
            source, target, mask, metadata = self.imputation.sample(
                step, 0, device="cuda", seed=seed, forced_shape=shape,
                component=component, transpose=transpose,
            )
            return Episode(source, target, mask, {**metadata, "domain": domain, "gradient_data": "synthetic"})
        if domain != "recsys":
            raise ValueError("Unknown synthetic domain")
        rows, columns = shape or (1000, 1000)
        ratings = sample_rating_episode(
            seed, rows=rows, cols=columns, device="cuda", query_limit=4096,
            density_range=(.004, .08), parent_rows=12288, parent_cols=4096,
            normalization="two_way_user_scale", point_target="expected_rating",
            item_popularity_sigma_range=(1.4, 2.8), context_ordering="first_observation",
            query_sampling="user_balanced", preference_family="mixed", exclude_unknown_queries=True,
            uniform_noise_range=(0., .15), user_noise_log_std_range=(.15, .6),
        )
        if ratings.metadata["gradient_data"] != "synthetic" or not ratings.rating_query.any():
            raise ValueError("Require nonempty synthetic rating queries")
        return Episode(ratings.source, ratings.target, ratings.rating_query,
                       {**ratings.metadata, "domain": domain, "rows": rows, "columns": columns}, ratings)


def blend_predictions(source, general, specialist):
    if source.shape[0] != 1:
        raise ValueError("Fit one observed-only routing decision per synthetic matrix")
    matrix = source[0].detach().float().cpu().numpy()
    decision = fit_routing(
        matrix, general[0].detach().float().cpu().numpy(), specialist[0].detach().float().cpu().numpy(),
        np.isfinite(matrix), observed_scales(matrix),
    )
    weight = decision["specialist_weight"]
    return (general * (1. - weight) + specialist * weight).detach(), decision


@torch.no_grad()
def teacher_target(teachers, criterion, episode, options=STREAM_OPTIONS):
    if any(model.training or any(parameter.requires_grad for parameter in model.parameters()) for model in teachers):
        raise ValueError("Teachers must be frozen in evaluation mode")
    with compute_context(episode.source.device):
        specialist = streamed_point_predictions(teachers[1], episode.source, criterion, **options)
        if episode.metadata["domain"] == "recsys":
            return specialist.detach(), {"specialist_weight": 1., "target": "specialist"}
        general = streamed_point_predictions(teachers[0], episode.source, criterion, **options)
    prediction, decision = blend_predictions(episode.source, general, specialist)
    return prediction, {**decision, "target": "same-input observed blend"}


def distillation_loss(criterion, logits, truth, teacher, *, nll_weight):
    logits, truth = logits.float(), truth.float()
    mean = criterion.mean(logits)
    point = functional.huber_loss(mean, teacher.detach().float(), delta=1., reduction="none")
    target_cdf = criterion.bucket_means.float().unsqueeze(0) >= truth.unsqueeze(1)
    cdf = (logits.softmax(-1).cumsum(-1) - target_cdf.float()).square().mean(-1)
    return nll_weight * criterion(logits, truth) + 2. * point + .05 * cdf


def episode_metrics(prediction, episode):
    if not episode.mask.any() or not torch.isfinite(prediction[episode.mask]).all():
        raise ValueError("Finite predictions and nonempty query masks are required")
    if episode.ratings is None:
        error = prediction[episode.mask].double() - episode.target[episode.mask].double()
        return {"rmse": float(error.square().mean().sqrt()), "mae": float(error.abs().mean())}
    ratings = episode.ratings
    location = ratings.means[None, None, :] + ratings.row_offsets[None, :, None]
    scale = ratings.stds[None, None, :] * ratings.row_scales[None, :, None]
    estimate = (prediction.double() * scale + location).clamp(1., 5.)
    truth = episode.target.double() * scale + location
    error = estimate - truth
    counts = episode.mask.sum(-1)
    supported = counts > 0
    squares = error.square().masked_fill(~episode.mask, 0.).sum(-1) / counts.clamp_min(1)
    absolute = error.abs().masked_fill(~episode.mask, 0.).sum(-1) / counts.clamp_min(1)
    return {"rmse": float(squares[supported].sqrt().mean()), "mae": float(absolute[supported].mean())}


def validation_specs():
    specifications = [dict(domain="imputation", shape=(240, 80), component=component, transpose=False)
                      for component in range(8)]
    specifications.extend(dict(domain="imputation", shape=(2000, 1000), component=component,
                               transpose=bool(component % 2)) for component in (0, 3))
    specifications.extend(dict(domain="recsys", shape=(1000, 1000)) for _ in range(4))
    return specifications


def selection_score(metrics, baseline, reference):
    domains = ("imputation", "recsys")
    values = [metrics[domain][metric] for domain in domains for metric in ("rmse", "mae")]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Nonfinite synthetic validation metrics")
    ratios = [metrics[domain]["rmse"] / max(reference[domain]["rmse"], 1e-12) for domain in domains]
    eligible = (metrics["imputation"]["rmse"] <= 1.03 * baseline["imputation"]["rmse"]
                and metrics["recsys"]["rmse"] <= baseline["recsys"]["rmse"]
                and metrics["recsys"]["mae"] <= baseline["recsys"]["mae"])
    return sum(ratios) / len(ratios), eligible
