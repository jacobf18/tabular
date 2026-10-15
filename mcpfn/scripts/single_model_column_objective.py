"""Synthetic column-NRMSE majorization and prospective stricter validation."""

from contextlib import contextmanager
from functools import lru_cache
import math

import numpy as np
import torch

import diagnose_single_model_metric_alignment as metric
from dwip_teacher_mean_training import compute_context
from recsys_evaluation_common import write_json
from streamed_large_matrix_training import TRAIN_OPTIONS
from tabimpute.streamed_training import streamed_training_loss
import train_single_matrixformer_probability_balance as balanced


continuation = balanced.continuation
COLUMN_OBJECTIVE = {"weight": 1., "thresholds": [1e-3, 1e-4], "reference_rmse_floor": 1e-3}
PRIMARY_METRICS = (("imputation", "missbench_nrmse"), ("imputation", "uci_nrmse"),
                   ("recsys", "rmse"), ("recsys", "mae"))
RETENTION_METRICS = PRIMARY_METRICS + (("imputation", "rmse"), ("imputation", "common_nrmse"))
BASE_BACKWARD = balanced.backward_episode
BASE_EPISODE_METRICS = continuation.episode_metrics
BASE_AGGREGATE = continuation.aggregate


@torch.no_grad()
def column_majorizer(truth, mask, teacher, specification=COLUMN_OBJECTIVE):
    if specification != COLUMN_OBJECTIVE:
        raise ValueError("Require the registered column objective")
    if (truth.ndim != 3 or truth.shape[0] != 1 or mask.shape != truth.shape or teacher.shape != truth.shape
            or mask.dtype != torch.bool or not mask.any() or truth.device != teacher.device
            or truth.device != mask.device or not torch.isfinite(truth[mask]).all()
            or not torch.isfinite(teacher[mask]).all()):
        raise ValueError("Require one finite synthetic truth/teacher grid and aligned nonempty query mask")
    selected_truth = truth.double().masked_fill(~mask, 0.)
    counts = mask.sum(dim=1)
    means = selected_truth.sum(dim=1) / counts.clamp_min(1)
    centered = (truth.double() - means[:, None, :]).masked_fill(~mask, 0.)
    scales = (centered.square().sum(dim=1) / counts.clamp_min(1)).sqrt()
    residual = (teacher.double() - truth.double()).masked_fill(~mask, 0.)
    reference_rmse = (residual.square().sum(dim=1) / counts.clamp_min(1)).sqrt()
    anchors = reference_rmse.clamp_min(specification["reference_rmse_floor"])
    coefficients = torch.zeros_like(scales)
    offset = torch.zeros((), dtype=torch.float64, device=truth.device)
    valid_counts = []
    total_queries = int(mask.sum())
    for threshold in specification["thresholds"]:
        valid = (counts > 0) & (scales >= threshold)
        valid_count = int(valid.sum())
        valid_counts.append(valid_count)
        if valid_count:
            coefficients[valid] += total_queries / (len(specification["thresholds"]) * 2 * valid_count
                                                      * counts[valid] * anchors[valid] * scales[valid])
            offset += (anchors[valid] / (2 * scales[valid])).mean() / len(specification["thresholds"])
    weights = coefficients[:, None, :].expand_as(truth).float()
    if not torch.isfinite(weights).all() or not torch.isfinite(offset):
        raise FloatingPointError("Nonfinite column-majorizer coefficients")
    return weights, offset.float(), {"thresholds": list(specification["thresholds"]),
        "scored_columns": valid_counts, "query_cells": total_queries,
        "reference_rmse_floor": specification["reference_rmse_floor"],
        "floor_active_columns": int(((counts > 0) & (reference_rmse < specification["reference_rmse_floor"])).sum()),
        "weight": specification["weight"], "maximum_cell_coefficient": float(weights.max())}


def backward_episode(student, criterion, episode, teacher, weight, *, probability_scales=balanced.PROBABILITY_SCALES):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Column supervision must use synthetic data only")
    if episode.metadata["domain"] == "recsys":
        return BASE_BACKWARD(student, criterion, episode, teacher, weight, probability_scales=probability_scales)
    if episode.metadata["domain"] != "imputation":
        raise ValueError("Unknown synthetic domain")
    coefficients, offset, diagnostics = column_majorizer(episode.target, episode.mask, teacher)
    episode.metadata["column_objective"] = diagnostics
    teacher_values, cell_weights = teacher.detach().reshape(-1), coefficients.reshape(-1)
    indices = torch.arange(episode.target.numel(), device=episode.target.device).reshape_as(episode.target)

    def loss_function(logits, truth, lookup):
        point = criterion.mean(logits.float())
        original = balanced.balanced_loss(criterion, logits, truth, teacher_values[lookup],
                                          domain="imputation", probability_scales=probability_scales)
        return original + COLUMN_OBJECTIVE["weight"] * cell_weights[lookup] * (point - truth.float()).square()

    with compute_context(episode.source.device):
        loss = streamed_training_loss(student, episode.source, episode.target, episode.mask, loss_function,
                                      auxiliary_targets=indices, **TRAIN_OPTIONS)
    loss = loss + COLUMN_OBJECTIVE["weight"] * offset
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite column-aligned loss")
    (weight * loss).backward()
    return float(loss.detach())


@lru_cache(maxsize=1)
def original_scorers():
    return {suite: metric.load_original(path, "compute_normalized_rmse_columnwise")
            for suite, path in metric.SCORERS.items()}


def episode_metrics(prediction, episode):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Only synthetic episodes may enter training validation")
    result = BASE_EPISODE_METRICS(prediction, episode)
    if episode.metadata["domain"] == "imputation":
        scores = metric.score_queries(episode.target[0].detach().cpu().numpy(),
                                      prediction[0].detach().float().cpu().numpy(),
                                      episode.mask[0].cpu().numpy(), original_scorers())
        if not math.isclose(scores["pooled_rmse"], result["rmse"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Column scoring changed the original pooled query metric")
        common = scores["common_nonconstant_nrmse"]
        if common is None or not math.isfinite(common):
            raise ValueError("Every imputation validation case requires a common nonconstant column")
        result.update(missbench_nrmse=scores["missbench_nrmse"], uci_nrmse=scores["uci_nrmse"], common_nrmse=common)
        result["column_coverage"] = {key: value for key, value in scores.items()
                                     if key.endswith("columns") or key == "queried_columns"}
    return result


def aggregate(cases, field):
    result = BASE_AGGREGATE(cases, field)
    for name in ("missbench_nrmse", "uci_nrmse", "common_nrmse"):
        values = [case[field][name] for case in cases if case["domain"] == "imputation"]
        if not values or not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Require complete finite column metrics on every imputation case")
        result["imputation"][name] = float(np.mean(values))
    return result


def selection_score(metrics, baseline, references):
    for domain, name in RETENTION_METRICS:
        for values in (metrics, baseline, references):
            if not math.isfinite(values[domain][name]) or values[domain][name] < 0:
                raise ValueError("Require complete finite prospective selection metrics")
    ratios = {f"{domain}_{name}": metrics[domain][name] / max(references[domain][name], 1e-12)
              for domain, name in PRIMARY_METRICS}
    retained = all(metrics[domain][name] <= baseline[domain][name] for domain, name in RETENTION_METRICS)
    return max(ratios.values()), ratios, retained


def confirmation_gate(student, initialization, general, references):
    score, ratios, retained = selection_score(student, initialization, references)
    initial_score, initial_ratios, _ = selection_score(initialization, initialization, references)
    _, _, general_retained = selection_score(student, general, references)
    return {"passed": retained and general_retained and score < initial_score,
            "score": score, "initialization_score": initial_score, "ratios": ratios,
            "initialization_ratios": initial_ratios, "initialization_retained": retained,
            "general_retained": general_retained,
            "metric_ratios_to_general": {f"{domain}_{name}": student[domain][name] / max(general[domain][name], 1e-12)
                                         for domain, name in RETENTION_METRICS},
            "primary_metrics": [list(pair) for pair in PRIMARY_METRICS],
            "retention_metrics": [list(pair) for pair in RETENTION_METRICS], "real_benchmark_success": False}


@contextmanager
def evaluation_metrics():
    original_episode, original_aggregate = continuation.episode_metrics, continuation.aggregate
    original_selection, original_bank = continuation.selection_score, continuation.prepare_bank

    def reference_bank(sampler, teachers, criterion, specifications, output):
        bank, reference = original_bank(sampler, teachers, criterion, specifications, output)
        for name in ("missbench_nrmse", "uci_nrmse", "common_nrmse"):
            reference["references"]["imputation"][name] = min(reference[role]["imputation"][name]
                                                               for role in ("general", "teacher"))
        write_json(output / "reference.json", reference)
        return bank, reference

    continuation.episode_metrics, continuation.aggregate = episode_metrics, aggregate
    continuation.selection_score, continuation.prepare_bank = selection_score, reference_bank
    try:
        yield
    finally:
        continuation.episode_metrics, continuation.aggregate = original_episode, original_aggregate
        continuation.selection_score, continuation.prepare_bank = original_selection, original_bank


@contextmanager
def training_objective():
    original_backward, original_preflight = continuation.backward_episode, balanced.backward_episode
    continuation.backward_episode = balanced.backward_episode = backward_episode
    try:
        yield
    finally:
        continuation.backward_episode, balanced.backward_episode = original_backward, original_preflight
