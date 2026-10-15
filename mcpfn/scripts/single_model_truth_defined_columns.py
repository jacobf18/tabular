"""Retain every canonical metric case while marking an undefined auxiliary guard explicitly."""

from contextlib import contextmanager
import math

import numpy as np

import single_model_column_objective as column


POLICY = {
    "canonical_metrics": "Both original NRMSE formulas and pooled RMSE include every original case unchanged",
    "common_column_guard": "Mean over the fixed truth-defined cases with at least one query column whose population std is >=1e-3",
    "undefined_case_value": None,
    "coverage": "The identical truth-only defined-case subset must be used by initialization, general, teacher and student",
    "all_undefined_bank": "Reject; never remove the common-column retention requirement",
    "case_resampling": False,
    "checkpoint_reselection": False,
    "real_benchmark_protocol_changed": False,
}


def episode_metrics(prediction, episode):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Only synthetic confirmation may use the undefined-common-case adapter")
    result = column.BASE_EPISODE_METRICS(prediction, episode)
    if episode.metadata["domain"] == "imputation":
        scores = column.metric.score_queries(episode.target[0].detach().cpu().numpy(),
            prediction[0].detach().float().cpu().numpy(), episode.mask[0].cpu().numpy(), column.original_scorers())
        if not math.isclose(scores["pooled_rmse"], result["rmse"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Canonical pooled metric changed")
        common = scores["common_nonconstant_nrmse"]
        if (common is None) != (scores["common_nonconstant_columns"] == 0):
            raise ValueError("Undefined common metric must depend only on target-column variance")
        if common is not None and (not math.isfinite(common) or common < 0):
            raise ValueError("Nonfinite predictions must not masquerade as undefined target variance")
        result.update(missbench_nrmse=scores["missbench_nrmse"], uci_nrmse=scores["uci_nrmse"], common_nrmse=common)
        result["column_coverage"] = {key: value for key, value in scores.items()
                                     if key.endswith("columns") or key == "queried_columns"}
    return result


def aggregate(cases, field):
    result = column.BASE_AGGREGATE(cases, field)
    imputation = [case[field] for case in cases if case["domain"] == "imputation"]
    for name in ("missbench_nrmse", "uci_nrmse"):
        values = [case[name] for case in imputation]
        if not values or not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Every original canonical NRMSE case is mandatory")
        result["imputation"][name] = float(np.mean(values))
    common = []
    for case in imputation:
        value, count = case["common_nrmse"], case["column_coverage"]["common_nonconstant_columns"]
        if (value is None) != (count == 0):
            raise ValueError("Auxiliary guard coverage must be truth-defined, never prediction-defined")
        if value is not None:
            if not math.isfinite(value) or value < 0:
                raise ValueError("Nonfinite defined auxiliary metric")
            common.append(value)
    if not common:
        raise ValueError("Cannot waive the common-column guard for an entirely undefined bank")
    result["imputation"].update(common_nrmse=float(np.mean(common)), common_nrmse_cases=len(common),
                                canonical_metric_cases=len(imputation))
    return result


@contextmanager
def confirmation_metrics():
    previous_episode, previous_aggregate = column.episode_metrics, column.aggregate
    column.episode_metrics, column.aggregate = episode_metrics, aggregate
    try:
        yield
    finally:
        column.episode_metrics, column.aggregate = previous_episode, previous_aggregate
