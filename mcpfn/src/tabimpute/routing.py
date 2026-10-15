"""Two-expert routing using reconstruction or masked observed-cell evidence."""

from dataclasses import dataclass
import time

import numpy as np


def observed_scales(matrix, normalization="column"):
    matrix = np.asarray(matrix, dtype=np.float64)
    if normalization not in {"column", "global"}:
        raise ValueError("Routing normalization must be column or global")
    observed = np.isfinite(matrix)
    axis = 0 if normalization == "column" else None
    counts = observed.sum(axis=axis)
    means = np.nansum(matrix, axis=axis) / np.maximum(counts, 1)
    residual = np.where(observed, matrix - means, 0.)
    scales = np.sqrt(np.sum(residual ** 2, axis=axis) / np.maximum(counts, 1))
    return np.broadcast_to(np.where(scales > 1e-8, scales, 1.), (matrix.shape[1],))


def observed_holdout(matrix, fraction=.1, seed=42):
    """Deterministically hide observations without emptying a row or column."""
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be between zero and one")
    observed = np.isfinite(matrix)
    row_counts, column_counts = observed.sum(axis=1), observed.sum(axis=0)
    candidates = np.flatnonzero(observed)
    budget = int(len(candidates) * fraction)
    heldout = np.zeros(matrix.shape, dtype=bool)
    for flat_index in np.random.default_rng(seed).permutation(candidates):
        if budget == 0:
            break
        row, column = divmod(int(flat_index), matrix.shape[1])
        if row_counts[row] > 1 and column_counts[column] > 1:
            heldout[row, column] = True
            row_counts[row] -= 1
            column_counts[column] -= 1
            budget -= 1
    return heldout


def fit_routing(target, general, specialist, evidence, scales):
    """Fit hard selection and a convex specialist weight without outer labels."""
    target, general, specialist = (np.asarray(array, dtype=np.float64) for array in (target, general, specialist))
    if target.shape != general.shape or target.shape != specialist.shape or evidence.shape != target.shape:
        raise ValueError("Routing arrays must have identical shapes")
    if not np.isfinite(target[evidence]).all():
        raise ValueError("Routing evidence must contain observed targets only")
    scales = np.asarray(scales)
    if scales.shape != (target.shape[1],) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Routing scales must be finite and positive, one per column")
    if not np.isfinite(general).all() or not np.isfinite(specialist).all():
        raise ValueError("Nonfinite expert predictions")
    count = int(evidence.sum())
    if not count:
        return {"evidence_cells": 0, "general_mse": None, "specialist_mse": None,
                "selected": "general", "specialist_weight": 0., "fallback": "no evidence"}
    general_error = ((general - target) / scales)[evidence]
    specialist_error = ((specialist - target) / scales)[evidence]
    difference = specialist_error - general_error
    denominator = float(np.dot(difference, difference))
    weight = float(np.clip(-np.dot(general_error, difference) / denominator, 0., 1.)) if denominator > 0 else 0.
    general_mse = float(np.mean(general_error ** 2))
    specialist_mse = float(np.mean(specialist_error ** 2))
    if not np.isfinite([weight, general_mse, specialist_mse]).all():
        raise ValueError("Nonfinite routing loss")
    return {"evidence_cells": count, "general_mse": general_mse, "specialist_mse": specialist_mse,
            "selected": "specialist" if specialist_mse < general_mse else "general",
            "specialist_weight": weight, "fallback": None}


@dataclass
class RoutingResult:
    predictions: dict
    decisions: dict
    seconds: dict


class TwoExpertRouter:
    """Route once per supplied matrix, never using its missing-entry targets.

    Experts must return (completed, raw_predictions) for return_full=True.
    Raw reconstruction is in-sample evidence, not a cross-validation estimate.
    Holdout evidence is scored after both experts refit preprocessing on the
    reduced context; final predictions always use every original observation.
    """

    def __init__(self, general, specialist, *, normalization="column", holdout_fraction=.1,
                 seed=42, policy="observed_hard"):
        if normalization not in {"column", "global"}:
            raise ValueError("Routing normalization must be column or global")
        if not 0 < holdout_fraction < 1:
            raise ValueError("Holdout fraction must be between zero and one")
        if policy not in {"observed_hard", "observed_blend", "holdout_hard", "holdout_blend"}:
            raise ValueError("Unknown routing policy")
        self.experts = {"general": general, "specialist": specialist}
        self.normalization = normalization
        self.holdout_fraction = holdout_fraction
        self.seed = seed
        self.policy = policy
        self.last_result = None

    def _predict(self, matrix):
        predictions, raw, seconds = {}, {}, {}
        for name, expert in self.experts.items():
            started = time.perf_counter()
            completed, full = expert.impute(matrix.copy(order="K"), return_full=True)
            completed, full = np.asarray(completed), np.asarray(full)
            if completed.shape != matrix.shape or full.shape != matrix.shape:
                raise ValueError("Expert output shape mismatch")
            if not np.isfinite(completed).all() or not np.isfinite(full).all():
                raise ValueError(f"Nonfinite expert predictions: {name}")
            predictions[name] = completed.copy()
            predictions[name][np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
            raw[name] = full.copy()
            seconds[name] = time.perf_counter() - started
        return predictions, raw, seconds

    def predict_all(self, matrix, *, include_holdout=True):
        matrix = np.asarray(matrix)
        if not np.issubdtype(matrix.dtype, np.floating):
            matrix = matrix.astype(np.float64)
        if matrix.ndim != 2 or min(matrix.shape) == 0 or np.isinf(matrix).any():
            raise ValueError("Expected a nonempty 2D matrix with finite observations or NaN")
        predictions, raw, seconds = self._predict(matrix)
        evidence = np.isfinite(matrix)
        decisions = {"observed": fit_routing(matrix, raw["general"], raw["specialist"], evidence,
                                             observed_scales(matrix, self.normalization))}
        if include_holdout:
            heldout = observed_holdout(matrix, self.holdout_fraction, self.seed)
            reduced = matrix.copy(order="K")
            reduced[heldout] = np.nan
            if heldout.any():
                validation, _, elapsed = self._predict(reduced)
                seconds.update({f"holdout_{name}": value for name, value in elapsed.items()})
            else:
                validation = raw
            decisions["holdout"] = fit_routing(matrix, validation["general"], validation["specialist"], heldout,
                                               observed_scales(reduced, self.normalization))
        for mode, decision in decisions.items():
            predictions[f"{mode}_hard"] = predictions[decision["selected"]]
            weight = decision["specialist_weight"]
            blend = (1. - weight) * predictions["general"] + weight * predictions["specialist"]
            blend[evidence] = matrix[evidence]
            predictions[f"{mode}_blend"] = blend
        self.last_result = RoutingResult(predictions, decisions, seconds)
        return self.last_result

    def impute(self, matrix):
        result = self.predict_all(matrix, include_holdout=self.policy.startswith("holdout"))
        return result.predictions[self.policy].copy()
