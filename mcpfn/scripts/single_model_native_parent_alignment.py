"""Describe saved output distances to both frozen parents without inference."""

from collections import defaultdict
import math
import types

import numpy as np

import analyze_single_model_teacher_alignment as previous
from analyze_single_model_teacher_alignment import (
    ROOT, bind, cohort, digest, json, paired_cases, read, save_csv, squared_distances, write,
)


METRICS = previous.METRICS + (
    "actual_v29_mse", "actual_v29_shrunk_mse", "structured_mean_mse", "structured_mean_shrunk_mse",
)


def structured_mean(arrays):
    if "actual_v29" not in arrays or "v8" not in arrays:
        raise ValueError("Keep both actual frozen parents distinct")
    first, second = arrays["v8"], arrays["actual_v29"]
    if (first.dtype != np.float32 or second.dtype != np.float32
            or first.shape != second.shape or first.ndim != 2):
        raise ValueError("Require identical full FP32 parent grids")
    return first * np.float32(.5) + second * np.float32(.5)


def score_case(source, truth, query, arrays):
    reference = structured_mean(arrays)
    rows = previous.score_case(source, truth, query, arrays)
    for row in rows:
        mask, target = (query, truth) if row["split"] == "query" else (np.isfinite(source), source)
        prediction = arrays[row["role"]]
        for label, values in (("actual_v29", arrays["actual_v29"]), ("structured_mean", reference)):
            pooled, shrunk = squared_distances(prediction, values, target, mask)
            row[f"{label}_mse"] = pooled
            row[f"{label}_shrunk_mse"] = shrunk
    return rows


summarize = types.FunctionType(previous.summarize.__code__, globals(), "summarize", previous.summarize.__defaults__)
