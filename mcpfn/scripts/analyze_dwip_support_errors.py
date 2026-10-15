"""Stratify cached independent synthetic errors by observed support, without inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_dwip_fold_regression import QUEUES
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json


def support_slices(case, before, after):
    target = case["target"][case["mask"]]
    indices = case["mask"].flatten().nonzero().flatten()
    for cached in (before, after):
        if not torch.equal(cached["target"], target) or not torch.equal(cached["flat_indices"], indices):
            raise ValueError("Cached targets or locations do not match source cases")
        if cached["mean"].shape != target.shape or not torch.isfinite(cached["mean"]).all():
            raise ValueError("Invalid cached means")
    observed = torch.isfinite(case["source"])
    errors = {"original": (before["mean"].double() - target.double()).square(),
              "candidate": (after["mean"].double() - target.double()).square()}
    result = {}
    for axis, dimension in (("row", 2), ("column", 1)):
        counts = observed.sum(dimension, keepdim=True).expand_as(observed)[case["mask"]]
        for name, subset in (("0", counts == 0), ("1", counts == 1),
                             ("2_4", (counts >= 2) & (counts <= 4)), ("5_plus", counts >= 5)):
            if subset.any():
                result[f"{axis}_{name}"] = {"n": int(subset.sum()),
                    **{f"{label}_sse": float(error[subset].sum()) for label, error in errors.items()}}
    return result


def compare_cases(cases, before, after):
    expected = {case["id"] for case in cases}
    if set(before) != expected or set(after) != expected:
        raise ValueError("Prediction case identities differ")
    groups = {}
    for case in cases:
        for name, values in support_slices(case, before[case["id"]], after[case["id"]]).items():
            group = groups.setdefault(name, {}).setdefault(case["group"],
                {"n": 0, "original_sse": 0., "candidate_sse": 0., "monitor_only": case["monitor_only"]})
            for key, value in values.items():
                group[key] += value
    summaries = {}
    for name, group_values in groups.items():
        for values in group_values.values():
            for label in ("original", "candidate"):
                values[f"{label}_rmse"] = (values[f"{label}_sse"] / values["n"]) ** .5
            values["relative_rmse"] = values["candidate_rmse"] / values["original_rmse"] if values["original_rmse"] else None
        selected = [values for values in group_values.values() if not values["monitor_only"]]
        if selected:
            ratios = [values["relative_rmse"] for values in selected if values["relative_rmse"] is not None]
            summaries[name] = {"non_monitor_groups": len(selected), "cells": sum(values["n"] for values in selected),
                               "groups_improved": sum(values["candidate_sse"] < values["original_sse"] for values in selected),
                               "mean_relative_group_rmse": float(np.mean(ratios)) if ratios else None,
                               "ratio_groups": len(ratios)}
    return {"summaries": summaries, "groups": groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "runs/information_flow_diagnostics/dwip_support_errors_20260916.json")
    args = parser.parse_args()
    original = REPO / "runs/objective_queues/dwip_spectral_20260915"
    source_paths = {"spectral_cases": original / "holdout.pth",
                    "general_cases": REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth"}
    payload = {"purpose": "Generic independent-error diagnosis; no benchmark inputs, selector changes or checkpoint selection",
               "source_sha256": sha256(Path(__file__)), "input_sha256": {}, "comparisons": {},
               "cautions": ["Row and column slices overlap; do not sum their counts together.",
                            "Monitor-only groups remain separate and are excluded from slice summaries.",
                            "Relative group RMSE is descriptive, not a new selection metric or significance test."]}
    for population in ("spectral", "general"):
        cases_path = source_paths[f"{population}_cases"]
        before_path = original / f"baseline_{population}_predictions.pth"
        cases = torch.load(cases_path, map_location="cpu", weights_only=True)["cases"]
        before = torch.load(before_path, map_location="cpu", weights_only=True)["predictions"]
        for path in (cases_path, before_path):
            payload["input_sha256"][str(path)] = sha256(path)
        payload["comparisons"][population] = {}
        for label, queue in QUEUES.items():
            suffix = "selected" if population == "spectral" else "general"
            after_path = REPO / f"runs/objective_queues/{queue}/dwip1e5_{suffix}_predictions.pth"
            payload["input_sha256"][str(after_path)] = sha256(after_path)
            after = torch.load(after_path, map_location="cpu", weights_only=True)["predictions"]
            result = compare_cases(cases, before, after)
            payload["comparisons"][population][label] = result
            print(population, label, json.dumps(result["summaries"], sort_keys=True), flush=True)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
