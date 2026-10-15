"""Frozen-cache teacher/student error decomposition, without model selection."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from diagnose_dwip_matched_objective_gradients import require_completed_stages
from diagnose_dwip_top2_fixed_predictions import compare_fixed_predictions
from run_dwip_budgeted_queue_v1 import immutable_json, RESERVE_BYTES
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK


EXPERIMENT = "dwip_teacher_compression_gap_20260917"
STUDENT = "dwip_spectral_anchor_mix25_teacher_mean_huber1_64k_20260917"
FIXED = "dwip_top2_fixed_complementarity_20260917"
INDEPENDENT = "dwip_top2_independent_complementarity_20260917"
PANELS = ("fixed", "independent_spectral", "independent_general")
BUCKETS = ("0", "1", "2_4", "5_plus")
SOURCES = (
    "scripts/diagnose_dwip_teacher_compression_gap.py",
    "tests/test_dwip_teacher_compression_gap.py",
    "scripts/diagnose_dwip_top2_fixed_predictions.py",
    "scripts/diagnose_dwip_matched_objective_gradients.py",
    "scripts/run_dwip_continuous_spectrum.py",
    "scripts/run_dwip_budgeted_queue_v1.py",
    "scripts/run_dwip_transfer_evaluation.py",
    "scripts/run_rowcol_controlled_experiments.py",
)
TERMS = (
    "rank1_sse", "rank2_sse", "teacher_sse", "student_sse",
    "compression_sse", "cross_sum", "rank1_compression_sse",
    "rank1_cross_sum", "teacher_disagreement_sse",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def support_buckets(case):
    source, mask = case["source"], case["mask"]
    if source.ndim != 3 or source.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Require matching B x R x C source and Boolean mask")
    if torch.isinf(source).any() or (mask & torch.isfinite(source)).any():
        raise ValueError("Scored cells must be missing and source cannot contain infinity")
    observed = torch.isfinite(source)
    rows = observed.sum(2, keepdim=True).expand_as(source)[mask]
    columns = observed.sum(1, keepdim=True).expand_as(source)[mask]
    minimum = torch.minimum(rows, columns)
    return dict(zip(BUCKETS, (minimum == 0, minimum == 1,
                            (minimum >= 2) & (minimum <= 4), minimum >= 5)))


def residual_terms(target, first, second, student):
    teacher = ((first.double() + second.double()) / 2).float().double()
    truth = target.double()
    teacher_error = teacher - truth
    compression = student.double() - teacher
    first_compression = first.double() - teacher
    return {
        "rank1_sse": (first.double() - truth).square(),
        "rank2_sse": (second.double() - truth).square(),
        "teacher_sse": teacher_error.square(),
        "student_sse": (student.double() - truth).square(),
        "compression_sse": compression.square(),
        "cross_sum": 2 * teacher_error * compression,
        "rank1_compression_sse": first_compression.square(),
        "rank1_cross_sum": 2 * teacher_error * first_compression,
        "teacher_disagreement_sse": (first.double() - second.double()).square(),
    }


def add_terms(groups, case, terms, subset):
    if not subset.any():
        return
    group = groups.setdefault(case["group"], {
        "n": 0, "case_count": 0, "monitor_only": case["monitor_only"],
        **dict.fromkeys(TERMS, 0.),
    })
    if group["monitor_only"] != case["monitor_only"]:
        raise ValueError("Inconsistent monitor membership")
    group["n"] += int(subset.sum())
    group["case_count"] += 1
    for name, values in terms.items():
        group[name] += float(values[subset].sum())


def finish_group(group):
    count = group["n"]
    terms = {name: group[name] / count for name in TERMS}
    gap = terms["student_sse"] - terms["teacher_sse"]
    reconstructed = terms["compression_sse"] + terms["cross_sum"]
    control_gap = terms["rank1_sse"] - terms["teacher_sse"]
    control_reconstructed = terms["rank1_compression_sse"] + terms["rank1_cross_sum"]
    scale = max(1., *(abs(value) for value in terms.values()))
    if max(abs(gap - reconstructed), abs(control_gap - control_reconstructed)) > 1e-12 * scale:
        raise ValueError("FP64 squared-error decomposition does not close")
    return {
        "n": count, "case_count": group["case_count"], "monitor_only": group["monitor_only"],
        "rmse_fp64_residual": {label: math.sqrt(terms[f"{label}_sse"])
                               for label in ("rank1", "rank2", "teacher", "student")},
        "teacher_mse": terms["teacher_sse"],
        "student_minus_teacher_mse": gap,
        "compression_mse": terms["compression_sse"],
        "teacher_residual_cross_term": terms["cross_sum"],
        "rank1_compression_mse": terms["rank1_compression_sse"],
        "rank1_teacher_residual_cross_term": terms["rank1_cross_sum"],
        "teacher_disagreement_mse": terms["teacher_disagreement_sse"],
        "student_minus_rank1_mse": terms["student_sse"] - terms["rank1_sse"],
        "excess_compression_mse_vs_rank1": terms["compression_sse"] - terms["rank1_compression_sse"],
        "cross_term_change_vs_rank1": terms["cross_sum"] - terms["rank1_cross_sum"],
        "identity_absolute_error": max(abs(gap - reconstructed), abs(control_gap - control_reconstructed)),
    }


def summarize_groups(groups):
    scored = [group for group in groups.values() if not group["monitor_only"]]
    if not scored:
        return None
    fields = ("compression_mse", "teacher_residual_cross_term", "student_minus_teacher_mse",
              "excess_compression_mse_vs_rank1", "cross_term_change_vs_rank1", "student_minus_rank1_mse")
    nonzero = [group for group in scored if group["rmse_fp64_residual"]["rank1"] > 0]
    return {
        "non_monitor_groups": len(scored), "cells": sum(group["n"] for group in scored),
        "student_beats_rank1_groups": sum(group["student_minus_rank1_mse"] < 0 for group in scored),
        "student_beats_teacher_groups": sum(group["student_minus_teacher_mse"] < 0 for group in scored),
        "macro_rmse_fp64_residual": {
            label: float(np.mean([group["rmse_fp64_residual"][label] for group in scored]))
            for label in ("rank1", "rank2", "teacher", "student")},
        "rank1_mse_normalized_group_mean": {
            field: float(np.mean([group[field] / group["rmse_fp64_residual"]["rank1"] ** 2
                                  for group in nonzero])) if nonzero else None
            for field in fields},
        "normalization_groups": len(nonzero),
    }


def compare_student(panel, cases, first, second, student):
    if panel not in PANELS:
        raise ValueError("Only the three registered synthetic panels are permitted")
    teachers = compare_fixed_predictions(cases, first, second)
    student_check = compare_fixed_predictions(cases, first, student)
    groups, per_case = {}, {}
    slices = {bucket: {} for bucket in BUCKETS}
    for case in cases:
        name = case["id"]
        target = case["target"][case["mask"]].float()
        predictions = [payload["predictions"][name]["mean"] for payload in (first, second, student)]
        terms = residual_terms(target, *predictions)
        all_cells = torch.ones_like(target, dtype=torch.bool)
        add_terms(groups, case, terms, all_cells)
        single = {}
        add_terms(single, case, terms, all_cells)
        per_case[name] = {"group": case["group"], **finish_group(single[case["group"]])}
        for bucket, subset in support_buckets(case).items():
            add_terms(slices[bucket], case, terms, subset)
    finished = {name: finish_group(group) for name, group in groups.items()}
    support = {bucket: {name: finish_group(group) for name, group in values.items()}
               for bucket, values in slices.items()}
    return {
        "panel": panel, "case_count": len(cases),
        "cached_macro_rmse_reproduced": {
            **teachers["macro_rmse"], "student": student_check["macro_rmse"]["rank2"]},
        "summary": summarize_groups(finished), "groups": finished, "cases": per_case,
        "minimum_axis_observed_support": {
            bucket: {"summary": summarize_groups(values), "groups": values}
            for bucket, values in support.items()},
    }


def verify_inputs(inputs):
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen diagnostic input changed: {path}")


def register(root):
    queues = root.parent
    paths = {
        "fixed": queues / FIXED / "prespecification.json",
        "independent": queues / INDEPENDENT / "prespecification.json",
        "inventory": queues / STUDENT / "final_saved_model_inventory.json",
        "comparison": queues / STUDENT / "teacher_full_comparison.json",
        "manifest": queues / STUDENT / "manifest.json",
    }
    documents = {name: read_json(path) for name, path in paths.items()}
    members = documents["fixed"]["members"]
    if members != documents["independent"]["members"]:
        raise ValueError("Require the same previously frozen teachers")
    student = documents["inventory"]["winner"]
    if student["queue"] != STUDENT or student["state"] != "complete":
        raise ValueError("Require the completed fixed-selected teacher-mean student")
    expected_hashes = {}
    for document in documents.values():
        for path, digest in document.get("input_sha256", {}).items():
            canonical = str(Path(path).resolve())
            if canonical in expected_hashes and expected_hashes[canonical] != digest:
                raise ValueError(f"Conflicting provenance: {canonical}")
            expected_hashes[canonical] = digest
    inputs = {}

    def capture(path, require_prior=True):
        canonical = str(Path(path).resolve())
        actual = sha256(canonical)
        if require_prior and expected_hashes.get(canonical) != actual:
            raise ValueError(f"Input is absent from or differs from frozen provenance: {path}")
        inputs[canonical] = actual
        return str(path)

    for path in (*paths.values(), *(REPO / source for source in SOURCES)):
        capture(path, require_prior=False)
    for member in (*members, student):
        queue = queues / member["queue"]
        require_completed_stages(queue)
        if sha256(member["checkpoint"]) != member["sha256"]:
            raise ValueError("Selected checkpoint changed")
        capture(member["checkpoint"], require_prior=False)
        for directory in (queue, *(queue / f"{stage}_evaluation" for stage in ("transfer", "retention", "runtime"))):
            capture(directory / "status.json", require_prior=False)
    panels = {}
    for panel in PANELS:
        if panel == "fixed":
            case_paths = [queues / member["queue"] / "validation.pth" for member in (*members, student)]
            prediction_paths = {label: WORK / f"tabimpute-{member['queue']}-dwip1e5/validation_predictions_best.pth"
                                for label, member in zip(("rank1", "rank2", "student"), (*members, student))}
            digest, count = student["case_digest"], 72
        else:
            previous = documents["independent"]["panels"][panel]
            case_paths = [Path(path) for path in previous["case_paths"]]
            if panel == "independent_spectral":
                case_paths.append(queues / STUDENT / "holdout.pth")
            suffix = "selected" if panel == "independent_spectral" else "general"
            prediction_paths = {**previous["predictions"], "student": queues / STUDENT / f"dwip1e5_{suffix}_predictions.pth"}
            digest = documents["independent"]["expected_case_digest"][panel]
            count = documents["independent"]["case_count"][panel]
        panels[panel] = {"case_paths": [capture(path) for path in case_paths],
                         "predictions": {label: capture(path) for label, path in prediction_paths.items()},
                         "case_digest": digest, "case_count": count}
    if len({panel["case_digest"] for panel in panels.values()}) != 3:
        raise ValueError("Fixed and independent populations must differ")
    registration = {
        "utc": datetime.now(timezone.utc).isoformat(), "members": members, "student": student,
        "panels": panels, "input_sha256": inputs,
        "arithmetic": "Frozen equal FP64 teacher mean rounded to FP32. Reproduce cached FP32-residual group/macro RMSE at1e-12. Separately decompose FP64 student SSE as teacher SSE + compression SSE + 2*teacher-residual dot compression; also subtract the same decomposition for rank1.",
        "support": "Partition scored cells by min(observed row count, observed column count):0,1,2-4,5+. Observed source only; no target-dependent strata. Macro summaries exclude monitor groups, weight groups equally and normalize decomposition terms by rank1 group MSE, excluding exact-zero denominators.",
        "limits": "CPU-only existing synthetic caches, no fresh/blinded claim or statistical significance claim. Known public and aggregate synthetic outcomes do not select pairs, weights, checkpoints or priors. No optimizer, training, GPU, ensemble deployment or checkpoint creation/promotion. Preserve all existing artifacts and20GiB reserve.",
        "decision_rule": "Describe compression error and residual alignment on every registered panel and support stratum before proposing any controlled learning change; this diagnostic cannot select or promote a checkpoint.",
        "allowance_bytes": 32 * 1024**2,
    }
    immutable_json(root / "prespecification.json", registration)
    immutable_json(root / "registration_status.json", {
        "state": "registered", "evaluation_started": False,
        "prespecification_sha256": sha256(root / "prespecification.json"),
    })
    return {"state": "registered", "verified_inputs": len(inputs)}


def evaluate(root):
    path = root / "prespecification.json"
    registration = read_json(path)
    if read_json(root / "registration_status.json")["prespecification_sha256"] != sha256(path):
        raise ValueError("Require unchanged separate registration before evaluation")
    verify_inputs(registration["input_sha256"])
    panels = {}
    for panel, paths in registration["panels"].items():
        cases = None
        for case_path in paths["case_paths"]:
            population = torch.load(case_path, map_location="cpu", weights_only=True)["cases"]
            if len(population) != paths["case_count"] or case_digest(population) != paths["case_digest"]:
                raise ValueError(f"Frozen synthetic population changed: {panel}")
            cases = population
        payloads = {label: torch.load(prediction_path, map_location="cpu", weights_only=True)
                    for label, prediction_path in paths["predictions"].items()}
        panels[panel] = compare_student(panel, cases, *(payloads[label] for label in ("rank1", "rank2", "student")))
    verify_inputs(registration["input_sha256"])
    result = {"utc": datetime.now(timezone.utc).isoformat(), "prespecification_sha256": sha256(path),
              "panels": panels, "checkpoint_created_or_promoted": False, "training_started": False}
    immutable_json(root / "results.json", result)
    immutable_json(root / "status.json", {"state": "complete", "gpu_evaluation_started": False,
                                          "checkpoint_created_or_promoted": False, "utc": result["utc"]})
    return {"state": "complete", "panels": {panel: result["cached_macro_rmse_reproduced"]
                                             for panel, result in panels.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = REPO / "runs/objective_queues" / EXPERIMENT
    root.mkdir(exist_ok=True)
    if (root / "results.json").exists():
        raise ValueError("Diagnostic already complete; preserve existing evidence")
    if shutil.disk_usage(root).free < RESERVE_BYTES + 32 * 1024**2:
        raise RuntimeError("Preserve20GiB reserve plus32MiB diagnostic allowance")
    print(json.dumps(register(root) if args.register_only else evaluate(root), indent=2))


if __name__ == "__main__":
    main()
