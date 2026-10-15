"""Independently reconstruct applied teacher-mean terms from every saved probe task."""

from collections import Counter
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch

import preflight_single_model_native_column_mean as probe
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = probe.ROOT / "native_column_mean_preflight_v1_mean_audit"
CONFIG_SHA256 = "216fa8bf6745f7bc9f37c06ac89b720dc4d1ebca153ec116a3c576360399f0c2"
POLICY = (
    "Read-only CPU reconstruction of teacher-mean Huber terms on all28 original probe tasks, "
    "all columns and both saved endpoints. Independently compute count-shrunk equal-column "
    "weights only for the two native tasks; preserve the original uniform mean20 term on all24 "
    "structured tasks and both ratings. Compare native coefficients and reweighted pure-mean "
    "terms against the registered float32 implementation. Retain zero-query, singleton and "
    "constant columns and adverse changes. These are normalized-network teacher distances, "
    "not truth accuracy, original-unit rating scores, full composite NLL/CDF loss, autograd "
    "replay or a new feasibility/selection gate. No inference, gradients, new seeds, training, "
    "qualification, real data or changed original artifacts."
)


def huber(residual):
    absolute = np.abs(residual)
    return np.where(absolute <= 1., .5 * residual ** 2, absolute - .5)


def describe(target, mask, teacher, predictions, *, native):
    if (target.ndim != 3 or target.shape[0] != 1 or mask.shape != target.shape or mask.dtype != torch.bool
            or teacher.shape != target.shape or set(predictions) != {"before", "candidate"}
            or not mask.any() or target.device.type != "cpu" or mask.device.type != "cpu"
            or teacher.device.type != "cpu" or any(value.shape != target.shape for value in predictions.values())
            or not torch.isfinite(target[mask]).all() or not torch.isfinite(teacher[mask]).all()
            or any(not np.isfinite(value[mask.numpy()]).all() for value in predictions.values())):
        raise ValueError("Use complete finite queried CPU grids with both original endpoints")
    labels, selected, means = target[0].double().numpy(), mask[0].numpy(), teacher[0].double().numpy()
    queries = int(selected.sum())
    queried_columns = int(selected.any(axis=0).sum())
    residuals = {role: prediction[0].astype(np.float64) - means for role, prediction in predictions.items()}
    actual_weights = probe.retention.column_coefficients(target, mask)[0].numpy() if native else None
    rows = []
    for column in range(target.shape[-1]):
        chosen = selected[:, column]
        count = int(chosen.sum())
        variance = float(np.var(labels[chosen, column], ddof=0)) if count else None
        denominator = max((count * variance + 1.) / (count + 1.), 1e-6) if count else None
        coefficient = queries / (queried_columns * count * denominator) if native and count else 1. if count else 0.
        if native and (not np.allclose(actual_weights[chosen, column], np.float32(coefficient), rtol=1e-6, atol=0.)
                or np.any(actual_weights[~chosen, column] != 0.)):
            raise ValueError("Every coefficient must match the original implementation without changing coverage")
        row = {"column": column, "queries": count, "variance": variance, "shrunk_variance": denominator,
            "applied_coefficient": coefficient, "single_query": count == 1, "constant_multi_query": count > 1 and variance == 0.}
        for role in predictions:
            summed = float(np.sum(huber(residuals[role][chosen, column])))
            row[role + "_uniform_mean20_contribution"] = 20. * summed / queries
            row[role + "_applied_mean20_contribution"] = 20. * summed * coefficient / queries
        row["applied_change"] = row["candidate_applied_mean20_contribution"] - row["before_applied_mean20_contribution"]
        rows.append(row)
    values = {role + "_" + weighting + "_mean20": math.fsum(row[role + "_" + weighting + "_mean20_contribution"] for row in rows)
        for role in predictions for weighting in ("uniform", "applied")}
    if native:
        for role, prediction in predictions.items():
            point = torch.from_numpy(prediction)[mask].float()
            teacher_values = teacher[mask].float()
            unweighted = 20. * torch.nn.functional.huber_loss(point, teacher_values, delta=1., reduction="none")
            compiled = probe.retention.reweight_teacher_mean(unweighted, point, teacher_values,
                torch.from_numpy(actual_weights)[mask[0]])
            if not math.isclose(float(compiled.mean()), values[role + "_applied_mean20"], rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError("The saved-grid independent native mean term must match the registered callback")
    values.update(queries=queries, columns=target.shape[-1], queried_columns=queried_columns,
        applied_change=values["candidate_applied_mean20"] - values["before_applied_mean20"])
    return rows, values


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized() or OUTPUT.exists():
        raise ValueError("Use one new CPU-only supplementary audit")
    if shutil.disk_usage(probe.ROOT).free < 18 * 1024**3 + 8 * 1024**2:
        raise OSError("Preserve18 GiB plus8 MiB supplementary audit capacity")
    config = probe.read(probe.OUTPUT / "config.json")
    verify_hashes({str(probe.OUTPUT / "config.json"): CONFIG_SHA256})
    if (probe.read(probe.OUTPUT / "registration.json")["config_sha256"] != CONFIG_SHA256
            or config["native_teacher_mean"] != probe.retention.MEAN_SPECIFICATION
            or probe.read(probe.OUTPUT / "status.json")["state"] != "complete"
            or not probe.read(probe.OUTPUT / "audit.json")["complete"]
            or not probe.read(probe.OUTPUT / "retention_audit.json")["complete"]):
        raise ValueError("Require the original complete independent probe audits before this descriptive recheck")
    tasks = probe.read(probe.OUTPUT / "training_batch/manifest.json")["tasks"]
    if len(tasks) != 28 or len(config["batch"]["tasks"]) != 28:
        raise ValueError("Never narrow the original task mixture")
    endpoints = {role: probe.read(probe.OUTPUT / f"{role}_predictions.json")["rows"] for role in ("before", "candidate")}
    if any(len(rows) != 28 or [row["index"] for row in rows] != list(range(28)) for rows in endpoints.values()):
        raise ValueError("Preserve both complete ordered endpoint manifests")
    paths = [probe.OUTPUT / name for name in ("config.json", "registration.json", "audit.json", "retention_audit.json",
        "status.json", "gate.json", "result.json", "training_batch/manifest.json", "before_predictions.json", "candidate_predictions.json")]
    inputs = {str(path): sha256(path) for path in paths}
    for task in tasks:
        inputs[task["artifact"]] = task["sha256"]
    for endpoint in endpoints.values():
        for record in endpoint:
            inputs[record["prediction"]] = record["sha256"]
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])
    all_columns, task_rows = [], []
    for registered, saved in zip(config["batch"]["tasks"], tasks):
        payload = torch.load(saved["artifact"], map_location="cpu", weights_only=False, mmap=True)
        if payload["task"] != registered or any(saved[key] != value for key, value in registered.items()):
            raise ValueError("Keep every actual registered task and its cached teacher")
        episode = payload["episode"]
        metadata = episode["metadata"]
        if metadata["gradient_data"] != "synthetic":
            raise ValueError("Never use real data in this objective diagnosis")
        reference_role = probe.retention.reference_role(metadata)
        native = reference_role == "general"
        predictions = {}
        for role, endpoint in endpoints.items():
            record = endpoint[registered["index"]]
            if record["task"] != registered["name"]:
                raise ValueError("Saved predictions must match the complete original task identity")
            with np.load(record["prediction"], allow_pickle=False) as archive:
                predictions[role] = archive["prediction"]
        rows, values = describe(episode["target"], episode["mask"], payload["teacher"], predictions, native=native)
        identity = {"task_index": registered["index"], "task": registered["name"], "seed": metadata["seed"],
            "cohort": registered["cohort"], "task_weight": registered["weight"], "native_reweighting": native}
        all_columns.extend({**identity, **row} for row in rows)
        task_rows.append({**identity, **values, "mixture_weighted_change": registered["weight"] * values["applied_change"]})
    if Counter(row["native_reweighting"] for row in task_rows) != {True: 2, False: 26}:
        raise ValueError("Reweight exactly both native tasks, retaining all24 structured and both ratings")
    verify_hashes(inputs)
    if torch.cuda.is_initialized():
        raise ValueError("No CUDA initialization is permitted")
    sources = {str(path): sha256(path) for path in (Path(__file__).resolve(), REPO / "tests/test_single_model_native_mean_probe_audit.py")}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"policy": POLICY, "input_sha256": inputs, "source_sha256": sources})
    save_csv(OUTPUT / "columns.csv", all_columns)
    save_csv(OUTPUT / "tasks.csv", task_rows)
    summary = {"complete": True, "policy": POLICY, "tasks": 28, "native_reweighted_tasks": 2,
        "unchanged_structured_tasks": 24, "unchanged_rating_tasks": 2, "columns": len(all_columns),
        "adverse_teacher_mean_tasks": sum(row["applied_change"] > 0 for row in task_rows),
        "native_teacher_mean_tasks": [row for row in task_rows if row["native_reweighting"]],
        "new_gate": False, "model_inference_performed": False, "full_composite_loss_replayed": False,
        "autograd_replayed": False, "cuda_initialized": False, "config_sha256": sha256(OUTPUT / "config.json"),
        "artifact_sha256": {str(OUTPUT / name): sha256(OUTPUT / name) for name in ("columns.csv", "tasks.csv")}}
    write_json(OUTPUT / "audit.json", summary)
    print({key: value for key, value in summary.items() if key not in ("policy", "artifact_sha256", "native_teacher_mean_tasks")})


if __name__ == "__main__":
    main()
