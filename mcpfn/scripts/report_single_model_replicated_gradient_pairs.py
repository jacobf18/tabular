"""Describe within-cohort gradient disagreement in the independently audited doubled-draw preflight."""

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path

import numpy as np
import torch

from dwip_teacher_mean_training import verify_hashes
import preflight_single_model_replicated_decoder as preflight
from recsys_evaluation_common import REPO, sha256, write_json
from report_single_model_guard_contributions import write_csv


POLICY = ("Descriptive CPU analysis of all paired raw decoder gradients in the completed28-task preflight. "
          "Use its independently remeasured84-gradient Gram matrix and preserve every cohort and objective. "
          "Compare two independent draws conditional on the same registered recipe at the same V8 weights. "
          "Report raw gradient geometry before task clipping, Adam and joint projection, not a counterfactual "
          "optimizer result or evidence of population variance reduction. No new predictions, training, "
          "model selection, confirmation, benchmark evaluation or change to the live V18 protocol.")


def paired_geometry(gram, entries, tasks):
    gram = np.asarray(gram, dtype=np.float64)
    if (gram.shape != (len(entries), len(entries)) or not np.isfinite(gram).all()
            or not np.allclose(gram, gram.T, rtol=1e-12, atol=0)
            or [entry["index"] for entry in entries] != list(range(len(entries)))
            or len({task["name"] for task in tasks}) != len(tasks)):
        raise ValueError("Require the complete finite symmetric Gram matrix and unique ordered gradient slots")
    coordinates = {(entry["task"], entry["kind"]): entry["index"] for entry in entries}
    if len(coordinates) != len(entries):
        raise ValueError("Never duplicate an objective or overwrite its gradient")
    cohorts = {}
    for task in tasks:
        key = task["cohort"], task["replicate"]
        if key in cohorts or task["replicate"] not in (0, 1):
            raise ValueError("Require two distinct draws for every cohort")
        cohorts[key] = task
    expected, rows = set(), []
    for cohort in sorted({task["cohort"] for task in tasks}):
        if any((cohort, draw) not in cohorts for draw in (0, 1)):
            raise ValueError("Never omit a paired draw")
        first, second = (cohorts[cohort, draw] for draw in (0, 1))
        recipes = [deepcopy(task["specification"]) for task in (first, second)]
        seeds = [recipe.get("structured", recipe).pop("seed") for recipe in recipes]
        if (recipes[0] != recipes[1] or seeds[0] == seeds[1] or first["gradient_seed"] == second["gradient_seed"]
                or first["weight"] != second["weight"]):
            raise ValueError("Pair independent equally weighted draws of the same conditional recipe")
        kinds = ("composite", "rmse", "mae" if recipes[0]["domain"] == "recsys" else "common_nrmse")
        for kind in kinds:
            slots = [(task["name"], kind) for task in (first, second)]
            expected.update(slots)
            if any(slot not in coordinates for slot in slots):
                raise ValueError("Every original composite and point gradient is mandatory")
            first_index, second_index = (coordinates[slot] for slot in slots)
            first_energy, second_energy = gram[first_index, first_index], gram[second_index, second_index]
            product = gram[first_index, second_index]
            if first_energy < 0 or second_energy < 0:
                raise ValueError("Squared gradient norms cannot be negative")
            denominator = math.sqrt(first_energy) * math.sqrt(second_energy)
            if abs(product) > denominator + 1e-12 * (first_energy + second_energy):
                raise ValueError("Raw gradient products must satisfy Cauchy-Schwarz")
            mean_energy = max(0., (first_energy + second_energy + 2 * product) / 4)
            disagreement_energy = max(0., (first_energy + second_energy - 2 * product) / 4)
            average_energy = (first_energy + second_energy) / 2
            rows.append({"cohort": cohort, "kind": kind, "first_index": first_index, "second_index": second_index,
                         "first_norm": math.sqrt(first_energy), "second_norm": math.sqrt(second_energy),
                         "cosine": None if denominator == 0 else max(-1., min(1., product / denominator)),
                         "mean_norm": math.sqrt(mean_energy), "half_difference_norm": math.sqrt(disagreement_energy),
                         "mean_energy_fraction": None if average_energy == 0 else mean_energy / average_energy})
    if set(coordinates) != expected:
        raise ValueError("Never ignore an extra task or objective")
    return rows


def report(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Use CPU-only analysis of saved audited gradients")
    if output.exists():
        raise FileExistsError("Preserve earlier diagnostics")
    config = preflight.verify(preflight.OUTPUT)
    audit_path = preflight.OUTPUT / "audit.json"
    audit = preflight.read(audit_path)
    if (not audit["complete"] or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
            or audit["raw_endpoint"]["gradient_vectors"] != 84 or audit["cuda_initialized"]):
        raise ValueError("Require the complete independent preflight audit, not unaudited live records")
    verify_hashes(audit["artifact_sha256"])
    result_path, manifest_path = [preflight.OUTPUT / name for name in ("result.json", "training_batch/manifest.json")]
    result, manifest = preflight.read(result_path), preflight.read(manifest_path)
    if manifest["step"] != 1 or len(manifest["tasks"]) != 28 or len(manifest["entries"]) != 84:
        raise ValueError("Retain the entire registered preflight")
    rows = paired_geometry(result["record"]["joint_gram"], manifest["entries"], config["batch"]["tasks"])
    if len(rows) != 42:
        raise ValueError("Report all42 pairs, not just adverse examples")
    paths = [audit_path, result_path, manifest_path, preflight.OUTPUT / "config.json"]
    sources = [Path(__file__).resolve(), Path(preflight.__file__).resolve(),
               REPO / "scripts/report_single_model_guard_contributions.py", REPO / "tests/test_single_model_replicated_gradient_pairs.py"]
    specification = {"policy": POLICY, "input_sha256": {str(path): sha256(path) for path in paths},
                     "source_sha256": {str(path): sha256(path) for path in sources}}
    output.mkdir(parents=True)
    write_json(output / "config.json", specification)
    write_csv(output / "pairs.csv", rows)
    write_json(output / "result.json", {"complete": True, "pairs": rows, "raw_gradients": 84,
               "prediction_performed": False, "selection_performed": False, "training_performed": False,
               "goal_success_not_assessed": True, "config_sha256": sha256(output / "config.json")})
    lines = ["# Paired preflight gradient geometry", "", POLICY, "",
             "Mean energy fraction is a two-sample geometric identity, not a measured generalization improvement.", "",
             "| Cohort | Objective | Cosine | Mean energy / average draw energy |",
             "|---|---|---:|---:|"]
    for row in rows:
        cosine = "undefined" if row["cosine"] is None else f'{row["cosine"]:.6f}'
        energy = "undefined" if row["mean_energy_fraction"] is None else f'{row["mean_energy_fraction"]:.6f}'
        lines.append(f'| {row["cohort"]} | {row["kind"]} | {cosine} | {energy} |')
    (output / "RESULTS.md").write_text("\n".join([*lines, ""]))
    verify_hashes(specification["input_sha256"])
    verify_hashes(specification["source_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.output.resolve())
