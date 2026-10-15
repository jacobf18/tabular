"""Audit and compare only NRMSE for the frozen MissBench/UCI checkpoint rerun."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from audit_dwip_missbench_uci import load_original
from evaluate_dwip_missbench_uci import PATTERNS, REPO, sha256, write_json
from recsys_evaluation_common import write_csv


DEFAULT_REFERENCE = REPO / "runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_missbench_uci_20260918"


def aggregate(reference, completed, dimensions):
    groups = defaultdict(list)
    for row in reference:
        groups[tuple(row[key] for key in dimensions)].append(row)
    summaries = []
    for key, expected in sorted(groups.items()):
        old_by_dataset, new_by_dataset = defaultdict(list), defaultdict(list)
        for row in expected:
            old_by_dataset[row["dataset"]].append(float(row["nrmse_original"]))
            if row["case_id"] in completed:
                new_by_dataset[row["dataset"]].append(completed[row["case_id"]]["new_nrmse"])
        count = sum(len(values) for values in new_by_dataset.values())
        complete = count == len(expected)
        previous = float(np.mean([np.mean(values) for values in old_by_dataset.values()]))
        current = float(np.mean([np.mean(values) for values in new_by_dataset.values()])) if complete else None
        summaries.append({
            **dict(zip(dimensions, key)), "previous_nrmse": previous, "new_nrmse": current,
            "delta_nrmse": current - previous if complete else None,
            "change_percent": 100 * (current - previous) / previous if complete and previous != 0 else None,
            "completed_cases": count, "expected_cases": len(expected), "datasets": len(old_by_dataset),
            "complete": complete,
        })
    return summaries


def write_report(output, summaries, expected, completed, failed):
    lines = ["# Larger-matrix checkpoint: MissBench and UCI NRMSE", "",
             f"Audited {completed}/{expected} cases; {failed} failed cases.",
             "Only MatrixFormer is rerun. Baselines, frozen inputs/masks, preprocessing, and inference geometry are unchanged.",
             "No row train/test split; no causal inference benchmarks are run.",
             "NRMSE uses the original masked-target column standard deviations and constant-column rules.",
             "Cases are averaged within each dataset, then datasets are weighted equally. Negative changes are improvements.",
             "Incomplete pattern groups are not assigned a headline score.", ""]
    for suite, title in (("missbench", "MissBench: 40% missingness"), ("uci", "UCI: rates 10–50%, ten repeats")):
        lines += [f"## {title}", "", "| Pattern | Previous NRMSE | New NRMSE | Change | Cases |",
                  "|---|---:|---:|---:|---:|"]
        group = sorted((row for row in summaries if row["suite"] == suite), key=lambda row: PATTERNS.index(row["pattern"]))
        for row in group:
            if row["complete"]:
                change = "—" if row["change_percent"] is None else f"{row['change_percent']:+.2f}%"
                lines.append(f"| {row['pattern']} | {row['previous_nrmse']:.6f} | {row['new_nrmse']:.6f} | "
                             f"{change} | {row['completed_cases']}/{row['expected_cases']} |")
            else:
                lines.append(f"| {row['pattern']} | {row['previous_nrmse']:.6f} | pending | — | "
                             f"{row['completed_cases']}/{row['expected_cases']} |")
        lines.append("")
    lines += ["Per-case, per-dataset, per-pattern/rate, and supplementary per-suite results are in the adjacent comparison CSVs.",
              "Every completed new prediction and archived parent prediction is independently rescored with the original suite's NRMSE function.",
              "Coverage is reported explicitly; no failed or missing case is silently treated as a successful comparison.",
              "The manuscript and its historical tables are not modified by this rerun."]
    (output / "NRMSE_COMPARISON.md").write_text("\n".join(lines) + "\n")


def compare(evaluation, reference, allow_partial=False):
    reference_manifest = json.loads((reference / "manifest.json").read_text())["protocol"]
    current_manifest = json.loads((evaluation / "manifest.json").read_text())["protocol"]
    changed_keys = {"checkpoint", "checkpoint_sha256", "config_sha256"}
    for key in set(reference_manifest) | set(current_manifest):
        if key not in changed_keys and reference_manifest.get(key) != current_manifest.get(key):
            raise ValueError(f"Evaluation protocol changed: {key}")
    campaign = json.loads((evaluation / "campaign.json").read_text())
    for path, expected_hash in campaign["source_sha256"].items():
        if sha256(Path(path)) != expected_hash:
            raise ValueError(f"Evaluation source changed: {path}")
    if sha256(Path(current_manifest["checkpoint"])) != current_manifest["checkpoint_sha256"]:
        raise ValueError("Evaluation checkpoint changed")
    config = json.loads((Path(current_manifest["checkpoint"]).parent / "config.json").read_text())
    if sha256(Path(config["parent_checkpoint"])) != reference_manifest["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint is not the new checkpoint's parent")
    previous = list(csv.DictReader((reference / "per_case_metrics.csv").open()))
    if len(previous) != len(reference_manifest["cases"]) or any(row["status"] != "ok" for row in previous):
        raise ValueError("Incomplete reference evaluation")
    scorer_paths = {"missbench": REPO / "benchmark/plot_options.py", "uci": REPO / "benchmark/uci/plot_uci_negative_rmse.py"}
    scorers = {suite: load_original(path, "compute_normalized_rmse_columnwise") for suite, path in scorer_paths.items()}
    completed, failures, missing = {}, [], []
    maximum_difference = 0.0
    artifacts = {}
    for old in previous:
        case_path = evaluation / "cases" / (old["case_id"] + ".json")
        if not case_path.exists():
            missing.append(old["case_id"])
            continue
        current = json.loads(case_path.read_text())
        if current["status"] != "ok":
            failures.append({"case_id": old["case_id"], "error": current.get("error")})
            continue
        source = Path(old["source_directory"])
        for filename, field in (("missing.npy", "missing_sha256"), ("true.npy", "true_sha256")):
            if not (sha256(source / filename) == old[field] == current[field]):
                raise ValueError(f"Input or mask changed: {old['case_id']}")
        observed = np.load(source / "missing.npy", allow_pickle=False)
        truth = np.load(source / "true.npy", allow_pickle=False)
        mask = np.isnan(observed)
        for root, record in ((reference, old), (evaluation, current)):
            prediction_path = root / record["prediction_path"]
            prediction_hash = sha256(prediction_path)
            if prediction_hash != record["prediction_sha256"]:
                raise ValueError(f"Prediction hash mismatch: {prediction_path}")
            with np.load(prediction_path, allow_pickle=False) as stored:
                values = stored["missing_values"]
                if (tuple(stored["shape"]) != truth.shape or values.shape != (int(mask.sum()),)
                        or not np.isfinite(values).all()):
                    raise ValueError(f"Invalid masked predictions: {prediction_path}")
                prediction = observed.astype(values.dtype)
                prediction[mask] = values
            independent = float(scorers[old["suite"]](truth, prediction, mask))
            reported = float(record["nrmse_original"])
            np.testing.assert_allclose(independent, reported, rtol=5e-7, atol=1e-7)
            maximum_difference = max(maximum_difference, abs(independent - reported))
            artifacts[str(prediction_path)] = prediction_hash
        before, after = float(old["nrmse_original"]), float(current["nrmse_original"])
        completed[old["case_id"]] = {
            **{key: old[key] for key in ("case_id", "suite", "dataset", "pattern", "missingness", "repeat")},
            "previous_nrmse": before, "new_nrmse": after, "delta_nrmse": after - before,
            "change_percent": 100 * (after - before) / before if before != 0 else None,
        }
    if not completed:
        raise ValueError("No completed cases to compare")
    outputs = {
        "per_pattern": ("suite", "pattern"), "per_pattern_rate": ("suite", "pattern", "missingness"),
        "per_dataset": ("suite", "dataset"), "per_dataset_pattern": ("suite", "dataset", "pattern"),
        "per_suite": ("suite",),
    }
    write_csv(evaluation / "comparison_per_case.csv", list(completed.values()))
    summaries = {}
    for name, dimensions in outputs.items():
        summaries[name] = aggregate(previous, completed, dimensions)
        write_csv(evaluation / f"comparison_{name}.csv", summaries[name])
    write_report(evaluation, summaries["per_pattern"], len(previous), len(completed), len(failures))
    audit = {
        "complete": not failures and not missing, "audited_cases": len(completed), "expected_cases": len(previous),
        "failed": failures, "pending_cases": missing, "same_frozen_inputs": True, "same_protocol_except_checkpoint": True,
        "maximum_original_scorer_difference": maximum_difference,
        "reference_manifest_sha256": sha256(reference / "manifest.json"),
        "evaluation_manifest_sha256": sha256(evaluation / "manifest.json"),
        "comparison_script_sha256": sha256(Path(__file__)),
        "scorer_source_sha256": {str(path): sha256(path) for path in scorer_paths.values()},
    }
    write_json(evaluation / "comparison_audit.json", audit)
    write_json(evaluation / "comparison_predictions_sha256.json", artifacts)
    print(json.dumps({key: audit[key] for key in ("complete", "audited_cases", "expected_cases", "maximum_original_scorer_difference")}))
    if not allow_partial and not audit["complete"]:
        raise RuntimeError("Comparison incomplete; see comparison_audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--allow-partial", action="store_true")
    arguments = parser.parse_args()
    compare(arguments.evaluation, arguments.reference, arguments.allow_partial)
