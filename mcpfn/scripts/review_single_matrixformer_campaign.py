"""Reaudit a completed frozen campaign without changing it, then expose every comparison."""

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from confirm_single_model_weight_average import inventory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_csv, write_json
from run_single_matrixformer_benchmarks import STAGES, require_full_audit
from train_single_matrixformer import storage_guard


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def compare(scope, key, metric, student, blend, *, higher_better=False, required=False):
    student, blend = float(student), float(blend)
    if not math.isfinite(student) or not math.isfinite(blend):
        raise ValueError("Never omit a nonfinite comparison")
    difference = student - blend
    regression = -difference if higher_better else difference
    return {"scope": scope, "key": key, "metric": metric, "student": student, "blend": blend,
            "delta": difference, "regression": regression, "higher_better": higher_better,
            "relative_change_percent": None if higher_better or blend == 0 else 100 * difference / blend,
            "strictly_improved": regression < 0, "retained": regression <= 0, "required_for_point_parity": required}


def unique_complete(records, dimensions, expected):
    keys = [tuple(row[name] for name in dimensions) for row in records]
    if len(records) != expected or len(set(keys)) != expected:
        raise ValueError("Missing, duplicate or unexpected comparison coverage")
    for row in records:
        if "complete" in row and str(row["complete"]).lower() != "true":
            raise ValueError("Never assess an incomplete summary")
    return records


def comparisons(root):
    result = []
    tabular = root / "tabular"
    dimensions = {"overall": ("suite",), "per_pattern": ("suite", "pattern"),
                  "per_dataset": ("suite", "dataset"), "per_pattern_rate": ("suite", "pattern", "missingness")}
    expected = {"overall": 2, "per_pattern": 14, "per_dataset": 54, "per_pattern_rate": 26}
    for name, keys in dimensions.items():
        for row in unique_complete(rows(tabular / f"{name}.csv"), keys, expected[name]):
            if int(row["completed_cases"]) != int(row["expected_cases"]):
                raise ValueError("Tabular comparison dropped registered cases")
            result.append(compare(f"tabular_{name}", "/".join(row[key] for key in keys), "NRMSE",
                                  row["new_nrmse"], row["previous_nrmse"], required=name in {"overall", "per_pattern"}))
    ratings = unique_complete(rows(root / "recsys/comparison.csv"), ("dataset", "role"), 8)
    expected_datasets = {"MovieLens 100K", "Netflix small", "MovieLens 1M", "MovieLens 10M"}
    if {(row["dataset"], row["role"]) for row in ratings} != {
            (dataset, role) for dataset in expected_datasets for role in ("student", "parent_same_recipe")}:
        raise ValueError("Ratings require all four datasets and separately declared preprocessing controls")
    for row in ratings:
        for metric in ("RMSE", "MAE"):
            result.append(compare(f"recsys_{row['role']}", row["dataset"], metric, row[metric], row[f"blend_{metric}"],
                                  required=row["role"] == "student"))
    for name, keys, expected_count in (("per_dataset", ("dataset",), 4), ("panels", ("dataset", "method", "budget"), 180)):
        for row in unique_complete(rows(root / "llm" / f"{name}.csv"), keys, expected_count):
            if name == "per_dataset" and (int(row["panels"]) != 45 or int(row["expected_panels"]) != 45):
                raise ValueError("Every LLM dataset requires its original 45 panels")
            if name == "panels" and (int(row["folds"]) != 10 or any(int(row[f"{role}_{metric}_folds"]) != 10
                    for role in ("student", "blend") for metric in ("r2", "rmse"))):
                raise ValueError("Every original panel requires all ten finite matched folds")
            for metric in ("r2", "rmse"):
                result.append(compare(f"llm_{name}", "/".join(row[key] for key in keys), metric.upper(),
                                      row[f"student_{metric}"], row[f"blend_{metric}"], higher_better=metric == "r2", required=True))
    return result


def decision(records):
    required = [row for row in records if row["required_for_point_parity"]]
    primary = [row for row in required if row["scope"] in {"tabular_overall", "recsys_student", "llm_per_dataset"}]
    if len(required) != 392 or len(primary) != 18:
        raise ValueError("Strict point comparison must retain the entire registered suite/pattern/panel scope")
    return {"strict_point_estimate_parity": all(row["retained"] for row in required),
            "all_primary_point_estimates_retained": all(row["retained"] for row in primary),
            "required_comparisons": len(required), "primary_comparisons": len(primary),
            "primary_regressions": sum(not row["retained"] for row in primary),
            "required_regressions": sum(not row["retained"] for row in required),
            "practical_equivalence_margin": None, "comparison_tolerance": 0,
            "uncertainty_estimated_here": False, "goal_success_not_assessed": True,
            "required_review": "Raw-prediction reaudit and strict point estimates are not a substitute for uncertainty and synthetic-only training/provenance review"}


def readonly_inputs(source, target, stage):
    target.mkdir()
    shutil.copy2(source / "manifest.json", target / "manifest.json")
    if stage in {"tabular", "llm"}:
        shutil.copy2(source / "parent_parity.json", target / "parent_parity.json")
        names = ["predictions", "cases"] if stage == "tabular" else ["predictions"]
    else:
        names = ["100K", "netflix", "1M", "10M"]
    for name in names:
        path = source / name
        if not path.is_dir():
            raise ValueError(f"Missing original audited input directory: {path}")
        (target / name).symlink_to(path.resolve(), target_is_directory=True)


def validate_campaign(campaign):
    status, audit, config = [read(campaign / name) for name in ("status.json", "campaign_audit.json", "config.json")]
    if status["state"] != "complete" or not audit["all_evaluations_complete"] or set(audit["audits"]) != set(STAGES):
        raise ValueError("Only a complete original three-suite campaign may be reviewed")
    candidate = audit["candidate"]
    if candidate != read(campaign / "candidate.json") or candidate["selected_step"] == 0 or not candidate["selected_before_benchmark_evaluation"]:
        raise ValueError("Require a single frozen nonzero synthetic-selected candidate")
    confirmation = Path(config["confirmation"])
    gate = read(confirmation / "gate.json")
    if (not gate["passed"] or gate["checkpoint_sha256"] != candidate["checkpoint_sha256"]
            or gate["selected_step"] != candidate["selected_step"] or sha256(confirmation / "gate.json") != candidate["confirmation_sha256"]):
        raise ValueError("Candidate no longer binds its independent synthetic gate")
    hashes = {str(campaign / name): sha256(campaign / name) for name in
              ("status.json", "campaign_audit.json", "config.json", "candidate.json")}
    hashes.update({candidate["checkpoint"]: candidate["checkpoint_sha256"],
                   str(confirmation / "gate.json"): candidate["confirmation_sha256"]})
    for stage in STAGES:
        directory = campaign / stage
        path = directory / "audit.json"
        if Path(audit["audits"][stage]["path"]).resolve() != path.resolve() or sha256(path) != audit["audits"][stage]["sha256"]:
            raise ValueError("Original campaign no longer binds its stage audits")
        stage_audit, manifest = read(path), read(directory / "manifest.json")
        require_full_audit(stage, stage_audit)
        if sha256(directory / "manifest.json") != stage_audit["manifest_sha256"]:
            raise ValueError("Stage manifest changed after its original audit")
        if (manifest["protocol_sha256"] != config["protocol_sha256"][stage]
                or Path(manifest["protocol"]).resolve() != Path(config["protocols"][stage]).resolve()
                or manifest["confirmation_sha256"] != candidate["confirmation_sha256"]
                or manifest["selected_step"] != candidate["selected_step"]):
            raise ValueError("Stages must use the same frozen protocols and candidate")
        candidate_paths = {str(Path(name).resolve()): digest for name, digest in manifest["candidate_sha256"].items()}
        if candidate_paths.get(str(Path(candidate["checkpoint"]).resolve())) != candidate["checkpoint_sha256"]:
            raise ValueError("Stage used a different model")
        hashes.update(manifest["candidate_sha256"])
        hashes[str(path)] = audit["audits"][stage]["sha256"]
        hashes[str(directory / "manifest.json")] = stage_audit["manifest_sha256"]
        for filename in directory.glob("*.csv"):
            hashes[str(filename)] = sha256(filename)
    verify_hashes(hashes)
    return config, candidate, hashes


def markdown(records, verdict):
    lines = ["# Single-checkpoint frozen benchmark review", "", "All numbers are recomputed from the original saved predictions without changing the source campaign.", "",
             f"Strict required point-estimate comparisons: **{verdict['required_comparisons']}**; regressions: **{verdict['required_regressions']}**.",
             f"All required point estimates retained: **{verdict['strict_point_estimate_parity']}**. This alone does not establish goal completion.", "",
             "No equivalence margin or rounded-number tolerance is applied. Uncertainty and training provenance require separate review.", "",
             "| Scope | Dataset | Metric | Single checkpoint | Frozen blend | Delta |", "|---|---|---|---:|---:|---:|"]
    for row in records:
        if row["scope"] in {"tabular_overall", "recsys_student", "llm_per_dataset"}:
            lines.append(f"| {row['scope']} | {row['key']} | {row['metric']} | {row['student']:.10g} | {row['blend']:.10g} | {row['delta']:+.10g} |")
    lines.extend(["", "Complete full-precision comparisons, including every missingness pattern, rate, dataset, LLM panel and preprocessing control, are in `comparisons.csv`.",
                  "The regenerated per-case and per-fold tables remain under `tabular/` and `llm/`. No failed comparison is excluded.", ""])
    return "\n".join(lines)


def review(campaign, output):
    if output.resolve().is_relative_to(campaign.resolve()) or campaign.resolve().is_relative_to(output.resolve()):
        raise ValueError("Review output must be separate from the unchanged source campaign")
    config, candidate, hashes = validate_campaign(campaign)
    evaluators = {stage: importlib.import_module(f"evaluate_single_matrixformer_{stage}") for stage in STAGES}
    sources = {Path(__file__).resolve()}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    registration = {"campaign": str(campaign), "candidate": candidate, "source_sha256": {str(path): sha256(path) for path in sources},
                    "input_sha256": hashes, "training_performed": False, "checkpoint_selection_performed": False,
                    "inference_performed": False, "original_artifacts_modified": False,
                    "comparison_policy": "No tolerance or retrospective equivalence margin; all 18 primary and 374 pattern/panel point estimates must retain the blend"}
    write_json(output / "config.json", registration)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time()})
    checkpoint = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=True)
    original = torch.load(evaluators["tabular"].GENERAL, map_location="cpu", weights_only=True)["model"]
    model_inventory = inventory(checkpoint["model"], original)
    if model_inventory["tensors"] != 254 or model_inventory["elements"] != 77998688 or "teachers" in checkpoint:
        raise ValueError("Require one original-size network, not an enlarged inference ensemble")
    del checkpoint, original
    for stage, evaluator in evaluators.items():
        write_json(output / "status.json", {"state": "reauditing", "stage": stage, "pid": os.getpid(), "updated_unix": time.time()})
        readonly_inputs(campaign / stage, output / stage, stage)
        protocol = evaluator.load_protocol(Path(config["protocols"][stage]))
        evaluator.audit(output / stage, protocol)
        require_full_audit(stage, read(output / stage / "audit.json"))
    records = comparisons(output)
    verdict = decision(records)
    verify_hashes(hashes)
    verify_hashes(registration["source_sha256"])
    if torch.cuda.is_initialized():
        raise ValueError("This read-only review must not initialize CUDA")
    write_csv(output / "comparisons.csv", records)
    (output / "RESULTS.md").write_text(markdown(records, verdict))
    write_json(output / "result.json", {**verdict, "candidate": candidate, "model_inventory": model_inventory,
        "all_raw_predictions_reaudited": True, "original_campaign_hashes_unchanged": True,
        "config_sha256": sha256(output / "config.json"), "comparisons_sha256": sha256(output / "comparisons.csv"),
        "cuda_initialized": False, "complete": True})
    print(json.dumps(verdict), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Require CUDA_VISIBLE_DEVICES='' for a CPU-only frozen-result review")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(4)
        storage_guard(output)
        review(arguments.campaign.resolve(), output)
        write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
