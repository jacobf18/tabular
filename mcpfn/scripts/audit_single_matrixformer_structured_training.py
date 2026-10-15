"""CPU raw-prediction and strict-selection audit of an immutable V9 training prefix."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import single_model_structured_mixture as mixture
from train_single_matrixformer_structured_mixture import read, validate_recipe
from streamed_large_matrix_training import learning_rate


def close_values(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Reported metric keys changed")
        for key in expected:
            close_values(actual[key], expected[key])
    elif expected is None:
        if actual is not None:
            raise ValueError("Undefined metric must remain null")
    elif isinstance(expected, (int, bool)):
        if actual != expected:
            raise ValueError("Exact truth-coverage count changed")
    elif not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("Saved raw predictions do not reproduce reported metrics")


def selection_trace(reports, reference):
    if not reports or reports[0]["step"] != 0:
        raise ValueError("Require the unchanged initialization before trained candidates")
    baseline = reports[0]["student"]
    best_step, best_score = 0, float("inf")
    result = []
    for report in reports:
        reconstructed = mixture.aggregate(report["cases"], "student")
        if reconstructed != report["student"]:
            raise ValueError("Recorded case aggregates changed")
        score, ratios, eligible = mixture.selection_score(reconstructed, baseline, reference["references"])
        if eligible and score < best_score:
            best_step, best_score = report["step"], score
        if (report["score"] != score or report["ratios"] != ratios or report["eligible"] != eligible
                or report["best_step"] != best_step):
            raise ValueError("Strict registered selection decision did not reproduce")
        result.append({"step": report["step"], "score": score, "eligible": eligible, "best_step": best_step})
    return result


def ready(run, through_step):
    report = run / f"validation_{through_step}.json"
    if report.exists():
        return True
    status = read(run / "status.json")
    if status["state"] in {"training_complete", "failed", "interrupted", "registered"}:
        raise ValueError("The requested completed prefix is unavailable; never restart training from this auditor")
    try:
        os.kill(status["pid"], 0)
    except ProcessLookupError:
        if report.exists():
            return True
        raise RuntimeError("Original training handle disappeared; do not infer an observation timeout stopped it") from None
    return False


def audit(run, output, through_step):
    config = read(run / "config.json")
    validate_recipe(config)
    mixture.backend.verify_run(config, run)
    if through_step not in range(0, config["steps"] + 1, config["validate_every"]):
        raise ValueError("Require a complete registered validation boundary")
    reference = read(run / "reference.json")
    verify_hashes({str(run / "synthetic_bank.pt"): reference["bank_sha256"]})
    bank = torch.load(run / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    reports = [read(run / f"validation_{step}.json") for step in range(0, through_step + 1, config["validate_every"])]
    expected = config["validation_cases"]
    if len(bank) != len(expected) or len(bank) != 152 or len(reference["cases"]) != 152:
        raise ValueError("Require all56 legacy and96 structured validation cases")
    original_paths = [run / name for name in ("config.json", "registration.json", "reference.json", "synthetic_bank.pt")]
    original_paths += [run / f"validation_{report['step']}.json" for report in reports]
    hashes = {str(path): sha256(path) for path in original_paths}
    prediction_grids = 0
    for index, cached in enumerate(bank):
        episode = mixture.backend.restore_episode(cached, "cpu")
        specification = expected[index]
        seed = specification["structured"]["seed"] if "structured" in specification else specification["seed"]
        if episode.metadata["seed"] != seed or bool(episode.metadata.get("structured", False)) != ("structured" in specification):
            raise ValueError("Bank inputs no longer bind the original ordered seeds and groups")
        row = reference["cases"][index]
        selected_rows = [(row, ("general", "teacher"))] + [(report["cases"][index], ("student",)) for report in reports]
        coverage = []
        for record, roles in selected_rows:
            if record["index"] != index or record["seed"] != seed:
                raise ValueError("Saved score/input identity changed")
            path = Path(record["prediction_file"])
            if not path.resolve().is_relative_to(run):
                raise ValueError("Predictions must belong to this run")
            hashes[str(path)] = record["prediction_sha256"]
            verify_hashes({str(path): record["prediction_sha256"]})
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(roles):
                    raise ValueError("Unexpected saved prediction roles")
                for role in roles:
                    prediction = torch.from_numpy(saved[role])
                    if prediction.shape != episode.source.shape or not torch.isfinite(prediction).all():
                        raise ValueError("Require full finite observed and queried grids")
                    score = mixture.defined.episode_metrics(prediction, episode)
                    close_values(score, record[role])
                    prediction_grids += 1
                    if episode.metadata["domain"] == "imputation":
                        coverage.append(tuple(score["column_coverage"][key] for key in
                                              ("queried_columns", "common_nonconstant_columns")))
        if coverage and len(set(coverage)) != 1:
            raise ValueError("Truth-defined coverage differs between roles or updates")
    for role in ("general", "teacher"):
        if mixture.aggregate(reference["cases"], role) != reference[role]:
            raise ValueError("Reference aggregates changed")
    trace = selection_trace(reports, reference)
    if through_step:
        lines = (run / "training.jsonl").read_text().splitlines(keepends=True)
        rows = [json.loads(line) for line in lines if line.endswith("\n")]
        rows = [row for row in rows if row["step"] <= through_step]
    else:
        rows = []
    if [row["step"] for row in rows] != list(range(1, through_step + 1)):
        raise ValueError("Every optimizer update in the requested prefix is mandatory")
    for row in rows:
        step = row["step"]
        if row["lr"] != learning_rate(step, config["steps"], config["peak_lr"], config["warmup_steps"]):
            raise ValueError("Learning-rate schedule changed")
        microsteps = row["microsteps"]
        if [micro["domain"] for micro in microsteps] != ["imputation", "recsys"]:
            raise ValueError("Every update must retain both original training domains")
        for offset, micro in enumerate(microsteps):
            if (micro["gradient_data"] != "synthetic" or micro["seed"] != config["data_seed"] + 16 * step + offset
                    or micro["teacher"]["target"] != ("initialization", "specialist")[offset]
                    or not math.isfinite(micro["loss"])):
                raise ValueError("Seed, synthetic-only domain, teacher policy or loss changed")
        imputation, rating = microsteps
        if bool(imputation.get("structured", False)) != (step % 2 == 0):
            raise ValueError("Structured sampling fraction changed")
        if step % 2 == 0:
            case = config["training_structured_cases"][step // 2 - 1]["case"]
            if any(imputation[key] != case[key] for key in ("family", "component", "seed", "regime", "requested_missingness")):
                raise ValueError("Structured training case changed")
            if imputation["requested_shape"] != case["shape"]:
                raise ValueError("Structured requested shape changed")
        if [rating["rows"], rating["columns"]] != config["rating_training_shapes"][(step - 1) % 2]:
            raise ValueError("Original alternating rating context changed")
    verify_hashes(hashes)
    mixture.backend.verify_run(config, run)
    result = {"complete": True, "scope": f"immutable training/validation prefix through update{through_step}",
              "through_step": through_step, "full_400_update_log_and_validation_audited": through_step == 400,
              "validation_cases": len(bank), "legacy_cases": 56, "structured_cases": 96,
              "prediction_grids_rescored": prediction_grids, "selection_trace": trace,
              "research_selection_tolerance": 0, "numerical_score_reproduction_rtol": 1e-6,
              "numerical_score_reproduction_atol": 1e-7, "training_updates_replayed": False,
              "logged_training_updates_checked": len(rows), "original_artifacts_unchanged": True,
              "original_artifact_sha256": hashes, "source_sha256": sha256(Path(__file__)),
              "goal_success_not_assessed": True, "cuda_initialized": torch.cuda.is_initialized()}
    write_json(output / "audit.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "original_artifact_sha256"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-step", type=int, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require CUDA_VISIBLE_DEVICES='' for an independent CPU audit")
    run, output = args.run.resolve(), args.output.resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Write audit outputs separately from original training artifacts")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    try:
        while not ready(run, args.through_step):
            if not args.wait:
                raise RuntimeError("Requested validation prefix is not complete")
            write_json(output / "status.json", {"state": "waiting_for_completed_prefix", "pid": os.getpid(), "updated_unix": time.time()})
            time.sleep(30)
        audit(run, output, args.through_step)
        write_json(output / "status.json", {"state": "complete", "through_step": args.through_step,
                                            "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
