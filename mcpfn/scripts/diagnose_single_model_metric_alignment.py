"""Compare synthetic pooled RMSE, frozen column NRMSE and teacher quality on CPU."""

import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from audit_dwip_missbench_uci import load_original
from dwip_teacher_mean_training import control_recipe, initialize_student, verify_hashes
from evaluate_single_matrixformer_tabular import SCORERS, checked_score
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_csv, write_json
from screen_single_matrixformer_initializers import restore_episode
from single_model_distillation import blend_predictions
from streamed_large_matrix_training import STREAM_OPTIONS
from tabimpute.streamed_inference import streamed_point_predictions
from train_single_matrixformer import GENERAL, SPECIALIST


ROOT = REPO / "runs/single_model_goal_20261005"
CASE_INDICES = tuple(range(24))
METRICS = ("pooled_rmse", "missbench_nrmse", "uci_nrmse")
ROLES = ("general", "specialist", "observed_blend", "student_v2")


def read(path):
    return json.loads(Path(path).read_text())


def validate_episode(episode):
    shape = tuple(episode.source.shape)
    metadata = episode.metadata
    if (len(shape) != 3 or shape[0] != 1 or episode.target.shape != episode.source.shape
            or episode.mask.shape != episode.source.shape or episode.mask.dtype != torch.bool
            or not episode.mask.any() or metadata.get("gradient_data") != "synthetic"
            or metadata.get("domain") != "imputation"
            or shape[1:] != (metadata.get("rows"), metadata.get("columns"))
            or not torch.isnan(episode.source[episode.mask]).all()):
        raise ValueError("Require each full, aligned, synthetic imputation context")
    expected = (80, 240) if metadata["transposed"] else (240, 80)
    anchor = metadata["prior"] in ("anchor_latent_mcar", "anchor_latent_profile_mixed")
    if (not anchor and shape[1:] != expected) or (anchor and (min(shape[1:]) < 1
            or max(shape[1:]) > 240 or shape[1] * shape[2] > 240 * 80)):
        raise ValueError("Require the full small spectral or intrinsic-size anchor context")


def score_queries(truth, prediction, mask, scorers):
    if (truth.ndim != 2 or prediction.shape != truth.shape or mask.shape != truth.shape
            or mask.dtype != np.bool_ or not mask.any()):
        raise ValueError("Require a nonempty, aligned two-dimensional synthetic query mask")
    missing = np.where(mask, np.nan, 0.)
    error = prediction[mask].astype(np.float64) - truth[mask].astype(np.float64)
    result = {"pooled_rmse": float(np.sqrt(np.mean(error**2))), "queries": int(mask.sum())}
    for suite, original in scorers.items():
        score, diagnostics = checked_score(truth, prediction, missing, suite, original)
        result[f"{suite}_nrmse"] = score
        result[f"{suite}_scored_columns"] = diagnostics["nrmse_scored_columns"]
        result[f"{suite}_skipped_constant_columns"] = diagnostics["nrmse_skipped_constant_columns"]
    scales, common_scores = [], []
    for column in range(truth.shape[1]):
        selected = mask[:, column]
        if not selected.any():
            continue
        scale = float(np.std(truth[selected, column]))
        scales.append(scale)
        if scale >= 1e-3:
            errors = prediction[selected, column].astype(np.float64) - truth[selected, column].astype(np.float64)
            common_scores.append(float(np.sqrt(np.mean(errors**2)) / scale))
    result.update(queried_columns=len(scales), target_std_min=float(min(scales)),
                  target_std_median=float(np.median(scales)), target_std_max=float(max(scales)),
                  common_nonconstant_columns=len(common_scores),
                  common_nonconstant_nrmse=float(np.mean(common_scores)) if common_scores else None)
    return result


def summarize(cases):
    if not cases or len({case["seed"] for case in cases}) != len(cases):
        raise ValueError("Require nonempty distinct synthetic cases")
    groups = defaultdict(list)
    for case in cases:
        if case["gradient_data"] != "synthetic" or set(case["scores"]) != set(ROLES):
            raise ValueError("Only complete matched synthetic predictions may enter this diagnostic")
        if any(not np.isfinite(case["scores"][role][metric]) or case["scores"][role][metric] < 0
               for role in ROLES for metric in METRICS):
            raise ValueError("Require finite nonnegative synthetic metrics")
        groups["ALL"].append(case)
        groups[case["prior"]].append(case)
    rows = []
    for family, selected in sorted(groups.items()):
        for role in ROLES:
            row = {"family": family, "role": role, "cases": len(selected)}
            for metric in METRICS:
                values = [case["scores"][role][metric] for case in selected]
                parents = [case["scores"]["general"][metric] for case in selected]
                row[metric] = float(np.mean(values))
                row[f"{metric}_wins_vs_general"] = sum(value < parent - 1e-10 for value, parent in zip(values, parents))
                row[f"{metric}_losses_vs_general"] = sum(value > parent + 1e-10 for value, parent in zip(values, parents))
            for suite in ("missbench", "uci"):
                flips = 0
                for case in selected:
                    parent, candidate = case["scores"]["general"], case["scores"][role]
                    rmse_delta = candidate["pooled_rmse"] - parent["pooled_rmse"]
                    nrmse_delta = candidate[f"{suite}_nrmse"] - parent[f"{suite}_nrmse"]
                    flips += abs(rmse_delta) > 1e-10 and abs(nrmse_delta) > 1e-10 and rmse_delta * nrmse_delta < 0
                row[f"{suite}_opposite_direction_cases"] = flips
            rows.append(row)
    return rows


def register(output):
    confirmation = ROOT / "mixed_distill_v2_confirmation"
    source_config = read(ROOT / "mixed_distill_v2/config.json")
    reference = read(confirmation / "reference.json")
    checkpoint = ROOT / "mixed_distill_v2/best_validation_checkpoint.pth"
    gate = read(confirmation / "gate.json")
    if (read(confirmation / "status.json")["state"] != "complete" or not gate["passed"]
            or gate["checkpoint_sha256"] != sha256(checkpoint)
            or gate["bank_sha256"] != reference["bank_sha256"] or gate["cases"] != 46):
        raise ValueError("Require the completed synthetic-confirmed fixed v2 checkpoint")
    inputs = {str(confirmation / "synthetic_bank.pt"): reference["bank_sha256"],
              str(confirmation / "reference.json"): sha256(confirmation / "reference.json"),
              str(confirmation / "student.json"): sha256(confirmation / "student.json"),
              str(confirmation / "gate.json"): sha256(confirmation / "gate.json"),
              str(ROOT / "mixed_distill_v2/config.json"): sha256(ROOT / "mixed_distill_v2/config.json"),
              **source_config["input_sha256"], str(checkpoint): gate["checkpoint_sha256"]}
    verify_hashes(inputs)
    bank = torch.load(confirmation / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    case_shapes = {}
    for index in CASE_INDICES:
        episode = restore_episode(bank[index], "cpu")
        validate_episode(episode)
        case_shapes[str(index)] = list(episode.source.shape[1:])
    del bank
    checkpoints = {"general": str(GENERAL), "specialist": str(SPECIALIST), "student_v2": str(checkpoint)}
    sources = {Path(__file__).resolve(), *SCORERS.values(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"bank": str(confirmation / "synthetic_bank.pt"), "case_indices": list(CASE_INDICES),
              "case_shapes": case_shapes,
              "checkpoints": checkpoints, "checkpoint_sha256": {path: sha256(Path(path)) for path in checkpoints.values()},
              "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
              "precision": "FP32 CPU, four threads; no CUDA, no optimizer updates",
              "scope": "The same first 24 full small-imputation cases: 18 spectral 240x80 or 80x240, plus six intrinsic-size anchor cases; no cropping or normalization refitting",
              "repair": "The first attempt stopped after six cases because its shape guard incorrectly excluded intrinsic-size anchor priors; all original case indices, checkpoints and metrics are unchanged",
              "metric": "Frozen MissBench and UCI masked-target column NRMSE formulas on the same synthetic supervision mask as pooled RMSE",
              "query_mask": "Supervision queries only, not all unobserved cells; scoring-only missing arrays are never fed to a model",
              "selection_performed": False, "training_performed": False, "real_benchmark_data_used": False,
              "limitations": "Reused completed synthetic confirmation bank and FP32 CPU small contexts; diagnostic only, not independent confirmation or large-context BF16 performance"}
    write_json(output / "config.json", config)
    for source in sources:
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return config


@torch.no_grad()
def run(config, output):
    if config["case_indices"] != list(CASE_INDICES):
        raise ValueError("Do not replace the predeclared case set")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    verify_hashes(config["checkpoint_sha256"])
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True)
    stored_reference = read(ROOT / "mixed_distill_v2_confirmation/reference.json")["cases"]
    stored_student = read(ROOT / "mixed_distill_v2_confirmation/student.json")["cases"]
    models = {}
    for role, checkpoint in config["checkpoints"].items():
        arguments = control_recipe()
        arguments.init_model_checkpoint = checkpoint
        model, optimizer, scheduler, criterion = initialize_student(arguments, torch.device("cpu"))
        del optimizer, scheduler
        models[role] = model.eval().requires_grad_(False)
    scorers = {suite: load_original(path, "compute_normalized_rmse_columnwise") for suite, path in SCORERS.items()}
    cases, flat_rows = [], []
    for index in config["case_indices"]:
        started = time.monotonic()
        episode = restore_episode(bank[index], "cpu")
        validate_episode(episode)
        if list(episode.source.shape[1:]) != config["case_shapes"][str(index)]:
            raise ValueError("Context dimensions changed after registration")
        prediction = {role: streamed_point_predictions(model, episode.source, criterion, **STREAM_OPTIONS)
                      for role, model in models.items()}
        prediction["observed_blend"], routing = blend_predictions(episode.source, prediction["general"], prediction["specialist"])
        arrays = {role: value[0].numpy() for role, value in prediction.items()}
        scores = {role: score_queries(episode.target[0].numpy(), arrays[role], episode.mask[0].numpy(), scorers) for role in ROLES}
        previous = {"general": stored_reference[index]["general"]["rmse"],
                    "observed_blend": stored_reference[index]["teacher"]["rmse"],
                    "student_v2": stored_student[index]["student"]["rmse"]}
        if stored_reference[index]["seed"] != episode.metadata["seed"] or stored_student[index]["seed"] != episode.metadata["seed"]:
            raise ValueError("Stored BF16 comparisons must refer to the same synthetic case")
        path = output / "predictions" / f"{index:03d}.npz"
        path.parent.mkdir(exist_ok=True)
        atomic_npz(path, **arrays)
        case = {"index": index, **episode.metadata, "routing": routing, "scores": scores,
                "stored_bf16_pooled_rmse": previous,
                "fp32_minus_bf16_pooled_rmse": {role: scores[role]["pooled_rmse"] - value for role, value in previous.items()},
                "prediction_sha256": sha256(path), "seconds": time.monotonic() - started}
        cases.append(case)
        write_json(output / "cases" / f"{index:03d}.json", case)
        flat_rows.extend({"index": index, "seed": case["seed"], "prior": case["prior"], "role": role, **scores[role]} for role in ROLES)
        write_json(output / "status.json", {"state": "diagnosing", "completed": len(cases), "expected": len(CASE_INDICES),
                                            "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case": index, "prior": case["prior"], "seconds": case["seconds"]}), flush=True)
    for role, model in models.items():
        original = torch.load(config["checkpoints"][role], map_location="cpu", weights_only=True)["model"]
        if any(not torch.equal(value, original[name]) for name, value in model.state_dict().items()):
            raise ValueError("Diagnostic must not update model state")
    rows = summarize(cases)
    write_csv(output / "per_case.csv", flat_rows)
    write_csv(output / "by_family.csv", rows)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    verify_hashes(config["checkpoint_sha256"])
    write_json(output / "summary.json", {"complete": True, "cases": len(cases), "summary": [row for row in rows if row["family"] == "ALL"],
                                         "cuda_initialized": torch.cuda.is_initialized(), "model_state_unchanged": True,
                                         "training_performed": False, "checkpoint_selection_performed": False,
                                         "config_sha256": sha256(output / "config.json"), "limitations": config["limitations"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise RuntimeError("Require CUDA_VISIBLE_DEVICES='' for CPU-only diagnosis")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(output / "status.json", {"state": "registering", "pid": os.getpid(), "updated_unix": time.time()})
            config = register(output)
            run(config, output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
