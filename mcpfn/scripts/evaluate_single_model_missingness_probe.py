"""Frozen-model diagnostic on all paired synthetic masks, never checkpoint selection."""

import argparse
import fcntl
import importlib.resources
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import diagnose_single_model_missingness_coverage as probe
from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_csv, write_json
from single_model_distillation import Episode, blend_predictions
from single_model_truth_defined_columns import episode_metrics
from streamed_large_matrix_training import STREAM_OPTIONS
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.streamed_inference import streamed_point_predictions
from train_dwip_spectral_geometry import DwipTrainingAdapter
from train_single_matrixformer import GENERAL, SPECIALIST, INPUTS
from train_streamed_large_matrices import wait_for_gpu


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
ROLES = ("general", "specialist", "v2", "v8", "blend")
METRICS = ("rmse", "mae", "missbench_nrmse", "uci_nrmse", "common_nrmse", "observed_rmse", "query_blend_rmse")


def read(path):
    return json.loads(path.read_text())


def register(bank, after, output):
    if output.exists():
        raise FileExistsError("Never replace a registered model diagnostic")
    audit = read(bank / "coverage_audit.json")
    bank_config = read(bank / "config.json")
    if not audit["complete"] or audit["cases"] != 192 or audit["config_sha256"] != sha256(bank / "config.json"):
        raise ValueError("Require the complete audited model-free probe")
    verify_hashes(audit["original_artifact_sha256"])
    verify_hashes(bank_config["source_hashes"])
    v8_config = read(ROOT / "query_budget_v8/config.json")
    checkpoints = {"general": str(GENERAL), "specialist": str(SPECIALIST),
                   "v2": v8_config["parent_checkpoint"], "v8": str(ROOT / "query_budget_v8/best_validation_checkpoint.pth")}
    hashes = dict(INPUTS)
    for role, folder in (("v2", REPO / "runs/single_model_goal_20261005/mixed_distill_v2_confirmation"),
                         ("v8", ROOT / "query_budget_v8_confirmation_defined1")):
        gate = read(folder / "gate.json")
        if not gate["passed"]:
            raise ValueError("Only the original synthetically selected checkpoints may be diagnosed")
        hashes[checkpoints[role]] = gate["checkpoint_sha256"]
        hashes[str(folder / "gate.json")] = sha256(folder / "gate.json")
    verify_hashes(hashes)
    hashes.update(audit["original_artifact_sha256"])
    hashes[str(bank / "coverage_audit.json")] = sha256(bank / "coverage_audit.json")
    hashes[str(after / "config.json")] = sha256(after / "config.json")
    borders = Path(str(importlib.resources.files("tabimpute") / "data/borders.pt"))
    hashes[str(borders)] = sha256(borders)
    config = {"schema": 1, "registered_unix": time.time(), "bank": str(bank), "after_campaign": str(after),
              "checkpoints": checkpoints, "model": v8_config["model"], "borders": str(borders),
              "roles": list(ROLES), "metrics": list(METRICS), "stream_options": STREAM_OPTIONS,
              "precision": "FP32 parameters, BF16 autocast, TF32 disabled", "cases": 192,
              "input_sha256": hashes, "source_sha256": probe.source_hashes(),
              "purpose": "Descriptive synthetic missingness/teacher-transfer diagnosis; not training, a gate, or model selection",
              "blend": "Unchanged observed-error router on identical saved inputs; hidden targets never passed to router",
              "scope": "All original 192 cases and 5 roles; no family omissions, replacement seeds or large-context claims",
              "undefined_guard": "Explicit truth-only nulls with coverage; both canonical NRMSE retain every case",
              "artifacts": "Full observed/query prediction grids for every role, raw-prediction rescoring and routing decisions"}
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"state": "registered", "config_sha256": sha256(output / "config.json")})


def campaign_ready(path):
    status = read(path / "status.json")
    if status["state"] in {"failed", "interrupted", "not_promoted"}:
        raise ValueError("Do not bypass the unfinished or failed original benchmark scope")
    if status["state"] != "complete":
        return False
    return bool(read(path / "campaign_audit.json")["all_evaluations_complete"])


def score_predictions(source, target, query, predictions):
    if set(predictions) != set(ROLES) or not query.any() or not torch.isfinite(source).any():
        raise ValueError("Require complete role coverage and observed/query cells; never omit an undefined case")
    if any(prediction.shape != source.shape or not torch.isfinite(prediction).all() for prediction in predictions.values()):
        raise ValueError("Require full finite prediction grids including observed entries")
    episode = Episode(source.unsqueeze(0), target.unsqueeze(0), query.unsqueeze(0),
                      {"domain": "imputation", "gradient_data": "synthetic"})
    records = {}
    observed = torch.isfinite(source)
    for role, prediction in predictions.items():
        metrics = episode_metrics(prediction.unsqueeze(0), episode)
        metrics["observed_rmse"] = float((prediction[observed].double() - source[observed].double()).square().mean().sqrt())
        metrics["query_blend_rmse"] = float((prediction[query].double() - predictions["blend"][query].double()).square().mean().sqrt())
        records[role] = metrics
    if len({json.dumps(metrics["column_coverage"], sort_keys=True) for metrics in records.values()}) != 1:
        raise ValueError("Metric coverage must be identical for every model")
    return records


def summarize(records):
    expected = {(context, family, role) for context in range(16) for family in probe.FAMILIES for role in ROLES}
    keys = {(row["context"], row["family"], row["role"]) for row in records}
    if keys != expected or len(records) != len(expected):
        raise ValueError("Do not summarize incomplete or duplicate model/family coverage")
    aggregates = []
    for family in ("overall", *probe.FAMILIES):
        for role in ROLES:
            rows = [row for row in records if row["role"] == role and (family == "overall" or row["family"] == family)]
            result = {"family": family, "role": role, "cases": len(rows)}
            for metric in METRICS:
                values = [row[metric] for row in rows]
                if metric != "common_nrmse" and any(value is None for value in values):
                    raise ValueError("Every canonical case must be scored")
                result[metric] = probe.average_defined(values)
                result[f"{metric}_defined_cases"] = sum(value is not None for value in values)
            aggregates.append(result)
    return aggregates


@torch.no_grad()
def evaluate(output):
    config = read(output / "config.json")
    status = read(output / "status.json")
    if status["state"] != "registered" or status["config_sha256"] != sha256(output / "config.json"):
        raise ValueError("Require untouched registration; no silent restarts or overwrites")
    while not campaign_ready(Path(config["after_campaign"])):
        write_json(output / "progress.json", {"state": "waiting_for_complete_real_campaign", "pid": os.getpid(),
                                               "updated_unix": time.time()})
        time.sleep(30)
    wait_for_gpu(output, True, False)
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(.8)
    write_json(output / "status.json", {"state": "evaluating", "pid": os.getpid(), "updated_unix": time.time()})
    model = DwipTrainingAdapter(**config["model"]).cuda().float().eval().requires_grad_(False)
    criterion = FullSupportBarDistribution(torch.load(config["borders"], map_location="cuda", weights_only=True))
    states = {role: torch.load(path, map_location="cpu", weights_only=True)["model"]
              for role, path in config["checkpoints"].items()}
    bank = Path(config["bank"])
    metadata = read(bank / "per_case.json")
    predictions_directory = output / "predictions"
    predictions_directory.mkdir()
    rows, routing, files = [], [], {}
    for index, case in enumerate(metadata):
        case_id = f"context_{case['context']:02d}/{case['family']}"
        with np.load(bank / f"{case_id}.npz", allow_pickle=False) as stored:
            source = torch.from_numpy(stored["model_source"])
            target = torch.from_numpy(stored["model_target"])
            query = torch.from_numpy(stored["query"])
        predictions = {}
        gpu_source = source.unsqueeze(0).cuda()
        for role, state in states.items():
            model.load_state_dict(state, strict=True)
            with compute_context(gpu_source.device):
                prediction = streamed_point_predictions(model, gpu_source, criterion, **config["stream_options"])
            predictions[role] = prediction[0].detach().float().cpu()
        blended, decision = blend_predictions(source.unsqueeze(0), predictions["general"].unsqueeze(0),
                                              predictions["specialist"].unsqueeze(0))
        predictions["blend"] = blended[0]
        destination = predictions_directory / f"case_{index:03d}.npz"
        atomic_npz(destination, **{role: prediction.numpy() for role, prediction in predictions.items()})
        files[str(destination)] = sha256(destination)
        scores = score_predictions(source, target, query, predictions)
        for role in ROLES:
            rows.append({"context": case["context"], "family": case["family"], "component": case["component"],
                         "role": role, **scores[role]})
        routing.append({"case_id": case_id, **decision})
        write_json(output / "progress.json", {"state": "evaluating", "cases": index + 1,
                                               "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case": case_id, "scores": scores}, allow_nan=False), flush=True)
    del model, states
    torch.cuda.empty_cache()
    for index, case in enumerate(metadata):
        case_id = f"context_{case['context']:02d}/{case['family']}"
        with np.load(bank / f"{case_id}.npz", allow_pickle=False) as stored:
            source, target, query = [torch.from_numpy(stored[name]) for name in ("model_source", "model_target", "query")]
        with np.load(predictions_directory / f"case_{index:03d}.npz", allow_pickle=False) as stored:
            predictions = {role: torch.from_numpy(stored[role]) for role in ROLES}
        blended, decision = blend_predictions(source.unsqueeze(0), predictions["general"].unsqueeze(0),
                                              predictions["specialist"].unsqueeze(0))
        if not torch.equal(blended[0], predictions["blend"]) or {"case_id": case_id, **decision} != routing[index]:
            raise ValueError("Saved blend or observed-only routing did not reproduce")
        scores = score_predictions(source, target, query, predictions)
        for offset, role in enumerate(ROLES):
            if any(rows[index * len(ROLES) + offset][key] != value for key, value in scores[role].items()):
                raise ValueError("Saved prediction metrics did not reproduce")
    aggregates = summarize(rows)
    verify_hashes(files)
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    write_json(output / "per_case.json", rows)
    write_csv(output / "per_case.csv", [{key: value for key, value in row.items() if key != "column_coverage"} for row in rows])
    write_csv(output / "per_family.csv", aggregates)
    write_json(output / "routing.json", routing)
    write_json(output / "audit.json", {"complete": True, "cases": 192, "model_case_scores": len(rows),
                                      "raw_predictions_rescored": True, "routing_reproduced": True,
                                      "config_sha256": sha256(output / "config.json"), "prediction_sha256": files,
                                      "training_performed": False, "checkpoint_selection_performed": False})
    lines = ["# Frozen-model synthetic missingness diagnostic", "", config["purpose"], "",
             "All 192 fixed cases and five roles retained. Compact synthetic contexts only; not benchmark parity or a promotion gate.", "",
             "## MissBench-formula NRMSE", "", "| Family | General | Specialist | V2 | V8 | Blend |",
             "|---|---:|---:|---:|---:|---:|"]
    indexed = {(row["family"], row["role"]): row for row in aggregates}
    for family in ("overall", *probe.FAMILIES):
        lines.append(f"| {family} | " + " | ".join(f"{indexed[family, role]['missbench_nrmse']:.7f}" for role in ROLES) + " |")
    lines += ["", "All canonical scores retain every case. The separate common-column guard reports truth-defined coverage.",
              "NRMSE, pooled normalized RMSE/MAE, observed reconstruction RMSE and query discrepancy to the blend are in the CSVs.",
              "Full prediction grids are retained and rescored; the observed-only router is reproduced from its saved inputs."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(output / "status.json", {"state": "complete", "cases": 192, "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "evaluate"), required=True)
    parser.add_argument("--bank", type=Path, default=ROOT / "missingness_coverage_probe_v1")
    parser.add_argument("--after", type=Path, default=ROOT / "query_budget_v8_benchmarks_defined1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(args.bank.resolve(), args.after.resolve(), output)
    else:
        with (output / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(output / "status.json")["state"] != "registered":
                raise ValueError("Never replace a started diagnostic")
            try:
                evaluate(output)
            except BaseException as error:
                write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                                     "pid": os.getpid(), "updated_unix": time.time()})
                raise


if __name__ == "__main__":
    main()
