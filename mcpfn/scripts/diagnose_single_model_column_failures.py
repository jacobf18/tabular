"""Decompose all V6 confirmation imputation errors without selecting another model."""

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

import diagnose_single_model_metric_alignment as metric
from dwip_teacher_mean_training import compute_context
from train_single_matrixformer import storage_guard
from train_single_matrixformer_continuation import verify_run
from train_streamed_large_matrices import wait_for_gpu


ROLES = ("general", "initialization", "candidate")
SCALE_EDGES = (0., 1e-4, 1e-3, 1e-2, .1, 1., float("inf"))
SUPPORT_EDGES = (0, 2, 8, 32, 128, float("inf"))
METRICS = ("pooled_rmse", "missbench_nrmse", "uci_nrmse", "common_nonconstant_nrmse")
STORED_NAMES = ("rmse", "missbench_nrmse", "uci_nrmse", "common_nrmse")


def band(value, edges):
    return next(f"[{lower},{upper})" for lower, upper in zip(edges, edges[1:]) if lower <= value < upper)


def column_rows(truth, mask, source, arrays):
    if (truth.ndim != 2 or mask.dtype != np.bool_ or source.shape != truth.shape or mask.shape != truth.shape
            or not mask.any() or set(arrays) != set(ROLES) or not np.isfinite(truth).all()
            or any(values.shape != truth.shape or not np.isfinite(values).all() for values in arrays.values())):
        raise ValueError("Require full finite predictions, truth and a nonempty aligned query mask")
    rows = []
    for column in range(truth.shape[1]):
        selected = mask[:, column]
        if not selected.any():
            continue
        labels = truth[selected, column]
        native_scale = np.std(labels)
        scale = float(native_scale)
        support = int(np.isfinite(source[:, column]).sum())
        row = {"column": column, "queries": int(selected.sum()), "observations": support,
               "target_std": scale, "scale_band": band(scale, SCALE_EDGES),
               "support_band": band(support, SUPPORT_EDGES), "common_eligible": scale >= 1e-3}
        parent = arrays["initialization"][selected, column].astype(np.float64)
        for role, prediction in arrays.items():
            values = prediction[selected, column]
            error = values.astype(np.float64) - labels.astype(np.float64)
            rmse = np.sqrt(np.mean((values - labels)**2))
            row[f"{role}_rmse"] = float(np.sqrt(np.mean(error**2)))
            row[f"{role}_bias"] = float(np.mean(error))
            row[f"{role}_drift_rmse"] = float(np.sqrt(np.mean((values.astype(np.float64) - parent)**2)))
            row[f"{role}_common_nrmse"] = row[f"{role}_rmse"] / scale if scale >= 1e-3 else None
            for suite, threshold in (("missbench", 1e-3), ("uci", 1e-4)):
                row[f"{role}_{suite}_nrmse"] = (float(rmse / native_scale) if scale >= threshold
                                                   else (0. if rmse < threshold else None))
        rows.append(row)
    for role in ROLES:
        for suite in ("missbench", "uci", "common"):
            name = f"{role}_{suite}_nrmse"
            count = sum(row[name] is not None for row in rows)
            if not count:
                raise ValueError("Require at least one scored column for each metric")
            for row in rows:
                row[f"{role}_{suite}_contribution"] = 0. if row[name] is None else row[name] / count
    return rows


def contribution_summary(rows, cases):
    groups = defaultdict(list)
    for row in rows:
        for kind in ("scale_band", "support_band", "context", "prior", "replica"):
            groups[kind, str(row[kind])].append(row)
    result = []
    for (kind, group), selected in sorted(groups.items()):
        for suite in ("missbench", "uci", "common"):
            values = {role: sum(row[f"{role}_{suite}_contribution"] for row in selected) / cases for role in ROLES}
            result.append({"grouping": kind, "group": group, "metric": suite, "columns": len(selected),
                           "cases": len({row["index"] for row in selected}), **values,
                           "candidate_minus_initialization": values["candidate"] - values["initialization"],
                           "candidate_minus_general": values["candidate"] - values["general"]})
    return result


def verify(config, output):
    if metric.sha256(output / "config.json") != metric.read(output / "registration.json")["config_sha256"]:
        raise ValueError("Diagnostic configuration changed")
    for field in ("input_sha256", "source_sha256", "checkpoint_sha256"):
        metric.verify_hashes(config[field])


def register(run, confirmation, output):
    training = metric.read(run / "config.json")
    verify_run(training, run)
    audit = metric.read(confirmation / "confirmation_audit.json")
    gate, reference = metric.read(confirmation / "gate.json"), metric.read(confirmation / "reference.json")
    if (metric.read(confirmation / "status.json")["state"] != "complete" or not audit["complete"]
            or not audit["gate_reproduced"] or audit["cases"] != 92 or gate["selected_step"] != 300
            or gate["checkpoint_sha256"] != metric.sha256(run / "best_validation_checkpoint.pth")):
        raise ValueError("Require the audited, complete, unchanged V6 confirmation")
    indices = [case["index"] for case in reference["cases"] if case["domain"] == "imputation"]
    if indices != list(range(30)) + list(range(46, 76)):
        raise ValueError("Require all sixty imputation cases in both replicas")
    checkpoints = {"general": training["teachers"][0], "initialization": training["parent_checkpoint"],
                   "candidate": str(run / "best_validation_checkpoint.pth")}
    inputs = [run / "config.json", run / "registration.json"]
    inputs.extend(confirmation / name for name in ("synthetic_bank.pt", "manifest.json", "reference.json",
                  "initialization.json", "student.json", "gate.json", "confirmation_audit.json"))
    sources = {Path(__file__).resolve(), metric.REPO / "src/tabimpute/data/borders.pt", *metric.SCORERS.values()}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(metric.REPO) and Path(module.__file__).is_file())
    config = {"run": str(run), "confirmation": str(confirmation), "bank": str(confirmation / "synthetic_bank.pt"),
              "case_indices": indices, "checkpoints": checkpoints, "roles": list(ROLES),
              "checkpoint_sha256": {path: metric.sha256(Path(path)) for path in checkpoints.values()},
              "input_sha256": {str(path): metric.sha256(path) for path in inputs},
              "source_sha256": {str(path): metric.sha256(path) for path in sources},
              "seed": training["seed"], "memory_fraction": training["memory_fraction"],
              "precision": "Original FP32 weights/BF16 CUDA compute; original full contexts and masks",
              "scale_edges": list(SCALE_EDGES[:-1]) + ["infinity"],
              "support_edges": list(SUPPORT_EDGES[:-1]) + ["infinity"],
              "metric_parity": {"rtol": 1e-6, "atol": 1e-6},
              "training_performed": False, "checkpoint_selection_performed": False,
              "real_benchmark_data_used": False, "gate_changed": False,
              "limitations": "Reused synthetic confirmation; column contributions are descriptive, not new validation. No case or column is removed and no reference or failed gate is changed. Raw blend predictions are not regenerated; its original recorded scores remain the reference."}
    output.mkdir(parents=True, exist_ok=False)
    storage_guard(output)
    metric.write_json(output / "config.json", config)
    for source in sources:
        destination = output / "sources" / source.relative_to(metric.REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    metric.write_json(output / "registration.json", {"config_sha256": metric.sha256(output / "config.json"), "registered_unix": time.time()})
    metric.write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})
    print(json.dumps({"registered": str(output), "cases": len(indices), "roles": list(ROLES)}))


@torch.no_grad()
def diagnose(config, output):
    verify(config, output)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True)
    stored = {role: metric.read(Path(config["confirmation"]) / name)["cases"] for role, name in
              (("general", "reference.json"), ("initialization", "initialization.json"), ("candidate", "student.json"))}
    models = {}
    for role, checkpoint in config["checkpoints"].items():
        arguments = metric.control_recipe()
        arguments.init_model_checkpoint, arguments.seed = checkpoint, config["seed"]
        model, optimizer, scheduler, criterion = metric.initialize_student(arguments, torch.device("cuda"))
        del optimizer, scheduler
        models[role] = model.eval().requires_grad_(False)
    scorers = {suite: metric.load_original(path, "compute_normalized_rmse_columnwise") for suite, path in metric.SCORERS.items()}
    (output / "columns").mkdir(exist_ok=True)
    all_rows, cases = [], []
    for index in config["case_indices"]:
        started = time.monotonic()
        episode = metric.restore_episode(bank[index], "cuda")
        metadata = episode.metadata
        if (metadata["domain"] != "imputation" or metadata["gradient_data"] != "synthetic"
                or tuple(episode.source.shape) != (1, metadata["rows"], metadata["columns"])
                or not torch.isnan(episode.source[episode.mask]).all()):
            raise ValueError("Require the complete original synthetic imputation episode")
        with compute_context(episode.source.device):
            arrays = {role: metric.streamed_point_predictions(model, episode.source, criterion, **metric.STREAM_OPTIONS)[0].float().cpu().numpy()
                      for role, model in models.items()}
        truth, mask, source = (values[0].cpu().numpy() for values in (episode.target, episode.mask, episode.source))
        scores = {role: metric.score_queries(truth, values, mask, scorers) for role, values in arrays.items()}
        parity_errors = []
        for role in ROLES:
            record = stored[role][index]
            if record["index"] != index or record["seed"] != metadata["seed"]:
                raise ValueError("Original recorded seed or case pairing changed")
            previous = record["general" if role == "general" else "student"]
            for current_name, previous_name in zip(METRICS, STORED_NAMES):
                np.testing.assert_allclose(scores[role][current_name], previous[previous_name], **config["metric_parity"])
                parity_errors.append(abs(scores[role][current_name] - previous[previous_name]))
        rows = column_rows(truth, mask, source, arrays)
        for role in ROLES:
            for suite in ("missbench", "uci", "common"):
                name = "common_nonconstant_nrmse" if suite == "common" else f"{suite}_nrmse"
                np.testing.assert_allclose(sum(row[f"{role}_{suite}_contribution"] for row in rows), scores[role][name], rtol=1e-6, atol=1e-7)
        context = "large" if max(metadata["rows"], metadata["columns"]) > 240 else "small"
        rows = [{"index": index, "seed": metadata["seed"], "replica": index // 46 + 1,
                 "prior": metadata["prior"], "context": context, **row} for row in rows]
        path = output / "predictions" / f"{index:03d}.npz"
        path.parent.mkdir(exist_ok=True)
        metric.atomic_npz(path, **arrays)
        case = {"index": index, **metadata, "scores": scores, "prediction_sha256": metric.sha256(path),
                "maximum_parity_error": max(parity_errors), "seconds": time.monotonic() - started}
        metric.write_json(output / "cases" / f"{index:03d}.json", case)
        metric.write_csv(output / "columns" / f"{index:03d}.csv", rows)
        all_rows.extend(rows)
        cases.append(case)
        metric.write_json(output / "status.json", {"state": "diagnosing", "completed": len(cases), "expected": 60, "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case": index, "columns": len(rows), "seconds": case["seconds"]}), flush=True)
        del episode, arrays, truth, mask, source
    for role, model in models.items():
        original = torch.load(config["checkpoints"][role], map_location="cpu", weights_only=True)["model"]
        if model.state_dict().keys() != original.keys() or any(not torch.equal(value.cpu(), original[name]) for name, value in model.state_dict().items()):
            raise ValueError("A diagnostic model state changed")
    verify(config, output)
    contributions = contribution_summary(all_rows, len(cases))
    metric.write_csv(output / "contributions.csv", contributions)
    metric.write_json(output / "summary.json", {"complete": True, "cases": len(cases), "queried_columns": len(all_rows),
        "metrics": {role: {name: float(np.mean([case["scores"][role][name] for case in cases])) for name in METRICS} for role in ROLES},
        "parity_checks": len(cases) * len(ROLES) * len(METRICS), "all_metric_parity_passed": True,
        "maximum_parity_error": max(case["maximum_parity_error"] for case in cases), "model_state_unchanged": True,
        "training_performed": False, "checkpoint_selection_performed": False, "gate_changed": False,
        "real_benchmark_success": False, "config_sha256": metric.sha256(output / "config.json"), "limitations": config["limitations"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        register(arguments.run.resolve(), arguments.confirmation.resolve(), output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_for_gpu(output, False, False)
            diagnose(metric.read(output / "config.json"), output)
            metric.write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            metric.write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
