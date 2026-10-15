"""Audit a completed continuation's synthetic confirmation under column NRMSE on CPU."""

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
from train_single_matrixformer_continuation import verify_run


ROLES = ("general", "observed_blend", "initialization", "candidate")
INDICES = tuple(range(24))


def summarize(cases):
    if not cases or len({case["seed"] for case in cases}) != len(cases):
        raise ValueError("Require distinct nonempty synthetic cases")
    groups = defaultdict(list)
    for case in cases:
        if (case.get("gradient_data") != "synthetic" or case.get("domain") != "imputation"
                or set(case["scores"]) != set(ROLES)
                or any(not np.isfinite(case["scores"][role][name]) or case["scores"][role][name] < 0
                       for role in ROLES for name in metric.METRICS)):
            raise ValueError("Require complete finite synthetic imputation comparisons")
        groups["ALL"].append(case)
        groups[case["prior"]].append(case)
    rows = []
    for family, selected in sorted(groups.items()):
        for role in ROLES:
            row = {"family": family, "role": role, "cases": len(selected)}
            for name in metric.METRICS:
                values = [case["scores"][role][name] for case in selected]
                initial = [case["scores"]["initialization"][name] for case in selected]
                row[name] = float(np.mean(values))
                row[f"{name}_wins_vs_initialization"] = sum(value < parent - 1e-10 for value, parent in zip(values, initial))
                row[f"{name}_losses_vs_initialization"] = sum(value > parent + 1e-10 for value, parent in zip(values, initial))
            for suite in ("missbench", "uci"):
                flips = 0
                for case in selected:
                    candidate, initial = case["scores"][role], case["scores"]["initialization"]
                    pooled = candidate["pooled_rmse"] - initial["pooled_rmse"]
                    columnwise = candidate[f"{suite}_nrmse"] - initial[f"{suite}_nrmse"]
                    flips += abs(pooled) > 1e-10 and abs(columnwise) > 1e-10 and pooled * columnwise < 0
                row[f"{suite}_opposite_direction_cases"] = flips
            rows.append(row)
    return rows


def register(run, confirmation, output):
    original = metric.read(run / "config.json")
    verify_run(original, run)
    gate = metric.read(confirmation / "gate.json")
    manifest = metric.read(confirmation / "manifest.json")
    reference = metric.read(confirmation / "reference.json")
    if (metric.read(run / "status.json")["state"] != "training_complete"
            or metric.read(confirmation / "status.json")["state"] != "complete"
            or gate["selected_step"] <= 0 or gate["cases"] != 46
            or manifest["config_sha256"] != metric.sha256(run / "config.json")
            or gate["checkpoint_sha256"] != metric.sha256(run / "best_validation_checkpoint.pth")
            or gate["bank_sha256"] != reference["bank_sha256"]):
        raise ValueError("Require a completed nonzero candidate and its bound synthetic confirmation")
    inputs = [run / "config.json", confirmation / "manifest.json", confirmation / "gate.json",
              confirmation / "reference.json", confirmation / "initialization.json", confirmation / "student.json"]
    hashes = {**original["input_sha256"], **{str(path): metric.sha256(path) for path in inputs},
              str(confirmation / "synthetic_bank.pt"): reference["bank_sha256"]}
    metric.verify_hashes(hashes)
    bank = torch.load(confirmation / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    shapes = {}
    for index in INDICES:
        episode = metric.restore_episode(bank[index], "cpu")
        metric.validate_episode(episode)
        if episode.metadata["seed"] != original["confirmation_cases"][index]["seed"]:
            raise ValueError("Synthetic cases differ from the original confirmation bank")
        shapes[str(index)] = list(episode.source.shape[1:])
    del bank
    checkpoints = {"general": str(metric.GENERAL), "specialist": str(metric.SPECIALIST),
                   "initialization": original["parent_checkpoint"],
                   "candidate": str(run / "best_validation_checkpoint.pth")}
    sources = {Path(__file__).resolve(), *metric.SCORERS.values(), metric.REPO / "src/tabimpute/data/borders.pt"}
    sources.update((metric.REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(metric.REPO) and Path(module.__file__).is_file())
    config = {"run": str(run), "confirmation": str(confirmation), "bank": str(confirmation / "synthetic_bank.pt"),
              "case_indices": list(INDICES), "case_shapes": shapes, "checkpoints": checkpoints,
              "checkpoint_sha256": {path: metric.sha256(Path(path)) for path in checkpoints.values()},
              "input_sha256": hashes, "source_sha256": {str(path): metric.sha256(path) for path in sorted(sources)},
              "original_gate_passed": gate["passed"], "original_selected_step": gate["selected_step"],
              "precision": "FP32 CPU, four threads; no CUDA",
              "scope": "All first 24 full small imputation confirmation contexts, including intrinsic-size anchors; no cropping or normalization refitting",
              "metric": "Original MissBench/UCI scorers and pooled RMSE on the exact synthetic supervision queries; common-column coverage also retained",
              "training_performed": False, "checkpoint_selection_performed": False,
              "real_benchmark_data_used": False, "gate_changed": False,
              "limitations": "Reused small synthetic contexts and CPU precision; not fresh confirmation, large-context BF16 evidence, or permission to override the original gate"}
    metric.write_json(output / "config.json", config)
    for source in sources:
        destination = output / "sources" / source.relative_to(metric.REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return config


@torch.no_grad()
def diagnose(config, output):
    if config["case_indices"] != list(INDICES):
        raise ValueError("Do not replace the fixed 24-case diagnostic scope")
    for field in ("input_sha256", "source_sha256", "checkpoint_sha256"):
        metric.verify_hashes(config[field])
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True)
    confirmation = Path(config["confirmation"])
    stored = {name: metric.read(confirmation / f"{name}.json")["cases"]
              for name in ("reference", "initialization", "student")}
    models = {}
    for role, checkpoint in config["checkpoints"].items():
        arguments = metric.control_recipe()
        arguments.init_model_checkpoint = checkpoint
        model, optimizer, scheduler, criterion = metric.initialize_student(arguments, torch.device("cpu"))
        del optimizer, scheduler
        models[role] = model.eval().requires_grad_(False)
    scorers = {suite: metric.load_original(path, "compute_normalized_rmse_columnwise") for suite, path in metric.SCORERS.items()}
    cases, flat = [], []
    for index in INDICES:
        started = time.monotonic()
        episode = metric.restore_episode(bank[index], "cpu")
        metric.validate_episode(episode)
        if list(episode.source.shape[1:]) != config["case_shapes"][str(index)]:
            raise ValueError("Registered context dimensions changed")
        if any(rows[index]["index"] != index or rows[index]["seed"] != episode.metadata["seed"] for rows in stored.values()):
            raise ValueError("Stored BF16 scores must refer to the exact same synthetic cases")
        predictions = {role: metric.streamed_point_predictions(model, episode.source, criterion, **metric.STREAM_OPTIONS)
                       for role, model in models.items()}
        predictions["observed_blend"], routing = metric.blend_predictions(episode.source, predictions["general"], predictions["specialist"])
        arrays = {role: predictions[role][0].numpy() for role in ROLES}
        scores = {role: metric.score_queries(episode.target[0].numpy(), arrays[role], episode.mask[0].numpy(), scorers) for role in ROLES}
        previous = {"general": stored["reference"][index]["general"]["rmse"],
                    "observed_blend": stored["reference"][index]["teacher"]["rmse"],
                    "initialization": stored["initialization"][index]["student"]["rmse"],
                    "candidate": stored["student"][index]["student"]["rmse"]}
        path = output / "predictions" / f"{index:03d}.npz"
        path.parent.mkdir(exist_ok=True)
        metric.atomic_npz(path, **arrays)
        case = {"index": index, **episode.metadata, "routing": routing, "scores": scores,
                "stored_bf16_pooled_rmse": previous,
                "fp32_minus_bf16_pooled_rmse": {role: scores[role]["pooled_rmse"] - value for role, value in previous.items()},
                "prediction_sha256": metric.sha256(path), "seconds": time.monotonic() - started}
        cases.append(case)
        metric.write_json(output / "cases" / f"{index:03d}.json", case)
        flat.extend({"index": index, "seed": case["seed"], "prior": case["prior"], "role": role, **scores[role]} for role in ROLES)
        metric.write_json(output / "status.json", {"state": "diagnosing", "completed": len(cases), "expected": len(INDICES),
                                                  "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case": index, "prior": case["prior"], "seconds": case["seconds"]}), flush=True)
    for role, model in models.items():
        original = torch.load(config["checkpoints"][role], map_location="cpu", weights_only=True)["model"]
        if any(not torch.equal(value, original[name]) for name, value in model.state_dict().items()):
            raise ValueError("No diagnostic may modify the frozen model states")
    for field in ("input_sha256", "source_sha256", "checkpoint_sha256"):
        metric.verify_hashes(config[field])
    rows = summarize(cases)
    metric.write_csv(output / "per_case.csv", flat)
    metric.write_csv(output / "by_family.csv", rows)
    metric.write_json(output / "summary.json", {"complete": True, "cases": len(cases),
        "summary": [row for row in rows if row["family"] == "ALL"], "cuda_initialized": torch.cuda.is_initialized(),
        "model_state_unchanged": True, "gate_changed": False, "training_performed": False,
        "checkpoint_selection_performed": False, "config_sha256": metric.sha256(output / "config.json"),
        "limitations": config["limitations"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
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
            metric.write_json(output / "status.json", {"state": "registering", "pid": os.getpid(), "updated_unix": time.time()})
            config = register(arguments.run.resolve(), arguments.confirmation.resolve(), output)
            diagnose(config, output)
            metric.write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            metric.write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
