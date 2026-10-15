"""Audit every full synthetic imputation confirmation case without changing its gate."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import torch

import diagnose_single_model_full_confirmation_metrics as full
from dwip_teacher_mean_training import compute_context
from train_single_matrixformer_continuation import verify_run
from train_single_matrixformer_probability_balance import campaign_ready
from train_streamed_large_matrices import wait_for_gpu


metric = full.metric
ROLES = full.cpu.ROLES


def confirmation_inputs(run, confirmation, campaign):
    config = metric.read(run / "config.json")
    verify_run(config, run)
    gate = metric.read(confirmation / "gate.json")
    reference = metric.read(confirmation / "reference.json")
    checkpoint = run / "best_validation_checkpoint.pth"
    bank = confirmation / "synthetic_bank.pt"
    if (metric.read(run / "status.json")["state"] != "training_complete"
            or metric.read(confirmation / "status.json")["state"] != "complete"
            or not campaign_ready(campaign)
            or metric.read(campaign / "cache_launcher_config.json")["run"] != str(run)
            or gate.get("cases") != 46 or gate["selected_step"] <= 0
            or gate["selected_step"] != metric.read(run / "status.json")["best_step"]
            or gate["checkpoint_sha256"] != metric.sha256(checkpoint)
            or gate["bank_sha256"] != reference["bank_sha256"]
            or reference["bank_sha256"] != metric.sha256(bank)
            or len(config["confirmation_cases"]) != 46 or len(reference["cases"]) != 46):
        raise ValueError("Require completed, unchanged 46-case synthetic confirmation and campaign")
    records = {name: metric.read(confirmation / f"{name}.json")["cases"]
               for name in ("reference", "initialization", "student")}
    if any(len(cases) != 46 for cases in records.values()):
        raise ValueError("Require all original confirmation records")
    if len({case["seed"] for case in reference["cases"]}) != 46:
        raise ValueError("Require distinct synthetic confirmation seeds")
    for index, specification in enumerate(config["confirmation_cases"]):
        expected_domain = "imputation" if index < 30 else "recsys"
        for cases in records.values():
            case = cases[index]
            if (case["index"] != index or case["seed"] != specification["seed"]
                    or case["domain"] != expected_domain or case["gradient_data"] != "synthetic"
                    or (case["rows"], case["columns"]) != (reference["cases"][index]["rows"], reference["cases"][index]["columns"])):
                raise ValueError("Require paired original synthetic confirmation cases")
    return config, gate, reference


def make_config(run, confirmation, campaign):
    training, gate, reference = confirmation_inputs(run, confirmation, campaign)
    checkpoints = {"general": training["teachers"][0], "specialist": training["teachers"][1],
                   "initialization": training["parent_checkpoint"], "candidate": str(run / "best_validation_checkpoint.pth")}
    artifacts = [run / "config.json", run / "registration.json", confirmation / "synthetic_bank.pt",
                 campaign / "cache_launcher_config.json"]
    artifacts.extend(confirmation / name for name in ("gate.json", "manifest.json", "reference.json",
                                                       "initialization.json", "student.json"))
    sources = {Path(__file__).resolve(), metric.REPO / "src/tabimpute/data/borders.pt"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(metric.REPO) and Path(module.__file__).is_file())
    sources.update(Path(path) for path in metric.SCORERS.values())
    return {"run": str(run), "confirmation": str(confirmation), "after_campaign": str(campaign),
            "bank": str(confirmation / "synthetic_bank.pt"), "checkpoints": checkpoints,
            "checkpoint_sha256": {path: metric.sha256(Path(path)) for path in checkpoints.values()},
            "case_indices": list(full.INDICES),
            "case_shapes": {str(index): [reference["cases"][index]["rows"], reference["cases"][index]["columns"]]
                            for index in full.INDICES},
            "input_sha256": {str(path): metric.sha256(path) for path in artifacts},
            "source_sha256": {str(path): metric.sha256(path) for path in sources},
            "model_initialization_seed": training["seed"], "memory_fraction": training["memory_fraction"],
            "original_gate_passed": gate["passed"], "original_selected_step": gate["selected_step"],
            "pooled_rmse_parity": {"rtol": full.PARITY_RTOL, "atol": full.PARITY_ATOL},
            "precision": "Original FP32 weights and BF16 CUDA autocast, TF32 and cuDNN SDP disabled",
            "scope": "All 30 original full imputation contexts, including all six 2000x1000/1000x2000 cases; no cropping, refitting or CPU substitution",
            "training_performed": False, "checkpoint_selection_performed": False,
            "real_benchmark_data_used": False, "gate_changed": False,
            "limitations": "Reused synthetic confirmation bank; diagnostic only, not a new confirmation or real benchmark success; any future metric-aligned selection requires new registration and fresh banks"}


@torch.no_grad()
def diagnose(config, output):
    full.verify_config(config, output)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True)
    stored = {name: metric.read(Path(config["confirmation"]) / f"{name}.json")["cases"]
              for name in ("reference", "initialization", "student")}
    models = {}
    for role, checkpoint in config["checkpoints"].items():
        arguments = metric.control_recipe()
        arguments.init_model_checkpoint = checkpoint
        arguments.seed = config["model_initialization_seed"]
        model, optimizer, scheduler, criterion = metric.initialize_student(arguments, torch.device("cuda"))
        del optimizer, scheduler
        models[role] = model.eval().requires_grad_(False)
    scorers = {suite: metric.load_original(path, "compute_normalized_rmse_columnwise") for suite, path in metric.SCORERS.items()}
    cases, flat = [], []
    for index in full.INDICES:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        episode = metric.restore_episode(bank[index], "cuda")
        full.validate_episode(episode, index)
        if list(episode.source.shape[1:]) != config["case_shapes"][str(index)]:
            raise ValueError("Original full context dimensions changed")
        if any(rows[index]["index"] != index or rows[index]["seed"] != episode.metadata["seed"] for rows in stored.values()):
            raise ValueError("Original BF16 records must match every case index and seed")
        with compute_context(episode.source.device):
            predictions = {role: metric.streamed_point_predictions(model, episode.source, criterion, **metric.STREAM_OPTIONS)
                           for role, model in models.items()}
        predictions["observed_blend"], routing = metric.blend_predictions(episode.source, predictions["general"], predictions["specialist"])
        arrays = {role: predictions[role][0].float().cpu().numpy() for role in ROLES}
        truth, mask = episode.target[0].cpu().numpy(), episode.mask[0].cpu().numpy()
        scores = {role: metric.score_queries(truth, arrays[role], mask, scorers) for role in ROLES}
        previous = {"general": stored["reference"][index]["general"]["rmse"],
                    "observed_blend": stored["reference"][index]["teacher"]["rmse"],
                    "initialization": stored["initialization"][index]["student"]["rmse"],
                    "candidate": stored["student"][index]["student"]["rmse"]}
        full.verify_pooled_parity(scores, previous)
        path = output / "predictions" / f"{index:03d}.npz"
        path.parent.mkdir(exist_ok=True)
        metric.atomic_npz(path, **arrays)
        case = {"index": index, **episode.metadata, "routing": routing, "scores": scores,
                "stored_bf16_pooled_rmse": previous,
                "pooled_parity_absolute_errors": {role: abs(scores[role]["pooled_rmse"] - value) for role, value in previous.items()},
                "prediction_sha256": metric.sha256(path), "seconds": time.monotonic() - started,
                "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
        cases.append(case)
        metric.write_json(output / "cases" / f"{index:03d}.json", case)
        flat.extend({"index": index, "seed": case["seed"], "prior": case["prior"], "role": role, **scores[role]} for role in ROLES)
        metric.write_json(output / "status.json", {"state": "diagnosing", "completed": len(cases), "expected": 30,
                                                  "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case": index, "prior": case["prior"], "seconds": case["seconds"], "peak_gib": case["peak_gib"]}), flush=True)
        del episode, predictions, arrays, truth, mask
    for role, model in models.items():
        original = torch.load(config["checkpoints"][role], map_location="cpu", weights_only=True)["model"]
        if model.state_dict().keys() != original.keys() or any(not torch.equal(value.cpu(), original[name]) for name, value in model.state_dict().items()):
            raise ValueError("Diagnostic modified a frozen model state")
    full.verify_config(config, output)
    rows, contexts = full.cpu.summarize(cases), full.context_summaries(cases)
    metric.write_csv(output / "per_case.csv", flat)
    metric.write_csv(output / "by_family.csv", rows)
    metric.write_csv(output / "by_context.csv", contexts)
    metric.write_json(output / "summary.json", {"complete": True, "cases": 30, "small_cases": 24, "large_cases": 6,
        "summary": [row for row in rows if row["family"] == "ALL"], "contexts": contexts,
        "pooled_parity_checks": 120, "pooled_parity_passed": True,
        "maximum_pooled_parity_absolute_error": max(error for case in cases for error in case["pooled_parity_absolute_errors"].values()),
        "model_state_unchanged": True, "gate_changed": False, "training_performed": False,
        "checkpoint_selection_performed": False, "config_sha256": metric.sha256(output / "config.json"),
        "limitations": config["limitations"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--wait", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output = arguments.output.resolve()
    if arguments.stage == "register":
        if any(value is None for value in (arguments.run, arguments.confirmation, arguments.after_campaign)) or arguments.wait:
            parser.error("Registration requires a completed run, confirmation and campaign; no wait")
        full.register(make_config(arguments.run.resolve(), arguments.confirmation.resolve(), arguments.after_campaign.resolve()), output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if metric.read(output / "status.json")["state"] != "registered":
            raise ValueError("Never overwrite a started diagnostic")
        try:
            config = metric.read(output / "config.json")
            full.verify_config(config, output)
            if not campaign_ready(Path(config["after_campaign"])):
                raise RuntimeError("Predecessor campaign must be terminal")
            wait_for_gpu(output, arguments.wait, False)
            metric.write_json(output / "status.json", {"state": "diagnosing", "completed": 0, "expected": 30,
                                                      "pid": os.getpid(), "updated_unix": time.time()})
            diagnose(config, output)
            metric.write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            metric.write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                                      "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
