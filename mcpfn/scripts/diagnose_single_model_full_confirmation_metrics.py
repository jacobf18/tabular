"""Reproduce all synthetic imputation confirmation cases in BF16 and audit column NRMSE."""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

import diagnose_single_model_confirmation_metrics as cpu
from dwip_teacher_mean_training import compute_context
from train_single_matrixformer import storage_guard
from train_single_matrixformer_probability_balance import campaign_ready
from train_streamed_large_matrices import wait_for_gpu


metric = cpu.metric
INDICES = tuple(range(30))
PARITY_RTOL = PARITY_ATOL = 1e-6


def validate_proposal(proposal):
    scope = proposal["scope"]
    if (scope["case_indices"] != list(INDICES) or scope["expected_total"] != 30
            or scope["small_imputation_cases"] != 24 or scope["large_imputation_cases"] != 6
            or scope["cropping"] or scope["normalization_refitting"]
            or proposal["roles"] != list(cpu.ROLES)
            or proposal["pooled_rmse_parity"]["rtol"] != PARITY_RTOL
            or proposal["pooled_rmse_parity"]["atol"] != PARITY_ATOL
            or any(proposal[key] for key in ("training_performed", "checkpoint_selection_performed",
                                            "real_benchmark_data_used", "original_gate_changed"))):
        raise ValueError("Require the complete fixed 30-case diagnostic and original parity tolerance")


def validate_episode(episode, index):
    if index not in INDICES:
        raise ValueError("Unexpected confirmation case index")
    if index < 24:
        metric.validate_episode(episode)
    else:
        metadata = episode.metadata
        expected = (1000, 2000) if metadata["transposed"] else (2000, 1000)
        if (tuple(episode.source.shape) != (1, *expected)
                or episode.source.shape != episode.target.shape or episode.source.shape != episode.mask.shape
                or episode.mask.dtype != torch.bool or not episode.mask.any()
                or metadata.get("gradient_data") != "synthetic" or metadata.get("domain") != "imputation"
                or (metadata.get("rows"), metadata.get("columns")) != expected
                or not torch.isnan(episode.source[episode.mask]).all()):
            raise ValueError("Require each full original large synthetic context")
    if not torch.isfinite(episode.target[episode.mask]).all():
        raise ValueError("Require finite original synthetic query targets")


def verify_pooled_parity(scores, stored):
    if set(scores) != set(cpu.ROLES) or set(stored) != set(cpu.ROLES):
        raise ValueError("Require all four matched prediction roles")
    for role in cpu.ROLES:
        current, previous = scores[role]["pooled_rmse"], stored[role]
        if not np.isfinite(current) or not np.isfinite(previous) or current < 0 or previous < 0:
            raise ValueError("Require finite nonnegative pooled scores")
        np.testing.assert_allclose(current, previous, rtol=PARITY_RTOL, atol=PARITY_ATOL,
                                   err_msg=f"Original BF16 pooled-RMSE parity: {role}")


def context_summaries(cases):
    groups = {"small": [case for case in cases if case["index"] < 24],
              "large": [case for case in cases if case["index"] >= 24]}
    for rows, columns in ((2000, 1000), (1000, 2000)):
        groups[f"large_{rows}x{columns}"] = [case for case in groups["large"]
                                             if (case["rows"], case["columns"]) == (rows, columns)]
    result = []
    for group, selected in groups.items():
        if selected:
            result.extend({"context_group": group, **row} for row in cpu.summarize(selected) if row["family"] == "ALL")
    return result


def make_config(proposal_path):
    proposal = metric.read(proposal_path)
    validate_proposal(proposal)
    directory = Path(proposal["cpu_diagnostic"])
    original = metric.read(directory / "config.json")
    summary = metric.read(directory / "summary.json")
    if (metric.read(directory / "status.json")["state"] != "complete"
            or not summary["complete"] or summary["cases"] != 24 or summary["cuda_initialized"]
            or not summary["model_state_unchanged"] or summary["gate_changed"]
            or summary["training_performed"] or summary["checkpoint_selection_performed"]
            or original["case_indices"] != list(cpu.INDICES)
            or original["run"] != proposal["run"] or original["confirmation"] != proposal["confirmation"]
            or summary["config_sha256"] != proposal["cpu_config_sha256"]
            or metric.sha256(directory / "config.json") != proposal["cpu_config_sha256"]
            or original["checkpoint_sha256"][original["checkpoints"]["candidate"]] != proposal["candidate_checkpoint_sha256"]
            or original["input_sha256"][original["bank"]] != proposal["bank_sha256"]):
        raise ValueError("Require the completed immutable CPU diagnostic bound to this proposal")
    for field in ("input_sha256", "source_sha256", "checkpoint_sha256"):
        metric.verify_hashes(original[field])
    bank = torch.load(original["bank"], map_location="cpu", weights_only=True)
    if len(bank) != 46:
        raise ValueError("Require the complete original confirmation bank")
    shapes = {}
    for index in INDICES:
        episode = metric.restore_episode(bank[index], "cpu")
        validate_episode(episode, index)
        shapes[str(index)] = list(episode.source.shape[1:])
    del bank
    predecessor = Path(proposal["after_campaign"]) / "cache_launcher_config.json"
    predecessor_config = metric.read(predecessor)
    if Path(predecessor_config["run"]).name != "domain_teachers_v4":
        raise ValueError("Serialize after the whole registered v4 campaign")
    training_config = metric.read(Path(original["run"]) / "config.json")
    config = deepcopy(original)
    config.update(case_indices=list(INDICES), case_shapes=shapes, cpu_diagnostic=str(directory),
                  proposal=str(proposal_path), after_campaign=proposal["after_campaign"],
                  precision=proposal["precision"], pooled_rmse_parity=dict(proposal["pooled_rmse_parity"]),
                  model_initialization_seed=training_config["seed"], memory_fraction=training_config["memory_fraction"],
                  scope="All 30 original full imputation contexts, including six 2000x1000/1000x2000 cases; original masks and normalization",
                  limitations=proposal["limitations"], implementation_status="registered_full_context_diagnostic")
    artifacts = [proposal_path, directory / "config.json", directory / "summary.json", predecessor]
    artifacts.extend(directory / "cases" / f"{index:03d}.json" for index in cpu.INDICES)
    config["input_sha256"].update({str(path): metric.sha256(path) for path in artifacts})
    sources = {Path(__file__).resolve()}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(metric.REPO) and Path(module.__file__).is_file())
    config["source_sha256"].update({str(path): metric.sha256(path) for path in sources})
    return config


def verify_config(config, output):
    if (config["case_indices"] != list(INDICES)
            or config["pooled_rmse_parity"]["rtol"] != PARITY_RTOL
            or config["pooled_rmse_parity"]["atol"] != PARITY_ATOL
            or metric.sha256(output / "config.json") != metric.read(output / "registration.json")["config_sha256"]):
        raise ValueError("Full diagnostic registration changed")
    for field in ("input_sha256", "source_sha256", "checkpoint_sha256"):
        metric.verify_hashes(config[field])


def register(config, output):
    output.mkdir(parents=True, exist_ok=False)
    storage_guard(output)
    metric.write_json(output / "config.json", config)
    for source in config["source_sha256"]:
        destination = output / "sources" / Path(source).relative_to(metric.REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    metric.write_json(output / "registration.json", {"config_sha256": metric.sha256(output / "config.json"),
                      "registered_unix": time.time(), "training_performed": False, "checkpoint_selection_performed": False})
    metric.write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})
    verify_config(config, output)


@torch.no_grad()
def diagnose(config, output):
    verify_config(config, output)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True)
    confirmation = Path(config["confirmation"])
    stored = {name: metric.read(confirmation / f"{name}.json")["cases"]
              for name in ("reference", "initialization", "student")}
    models = {}
    for role, checkpoint in config["checkpoints"].items():
        arguments = metric.control_recipe()
        arguments.init_model_checkpoint = checkpoint
        arguments.seed = config["model_initialization_seed"]
        model, optimizer, scheduler, criterion = metric.initialize_student(arguments, device)
        del optimizer, scheduler
        models[role] = model.eval().requires_grad_(False)
    scorers = {suite: metric.load_original(path, "compute_normalized_rmse_columnwise") for suite, path in metric.SCORERS.items()}
    cases, flat, comparisons = [], [], []
    for index in INDICES:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        episode = metric.restore_episode(bank[index], device)
        validate_episode(episode, index)
        if list(episode.source.shape[1:]) != config["case_shapes"][str(index)]:
            raise ValueError("Original full context dimensions changed")
        if any(rows[index]["index"] != index or rows[index]["seed"] != episode.metadata["seed"] for rows in stored.values()):
            raise ValueError("Original BF16 records must match every case index and seed")
        with compute_context(device):
            predictions = {role: metric.streamed_point_predictions(model, episode.source, criterion, **metric.STREAM_OPTIONS)
                           for role, model in models.items()}
        predictions["observed_blend"], routing = metric.blend_predictions(episode.source, predictions["general"], predictions["specialist"])
        arrays = {role: predictions[role][0].float().cpu().numpy() for role in cpu.ROLES}
        truth, mask = episode.target[0].cpu().numpy(), episode.mask[0].cpu().numpy()
        scores = {role: metric.score_queries(truth, arrays[role], mask, scorers) for role in cpu.ROLES}
        previous = {"general": stored["reference"][index]["general"]["rmse"],
                    "observed_blend": stored["reference"][index]["teacher"]["rmse"],
                    "initialization": stored["initialization"][index]["student"]["rmse"],
                    "candidate": stored["student"][index]["student"]["rmse"]}
        verify_pooled_parity(scores, previous)
        path = output / "predictions" / f"{index:03d}.npz"
        path.parent.mkdir(exist_ok=True)
        metric.atomic_npz(path, **arrays)
        case = {"index": index, **episode.metadata, "routing": routing, "scores": scores,
                "stored_bf16_pooled_rmse": previous,
                "pooled_parity_absolute_errors": {role: abs(scores[role]["pooled_rmse"] - value) for role, value in previous.items()},
                "prediction_sha256": metric.sha256(path), "seconds": time.monotonic() - started,
                "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
        if index < 24:
            original = metric.read(Path(config["cpu_diagnostic"]) / "cases" / f"{index:03d}.json")
            if original["index"] != index or original["seed"] != case["seed"]:
                raise ValueError("CPU comparison must use the identical case")
            for role in cpu.ROLES:
                row = {"index": index, "seed": case["seed"], "prior": case["prior"], "role": role}
                for name in metric.METRICS:
                    row[f"cpu_{name}"] = original["scores"][role][name]
                    row[f"bf16_{name}"] = scores[role][name]
                    row[f"bf16_minus_cpu_{name}"] = scores[role][name] - original["scores"][role][name]
                comparisons.append(row)
        cases.append(case)
        metric.write_json(output / "cases" / f"{index:03d}.json", case)
        flat.extend({"index": index, "seed": case["seed"], "prior": case["prior"], "role": role, **scores[role]} for role in cpu.ROLES)
        metric.write_json(output / "status.json", {"state": "diagnosing", "completed": len(cases), "expected": len(INDICES),
                                                  "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case": index, "prior": case["prior"], "seconds": case["seconds"], "peak_gib": case["peak_gib"]}), flush=True)
        del episode, predictions, arrays, truth, mask
    for role, model in models.items():
        original = torch.load(config["checkpoints"][role], map_location="cpu", weights_only=True)["model"]
        if any(not torch.equal(value.cpu(), original[name]) for name, value in model.state_dict().items()):
            raise ValueError("Diagnostic modified a frozen model state")
    verify_config(config, output)
    rows = cpu.summarize(cases)
    contexts = context_summaries(cases)
    metric.write_csv(output / "per_case.csv", flat)
    metric.write_csv(output / "by_family.csv", rows)
    metric.write_csv(output / "by_context.csv", contexts)
    metric.write_csv(output / "cpu_bf16_comparison.csv", comparisons)
    metric.write_json(output / "summary.json", {"complete": True, "cases": len(cases), "small_cases": 24, "large_cases": 6,
        "summary": [row for row in rows if row["family"] == "ALL"], "contexts": contexts,
        "pooled_parity_checks": len(cases) * len(cpu.ROLES), "pooled_parity_passed": True,
        "maximum_pooled_parity_absolute_error": max(error for case in cases for error in case["pooled_parity_absolute_errors"].values()),
        "model_state_unchanged": True, "gate_changed": False, "training_performed": False,
        "checkpoint_selection_performed": False, "config_sha256": metric.sha256(output / "config.json"),
        "limitations": config["limitations"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output = arguments.output.resolve()
    if arguments.stage == "register":
        if arguments.proposal is None or arguments.wait:
            parser.error("Registration requires a proposal and does not wait or use CUDA")
        register(make_config(arguments.proposal.resolve()), output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if metric.read(output / "status.json")["state"] != "registered":
            raise ValueError("Never overwrite a started diagnostic")
        try:
            config = metric.read(output / "config.json")
            verify_config(config, output)
            while not campaign_ready(Path(config["after_campaign"])):
                if not arguments.wait:
                    raise RuntimeError("Original v4 campaign is still active")
                metric.write_json(output / "status.json", {"state": "waiting_for_campaign", "pid": os.getpid(), "updated_unix": time.time()})
                time.sleep(30)
            wait_for_gpu(output, arguments.wait, False)
            storage_guard(output)
            metric.write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
            metric.write_json(output / "status.json", {"state": "loading_models", "pid": os.getpid(), "updated_unix": time.time()})
            diagnose(config, output)
            metric.write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            metric.write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
