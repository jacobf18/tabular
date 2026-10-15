"""Measure the declared loss gradients on frozen generic cases without updates."""
import argparse
import fcntl
import importlib.resources
import json
import math
from pathlib import Path
import time

import torch

from diagnose_dwip_parameter_gradients import gradient_alignment, select_cases, summarize
from rowcol_fixed_validation import preserved_rng
from run_dwip_anchor_mix_point_objective import verify_point_only_change
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable


def require_completed_stages(root):
    for folder in (root, *(root / f"{stage}_evaluation" for stage in ("transfer", "retention", "runtime"))):
        if json.loads((folder / "status.json").read_text())["state"] != "complete":
            raise ValueError(f"Registered stage is not complete: {folder}")
        if not (folder / "results.json").is_file():
            raise ValueError(f"Completed stage lacks results: {folder}")


def matched_commands(control, candidate):
    commands = [list(command) for command in (control, candidate)]
    for command in commands:
        for flag in ("--run-name", "--fixed-validation-cache"):
            if command.count(flag) != 1:
                raise ValueError(f"Require one explicit output flag: {flag}")
            command[command.index(flag) + 1] = "<output-specific>"
    verify_point_only_change(*commands)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen diagnostic input changed: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    queues = {label: REPO / "runs/objective_queues" / name for label, name in
              (("control", args.control), ("candidate", args.queue))}
    for root in queues.values():
        require_completed_stages(root)
    commands = {label: json.loads((root / "dwip1e5_command.json").read_text()) for label, root in queues.items()}
    matched_commands(commands["control"], commands["candidate"])
    if gpu_processes():
        raise RuntimeError("GPU is busy; no diagnostic started")
    output = REPO / "runs/objective_queues" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((output / name).exists() for name in ("manifest.json", "results.json")):
        raise ValueError("Preserve existing diagnostic; use a new experiment")
    spec = dwip_spec()
    checkpoints = {"original": Path(spec["checkpoint_path"]), **{
        label: WORK / f"tabimpute-{root.name}-dwip1e5/best_validation_checkpoint.pth"
        for label, root in queues.items()}}
    weights = {"original": .2, "control": .2, "candidate": 2.}
    cache = REPO / "runs/objective_queues/dwip_spectral_20260915/validation.pth"
    prior_manifest = REPO / "runs/objective_queues/dwip_parameter_gradients_20260916/manifest.json"
    reference = json.loads(prior_manifest.read_text())
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    if sha256(checkpoints["original"]) != reference["checkpoint_sha256"]["original"]:
        raise ValueError("Original checkpoint changed")
    if spec["model_config"] != reference["model_config"]:
        raise ValueError("Original diagnostic architecture changed")
    verify_hashes(reference["source_sha256"])
    if sha256(cache) != reference["cache_sha256"] or sha256(borders) != reference["borders_sha256"]:
        raise ValueError("Original generic diagnostic inputs changed")
    cases = select_cases(torch.load(cache, map_location="cpu", weights_only=True)["cases"], 2)
    if [case["id"] for case in cases] != reference["case_ids"]:
        raise ValueError("Original diagnostic case identities changed")
    paths = [Path(__file__), prior_manifest, cache, borders, *checkpoints.values(),
             REPO / "scripts/diagnose_dwip_parameter_gradients.py",
             REPO / "scripts/run_dwip_anchor_mix_point_objective.py",
             REPO / "scripts/rowcol_fixed_validation.py",
             REPO / "src/tabimpute/model/model_new_stable.py",
             REPO / "src/tabimpute/model/bar_distribution.py"]
    selected_steps = {}
    for label, root in queues.items():
        metrics = checkpoints[label].parent / "best_validation_metrics.json"
        selected_steps[label] = json.loads(metrics.read_text())["step"]
        paths.extend([metrics, root / "manifest.json", root / "dwip1e5_command.json"])
        expected = json.loads((root / "runtime_evaluation/checkpoint_sha256.json").read_text())["selected"]
        if sha256(checkpoints[label]) != expected:
            raise ValueError("Checkpoint differs from completed evaluations")
    manifest = {"input_sha256": {str(path): sha256(path) for path in paths},
        "checkpoint_paths": {label: str(path) for label, path in checkpoints.items()},
        "selected_steps": selected_steps, "point_mse_weights": weights, "nll_weight": 1.,
        "case_ids": [case["id"] for case in cases], "model_config": spec["model_config"],
        "protocol": "Same24 generic cases and RNG20260916 as original diagnostic. FP32 weights, BF16 autocast, eval mode, cuDNN SDPA disabled. Actual declared weighted-point and NLL backward passes; no optimizer or updates.",
        "limitations": "Fixed spectral masks, not the full training mixture or independent holdout. No CDF, clipping, Adam preconditioning or actual training-mode gradient measurement. Selected steps differ. Norm ratios reflect both loss weight and checkpoint state; not causal alignment evidence for training batches.",
        "selection": "Report-only diagnostic. No benchmark inputs or new training coefficient/seed/checkpoint choices."}
    write_json(output / "manifest.json", manifest)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    criterion = FullSupportBarDistribution(torch.load(borders, weights_only=True).cuda())
    results = {}
    try:
        verify_hashes(manifest["input_sha256"])
        for label, checkpoint in checkpoints.items():
            started = time.monotonic()
            model = TabImputeModelStable(**spec["model_config"]).cuda().eval()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise ValueError("Expected FP32 weights")
            names, parameters = zip(*model.named_parameters())
            records = []
            torch.cuda.reset_peak_memory_stats()
            for case in cases:
                write_json(output / "status.json", {"state": "diagnosing", "model": label, "case": case["id"]})
                source, mask = case["source"].cuda(), case["mask"].cuda()
                target = case["target"].cuda()[mask].float()
                with preserved_rng(20260916), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(source, decode_mask=mask)
                nll = criterion(logits.float(), target).mean()
                point = (criterion.mean(logits.float()) - target).square().mean()
                nll_gradients = torch.autograd.grad(nll, parameters, retain_graph=True, allow_unused=True)
                point_gradients = torch.autograd.grad(weights[label] * point, parameters, allow_unused=True)
                record = {"id": case["id"], "group": case["group"], "shape": list(source.shape),
                    "target_count": target.numel(), "nll": float(nll.detach()), "mse": float(point.detach()),
                    "point_weight": weights[label], "alignment": gradient_alignment(names, nll_gradients, point_gradients)}
                if not all(math.isfinite(record[key]) for key in ("nll", "mse")):
                    raise FloatingPointError("Nonfinite diagnostic losses")
                records.append(record)
                write_json(output / f"{label}_cases.json", records)
                del logits, nll, point, nll_gradients, point_gradients
            results[label] = {"case_count": len(records), "summary": summarize(records),
                "group_summaries": {group: summarize([record for record in records if record["group"] == group])["all"]
                                    for group in sorted({record["group"] for record in records})},
                "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
            verify_hashes(manifest["input_sha256"])
            write_json(output / "results.json", results)
            print(label, json.dumps(results[label]["summary"]["all"]), flush=True)
            del model, parameters
            torch.cuda.empty_cache()
        write_json(output / "status.json", {"state": "complete", "models": list(results)})
    except Exception as error:
        write_json(output / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
