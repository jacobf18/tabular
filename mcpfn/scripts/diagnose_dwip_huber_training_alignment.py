"""Audit exact mean losses and terminal Adam alignment on frozen training batches."""
import argparse
from collections import Counter
import fcntl
import importlib.resources
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as functional

from diagnose_dwip_matched_objective_gradients import require_completed_stages, verify_hashes
from diagnose_dwip_parameter_gradients import gradient_alignment
from diagnose_dwip_residual_influence import residual_influence
from diagnose_dwip_training_objective import combined_report, direction_report, validate_training_batch
from rowcol_fixed_validation import preserved_rng
from run_dwip_anchor_mix_huber_objective import verify_huber_only_change
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from train_direct_pure_rabel_curriculum import cdf_alignment_loss
from train_dwip_spectral_geometry import DwipTrainingAdapter
from tabimpute.model.bar_distribution import FullSupportBarDistribution


def read_json(path):
    return json.loads(Path(path).read_text())


def matched_commands(control, candidate):
    normalized = [list(command) for command in (control, candidate)]
    for command in normalized:
        for flag in ("--run-name", "--fixed-validation-cache"):
            if command.count(flag) != 1:
                raise ValueError(f"Require one output-specific argument: {flag}")
            command[command.index(flag) + 1] = "<output-specific>"
    verify_huber_only_change(*normalized)


def point_losses(means, target, logit_dtype):
    means, target = means.float(), target.float()
    if means.shape != target.shape or not means.numel():
        raise ValueError("Require nonempty paired mean predictions and targets")
    if not torch.isfinite(means).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite mean prediction or target")
    return {"mse": (means - target).square().mean(),
        "huber": functional.huber_loss(means, target, delta=1.),
        "rounded_mse_proxy": (means.to(logit_dtype).float() - target).square().mean()}


def validate_cases(cases):
    identities = [(case["cohort"], case["id"]) for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("Require unique frozen batch identities")
    fresh = [case for case in cases if case["cohort"] == "frozen32"]
    stress = [case for case in cases if case["cohort"] == "targeted_training_tail"]
    families = Counter(case["family"] for case in fresh)
    if len(cases) != 34 or len(families) != 8 or set(families.values()) != {4}:
        raise ValueError("Require the exact eight-family32-batch panel plus two stress batches")
    if [case["id"] for case in fresh] != list(range(32)) or {case["id"] for case in stress} != {2250, 61275}:
        raise ValueError("Frozen case order or training-tail identities differ")
    for case in cases:
        validate_training_batch(case["source"], case["target"], case["mask"])


def statistics(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Require nonempty finite diagnostic values")
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "negative_count": sum(value < 0 for value in values)}


def summarize(records):
    groups = {cohort: [record for record in records if record["cohort"] == cohort]
        for cohort in ("frozen32", "targeted_training_tail")}
    groups.update({f"frozen32/{family}": [record for record in records
        if record["cohort"] == "frozen32" and record["family"] == family]
        for family in sorted({record["family"] for record in records if record["cohort"] == "frozen32"})})
    result = {}
    for group, selected in groups.items():
        values = {f"loss_{name}": [record["losses"][name] for record in selected]
            for name in ("nll", "mse", "huber", "rounded_mse_proxy", "cdf")}
        values.update({f"adam_{name}_descent_cosine": [record["adam"]["objectives"][name]["descent_cosine"]
            for record in selected] for name in ("nll", "mse", "huber", "cdf", "total")})
        values.update({"huber_to_mse_parameter_gradient_ratio": [record["huber_vs_mse"]["groups"]["all"]["point_to_nll_ratio"] for record in selected],
            "huber_mse_parameter_gradient_cosine": [record["huber_vs_mse"]["groups"]["all"]["cosine"] for record in selected],
            "clip_coefficient": [record["clipping"]["coefficient"] for record in selected],
            "gradient_additivity_relative_error": [record["clipping"]["separate_vs_combined_relative_error"] for record in selected]})
        result[group] = {}
        for name, samples in values.items():
            defined = [value for value in samples if value is not None]
            result[group][name] = {"total_count": len(samples), "undefined_count": len(samples) - len(defined),
                **(statistics(defined) if defined else {"count": 0})}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    if any(Path(name).name != name or name in {"", ".", ".."} for name in (args.queue, args.control, args.experiment)):
        raise ValueError("Require single queue directory names")
    roots = {label: REPO / "runs/objective_queues" / name
        for label, name in (("control", args.control), ("candidate", args.queue))}
    for root in roots.values():
        require_completed_stages(root)
    matched_commands(*(read_json(roots[label] / "dwip1e5_command.json") for label in ("control", "candidate")))
    if gpu_processes():
        raise RuntimeError("GPU busy; no competing diagnostic started")
    output = REPO / "runs/objective_queues" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "registration.json").exists():
        raise ValueError("Preserve existing diagnostic; no restart")
    if shutil.disk_usage(output).free < 21 * 1024**3:
        raise RuntimeError("Require20GiB reserve plus1GiB diagnostic allowance")
    previous = output.parent / "dwip_training_objective_diagnostic_20260916_v3"
    residual = output.parent / "dwip_training_residual_influence_20260916_v2"
    for folder in (previous, residual):
        if read_json(folder / "status.json")["state"] != "complete":
            raise ValueError("Frozen source diagnostics must be complete")
    prior_registration = read_json(previous / "registration.json")
    cache = residual / "cases.pth"
    if sha256(cache) != read_json(residual / "cases_sha256.json")["sha256"]:
        raise ValueError("Frozen training batch cache changed")
    hashes = {**prior_registration["input_sha256"], **read_json(residual / "registration.json")["input_sha256"]}
    checkpoints = {label: WORK / f"tabimpute-{root.name}-dwip1e5/latest_checkpoint.pth" for label, root in roots.items()}
    if sha256(checkpoints["control"]) != hashes[prior_registration["checkpoint_paths"]["candidate"]]:
        raise ValueError("Matched MSE control differs from the previous terminal64k diagnostic")
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    paths = [Path(__file__), cache, borders, residual / "cases_sha256.json", previous / "candidate_cases.json",
        previous / "registration.json", residual / "registration.json", *checkpoints.values()]
    for label, root in roots.items():
        hashes.update(read_json(root / "manifest.json")["source_sha256"])
        paths.extend([root / "manifest.json", root / "dwip1e5_command.json", root / "completed_stage_audit.json",
            checkpoints[label].parent / "config.json"])
        if read_json(checkpoints[label].parent / "config.json")["model"] != prior_registration["model_config"]:
            raise ValueError("Terminal model architecture changed")
    paths.extend(REPO / "scripts" / name for name in ("diagnose_dwip_training_objective.py",
        "diagnose_dwip_residual_influence.py", "run_dwip_anchor_mix_huber_objective.py"))
    hashes.update({str(path): sha256(path) for path in paths})
    verify_hashes(hashes)
    registration = {"input_sha256": hashes, "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "model_config": prior_registration["model_config"], "point_functions": {"control": "mse", "candidate": "huber"},
        "point_coefficient": 2., "huber_delta": 1., "nll_coefficient": 1., "cdf_coefficient": .05,
        "clip_norm": .5, "forward_seed": 1201, "chunk_size": prior_registration["chunk_size"],
        "control_reproduction_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "basis": "All six matched training windows show lower NLL but slightly higher BF16-rounded mean-MSE proxy with Huber. Historical exact FP32 MSE is unavailable. Measure exact MSE and Huber parameter/Adam alignment before claiming useful mean descent is suppressed or choosing another factor.",
        "protocol": "Reuse all32 frozen stratified training-shaped batches and the two previously training-selected stress batches, with original CUDA-before-mask provenance. Terminal64k model/moment pairs, never selected checkpoints. FP32 parameters/moments/means, BF16 autocast, train mode, cuDNN SDPA disabled, identical forward seeds. No optimizer updates, new data, hyperparameter grid, or benchmark inputs.",
        "gradient_labels": "huber_vs_mse uses legacy gradient_alignment keys: nll means weighted2*MSE, point means weighted2*Huber1. Adam total uses each model's actual registered point loss, and tests both point objectives at that model.",
        "limits": "Local analytical AdamW next-step vectors including clipping and weight decay, not demonstrated loss changes or gradient-descent convergence. Reused32 cases and two stress cases are descriptive, not fresh confirmation or prevalence estimates. Point losses use exact FP32 means; rounded_mse_proxy remains a different metric. No selection or next training factor."}
    write_json(output / "registration.json", registration)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        cases = torch.load(cache, map_location="cpu", weights_only=True)
        validate_cases(cases)
        reference = {record["id"]: record for record in read_json(previous / "candidate_cases.json")}
        criterion = FullSupportBarDistribution(torch.load(borders, weights_only=True).cuda())
        results = {}
        for label, checkpoint in checkpoints.items():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["epoch"] != 64000 or "optimizer" not in payload:
                raise ValueError("Require terminal64k model and AdamW moments")
            model = DwipTrainingAdapter(**registration["model_config"]).cuda().train()
            model.load_state_dict(payload["model"], strict=True)
            names, parameters = zip(*model.named_parameters())
            if any(parameter.dtype != torch.float32 for parameter in parameters):
                raise ValueError("Require FP32 parameters")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            records = []
            for case in cases:
                if shutil.disk_usage(output).free < 20 * 1024**3:
                    raise RuntimeError("Diagnostic reserve violated")
                write_json(output / "status.json", {"state": "diagnosing", "model": label, "cohort": case["cohort"], "case": case["id"]})
                source, mask = case["source"].cuda(), case["mask"].cuda()
                target = case["target"].cuda()[mask]
                with preserved_rng(registration["forward_seed"] + case["id"]), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(source, decode_mask=mask, prior_index=None)
                means = criterion.mean(logits.float())
                losses = {"nll": criterion(logits.float(), target.float()).mean(),
                    **point_losses(means, target, logits.dtype),
                    "cdf": cdf_alignment_loss(logits, target, criterion, chunk_size=registration["chunk_size"])}
                weighted = {"nll": losses["nll"], "mse": 2 * losses["mse"], "huber": 2 * losses["huber"], "cdf": .05 * losses["cdf"]}
                point_kind = registration["point_functions"][label]
                weighted["total"] = weighted["nll"] + weighted[point_kind] + weighted["cdf"]
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite diagnostic loss")
                gradients = {name: torch.autograd.grad(loss, parameters, retain_graph=name != "total", allow_unused=True)
                    for name, loss in weighted.items()}
                gradients["point"] = gradients[point_kind]
                clipping = combined_report(parameters, gradients, registration["clip_norm"])
                record = {"id": case["id"], "family": case["family"], "cohort": case["cohort"],
                    "shape": list(source.shape), "support": case["support"],
                    "losses": {name: float(loss.detach()) for name, loss in losses.items()},
                    "huber_vs_mse": gradient_alignment(names, gradients["mse"], gradients["huber"]),
                    "nll_vs_mse": gradient_alignment(names, gradients["nll"], gradients["mse"]),
                    "nll_vs_huber": gradient_alignment(names, gradients["nll"], gradients["huber"]),
                    "clipping": clipping, "adam": direction_report(parameters, gradients, payload["optimizer"], clipping["coefficient"]),
                    "residual_influence": residual_influence((means - target.float()).detach())}
                if label == "control" and case["cohort"] == "frozen32":
                    previous_record = reference[case["id"]]
                    if any(record[key] != previous_record[key] for key in ("family", "shape", "support")):
                        raise ValueError("Frozen control batch identity differs")
                    for name, old_name in (("nll", "nll"), ("mse", "point"), ("cdf", "cdf")):
                        if not np.isclose(record["losses"][name], previous_record["losses"][old_name], atol=1e-6, rtol=1e-5):
                            raise ValueError(f"Frozen control loss does not reproduce: {case['id']}/{name}")
                    record["previous_control_losses_reproduced"] = True
                records.append(record)
                write_json(output / f"{label}_cases.json", records)
                del logits, means, losses, weighted, gradients
            if any(not torch.equal(parameter.detach().cpu(), payload["model"][name]) for name, parameter in model.named_parameters()):
                raise ValueError("Read-only diagnostic changed parameters")
            verify_hashes(hashes)
            results[label] = {"summary": summarize(records), "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
            write_json(output / "results.json", results)
            print(label, json.dumps(results[label]["summary"]["frozen32"]), flush=True)
            del model, parameters, payload
            torch.cuda.empty_cache()
        write_json(output / "status.json", {"state": "complete", "models": list(results), "cases_per_model": len(cases),
            "parameters_unchanged": True, "previous_control_losses_reproduced": True})
    except Exception as error:
        write_json(output / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
