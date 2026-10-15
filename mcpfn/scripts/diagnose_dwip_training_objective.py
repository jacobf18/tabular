"""Read-only training-shaped objective and terminal-Adam diagnostics."""
import argparse
from collections import defaultdict
import fcntl
import importlib.resources
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from diagnose_dwip_matched_objective_gradients import matched_commands, require_completed_stages, verify_hashes
from diagnose_dwip_parameter_gradients import gradient_alignment
from general_spectral_geometry import install_geometry_generator
from rowcol_fixed_validation import preserved_rng
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from spectral_support_normalization import apply_shrunk_support_mask
from train_direct_pure_rabel_curriculum import CurriculumPhase, cdf_alignment_loss
from train_dwip_anchor_mix import ShapeAwareMixedMatrixPrior
from train_dwip_spectral_geometry import DwipTrainingAdapter
from tabimpute.model.bar_distribution import FullSupportBarDistribution


def argument(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f"Require one explicit argument: {flag}")
    return command[command.index(flag) + 1]


def clipping_coefficient(norm, maximum):
    if not math.isfinite(norm) or norm < 0 or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Require finite nonnegative norm and positive clipping threshold")
    return min(1., maximum / (norm + 1e-6))


def validate_training_batch(source, target, mask):
    if source.ndim != 3 or source.shape != target.shape or source.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Require matching BxRxC source, target and boolean decode mask")
    if not mask.any() or not torch.isfinite(target).all() or torch.isinf(source).any() or not torch.isnan(source[mask]).all():
        raise FloatingPointError("Invalid generated training batch")


def prepare_training_batch(source, target, registration, seed, device="cuda"):
    source = source.to(device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    source, target, mask, stats = apply_shrunk_support_mask(source, target,
        registration["support_mode"], seed, registration["support_variance_df"])
    validate_training_batch(source, target, mask)
    return source, target, mask, stats


def adam_descent(parameter, gradient, state, group):
    if group.get("amsgrad") or group.get("maximize"):
        raise ValueError("Only ordinary minimizing AdamW is supported")
    first_beta, second_beta = group["betas"]
    step = float(state.get("step", 0)) + 1
    first = state.get("exp_avg", torch.zeros_like(parameter)).to(parameter.device)
    second = state.get("exp_avg_sq", torch.zeros_like(parameter)).to(parameter.device)
    if first.dtype != parameter.dtype or second.dtype != parameter.dtype:
        raise ValueError("Adam moment precision must match parameter precision")
    if not all(torch.isfinite(value).all() for value in (parameter, gradient, first, second)):
        raise FloatingPointError("Nonfinite Adam inputs")
    first = first_beta * first + (1 - first_beta) * gradient
    second = second_beta * second + (1 - second_beta) * gradient.square()
    direction = (first / (1 - first_beta ** step)) / ((second / (1 - second_beta ** step)).sqrt() + group["eps"])
    return group["lr"] * (direction + group["weight_decay"] * parameter)


def direction_report(parameters, gradients, optimizer, coefficient):
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("Require the frozen single-group AdamW configuration")
    group = optimizer["param_groups"][0]
    if group["params"] != list(range(len(parameters))):
        raise ValueError("Adam parameter IDs must match the model's canonical order")
    squared_norm, dots, norms = 0., defaultdict(float), defaultdict(float)
    with torch.no_grad():
        for index, parameter in enumerate(parameters):
            total = gradients["total"][index]
            if total is None:
                continue
            direction = adam_descent(parameter, coefficient * total, optimizer["state"].get(index, {}), group)
            if not torch.isfinite(direction).all():
                raise FloatingPointError("Nonfinite analytical Adam direction")
            squared_norm += float(direction.double().square().sum())
            for name, values in gradients.items():
                gradient = values[index]
                if gradient is not None:
                    dots[name] += float((gradient.double() * direction.double()).sum())
                    norms[name] += float(gradient.double().square().sum())
    norm = math.sqrt(squared_norm)
    return {"norm": norm, "lr": group["lr"], "objectives": {name: {
        "descent_cosine": dots[name] / (math.sqrt(norms[name]) * norm) if norms[name] and norm else None,
        "linearized_loss_delta": -dots[name]} for name in gradients}}


def combined_report(parameters, gradients, maximum):
    squared_error, squared_sum = 0., 0.
    for index, total in enumerate(gradients["total"]):
        if total is None:
            continue
        separate = torch.zeros_like(total)
        for name in ("nll", "point", "cdf"):
            if gradients[name][index] is not None:
                separate = separate + gradients[name][index]
        squared_error += float((separate.double() - total.double()).square().sum())
        squared_sum += float(total.double().square().sum())
    for parameter, gradient in zip(parameters, gradients["total"]):
        parameter.grad = gradient.clone() if gradient is not None else None
    actual_norm = float(torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True))
    coefficient = clipping_coefficient(actual_norm, maximum)
    clipped_norm = math.sqrt(sum(float(parameter.grad.double().square().sum()) for parameter in parameters if parameter.grad is not None))
    for parameter in parameters:
        parameter.grad = None
    return {"preclip_norm": actual_norm, "postclip_norm": clipped_norm, "coefficient": coefficient,
        "separate_vs_combined_relative_error": math.sqrt(squared_error / squared_sum) if squared_sum else None}


def summarize(records):
    result = {}
    for family in ["all", *sorted({record["family"] for record in records})]:
        selected = records if family == "all" else [record for record in records if record["family"] == family]
        values = {"clip_coefficient": [record["clipping"]["coefficient"] for record in selected],
            "point_to_nll_ratio": [record["point_vs_nll"]["groups"]["all"]["point_to_nll_ratio"] for record in selected],
            "point_nll_cosine": [record["point_vs_nll"]["groups"]["all"]["cosine"] for record in selected],
            "cdf_to_nll_ratio": [record["cdf_vs_nll"]["groups"]["all"]["point_to_nll_ratio"] for record in selected],
            "cdf_nll_cosine": [record["cdf_vs_nll"]["groups"]["all"]["cosine"] for record in selected],
            "gradient_additivity_relative_error": [record["clipping"]["separate_vs_combined_relative_error"] for record in selected]}
        if "adam" in selected[0]:
            values.update({f"adam_{name}_descent_cosine": [record["adam"]["objectives"][name]["descent_cosine"] for record in selected]
                           for name in ("nll", "point", "cdf", "total")})
        result[family] = {"case_count": len(selected), "statistics": {name: {
            "min": float(np.min(finite)), "median": float(np.median(finite)), "max": float(np.max(finite)),
            "negative_count": sum(value < 0 for value in finite), "finite_count": len(finite)}
            for name, samples in values.items() if (finite := [value for value in samples if value is not None and math.isfinite(value)])}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    roots = {label: REPO / "runs/objective_queues" / name for label, name in (("control", args.control), ("candidate", args.queue))}
    for root in roots.values():
        require_completed_stages(root)
    commands = {label: json.loads((root / "dwip1e5_command.json").read_text()) for label, root in roots.items()}
    matched_commands(commands["control"], commands["candidate"])
    command = commands["control"]
    for flag, expected in (("--precision-mode", "amp_bfloat16"), ("--point-loss-kind", "mse"),
                           ("--point-loss-prediction", "mean")):
        if argument(command, flag) != expected:
            raise ValueError(f"Unexpected training objective or precision: {flag}")
    if gpu_processes():
        raise RuntimeError("GPU busy; diagnostic not started")
    output = REPO / "runs/objective_queues" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "registration.json").exists():
        raise ValueError("Preserve existing diagnostic; no restart")
    if shutil.disk_usage(output).free < 21 * 1024**3:
        raise RuntimeError("Require20GiB reserve plus1GiB diagnostic allowance")
    spec = dwip_spec()
    checkpoints = {"original": Path(spec["checkpoint_path"]), **{
        label: WORK / f"tabimpute-{root.name}-dwip1e5/latest_checkpoint.pth" for label, root in roots.items()}}
    phases = json.loads(argument(command, "--phases-json"))
    if len(phases) != 1 or phases[0]["end_step"] != 64000:
        raise ValueError("Require matched stationary64k training")
    prior = json.loads(argument(command, "--prior-overrides-json"))
    if len(prior["mixed_priors"]) != 8 or any(component["weight"] != 1. for component in prior["mixed_priors"]):
        raise ValueError("Require frozen equal-mass eight-component mixture")
    phase = CurriculumPhase(**phases[0])
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    hashes = {str(Path(__file__)): sha256(__file__), str(borders): sha256(borders)}
    for label, root in roots.items():
        frozen = json.loads((root / "manifest.json").read_text())
        verify_hashes(frozen["source_sha256"])
        if sha256(checkpoints["original"]) != frozen["initial_checkpoint_sha256"]:
            raise ValueError("Original checkpoint changed")
        hashes.update(frozen["source_sha256"])
        for path in (root / "manifest.json", root / "dwip1e5_command.json", checkpoints[label]):
            hashes[str(path)] = sha256(path)
        config = checkpoints[label].parent / "config.json"
        if json.loads(config.read_text())["model"] != spec["model_config"]:
            raise ValueError("Training checkpoint architecture differs from original")
        hashes[str(config)] = sha256(config)
    hashes[str(checkpoints["original"])] = sha256(checkpoints["original"])
    for name in ("diagnose_dwip_matched_objective_gradients.py", "diagnose_dwip_parameter_gradients.py"):
        path = REPO / "scripts" / name
        hashes[str(path)] = sha256(path)
    weights = {"original": .2, "control": .2, "candidate": 2.}
    seed, per_component = 20260922, 4
    registration = {"input_sha256": hashes, "phase": phases[0], "prior": prior,
        "model_config": spec["model_config"], "checkpoint_paths": {label: str(path) for label, path in checkpoints.items()},
        "point_weights": weights, "nll_weight": 1., "cdf_weight": .05, "clip_norm": .5,
        "batch_seed": seed, "batches_per_component": per_component, "prior_init_seed": int(argument(command, "--data-seed")),
        "forward_seed": 1201, "support_mode": argument(command, "--support-mask-mode"),
        "support_variance_df": float(argument(command, "--support-variance-df")),
        "chunk_size": int(argument(command, "--distribution-loss-chunk-size")),
        "protocol": "Register before generation. Four batches per each frozen equal-mass component, batch2 and unchanged component shapes. Child index then replicate order; batch RNG20260922+index, support RNG+10000000. Transfer generated CPU inputs/targets to CUDA before support masking, exactly as trainer. Train mode, FP32 weights/input/Adam moments and BF16 autocast, cuDNN SDPA disabled. Same cached batches and forward RNG for all models. No optimizer updates.",
        "adam": "Both terminal64k model/moment pairs; analytical next AdamW descent vector at stored terminal LR, including weight decay and actual global clip coefficient. No model writes or fused-kernel rounding reproduction. Original has no Adam diagnostic.",
        "limitations": "Stratified fresh training-shaped sample, not historical batches, selected checkpoints or independent validation. Four cases per family are descriptive, not significance. Terminal Adam directions are local linearizations, not demonstrated loss improvements or a continuation proposal.",
        "selection": "No benchmark inputs, seed/checkpoint/coefficient selection or new training factor."}
    for flag, expected in (("--nll-loss-weight", 1.), ("--cdf-loss-weight", .05), ("--grad-clip-norm", .5)):
        if float(argument(command, flag)) != expected:
            raise ValueError(f"Unexpected frozen loss configuration: {flag}")
    write_json(output / "registration.json", registration)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    install_geometry_generator()
    try:
        write_json(output / "status.json", {"state": "generating"})
        with preserved_rng(registration["prior_init_seed"]):
            generator = ShapeAwareMixedMatrixPrior(phase, "cuda", argument(command, "--missingness-type"), prior)
        cases = []
        for component, child in enumerate(generator.children):
            for replicate in range(per_component):
                index = component * per_component + replicate
                with preserved_rng(seed + index):
                    (source, target, _, _, _), _ = child.get_batch()
                source, target, mask, stats = prepare_training_batch(source, target, registration, seed + 10000000 + index)
                cases.append({"id": index, "family": generator.names[component], "component": component,
                    "source": source.cpu(), "target": target.cpu(), "mask": mask.cpu(), "support": stats})
        cache = output / "cases.pth"
        torch.save(cases, cache)
        hashes[str(cache)] = sha256(cache)
        write_json(output / "cases_manifest.json", {"sha256": hashes[str(cache)], "cases": [
            {key: value for key, value in case.items() if key not in ("source", "target", "mask")} |
            {"shape": list(case["source"].shape), "target_count": int(case["mask"].sum())} for case in cases]})
        del generator, source, target, mask
        criterion = FullSupportBarDistribution(torch.load(borders, weights_only=True).cuda())
        results = {}
        for label, checkpoint in checkpoints.items():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if label != "original" and payload["epoch"] != 64000:
                raise ValueError("Require terminal64k model and optimizer state")
            model = DwipTrainingAdapter(**spec["model_config"]).cuda().train()
            model.load_state_dict(payload["model"], strict=True)
            names, parameters = zip(*model.named_parameters())
            if any(parameter.dtype != torch.float32 for parameter in parameters):
                raise ValueError("Require FP32 parameters")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            records = []
            for case in cases:
                write_json(output / "status.json", {"state": "diagnosing", "model": label, "case": case["id"]})
                source, mask = case["source"].cuda(), case["mask"].cuda()
                target = case["target"].cuda()[mask]
                with preserved_rng(1201 + case["id"]), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(source, decode_mask=mask, prior_index=None)
                losses = {"nll": criterion(logits.float(), target.float()).mean(),
                    "point": (criterion.mean(logits.float()) - target.float()).square().mean(),
                    "cdf": cdf_alignment_loss(logits, target, criterion, chunk_size=registration["chunk_size"])}
                weighted = {"nll": losses["nll"], "point": weights[label] * losses["point"], "cdf": .05 * losses["cdf"]}
                weighted["total"] = sum(weighted.values())
                if not all(torch.isfinite(value) for value in weighted.values()):
                    raise FloatingPointError("Nonfinite training-shaped losses")
                gradients = {name: torch.autograd.grad(loss, parameters, retain_graph=name != "total", allow_unused=True)
                             for name, loss in weighted.items()}
                record = {"id": case["id"], "family": case["family"], "shape": list(source.shape),
                    "support": case["support"], "losses": {name: float(loss.detach()) for name, loss in losses.items()},
                    "point_vs_nll": gradient_alignment(names, gradients["nll"], gradients["point"]),
                    "cdf_vs_nll": gradient_alignment(names, gradients["nll"], gradients["cdf"]),
                    "point_vs_cdf": gradient_alignment(names, gradients["point"], gradients["cdf"]),
                    "clipping": combined_report(parameters, gradients, .5)}
                if label != "original":
                    record["adam"] = direction_report(parameters, gradients, payload["optimizer"], record["clipping"]["coefficient"])
                records.append(record)
                write_json(output / f"{label}_cases.json", records)
                del logits, losses, weighted, gradients
            if any(not torch.equal(parameter.detach().cpu(), payload["model"][name]) for name, parameter in model.named_parameters()):
                raise ValueError("Diagnostic changed parameters")
            verify_hashes(hashes)
            results[label] = {"summary": summarize(records), "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
            write_json(output / "results.json", results)
            print(label, json.dumps(results[label]["summary"]["all"]), flush=True)
            del model, parameters, payload
            torch.cuda.empty_cache()
        write_json(output / "status.json", {"state": "complete", "models": list(results), "parameters_unchanged": True})
    except Exception as error:
        write_json(output / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
