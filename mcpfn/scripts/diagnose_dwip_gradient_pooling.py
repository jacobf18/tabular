"""Read-only synthetic audit of gradient pooling, clipping order and frozen AdamW."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.resources
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from diagnose_dwip_huber_training_alignment import point_losses
from diagnose_dwip_matched_objective_gradients import require_completed_stages, verify_hashes
from diagnose_dwip_training_objective import adam_descent, argument, clipping_coefficient, prepare_training_batch
from general_spectral_geometry import install_geometry_generator
from rowcol_fixed_validation import preserved_rng
from run_dwip_anchor_mix_huber_curvature import verify_curvature_only_change
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from train_direct_pure_rabel_curriculum import CurriculumPhase, cdf_alignment_loss
from train_dwip_anchor_mix import ShapeAwareMixedMatrixPrior
from train_dwip_spectral_geometry import DwipTrainingAdapter
from tabimpute.model.bar_distribution import FullSupportBarDistribution


def read_json(path):
    return json.loads(path.read_text())


def norm(vector):
    if vector.ndim != 1 or not vector.numel() or not torch.isfinite(vector).all():
        raise ValueError("Require a finite nonempty vector")
    return float(torch.linalg.vector_norm(vector.double()))


def projection(direction, reference):
    if direction.shape != reference.shape:
        raise ValueError("Projection shapes differ")
    direction_norm, reference_norm = norm(direction), norm(reference)
    product = float(torch.dot(direction.double(), reference.double()))
    return {"norm": direction_norm, "reference_norm": reference_norm,
            "descent_cosine": product / (direction_norm * reference_norm) if direction_norm and reference_norm else None,
            "linearized_loss_delta": -product}


def clipped(vector, maximum=.5):
    return vector * clipping_coefficient(norm(vector), maximum)


def flatten_gradients(parameters, gradients):
    if len(parameters) != len(gradients):
        raise ValueError("Parameter and gradient counts differ")
    active = tuple(gradient is not None for gradient in gradients)
    values = [torch.zeros_like(parameter) if gradient is None else gradient for parameter, gradient in zip(parameters, gradients)]
    if any(value.shape != parameter.shape for parameter, value in zip(parameters, values)):
        raise ValueError("Gradient shapes differ")
    vector = torch.cat([value.detach().flatten() for value in values])
    norm(vector)
    return vector, active


def frozen_adam_direction(parameters, vector, active, optimizer):
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("Require the original single AdamW parameter group")
    group = optimizer["param_groups"][0]
    if group["params"] != list(range(len(parameters))) or len(active) != len(parameters):
        raise ValueError("Canonical parameter order differs")
    if vector.numel() != sum(parameter.numel() for parameter in parameters):
        raise ValueError("Flattened gradient length differs")
    vectors, offset = [], 0
    with torch.no_grad():
        for index, parameter in enumerate(parameters):
            end = offset + parameter.numel()
            gradient = vector[offset:end].view_as(parameter)
            direction = adam_descent(parameter, gradient, optimizer["state"].get(index, {}), group) if active[index] else torch.zeros_like(parameter)
            vectors.append(direction.flatten())
            offset = end
    result = torch.cat(vectors)
    norm(result)
    return result


def pooling_vectors(vectors, maximum=.5):
    if len(vectors) != 4 or any(vector.shape != vectors[0].shape for vector in vectors):
        raise ValueError("Require four equal-shaped microbatch gradients")
    before = [clipped(vector, maximum) for vector in vectors]
    return {"clip_mean_gradient": clipped(torch.stack(vectors).mean(0), maximum),
            "mean_clipped_gradient": torch.stack(before).mean(0)}


def summarize(blocks):
    result = {}
    for method in ("mean_single_adam", "adam_clip_mean", "adam_mean_clipped"):
        result[method] = {}
        for objective in ("reference_mse", "reference_total"):
            values = [block[method][objective]["descent_cosine"] for block in blocks]
            finite = [value for value in values if value is not None]
            result[method][objective] = {"defined_count": len(finite), "total_count": len(values),
                "mean_cosine": float(np.mean(finite)) if finite else None,
                "median_cosine": float(np.median(finite)) if finite else None,
                "negative_count": sum(value < 0 for value in finite)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.experiment in {"", ".", ".."}:
        raise ValueError("Require a fresh single-directory experiment name")
    roots = {"huber2": REPO / "runs/objective_queues/dwip_spectral_anchor_mix25_huber1_64k_20260916",
             "huber4": REPO / "runs/objective_queues/dwip_spectral_anchor_mix25_huber1_point4_64k_20260917"}
    for root in roots.values():
        require_completed_stages(root)
    if gpu_processes():
        raise RuntimeError("GPU busy; no diagnostic launched")
    commands = {label: read_json(root / "dwip1e5_command.json") for label, root in roots.items()}
    normalized = [list(commands[label]) for label in roots]
    for command in normalized:
        for flag in ("--run-name", "--fixed-validation-cache"):
            command[command.index(flag) + 1] = "<output-specific>"
    verify_curvature_only_change(*normalized)
    command = commands["huber2"]
    for flag, expected in (("--nll-loss-weight", "1.0"), ("--cdf-loss-weight", "0.05"),
                           ("--grad-clip-norm", "0.5"), ("--precision-mode", "amp_bfloat16"),
                           ("--point-loss-prediction", "mean")):
        actual = argument(command, flag)
        if actual != expected and (expected in {"amp_bfloat16", "mean"} or float(actual) != float(expected)):
            raise ValueError(f"Frozen objective or precision differs: {flag}")
    phases = json.loads(argument(command, "--phases-json"))
    if len(phases) != 1 or phases[0]["end_step"] != 64000 or phases[0]["batch_size"] != 2:
        raise ValueError("Require stationary64k batch2 control")
    output = REPO / "runs/objective_queues" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "registration.json").exists():
        raise ValueError("Preserve existing diagnostic; no restart")
    if shutil.disk_usage(output).free < 21 * 1024**3:
        raise RuntimeError("Require20GiB reserve plus1GiB allowance")
    checkpoints = {label: WORK / f"tabimpute-{root.name}-dwip1e5/latest_checkpoint.pth" for label, root in roots.items()}
    borders = Path(importlib.resources.files("tabimpute") / "data/borders.pt")
    hashes = {}
    for label, root in roots.items():
        hashes.update({str(REPO / path): expected for path, expected in read_json(root / "manifest.json")["source_sha256"].items()})
        for path in (root / "manifest.json", root / "dwip1e5_command.json", checkpoints[label], checkpoints[label].parent / "config.json"):
            hashes[str(path)] = sha256(path)
    paths = [Path(__file__), borders, *[REPO / "scripts" / name for name in (
        "diagnose_dwip_training_objective.py", "diagnose_dwip_huber_training_alignment.py",
        "diagnose_dwip_matched_objective_gradients.py", "run_dwip_anchor_mix_huber_curvature.py")]]
    hashes.update({str(path): sha256(path) for path in paths})
    verify_hashes(hashes)
    config = read_json(checkpoints["huber2"].parent / "config.json")["model"]
    if config != read_json(checkpoints["huber4"].parent / "config.json")["model"]:
        raise ValueError("Matched architecture differs")
    registration = {"utc": datetime.now(timezone.utc).isoformat(), "input_sha256": hashes,
        "checkpoint_paths": {label: str(path) for label, path in checkpoints.items()}, "model_config": config,
        "prior": json.loads(argument(command, "--prior-overrides-json")), "phase": phases[0],
        "prior_init_seed": int(argument(command, "--data-seed")), "forward_seed": 1201,
        "support_mode": argument(command, "--support-mask-mode"), "support_variance_df": float(argument(command, "--support-variance-df")),
        "chunk_size": int(argument(command, "--distribution-loss-chunk-size")), "point_coefficients": {"huber2": 2., "huber4": 4.},
        "reference_steps": [64001, 64064], "probe_steps": [64065, 64128], "pool_size": 4,
        "basis": "Both completed64k arms clip every sampled training gradient; fixed-case fluctuations and training objective tradeoffs remain. Measure whether batch-direction variation and nonlinear clipping order matter before proposing accumulation or another optimizer factor.",
        "protocol": "128 new unstratified training-mixture batches, actual per-step data seeds just beyond64k. First64 form disjoint mean MSE/total-gradient references, last64 form16 consecutive groups of4 batch2 episodes. Generate on CPU and transfer to CUDA before support masking, exactly as trainer. Same frozen cases and train-mode forward RNG across both terminal64k models. No optimizer updates or checkpoint selection.",
        "numerics": "FP32 weights, moments and gradient pooling, BF16 autocast; exact FP32 means for MSE/Huber. FP64 norms and projections. Analytical AdamW with stored LR, clipping and decay, not fused-kernel rounding reproduction. Inactive parameters retain zero update.",
        "comparisons": "Mean of four independent frozen single-batch Adam directions versus Adam(clip(mean raw gradient)) versus Adam(mean clipped gradient), all using identical frozen moments. Not four sequential optimizer steps or evidence that a larger-batch training schedule works.",
        "limits": "One fixed pool size4, not a grid. Shared finite64-batch references make block comparisons correlated and potentially noisy. Local derivative evidence, not observed loss changes or generalization. No benchmark data, threshold tuning, new prior, training factor or real-data evaluation. At most1GiB new artifacts; preserve20GiB reserve and all checkpoints."}
    write_json(output / "registration.json", registration)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    install_geometry_generator()
    try:
        write_json(output / "status.json", {"state": "generating", "updated_unix": time.time()})
        with preserved_rng(registration["prior_init_seed"]):
            generator = ShapeAwareMixedMatrixPrior(CurriculumPhase(**phases[0]), "cuda", argument(command, "--missingness-type"), registration["prior"])
        cases = []
        for step in range(64001, 64129):
            with preserved_rng(registration["prior_init_seed"] + step):
                (source, target, _, _, _), _ = generator.get_batch()
            if source.device.type != "cpu" or target.device.type != "cpu":
                raise ValueError("Expected original CPU generation before CUDA transfer and support masking")
            source, target, mask, support = prepare_training_batch(source, target, registration, registration["prior_init_seed"] + 10000000 + step)
            cases.append({"id": step, "family": generator.last_prior_name, "source": source.cpu(), "target": target.cpu(),
                          "mask": mask.cpu(), "support": support, "cohort": "reference" if step <= 64064 else "probe"})
        cache = output / "cases.pth"
        torch.save(cases, cache)
        hashes[str(cache)] = sha256(cache)
        write_json(output / "cases_manifest.json", {"sha256": hashes[str(cache)], "cases": [
            {key: value for key, value in case.items() if key not in ("source", "target", "mask")} |
            {"shape": list(case["source"].shape), "target_count": int(case["mask"].sum())} for case in cases],
            "families": {cohort: dict(Counter(case["family"] for case in cases if case["cohort"] == cohort)) for cohort in ("reference", "probe")}})
        del generator, source, target, mask
        criterion = FullSupportBarDistribution(torch.load(borders, weights_only=True).cuda())
        results = {}
        for label, checkpoint in checkpoints.items():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["epoch"] != 64000 or "optimizer" not in payload:
                raise ValueError("Require terminal64k weights and moments")
            model = DwipTrainingAdapter(**config).cuda().train()
            model.load_state_dict(payload["model"], strict=True)
            names, parameters = zip(*model.named_parameters())
            if any(parameter.dtype != torch.float32 for parameter in parameters):
                raise ValueError("Expected FP32 parameters")
            optimizer = payload["optimizer"]
            for state in optimizer["state"].values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all():
                        raise ValueError("Expected finite FP32 moments")
                    state[key] = state[key].cuda()
            references, records, blocks, pending, pending_active, pending_directions = {}, [], [], [], [], []
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for case in cases:
                if shutil.disk_usage(output).free < 20 * 1024**3:
                    raise RuntimeError("Diagnostic storage reserve violated")
                write_json(output / "status.json", {"state": "diagnosing", "model": label, "case": case["id"], "updated_unix": time.time()})
                source, mask = case["source"].cuda(), case["mask"].cuda()
                target = case["target"].cuda()[mask]
                with preserved_rng(registration["forward_seed"] + case["id"]), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(source, decode_mask=mask, prior_index=None)
                means = criterion.mean(logits.float())
                losses = {"nll": criterion(logits.float(), target.float()).mean(), **point_losses(means, target, logits.dtype),
                          "cdf": cdf_alignment_loss(logits, target, criterion, chunk_size=registration["chunk_size"])}
                losses["total"] = losses["nll"] + registration["point_coefficients"][label] * losses["huber"] + .05 * losses["cdf"]
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite diagnostic loss")
                is_reference = case["cohort"] == "reference"
                gradient, active = flatten_gradients(parameters, torch.autograd.grad(losses["total"], parameters, retain_graph=is_reference, allow_unused=True))
                gradient_norm = norm(gradient)
                record = {"id": case["id"], "cohort": case["cohort"], "family": case["family"], "shape": list(source.shape),
                          "support": case["support"], "losses": {name: float(value.detach()) for name, value in losses.items()},
                          "preclip_norm": gradient_norm, "clip_coefficient": clipping_coefficient(gradient_norm, .5)}
                if is_reference:
                    mse_gradient, _ = flatten_gradients(parameters, torch.autograd.grad(losses["mse"], parameters, allow_unused=True))
                    for key, value in (("reference_total", gradient), ("reference_mse", mse_gradient)):
                        if key not in references:
                            references[key] = torch.zeros_like(value)
                        references[key].add_(value / 64)
                    del mse_gradient
                else:
                    direction = frozen_adam_direction(parameters, clipped(gradient), active, optimizer)
                    record["adam"] = {name: projection(direction, reference) for name, reference in references.items()}
                    pending.append(gradient)
                    pending_active.append(active)
                    pending_directions.append(direction)
                    if len(pending) == 4:
                        pooled = pooling_vectors(pending)
                        combined_active = tuple(any(values) for values in zip(*pending_active))
                        directions = {"mean_single_adam": torch.stack(pending_directions).mean(0),
                            "adam_clip_mean": frozen_adam_direction(parameters, pooled["clip_mean_gradient"], combined_active, optimizer),
                            "adam_mean_clipped": frozen_adam_direction(parameters, pooled["mean_clipped_gradient"], combined_active, optimizer)}
                        block = {"ids": list(range(case["id"] - 3, case["id"] + 1)),
                                 "clip_order": projection(pooled["clip_mean_gradient"], pooled["mean_clipped_gradient"]),
                                 "raw_mean_norm": norm(torch.stack(pending).mean(0)),
                                 "mean_microbatch_norm": float(np.mean([norm(value) for value in pending])),
                                 **{method: {name: projection(value, reference) for name, reference in references.items()}
                                    for method, value in directions.items()}}
                        blocks.append(block)
                        write_json(output / f"{label}_blocks.json", blocks)
                        pending, pending_active, pending_directions = [], [], []
                        del pooled, directions, direction
                records.append(record)
                write_json(output / f"{label}_cases.json", records)
                del logits, means, losses, gradient
            if len(blocks) != 16 or pending:
                raise ValueError("Incomplete frozen pooling blocks")
            if any(not torch.equal(parameter.detach().cpu(), payload["model"][name]) for name, parameter in zip(names, parameters)):
                raise ValueError("Read-only diagnostic changed model weights")
            verify_hashes(hashes)
            results[label] = {"summary": summarize(blocks), "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3, "sampled_clipping_fraction": sum(record["clip_coefficient"] < 1 for record in records) / len(records),
                "reference_norms": {name: norm(value) for name, value in references.items()}, "parameters_unchanged": True,
                "limitations": registration["limits"]}
            write_json(output / "results.json", results)
            print(label, json.dumps(results[label]), flush=True)
            del model, parameters, payload, optimizer, references
            torch.cuda.empty_cache()
        write_json(output / "status.json", {"state": "complete", "models": list(results), "cases_per_model": len(cases),
                   "all_inputs_unchanged": True, "parameters_unchanged": True, "updated_unix": time.time()})
    except Exception as error:
        write_json(output / "status.json", {"state": "error", "error": repr(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
