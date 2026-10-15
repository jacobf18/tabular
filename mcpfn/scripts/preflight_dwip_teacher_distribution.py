"""Preregistered generic-batch gradients and bounded distribution-teacher feasibility."""
import argparse
from collections import Counter
from copy import copy
from datetime import datetime, timezone
import fcntl
import io
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from dwip_teacher_distribution_training import FrozenDistributionTeachers, objective_terms
from dwip_teacher_mean_training import (
    base, build_training_prior, compute_context, control_recipe, initialize_student,
    objective_terms as mean_objective, optimizer_step, read_json, sample_training_batch, verify_hashes,
)
from preflight_dwip_teacher_mean import reference_batch
from rowcol_fixed_validation import preserved_rng
from run_dwip_budgeted_queue_v1 import immutable_json, regular_bytes, RESERVE_BYTES
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json


EXPERIMENT = "dwip_teacher_distribution_preflight_20260918"
DESIGN = "dwip_fresh_compression_replication_20260917"
STUDENT = "dwip_spectral_anchor_mix25_teacher_mean_huber1_64k_20260917"
SOURCES = (
    "scripts/preflight_dwip_teacher_distribution.py",
    "scripts/dwip_teacher_distribution_training.py",
    "tests/test_dwip_teacher_distribution_training.py",
    "tests/test_dwip_teacher_distribution_preflight.py",
)
ALLOWANCE = 32 * 1024**2


def gradient_sum(gradients, weights):
    if not gradients or len(gradients) != len(weights) or len({len(values) for values in gradients}) != 1:
        raise ValueError("Require matching nonempty gradient lists and weights")
    if not all(math.isfinite(weight) for weight in weights):
        raise ValueError("Require finite gradient weights")
    result = []
    for parts in zip(*gradients):
        present = [(value, weight) for value, weight in zip(parts, weights) if value is not None]
        if not present:
            result.append(None)
            continue
        if any(value.shape != present[0][0].shape or value.dtype != torch.float32 or
               not torch.isfinite(value).all() for value, weight in present):
            raise ValueError("Require aligned finite FP32 parameter gradients")
        total = torch.zeros_like(present[0][0])
        for value, weight in present:
            total.add_(value, alpha=weight)
        result.append(total)
    return tuple(result)


def gradient_dot(first, second):
    if len(first) != len(second):
        raise ValueError("Gradient lists must share parameter order and length")
    products = []
    for before, after in zip(first, second):
        for value in (before, after):
            if value is not None and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError("Require finite FP32 parameter gradients")
        if before is not None and after is not None:
            if before.shape != after.shape:
                raise ValueError("Parameter gradient shapes differ")
            products.append((before.double() * after.double()).sum())
    return float(torch.stack(products).sum()) if products else 0.


def gradient_gram(gradients):
    if not gradients or len({len(values) for values in gradients.values()}) != 1:
        raise ValueError("Require aligned nonempty named parameter-gradient lists")
    labels = tuple(gradients)
    first = next((value for values in gradients.values() for value in values if value is not None), None)
    device = first.device if first is not None else torch.device("cpu")
    matrix = torch.zeros(len(labels), len(labels), dtype=torch.float64, device=device)
    for parts in zip(*gradients.values()):
        present = [value for value in parts if value is not None]
        if not present:
            continue
        if any(value.shape != present[0].shape or value.dtype != torch.float32 or value.device != device
               for value in present):
            raise ValueError("Require matching FP32 parameter tensors on one device")
        block = torch.stack([value.flatten() if value is not None else torch.zeros_like(present[0]).flatten()
                             for value in parts]).double()
        if not torch.isfinite(block).all():
            raise ValueError("Require finite FP32 parameter gradients")
        matrix.add_(block @ block.T)
    values = matrix.cpu().tolist()
    return {name: dict(zip(labels, row)) for name, row in zip(labels, values)}


def metrics_from_gram(label, direction, gram, references):
    norm = math.sqrt(gram[label][label])
    result = {"norm": norm, "unused_parameters": sum(value is None for value in direction), "references": {}}
    for name in references:
        reference_norm = math.sqrt(gram[name][name])
        dot = gram[label][name]
        result["references"][name] = {
            "dot": dot, "reference_norm": reference_norm,
            "cosine": dot / (norm * reference_norm) if norm * reference_norm else None,
            "unit_descent_reference_decrease": dot / norm if norm else None,
        }
    return result


def direction_metrics(direction, references):
    gram = gradient_gram({"direction": direction, **references})
    return metrics_from_gram("direction", direction, gram, references)


def batch_digest(source, target, mask, stats):
    return case_digest([{"source": source.detach().cpu(), "target": target.detach().cpu(),
                         "mask": mask.detach().cpu(), "stats": stats}])


def guard(root):
    allocated = regular_bytes(root)
    free = shutil.disk_usage(root).free
    if allocated > ALLOWANCE or free < RESERVE_BYTES + ALLOWANCE - allocated:
        raise RuntimeError("Preserve20GiB reserve and32MiB owned preflight allowance")
    return {"allocated_bytes": allocated, "free_bytes": free, "allowance_bytes": ALLOWANCE,
            "reserve_bytes": RESERVE_BYTES}


def register(root):
    args = control_recipe()
    proposal_path = root.parent / DESIGN / "next_distribution_compression_proposal.json"
    proposal = read_json(proposal_path)
    verify_hashes(proposal["input_sha256"])
    previous_path = root.parent / STUDENT / "manifest.json"
    previous = read_json(previous_path)
    verify_hashes(previous["source_sha256"])
    paths = [proposal_path, previous_path, args.init_model_checkpoint,
             root.parent / DESIGN / "completion_review.json",
             *(REPO / source for source in SOURCES), *previous["source_sha256"]]
    models = {"original": str(Path(args.init_model_checkpoint).resolve()),
              "mean_student": str(WORK / f"tabimpute-{STUDENT}-dwip1e5/best_validation_checkpoint.pth")}
    paths.extend(models.values())
    for member in proposal["preserved"]["teachers"]:
        path = WORK / f"tabimpute-{member['queue']}-dwip1e5/best_validation_checkpoint.pth"
        verify_hashes({path: member["sha256"]})
        paths.append(path)
    fresh = read_json(root.parent / DESIGN / "prespecification.json")
    inputs = {**fresh["input_sha256"], **proposal["input_sha256"],
              **{str(Path(path).resolve()): sha256(path) for path in paths}}
    verify_hashes(inputs)
    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(), "input_sha256": inputs, "models": models,
        "recipe_equivalence_steps": list(range(1, 17)),
        "gradient_steps": list(range(1, 17)), "gradient_data_seed": 2026091801,
        "reference_steps": list(range(1, 9)), "reference_data_seed": 2026091811,
        "corner_shapes": [[2, 320, 100], [2, 50, 50], [2, 80, 35]],
        "corner_missing_probability": .95, "corner_seed": 20260918, "corner_repeats": 3,
        "arithmetic": "FP32 weights/gradients/moments and BF16 forwards. Weighted loss-component gradients and control/candidate total gradients over actual parameters. Equal-batch pooled true-MSE and teacher-mean-MSE references from eight disjoint fresh online batches; report per-batch and pooled-direction projections. Positive dot means an infinitesimal raw-gradient descent direction decreases reference loss; no Adam or finite-update guarantee.",
        "scope": "Two frozen student states, sixteen diagnostic batches and eight reference batches per state, using the unchanged generic online training sampler with preregistered fresh seeds. No validation or public cases are used for gradients. First16 original-seed batches verify sampler equivalence only. Models are not updated during gradient diagnostics; twelve stress-corner Adam steps are discarded in memory without writing checkpoints.",
        "gates": "Exact true-target NLL/CDF preservation, equal teacher-mean numerical identity, detached external teachers, finite actual parameter gradients and unchanged state after diagnostics, finite FP32 weights/moments after stress updates, largest observed stress-step*64000 below20h leaving4h overhead. Gradients are descriptive evidence, not a performance guarantee or coefficient search. Full training orchestration and an admitted storage budget remain separate gates.",
        "storage": guard(root), "training_ready": False,
    }
    immutable_json(root / "manifest.json", manifest)
    immutable_json(root / "registration_status.json", {"state": "registered", "gpu_started": False,
                  "manifest_sha256": sha256(root / "manifest.json")})
    return {"state": "registered", "verified_inputs": len(inputs)}


def pooled_references(model, criterion, teachers, args, manifest, device):
    configured = copy(args)
    configured.data_seed = manifest["reference_data_seed"]
    prior = build_training_prior(configured, device=device)
    parameters = tuple(model.parameters())
    pool, records = {}, []
    for step in manifest["reference_steps"]:
        source, target, mask, stats = sample_training_batch(prior, configured, step, device)
        teachers_output = teachers.distribution_target(source, mask)
        with compute_context(device):
            logits = model(source, decode_mask=mask)
        mean = criterion.mean(logits.float())
        losses = {"pooled_true_mse": (mean - target[mask]).square().mean(),
                  "pooled_teacher_mse": (mean - teachers_output["mean"]).square().mean()}
        for index, (name, loss) in enumerate(losses.items()):
            gradient = torch.autograd.grad(loss, parameters, retain_graph=index == 0, allow_unused=True)
            pool[name] = gradient_sum([pool[name], gradient], [1., 1/len(manifest["reference_steps"])]) if name in pool else gradient_sum([gradient], [1/len(manifest["reference_steps"])])
        records.append({"step": step, "batch_digest": batch_digest(source, target, mask, stats),
                        "shape": list(source.shape), "stats": stats,
                        "losses": {name: float(loss.detach()) for name, loss in losses.items()}})
    return pool, records


def diagnose(model, criterion, teachers, args, manifest, device, root, label):
    pool, reference_records = pooled_references(model, criterion, teachers, args, manifest, device)
    configured = copy(args)
    configured.data_seed = manifest["gradient_data_seed"]
    prior = build_training_prior(configured, device=device)
    parameters = tuple(model.parameters())
    records = []
    for step in manifest["gradient_steps"]:
        write_json(root / "status.json", {"state": "parameter_gradients", "model": label, "step": step})
        source, target, mask, stats = sample_training_batch(prior, configured, step, device)
        teachers_output = teachers.distribution_target(source, mask)
        with compute_context(device):
            logits = model(source, decode_mask=mask)
        candidate = objective_terms(model, criterion, logits, target[mask], teachers_output["probabilities"])
        control = mean_objective(model, criterion, logits, target[mask], teachers_output["mean"])
        if not all(torch.equal(candidate[name], control[name]) for name in ("nll", "cdf", "true_rmse")):
            raise ValueError("Auxiliary substitution changed true-target NLL/CDF")
        mean = criterion.mean(logits.float())
        components = {"nll": candidate["nll"], "huber": 2 * control["point"], "kl": 2 * candidate["kl"],
                      "cdf": .05 * candidate["cdf"], "true_mse": (mean - target[mask]).square().mean(),
                      "teacher_mse": (mean - teachers_output["mean"]).square().mean()}
        gradients = {name: torch.autograd.grad(loss, parameters, retain_graph=index < len(components) - 1, allow_unused=True)
                     for index, (name, loss) in enumerate(components.items())}
        gradients["control_total"] = gradient_sum([gradients[name] for name in ("nll", "huber", "cdf")], [1., 1., 1.])
        gradients["candidate_total"] = gradient_sum([gradients[name] for name in ("nll", "kl", "cdf")], [1., 1., 1.])
        references = {**pool, **{name: gradients[name] for name in ("true_mse", "teacher_mse", "nll")}}
        gram = gradient_gram({**gradients, **pool})
        metrics = {name: metrics_from_gram(name, gradients[name], gram, references)
                   for name in ("nll", "huber", "kl", "cdf", "control_total", "candidate_total")}
        record = {"step": step, "shape": list(source.shape), "stats": stats,
                  "batch_digest": batch_digest(source, target, mask, stats),
                  "components": {name: float(value.detach()) for name, value in components.items()},
                  "directions": metrics,
                  "mean_identity_max_absolute_error": teachers_output["mean_identity_max_absolute_error"],
                  "probability_sum_max_absolute_error": teachers_output["probability_sum_max_absolute_error"]}
        records.append(record)
        immutable_json(root / f"{label}_gradient_{step:02d}.json", record)
        del gradients, references, logits, components, candidate, control, teachers_output, mean
    summary = {}
    for direction in ("control_total", "candidate_total", "huber", "kl"):
        summary[direction] = {}
        for reference in ("pooled_teacher_mse", "pooled_true_mse", "teacher_mse", "true_mse", "nll"):
            values = [record["directions"][direction]["references"][reference] for record in records]
            summary[direction][reference] = {
                "positive_dot_batches": sum(value["dot"] > 0 for value in values),
                "median_cosine": float(np.median([value["cosine"] for value in values if value["cosine"] is not None])),
                "median_unit_descent_decrease": float(np.median([value["unit_descent_reference_decrease"] for value in values])),
            }
    return {"records": records, "reference_records": reference_records, "summary": summary,
            "component_coverage": dict(Counter(record["stats"]["sampled_prior_name"] for record in records))}


def resource_preflight(args, device, teachers, manifest):
    model, optimizer, scheduler, criterion = initialize_student(args, device)
    timings = []
    for shape in manifest["corner_shapes"]:
        with preserved_rng(manifest["corner_seed"]):
            target = torch.randn(shape, device=device)
            mask = torch.rand(shape, device=device) < manifest["corner_missing_probability"]
            source = target.masked_fill(mask, float("nan"))
        seconds = []
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(manifest["corner_repeats"] + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            output = teachers.distribution_target(source, mask)
            with compute_context(device):
                logits = model(source, decode_mask=mask)
            terms = objective_terms(model, criterion, logits, target[mask], output["probabilities"])
            gradient_norm = optimizer_step(model, optimizer, scheduler, terms, args.grad_clip_norm)
            torch.cuda.synchronize()
            if repeat:
                seconds.append(time.monotonic() - started)
            del logits, terms, output
        timings.append({"shape": shape, "missing_cells": int(mask.sum()), "seconds": seconds,
                        "max_seconds": max(seconds), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "last_gradient_norm": gradient_norm})
    states = list(model.state_dict().values())
    moments = [state for state in optimizer.state.values() if "exp_avg" in state]
    if len(states) != 254 or len(moments) != 238 or any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in states):
        raise ValueError("Student FP32 state or optimizer parameter set differs from the control")
    if any(state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all()
           for state in moments for key in ("exp_avg", "exp_avg_sq")):
        raise ValueError("Adam moments must remain finite FP32")
    if any(parameter.grad is not None or parameter.requires_grad for teacher in teachers.models for parameter in teacher.parameters()):
        raise ValueError("Teacher gradients leaked into stress updates")
    sizes = {}
    for label, payload in (
            ("model", base.model_checkpoint(model, step=12, metrics={})),
            ("optimizer", base.training_checkpoint(model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False),
                step=12, smoothed_loss=0., best_smoothed_loss=0., metrics={}))):
        with io.BytesIO() as buffer:
            torch.save(payload, buffer)
            sizes[label] = buffer.tell()
    hours = max(item["max_seconds"] for item in timings) * 64000 / 3600
    if hours >= 20:
        raise RuntimeError("Proposed run exceeds20h training feasibility allowance")
    return {"timings": timings, "projected_max_shape_training_hours": hours, "serialization_bytes": sizes,
            "finite_fp32_model_tensors": len(states), "finite_fp32_adam_moment_pairs": len(moments)}


def evaluate(root):
    manifest = read_json(root / "manifest.json")
    if read_json(root / "registration_status.json")["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("Preflight registration changed")
    verify_hashes(manifest["input_sha256"])
    if gpu_processes():
        raise RuntimeError("GPU is busy; preserve existing jobs")
    if any(root.glob("*_gradient_*.json")):
        raise ValueError("Preserve partial gradient evidence; audit before resuming")
    started = time.monotonic()
    device = torch.device("cuda")
    args = control_recipe()
    model, optimizer, scheduler, criterion = initialize_student(args, device)
    del optimizer, scheduler
    teachers = FrozenDistributionTeachers.load_registered(args, criterion, device)
    prior = build_training_prior(args, device=device)
    reference_prior = build_training_prior(args, device=device)
    equivalent = []
    for step in manifest["recipe_equivalence_steps"]:
        source, target, mask, stats = sample_training_batch(prior, args, step, device)
        expected_source, expected_target, expected_mask, expected_stats = reference_batch(reference_prior, args, step, device)
        torch.testing.assert_close(source, expected_source, rtol=0., atol=0., equal_nan=True)
        if not (torch.equal(target, expected_target) and torch.equal(mask, expected_mask) and stats == expected_stats):
            raise ValueError("Original sampled data/support recipe changed")
        equivalent.append({"step": step, "shape": list(source.shape), "stats": stats,
                           "batch_digest": batch_digest(source, target, mask, stats)})
    results = {}
    for label, checkpoint_path in manifest["models"].items():
        original = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(original, strict=True)
        model.train()
        result = diagnose(model, criterion, teachers, args, manifest, device, root, label)
        if any(not torch.equal(value.detach().cpu(), original[name]) for name, value in model.state_dict().items()):
            raise ValueError("Read-only gradient diagnostic changed student state")
        result["unchanged_model_state"] = True
        results[label] = result
        immutable_json(root / f"{label}_results.json", result)
        del original
    for field in ("records", "reference_records"):
        if [record["batch_digest"] for record in results["original"][field]] != [record["batch_digest"] for record in results["mean_student"][field]]:
            raise ValueError("Student states did not use the same frozen online batches")
    del model
    torch.cuda.empty_cache()
    resources = resource_preflight(args, device, teachers, manifest)
    verify_hashes(manifest["input_sha256"])
    output = {"utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha256(root / "manifest.json"),
              "recipe_equivalence": equivalent, "gradients": results, "resources": resources,
              "elapsed_seconds": time.monotonic() - started, "storage": guard(root),
              "core_preflight_passed": True, "training_ready": False, "checkpoint_files_created": False,
              "remaining_gates": "Interpret the registered parameter-gradient evidence, then implement and admit one unchanged-schedule trainer and full follow-up controller under a freshly reviewed storage budget. No training is admitted by this core preflight."}
    immutable_json(root / "results.json", output)
    write_json(root / "status.json", {"state": "complete", "utc": output["utc"], "training_ready": False})
    return {"state": "complete", "core_preflight_passed": True, "training_ready": False, "resources": resources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = REPO / "runs/objective_queues" / EXPERIMENT
    root.mkdir(exist_ok=True)
    with (root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "results.json").exists():
            raise ValueError("Preserve completed preflight evidence")
        guard(root)
        print(json.dumps(register(root) if args.register_only else evaluate(root), indent=2))


if __name__ == "__main__":
    main()
