"""CPU-only audit of paired training, cached tasks, endpoint updates and selection."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from audit_single_matrixformer_structured_training import close_values
from confirm_single_model_weight_average import inventory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import train_single_matrixformer_family_projection as training


def read_rows(path):
    lines = path.read_text().splitlines(keepends=True)
    if any(not line.endswith("\n") for line in lines):
        raise ValueError("Do not audit incomplete training records")
    return [json.loads(line) for line in lines]


def check_record(record, weights, *, projected):
    gram = np.asarray(record["raw_gram"], dtype=np.float64)
    products = np.asarray(record["raw_products_before"], dtype=np.float64)
    after = np.asarray(record["actual_products_after"], dtype=np.float64)
    if (record["projected"] != projected or gram.shape != (len(weights), len(weights))
            or products.shape != (len(weights),) or after.shape != products.shape
            or not all(np.isfinite(values).all() for values in (gram, products, after))
            or not math.isfinite(record["global_norm_before_clip"]) or record["global_norm_before_clip"] < 0
            or not math.isfinite(record["actual_step_norm"]) or record["actual_step_norm"] <= 0):
        raise ValueError("Invalid logged gradient or parameter update")
    np.testing.assert_allclose(gram, gram.T, rtol=1e-10, atol=1e-12)
    if np.diag(gram).min() < 0:
        raise ValueError("Negative squared gradient norm")
    norms = np.sqrt(np.diag(gram))
    combination = record["combination"]
    if combination["weights"] != weights or combination["unweighted_task_clip"] != .5:
        raise ValueError("Original task weights or clipping changed")
    np.testing.assert_allclose(combination["raw_gradient_norms"], norms, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(combination["weighted_clip_coefficients"], np.asarray(weights) * np.minimum(1., .5 / (norms + 1e-6)), rtol=1e-10, atol=1e-12)
    if record["adverse_tasks_before"] != int(np.sum(products < 0)) or record["adverse_tasks_after"] != int(np.sum(after < 0)):
        raise ValueError("Incorrect reported adverse task count")
    if projected:
        training.preflight.verify_certificate(gram, products, record["proposed_step_norm"], record["projection"],
                                              training.projection.SPECIFICATION["relative_inward_margin"])
        if np.any(after < 0) or not np.any(after > 0):
            raise ValueError("An actual protected update is adverse or zero")
    elif (record["projection"] is not None or not np.array_equal(products, after)
          or record["actual_step_norm"] != record["proposed_step_norm"]):
        raise ValueError("Control must retain its actual Adam proposal unchanged")


def check_optimizer(snapshot, sizes, step, weight_decay):
    before, after = snapshot["optimizer_before"], snapshot["optimizer_after"]
    if len(before["param_groups"]) != 1 or before["param_groups"] != after["param_groups"]:
        raise ValueError("Optimizer parameter groups changed during update")
    group = before["param_groups"][0]
    if (tuple(group["betas"]) != (.9, .999) or group["eps"] != 1e-8 or group["weight_decay"] != weight_decay
            or group["lr"] != snapshot["record"]["lr"] or group.get("amsgrad") or group.get("maximize")
            or len(group["params"]) != len(sizes)):
        raise ValueError("Original AdamW recipe changed")
    if set(after["state"]) != set(group["params"]) or (step == 1 and before["state"]):
        raise ValueError("Endpoint optimizer inventory or first-step initialization changed")
    if step > 1 and set(before["state"]) != set(group["params"]):
        raise ValueError("Missing accumulated optimizer state")
    offset, maximum_error = 0, 0.
    for key, size in zip(group["params"], sizes):
        applied = snapshot["applied_gradient"][offset:offset + size].double()
        original = snapshot["before"][offset:offset + size].double()
        tentative = snapshot["tentative"][offset:offset + size].double()
        state = after["state"][key]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != step:
            raise ValueError("Original AdamW state or update count changed")
        old = before["state"].get(key)
        if old is not None and float(old["step"]) != step - 1:
            raise ValueError("Endpoint is not the registered optimizer update")
        first = torch.zeros_like(applied) if old is None else old["exp_avg"].flatten().double()
        second = torch.zeros_like(applied) if old is None else old["exp_avg_sq"].flatten().double()
        first_expected = .9 * first + .1 * applied
        second_expected = .999 * second + .001 * applied.square()
        epsilon = torch.finfo(torch.float32).eps
        first_bound = 4 * epsilon * (.9 * first.abs() + .1 * applied.abs())
        second_bound = 4 * epsilon * (.999 * second.abs() + .001 * applied.square())
        if (not ((state["exp_avg"].flatten().double() - first_expected).abs() <= first_bound).all()
                or not ((state["exp_avg_sq"].flatten().double() - second_expected).abs() <= second_bound).all()):
            raise ValueError("AdamW moments exceed the FP32 rounding envelope")
        adaptive = group["lr"] * (state["exp_avg"].flatten().double() / (1 - .9 ** step)) / (
            (state["exp_avg_sq"].flatten().double() / (1 - .999 ** step)).sqrt() + 1e-8)
        decay = group["lr"] * weight_decay * original
        error = (tentative - (original - decay - adaptive)).abs()
        bound = 4 * epsilon * (original.abs() + decay.abs() + adaptive.abs())
        if not (error <= bound).all():
            raise ValueError("Actual AdamW proposal exceeds the analytic rounding envelope")
        maximum_error = max(maximum_error, float(error.max()))
        offset += size
    return maximum_error


def check_snapshot(snapshot, record, sizes, step, weight_decay):
    if any(record[key] != value for key, value in snapshot["record"].items()):
        raise ValueError("Snapshot and logged optimizer certificate differ")
    gradients = snapshot["task_gradients"]
    if gradients.dtype != torch.float32 or gradients.shape != (len(record["combination"]["weights"]), sum(sizes)):
        raise ValueError("Preserve full task gradients in original FP32 coordinates")
    for name in ("before", "tentative", "after", "combined_gradient", "applied_gradient"):
        vector = snapshot[name]
        if vector.shape != (sum(sizes),) or vector.dtype != torch.float32 or not torch.isfinite(vector).all():
            raise ValueError("Require finite full FP32 endpoint vectors")
    combined, combination = training.projection.combined_gradient(gradients, record["combination"]["weights"])
    torch.testing.assert_close(combined, snapshot["combined_gradient"], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(combination["weighted_clip_coefficients"], record["combination"]["weighted_clip_coefficients"], rtol=1e-10, atol=1e-12)
    norm = float(torch.linalg.vector_norm(snapshot["combined_gradient"].double()))
    np.testing.assert_allclose(norm, record["global_norm_before_clip"], rtol=1e-6, atol=1e-10)
    applied = snapshot["combined_gradient"] * min(1., .5 / (record["global_norm_before_clip"] + 1e-6))
    torch.testing.assert_close(applied, snapshot["applied_gradient"], rtol=1e-6, atol=1e-10)
    gram, products, norm = training.projection.gram_and_products(gradients, snapshot["before"].double() - snapshot["tentative"].double())
    np.testing.assert_allclose(gram, record["raw_gram"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(products, record["raw_products_before"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(norm, record["proposed_step_norm"], rtol=1e-12, atol=1e-12)
    check_record(record, record["combination"]["weights"], projected=record["projected"])
    if record["projected"]:
        expected = training.projection.projected_parameters(snapshot["tentative"], gradients, record["projection"]["raw_gradient_coefficients"])
        lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
        upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
        if not ((snapshot["after"] >= lower) & (snapshot["after"] <= upper)).all():
            raise ValueError("Projection arithmetic exceeds one FP32 output ULP")
    elif not torch.equal(snapshot["after"], snapshot["tentative"]):
        raise ValueError("Control parameters changed after the proposal")
    _, products, norm = training.projection.gram_and_products(gradients, snapshot["before"].double() - snapshot["after"].double())
    np.testing.assert_allclose(products, record["actual_products_after"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(norm, record["actual_step_norm"], rtol=1e-12, atol=1e-12)
    maximum_error = check_optimizer(snapshot, sizes, step, weight_decay)
    return {"step": step, "gradient_vectors": gradients.shape[0], "maximum_analytic_adamw_error": maximum_error,
            "combined_gradient_norm_accumulation_dtype": "float64", "allowed_projection_output_ulps": 1}


def check_episode(payload, task, logged):
    if payload["task"] != task or payload["routing"] != logged["routing"]:
        raise ValueError("Cached episode does not bind the registered task and teacher routing")
    episode = training.mixture.backend.restore_episode(payload["episode"], "cpu")
    seed = task["specification"].get("structured", task["specification"])["seed"]
    if (episode.metadata.get("seed") != seed or episode.metadata.get("gradient_data") != "synthetic"
            or episode.metadata.get("domain") != task["specification"]["domain"]
            or episode.target.shape != episode.source.shape or episode.mask.shape != episode.source.shape
            or episode.mask.dtype != torch.bool or not episode.mask.any()
            or not torch.isfinite(episode.target).all() or not torch.isfinite(episode.source[~episode.mask]).all()
            or payload["teacher"].shape != episode.target.shape or not torch.isfinite(payload["teacher"]).all()):
        raise ValueError("Incomplete or invalid synthetic training episode")
    if set(logged["losses"]) != set(training.ARMS) or set(logged["diagnostics"]) != set(training.ARMS):
        raise ValueError("Both arms must evaluate the identical cached task")
    for name in training.ARMS:
        if not math.isfinite(logged["losses"][name]) or any(logged["diagnostics"][name][key] != value for key, value in episode.metadata.items()):
            raise ValueError("Training provenance or scalar loss changed")


def audit(pair, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Independent audit must be CPU-only")
    config = training.verify_registration(pair)
    if training.read(pair / "status.json")["state"] != "training_complete":
        raise ValueError("Require both completed arms; never silently audit a shortened trial")
    hashes = {str(pair / name): sha256(pair / name) for name in ("config.json", "registration.json", "training.jsonl", "parameter_inventory.json")}
    parameters = training.read(pair / "parameter_inventory.json")
    if parameters != training.read(training.ROOT / "domain_gradient_probe_v1/parameter_inventory.json"):
        raise ValueError("Original parameter inventory changed")
    paired = read_rows(pair / "training.jsonl")
    rows = {name: read_rows(arm / "training.jsonl") for name, arm in training.ARMS.items()}
    for reports in (paired, *rows.values()):
        if [report["step"] for report in reports] != list(range(1, training.STEPS + 1)):
            raise ValueError("All one hundred matched updates are mandatory")
    for batch, report in zip(config["batches"], paired):
        manifest_path = pair / "training_batches" / f'step_{batch["step"]:03d}' / "manifest.json"
        hashes[str(manifest_path)] = report["manifest_sha256"]
        if report["manifest"] != str(manifest_path):
            raise ValueError("Training manifest location changed")
        verify_hashes({str(manifest_path): hashes[str(manifest_path)]})
        manifest = training.read(manifest_path)
        if manifest["step"] != batch["step"] or len(manifest["tasks"]) != 14 or not manifest["same_cached_inputs_and_targets_for_both_arms"]:
            raise ValueError("Every update must retain all fourteen shared tasks")
        for task, logged in zip(batch["tasks"], manifest["tasks"]):
            if {key: logged[key] for key in task} != task:
                raise ValueError("Registered training task changed")
            path = manifest_path.parent / f'task_{task["index"]:02d}.pt'
            if logged["artifact"] != str(path):
                raise ValueError("Unexpected training artifact location")
            hashes[str(path)] = logged["sha256"]
            verify_hashes({str(path): hashes[str(path)]})
            check_episode(torch.load(path, map_location="cpu", weights_only=True), task, logged)
        for name in training.ARMS:
            record = rows[name][batch["step"] - 1]
            if (record["task_manifest"] != str(manifest_path) or record["task_manifest_sha256"] != hashes[str(manifest_path)]
                    or record["task_losses"] != [task["losses"][name] for task in manifest["tasks"]]
                    or record["lr"] != training.learning_rate(batch["step"], training.STEPS, config["peak_lr"], config["warmup_steps"])):
                raise ValueError("Both arms must bind the same inputs and learning-rate schedule")
            check_record(record, [task["weight"] for task in batch["tasks"]], projected=name == "projected")
    reference = training.read(training.fixed.PRODUCER / "reference.json")
    baseline = training.read(training.fixed.PRODUCER / "validation_0.json")["student"]
    bank = torch.load(training.fixed.PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    if len(bank) != 152 or len(reference["cases"]) != 152:
        raise ValueError("Require every original development case")
    for cached, specification in zip(bank, config["base"]["validation_cases"]):
        if cached["metadata"]["seed"] != specification.get("structured", specification)["seed"]:
            raise ValueError("Development case order or synthetic seed changed")
    for role in ("general", "teacher"):
        if training.mixture.aggregate(reference["cases"], role) != reference[role]:
            raise ValueError("Frozen reference aggregation changed")
    if training.fixed.reference_values(reference["general"], reference["teacher"]) != reference["references"]:
        raise ValueError("Original reference floors changed")
    fields = [(reference, ("general", "teacher"))]
    results = {}
    parent = torch.load(training.fixed.PARENT, map_location="cpu", weights_only=True)["model"]
    for name, arm in training.ARMS.items():
        best = {"step": 0, "score": training.fixed.selection_decision(baseline, baseline, reference["references"], config["base"])["initialization_score"]}
        reports = []
        for step in range(0, training.STEPS + 1, training.EVERY):
            path = arm / f"validation_{step}.json"
            hashes[str(path)] = sha256(path)
            report = training.read(path)
            if report["step"] != step or len(report["cases"]) != 152 or training.mixture.aggregate(report["cases"], "student") != report["student"]:
                raise ValueError("Validation coverage or aggregate changed")
            if step == 0:
                close_values(report["student"], baseline)
            decision = training.fixed.selection_decision(report["student"], baseline, reference["references"], config["base"])
            if report["decision"] != decision or len(decision["comparisons"]) != 22:
                raise ValueError("Original zero-tolerance selection guards changed")
            best = training.selected_boundary(best, step, decision)
            if report["best_step"] != best["step"]:
                raise ValueError("Strict minimax selection or earlier-boundary tie rule changed")
            reports.append(report)
            fields.append((report, ("student",)))
        if read_rows(arm / "validation.jsonl") != [{key: value for key, value in report.items() if key != "cases"} for report in reports]:
            raise ValueError("Validation trace changed")
        status = training.read(arm / "status.json")
        if status["state"] != "training_complete" or status["step"] != training.STEPS or status["best_step"] != best["step"] or status["best_score"] != best["score"]:
            raise ValueError("Terminal selection does not reproduce")
        checkpoints = [arm / "latest_checkpoint.pth"]
        if best["step"]:
            checkpoints.append(arm / "selected_checkpoint.pth")
        elif (arm / "selected_checkpoint.pth").exists():
            raise ValueError("No eligible trained checkpoint was selected")
        if (arm / "best_validation_checkpoint.pth").exists():
            raise ValueError("Never publish an alias before independent confirmation")
        for path in checkpoints:
            state = torch.load(path, map_location="cpu", weights_only=True)
            inventory(state["model"], parent)
            parameter_names = {key for key, shape, size in parameters}
            if any(not torch.equal(value, parent[key]) for key, value in state["model"].items() if key not in parameter_names):
                raise ValueError("A nonparameter model buffer changed")
            expected_step = best["step"] if path.name == "selected_checkpoint.pth" else training.STEPS
            if (state["step"] != expected_step or state["gradient_updates_performed"] != expected_step
                    or state["config_sha256"] != sha256(arm / "config.json") or state["bank_sha256"] != reference["bank_sha256"]
                    or state["arm"] != name or state["independent_confirmation_passed"] or state["preflight_only"]):
                raise ValueError("Training checkpoint provenance changed")
            hashes[str(path)] = sha256(path)
        endpoints = []
        for step in training.RAW_STEPS:
            record = rows[name][step - 1]
            path = arm / f"raw_update_{step}.pt"
            if record["raw_snapshot"] != str(path):
                raise ValueError("Missing raw endpoint snapshot")
            hashes[str(path)] = record["raw_snapshot_sha256"]
            verify_hashes({str(path): hashes[str(path)]})
            snapshot = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if not snapshot["optimizer_before"]["param_groups"][0]["fused"]:
                raise ValueError("Actual training must preserve fused CUDA AdamW")
            endpoints.append(check_snapshot(snapshot, record, [item[2] for item in parameters], step, config["base"]["weight_decay"]))
            if step == 1:
                expected = torch.cat([parent[key].flatten() for key, shape, size in parameters])
                if not torch.equal(snapshot["before"], expected):
                    raise ValueError("Both arms must start from actual selected V8")
            if step == training.STEPS:
                latest = torch.load(arm / "latest_checkpoint.pth", map_location="cpu", weights_only=True)
                expected = torch.cat([latest["model"][key].flatten() for key, shape, size in parameters])
                if not torch.equal(snapshot["after"], expected):
                    raise ValueError("Final checkpoint is not the audited endpoint")
                del latest
            del snapshot, expected
        hashes.update({str(arm / filename): sha256(arm / filename) for filename in ("training.jsonl", "validation.jsonl", "status.json")})
        results[name] = {"best": best, "raw_endpoints": endpoints, "retained_boundaries": [report["step"] for report in reports if report["step"] and report["decision"]["passed"]]}
    initial = {name: torch.load(arm / "raw_update_1.pt", map_location="cpu", weights_only=True, mmap=True) for name, arm in training.ARMS.items()}
    for key in ("before", "task_gradients", "combined_gradient", "applied_gradient", "tentative"):
        if not torch.equal(initial["control"][key], initial["projected"][key]):
            raise ValueError("The first paired Adam proposal must be exactly matched before projection")
    del initial
    count, prediction_hashes = training.fixed.raw_audit(bank, fields)
    if count != 1824:
        raise ValueError("Audit must check every saved full prediction grid")
    hashes.update(prediction_hashes)
    training.verify_registration(pair)
    verify_hashes(hashes)
    write_json(output / "audit.json", {"complete": True, "arms": results, "training_episodes": 1400, "optimizer_updates_per_arm": 100,
        "prediction_grids": count, "logged_projection_certificates": 100, "raw_gradient_vectors_remeasured": 56,
        "artifact_sha256": hashes, "cuda_initialized": torch.cuda.is_initialized(), "autograd_replayed": False,
        "full_optimizer_trajectory_replayed": False, "fused_cuda_adamw_replayed": False, "research_selection_tolerance": 0,
        "goal_success_not_assessed": True, "source_sha256": {str(Path(__file__).resolve()): sha256(Path(__file__))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, default=training.PAIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "status.json", {"state": "auditing", "pid": os.getpid(), "updated_unix": time.time()})
        audit(args.pair.resolve(), output)
        write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
