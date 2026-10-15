"""Fresh synthetic parameter-gradient diagnosis, never a restarted training trial."""

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
import preflight_single_model_family_projection as preflight
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_family_projection as projection
import single_model_fixed_contraction as fixed
from single_model_point_metric_gradients import point_metric_terms
from single_model_realized_projection import MARGINS, realized_projection
from single_model_streamed_point_gradients import streamed_point_metric_losses
import single_model_structured_mixture as mixture
from single_model_truth_defined_columns import episode_metrics
from streamed_large_matrix_training import TRAIN_OPTIONS, atomic_torch_save, learning_rate
from train_single_matrixformer import storage_guard
import train_single_matrixformer_family_projection as training
from train_streamed_large_matrices import append_json, wait_for_gpu


ROOT = fixed.ROOT
DATA_SEED = 161100901
GRADIENT_SEED = 162100901
STATES = {"v8": {"checkpoint": str(fixed.PARENT), "next_step": 1, "restore_optimizer": False},
          "projected80": {"checkpoint": str(training.ARMS["projected"] / "latest_checkpoint.pth"),
                          "next_step": 81, "restore_optimizer": True}}
POLICY = ("Fourteen fresh synthetic tasks at fixed V8 and the last persisted V12 projected update80; "
          "unchanged composite objective versus exact-forward RMSE/common-NRMSE or equal-user RMSE/MAE parameter gradients. "
          "Actual AdamW proposals use the original next scheduled learning rate and optimizer state. "
          "Compare composite-only and all-composite-plus-point constraints, with four prospectively bounded inward margins; "
          "save every attempted rounded projection, accept no adverse or all-zero protected direction. "
          "Restore original parameters and optimizer immediately; no new trained checkpoint, candidate selection, "
          "validation, confirmation, or real benchmarks. No claim to reproduce the unsaved V12 update87.")


def read(path):
    return json.loads(Path(path).read_text())


def specifications():
    cases = deepcopy(preflight.specifications())
    for index, case in enumerate(cases):
        case["gradient_seed"] = GRADIENT_SEED + index * 100
        specification = case["specification"].get("structured", case["specification"])
        specification["seed"] = DATA_SEED + index * 100
    return cases


def verify_registration(output):
    config = read(output / "config.json")
    if (sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]
            or config["cases"] != specifications() or config["states"] != STATES or config["policy"] != POLICY
            or config["margins"] != list(MARGINS) or config["streaming"] != TRAIN_OPTIONS):
        raise ValueError("Never alter the registered diagnostic")
    fixed.validate_recipe(config["base"])
    if config["base"] != read(fixed.PRODUCER / "config.json"):
        raise ValueError("Original objective, teachers or normalization changed")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    return config


def register(output):
    if output.exists():
        raise FileExistsError("Never overwrite or restart a diagnostic")
    if shutil.disk_usage(ROOT).free < 55 * 1024**3:
        raise OSError("Reserve enough storage for full vectors while preserving18 GiB free")
    training.verify_registration(training.PAIR)
    evidence = ROOT / "family_projection_v12_terminal_audit/audit.json"
    parity = ROOT / "point_metric_parity_v1/audit.json"
    if not read(evidence)["complete"] or not read(parity)["complete"]:
        raise ValueError("Require completed terminal and scorer parity audits")
    base = read(fixed.PRODUCER / "config.json")
    old_configs = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    old_seeds = set().union(*(fixed.seeds_in(read(path)) for path in old_configs))
    seeds = fixed.seeds_in(specifications())
    if len(seeds) != 28 or old_seeds.intersection(seeds):
        raise ValueError("Require fresh distinct task and gradient seeds")
    inputs = [evidence, parity, fixed.PRODUCER / "config.json", ROOT / "domain_gradient_probe_v1/parameter_inventory.json"]
    inputs.extend(Path(record["checkpoint"]) for record in STATES.values())
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    sources.update(REPO / "tests" / name for name in ("test_single_model_streamed_point_gradients.py",
                   "test_single_model_realized_projection.py", "test_single_model_point_metric_gradients.py",
                   "test_single_model_point_alignment.py"))
    config = {"base": base, "cases": specifications(), "states": STATES, "policy": POLICY,
              "margins": list(MARGINS), "streaming": TRAIN_OPTIONS,
              "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in inputs}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in old_configs},
              "metric_value_rtol": 1e-12, "metric_value_atol": 1e-12,
              "geometry_rtol": 1e-9, "geometry_atol": 1e-12,
              "goal_guards_or_benchmarks_changed": False}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "neural_forward_performed": False, "candidate_selection_performed": False})
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def metric_summary(prediction, episode, records):
    original = episode_metrics(prediction, episode)
    result = {}
    for name, record in records.items():
        if record is None:
            if original[name] is not None:
                raise ValueError("Never discard a defined original point metric")
            result[name] = None
            continue
        np.testing.assert_allclose(record["value"], original[name], rtol=1e-12, atol=1e-12)
        result[name] = {key: value for key, value in record.items() if key not in ("gradient", "query_mask")}
        result[name]["original_scorer_value"] = original[name]
    return result


def compare_metric_records(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Metric record keys changed")
        for key in expected:
            compare_metric_records(actual[key], expected[key])
    elif expected is None or isinstance(expected, (str, bool, int)):
        if actual != expected:
            raise ValueError("Metric definition or exact coverage changed")
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def state_diagnostic(output, name, config, model, optimizer, criterion, cached_tasks):
    directory = output / name
    directory.mkdir()
    specification = config["states"][name]
    saved = torch.load(specification["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    if specification["restore_optimizer"]:
        if saved["step"] != 80 or saved["best_step"] != 0:
            raise ValueError("Require exactly the unselected persisted update80")
        optimizer.load_state_dict(saved["optimizer"])
    elif optimizer.state:
        raise ValueError("V8 diagnostic must start with fresh AdamW moments")
    del saved
    model.train()
    elements = sum(parameter.numel() for parameter in model.parameters())
    gradients = torch.empty((42, elements), dtype=torch.float32, device="cuda")
    entries = [None] * 14
    manifest = []
    torch.cuda.reset_peak_memory_stats()
    for task, cached_path in zip(config["cases"], cached_tasks):
        started = time.monotonic()
        write_json(output / "status.json", {"state": "capturing_gradients", "checkpoint": name, "task": task["name"],
                   "completed": len(manifest), "total": 14, "pid": os.getpid(), "updated_unix": time.time()})
        cached = torch.load(cached_path, map_location="cpu", weights_only=False)
        episode = mixture.backend.restore_episode(cached["episode"], "cuda")
        teacher = cached["teacher"].to("cuda")
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(task["gradient_seed"])
        loss = mixture.backend.backward_episode(model, criterion, episode, teacher, 1.)
        gradients[task["index"]].copy_(preflight.flatten(model, gradients=True))
        entries[task["index"]] = {"task": task["name"], "kind": "composite", "index": task["index"]}
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(task["gradient_seed"])
        with compute_context(episode.source.device):
            prediction, records, losses = streamed_point_metric_losses(model, criterion, episode, **TRAIN_OPTIONS)
        metrics = metric_summary(prediction.detach(), episode, records)
        selected = [(metric, objective) for metric, objective in losses.items() if objective is not None]
        indices = [task["index"]]
        for position, (metric, objective) in enumerate(selected):
            if float(objective.detach()) != records[metric]["value"]:
                raise ValueError("Metric anchoring must preserve the exact reported value")
            optimizer.zero_grad(set_to_none=True)
            objective.backward(retain_graph=position + 1 < len(selected))
            index = len(entries)
            gradients[index].copy_(preflight.flatten(model, gradients=True))
            entries.append({"task": task["name"], "kind": metric, "index": index})
            indices.append(index)
        optimizer.zero_grad(set_to_none=True)
        path = directory / f'task_{task["index"]:02d}.pt'
        storage_guard(output)
        atomic_torch_save({"gradients": gradients[indices].cpu(), "entries": [entries[index] for index in indices],
                           "prediction": prediction.detach().float().cpu(), "metrics": metrics, "composite_loss": loss,
                           "task": task, "cached_task": str(cached_path), "cached_task_sha256": sha256(cached_path)}, path)
        record = {"task": task["name"], "artifact": str(path), "sha256": sha256(path), "indices": indices,
                  "metrics": metrics, "composite_loss": loss, "seconds": time.monotonic() - started}
        manifest.append(record)
        append_json(directory / "tasks.jsonl", record)
        print(json.dumps({"checkpoint": name, "task": task["name"], "captured_gradients": len(entries) - 14 + len(manifest),
                          "seconds": record["seconds"]}), flush=True)
        del prediction, records, losses, selected, objective, episode, teacher, cached
    gradients = gradients[:len(entries)]
    rate = learning_rate(specification["next_step"], 100, 1e-5, 5)
    _, snapshot = training.update_model(model, optimizer, gradients[:14], [task["weight"] for task in config["cases"]],
                                         False, rate, raw=True)
    before, tentative = snapshot["before"].to("cuda"), snapshot["tentative"].to("cuda")
    preflight.assign(model, before)
    optimizer.load_state_dict(snapshot["optimizer_before"])
    optimizer.zero_grad(set_to_none=True)
    if not torch.equal(preflight.flatten(model), before):
        raise ValueError("The diagnostic must restore original parameters")
    path = directory / "proposal.pt"
    storage_guard(output)
    atomic_torch_save(snapshot, path)
    del snapshot
    gram, products, norm = projection.gram_and_products(gradients, before.double() - tentative.double())
    result = {"checkpoint": name, "entries": entries, "tasks": manifest, "proposal": str(path), "proposal_sha256": sha256(path),
              "rate": rate, "raw_gram": gram.tolist(), "adam_products": products.tolist(), "adam_norm": norm,
              "point_gradients": len(entries) - 14, "projections": {}, "parameters_and_optimizer_restored": True,
              "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
    for variant, protected in (("composite_only", gradients[:14]), ("composite_and_points", gradients)):
        attempts = realized_projection(before, tentative, protected)
        summaries = []
        for index, attempt in enumerate(attempts["attempts"]):
            artifact = directory / f"{variant}_attempt_{index}.pt"
            storage_guard(output)
            atomic_torch_save(attempt, artifact)
            summary = {key: value for key, value in attempt.items() if key != "parameters"}
            if attempt["parameters"] is not None:
                _, all_products, actual_norm = projection.gram_and_products(gradients, before.double() - attempt["parameters"].double())
                summary.update(all_products=all_products.tolist(), all_direction_norm=actual_norm)
            summary.update(artifact=str(artifact), sha256=sha256(artifact))
            summaries.append(summary)
        result["projections"][variant] = {"accepted_index": attempts["accepted_index"], "attempts": summaries}
        del attempts
    write_json(directory / "result.json", result)
    del gradients, before, tentative
    gc.collect()
    torch.cuda.empty_cache()
    return result


def execute(output):
    config = verify_registration(output)
    wait_for_gpu(output, True, False)
    cached_tasks = []
    (output / "tasks").mkdir()
    with mixture.runtime(config["base"], output):
        model, optimizer, criterion, teachers, sampler = mixture.backend.setup(config["base"])
        parameters = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
        if parameters != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json"):
            raise ValueError("Original network inventory changed")
        write_json(output / "parameter_inventory.json", parameters)
        for task in config["cases"]:
            episode = sampler.sample(step=1, **task["specification"])
            expected_seed = task["specification"].get("structured", task["specification"])["seed"]
            if episode.metadata.get("seed") != expected_seed or episode.metadata.get("gradient_data") != "synthetic":
                raise ValueError("Fresh task must bind its synthetic seed")
            teacher, routing = mixture.backend.teacher_target(teachers, criterion, episode)
            path = output / "tasks" / f'task_{task["index"]:02d}.pt'
            atomic_torch_save({"episode": mixture.backend.cache_episode(episode), "teacher": teacher.detach().float().cpu(),
                               "routing": routing, "task": task}, path)
            cached_tasks.append(path)
            del teacher, episode
        for name in STATES:
            state_diagnostic(output, name, config, model, optimizer, criterion, cached_tasks)
    verify_registration(output)
    write_json(output / "result.json", {"complete": True, "states": list(STATES), "synthetic_tasks": 14,
               "training_restarted": False, "trained_checkpoint_created": False, "validation_or_confirmation_used": False,
               "real_benchmarks_used": False, "finite_step_accuracy_assessed": False, "goal_achieved": False})


def audit(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Audit must use CPU only")
    config = verify_registration(output)
    if not read(output / "result.json")["complete"] or (output / "audit.json").exists():
        raise ValueError("Require completed unaudited capture")
    hashes = {}
    results = {}
    for name in STATES:
        result_path = output / name / "result.json"
        result = read(result_path)
        hashes[str(result_path)] = sha256(result_path)
        snapshot = torch.load(result["proposal"], map_location="cpu", weights_only=True, mmap=True)
        hashes[result["proposal"]] = result["proposal_sha256"]
        gradients = torch.empty((len(result["entries"]), snapshot["before"].numel()), dtype=torch.float32)
        seen = set()
        for task in result["tasks"]:
            hashes[task["artifact"]] = task["sha256"]
            saved = torch.load(task["artifact"], map_location="cpu", weights_only=False, mmap=True)
            hashes[saved["cached_task"]] = saved["cached_task_sha256"]
            cached = torch.load(saved["cached_task"], map_location="cpu", weights_only=False)
            episode = mixture.backend.restore_episode(cached["episode"], "cpu")
            records = point_metric_terms(saved["prediction"], episode)
            metrics = metric_summary(saved["prediction"], episode, records)
            compare_metric_records(metrics, task["metrics"])
            for row, entry in zip(saved["gradients"], saved["entries"]):
                if (entry != result["entries"][entry["index"]] or not torch.isfinite(row).all()
                        or entry["index"] in seen):
                    raise ValueError("Saved parameter-gradient identity or finiteness changed")
                seen.add(entry["index"])
                gradients[entry["index"]].copy_(row)
            del saved, cached, episode, records
        if seen != set(range(len(result["entries"]))):
            raise ValueError("Every recorded full parameter gradient is mandatory")
        before, tentative = snapshot["before"], snapshot["tentative"]
        gram, products, norm = projection.gram_and_products(gradients, before.double() - tentative.double())
        np.testing.assert_allclose(gram, result["raw_gram"], rtol=config["geometry_rtol"], atol=config["geometry_atol"])
        np.testing.assert_allclose(products, result["adam_products"], rtol=config["geometry_rtol"], atol=config["geometry_atol"])
        np.testing.assert_allclose(norm, result["adam_norm"], rtol=1e-12, atol=1e-12)
        summary = {"gradients": len(result["entries"]), "point_gradients": result["point_gradients"],
                   "adam_adverse_composite": int(np.sum(products[:14] < 0)), "adam_adverse_point": int(np.sum(products[14:] < 0)),
                   "projections": {}}
        for variant, projection_result in result["projections"].items():
            protected_count = 14 if variant == "composite_only" else len(result["entries"])
            accepted = []
            details = []
            for index, recorded in enumerate(projection_result["attempts"]):
                hashes[recorded["artifact"]] = recorded["sha256"]
                attempt = torch.load(recorded["artifact"], map_location="cpu", weights_only=True, mmap=True)
                if attempt["margin"] != MARGINS[index]:
                    raise ValueError("Prospective bounded margin schedule changed")
                if attempt["parameters"] is None:
                    if attempt["accepted"] or "solver_error" not in attempt:
                        raise ValueError("Solver failure cannot be accepted")
                    details.append({"index": index, "solver_error": attempt["solver_error"]})
                    continue
                preflight.verify_certificate(gram[:protected_count, :protected_count], products[:protected_count], norm,
                                             attempt["solution"], attempt["margin"])
                expected = projection.projected_parameters(tentative, gradients[:protected_count],
                                                           attempt["solution"]["raw_gradient_coefficients"])
                lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
                upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
                if not ((attempt["parameters"] >= lower) & (attempt["parameters"] <= upper)).all():
                    raise ValueError("Rounded projection arithmetic exceeds one output ULP")
                _, actual, actual_norm = projection.gram_and_products(gradients, before.double() - attempt["parameters"].double())
                np.testing.assert_allclose(actual, recorded["all_products"], rtol=config["geometry_rtol"], atol=config["geometry_atol"])
                np.testing.assert_allclose(actual[:protected_count], attempt["actual_products"],
                                           rtol=config["geometry_rtol"], atol=config["geometry_atol"])
                np.testing.assert_allclose(actual_norm, attempt["actual_norm"], rtol=1e-12, atol=1e-12)
                passed = bool(np.all(actual[:protected_count] >= 0) and np.any(actual[:protected_count] > 0))
                if passed != attempt["accepted"]:
                    raise ValueError("Realized protection decision does not reproduce")
                if passed:
                    accepted.append(index)
                details.append({"index": index, "margin": attempt["margin"], "accepted": passed,
                                "adverse_composite": int(np.sum(actual[:14] < 0)), "adverse_point": int(np.sum(actual[14:] < 0)),
                                "direction_norm": actual_norm})
                del expected, lower, upper, attempt
            if (accepted[0] if accepted else None) != projection_result["accepted_index"] or len(accepted) > 1:
                raise ValueError("Accept only the first bounded feasible attempt")
            summary["projections"][variant] = {"accepted_index": projection_result["accepted_index"], "attempts": details}
        results[name] = summary
        del gradients, snapshot, before, tentative
        gc.collect()
    verify_hashes(hashes)
    verify_registration(output)
    write_json(output / "audit.json", {"complete": True, "states": results, "input_sha256": hashes,
               "cuda_initialized": False, "autograd_replayed": False, "optimizer_replayed": False,
               "saved_full_gradient_geometry_recomputed": True, "metric_values_recomputed": True,
               "finite_step_accuracy_assessed": False, "goal_achieved": False})
    lines = ["# Point-metric alignment diagnostic", "", POLICY, "",
             "| Fixed state | Adam adverse composite /14 | Adam adverse point | Composite protection: adverse point | Joint protection: adverse point |",
             "|---|---:|---:|---:|---:|"]
    for name, result in results.items():
        counts = []
        for variant in ("composite_only", "composite_and_points"):
            selected = result["projections"][variant]
            counts.append("not feasible" if selected["accepted_index"] is None else str(selected["attempts"][selected["accepted_index"]]["adverse_point"]))
        lines.append(f'| {name} | {result["adam_adverse_composite"]} | {result["adam_adverse_point"]}/{result["point_gradients"]} | {counts[0]} | {counts[1]} |')
    lines.extend(["", "CPU remeasurement checks every saved full parameter vector, all attempted rounded projections, and original point-metric values.",
                  "This is first-order synthetic training evidence only, not held-out retention or a finite-step accuracy improvement.",
                  "The audit does not replay autograd or the CUDA optimizer. No checkpoint is promoted and the single-model goal remains unmet.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = "registered" if args.stage == "run" else "capture_complete"
        if read(output / "status.json")["state"] != expected:
            raise ValueError("Never restart a started stage")
        try:
            write_json(output / "status.json", {"state": "starting" if args.stage == "run" else "auditing",
                       "pid": os.getpid(), "updated_unix": time.time()})
            (execute if args.stage == "run" else audit)(output)
            write_json(output / "status.json", {"state": "capture_complete" if args.stage == "run" else "complete",
                       "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
