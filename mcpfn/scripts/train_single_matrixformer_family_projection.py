"""Paired hundred-update training with matched synthetic batches and family-protected AdamW."""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time

import numpy as np
import torch

from audit_single_matrixformer_structured_training import close_values
from dwip_teacher_mean_training import verify_hashes
import preflight_single_model_family_projection as preflight
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_family_projection as projection
import single_model_fixed_contraction as fixed
import single_model_structured_mixture as mixture
from streamed_large_matrix_training import atomic_torch_save, learning_rate, rng_state
from train_single_matrixformer import storage_guard
from train_single_matrixformer_column_aligned import repeated_bank
from train_streamed_large_matrices import append_json, optimizer_update, wait_for_gpu


ROOT = fixed.ROOT
PAIR = ROOT / "family_projection_v12_pair"
ARMS = {name: ROOT / f"family_projection_v12_{name}" for name in ("control", "projected")}
STEPS = 100
EVERY = 25
RAW_STEPS = (1, 100)
DATA_SEED = 141100901
GRADIENT_SEED = 142100901
CONFIRMATION_SEEDS = {"control": (151100901, 152100901), "projected": (153100901, 154100901)}
POLICY = "Two ordinary single-network arms from selected V8; identical1400 fresh cached synthetic episodes, teacher targets, query seeds,100 AdamW updates and validation bank. Full14-task batches: legacy .25, twelve structured families .25/12 each, ratings .5. Only the projected arm changes the actual Adam proposal using the registered family projection. No inference ensemble."
QUALIFICATION = "Complete both100-update arms and their audits; select each arm's lowest minimax trained boundary among all22-guard-retained strict improvements versus frozen V8, exact ties keep the earlier boundary. Independently confirm each selected arm on its reserved284-case bank with the unchanged gate, without fallback. Run the projected frozen real campaign first and the control campaign second, only for confirmed arms."


def read(path):
    return json.loads(Path(path).read_text())


def training_batches():
    result = []
    for step in range(1, STEPS + 1):
        seed = DATA_SEED + (step - 1) * 1400
        legacy = {"domain": "imputation", "seed": seed, "component": (step - 1) % 8,
                  "shape": list(((256, 128), (2000, 1000))[((step - 1) // 8) % 2]),
                  "regime": ("legacy", "area_scaled")[((step - 1) // 16) % 2], "transpose": False}
        tasks = [{"name": "legacy", "weight": .25, "specification": legacy}]
        cycle = (step - 1) // 4
        for index, family in enumerate(mixture.FAMILIES):
            shapes = mixture.COMPACT_SHAPES if family in mixture.NEURAL_FAMILIES else mixture.SPECTRAL_SHAPES
            specification = {"domain": "imputation", "structured": {"family": family,
                "component": mixture.COMPONENTS[(step - 1 + index) % 4], "seed": seed + 100 * len(tasks),
                "shape": list(shapes[(cycle + index) % 2]), "regime": ("legacy", "area_scaled")[(cycle // 2 + index) % 2],
                "requested_missingness": (.2, .6)[(cycle // 4 + index) % 2]}}
            tasks.append({"name": family, "weight": .25 / 12, "specification": specification})
        tasks.append({"name": "recsys", "weight": .5, "specification": {"domain": "recsys", "seed": seed + 100 * len(tasks),
                      "shape": list(((1000, 1000), (2000, 1000))[(step - 1) % 2])}})
        for index, task in enumerate(tasks):
            task.update(index=index, gradient_seed=GRADIENT_SEED + (step - 1) * 1400 + index * 100)
        result.append({"step": step, "tasks": tasks})
    return result


def confirmation_cases(arm):
    legacy, structured = CONFIRMATION_SEEDS[arm]
    return repeated_bank(legacy, confirmation=True) + mixture.structured_bank(structured, confirmation=True)


def verify_registration(pair):
    config = read(pair / "config.json")
    registration = read(pair / "registration.json")
    if registration["config_sha256"] != sha256(pair / "config.json"):
        raise ValueError("Paired training registration changed")
    if (config["batches"] != training_batches() or config["steps"] != STEPS or config["validate_every"] != EVERY
            or config["raw_steps"] != list(RAW_STEPS) or config["policy"] != POLICY or config["qualification"] != QUALIFICATION
            or config["projection"] != projection.SPECIFICATION or config["peak_lr"] != 1e-5 or config["warmup_steps"] != 5
            or config["arms"] != {name: str(path) for name, path in ARMS.items()}):
        raise ValueError("Never change the prospective data, optimizer, selection or confirmation recipe")
    fixed.validate_recipe(config["base"])
    if config["base"] != read(fixed.PRODUCER / "config.json"):
        raise ValueError("Original scalar losses, teachers, architecture or normalization changed")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    verify_hashes(registration["arm_config_sha256"])
    for name, arm in ARMS.items():
        current = read(arm / "config.json")
        if (current["pair_config_sha256"] != registration["config_sha256"] or current["arm"] != name
                or current["pair"] != str(pair) or current["model"] != config["base"]["model"]
                or current["confirmation_cases"] != json.loads(json.dumps(confirmation_cases(name)))
                or current["qualification"] != QUALIFICATION):
            raise ValueError("Arm configuration or fresh confirmation reservation changed")
    return config


def register(pair):
    if pair != PAIR or any(path.exists() for path in (pair, *ARMS.values())):
        raise FileExistsError("Use only the fresh registered pair and arm locations")
    base = read(fixed.PRODUCER / "config.json")
    fixed.validate_recipe(base)
    mixture.backend.verify_run(base, fixed.PRODUCER)
    evidence = ROOT / "family_projection_preflight_v1_audit_recovery1"
    audit = read(evidence / "audit.json")
    if (read(evidence / "status.json")["state"] != "complete" or not audit["complete"] or not audit["recovery_complete"]
            or audit["gradient_vectors_remeasured"] != 14 or not audit["kkt_conditions_verified"]
            or audit["actual_adverse_tasks_after"] != 0 or not audit["original_artifacts_unchanged"]
            or audit["training_restarted"] or audit["selection_tolerance_changed"] or audit["cuda_initialized"]):
        raise ValueError("Require the completed independent real-model projection feasibility audit")
    verify_hashes(audit["artifact_sha256"])
    verify_hashes(audit["recovery_input_sha256"])
    old_configs = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    old_seeds = set().union(*(fixed.seeds_in(read(path)) for path in old_configs))
    batches = training_batches()
    cases = {name: confirmation_cases(name) for name in ARMS}
    seeds = fixed.seeds_in([batches, *cases.values()])
    if len(seeds) != 3368 or old_seeds.intersection(seeds):
        raise ValueError("Reserve all fresh training, gradient and confirmation seeds before execution")
    paths = [fixed.PARENT, fixed.PRODUCER / "config.json", fixed.PRODUCER / "synthetic_bank.pt",
             fixed.PRODUCER / "reference.json", fixed.PRODUCER / "validation_0.json", evidence / "audit.json",
             ROOT / "family_projection_preflight_v1/result.json", ROOT / "domain_gradient_probe_v1/parameter_inventory.json"]
    inputs = {**base["input_sha256"], **{str(path): sha256(path) for path in paths}}
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"base": base, "model": base["model"], "steps": STEPS, "validate_every": EVERY, "peak_lr": 1e-5,
              "warmup_steps": 5, "raw_steps": list(RAW_STEPS), "batches": batches, "policy": POLICY, "qualification": QUALIFICATION,
              "projection": projection.SPECIFICATION, "arms": {name: str(path) for name, path in ARMS.items()},
              "input_sha256": inputs, "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in old_configs},
              "data_policy": "Both arms restore independent copies of the same cached episode and use the identical cached teacher and query seed; no independently regenerated rating bank is asserted identical",
              "audit_scope": "Every saved prediction grid and training episode; all100 logged projection certificates; full raw-gradient snapshots at updates1 and100. No claim of a full autograd or fused-CUDA trajectory replay."}
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])
    pair.mkdir(parents=True)
    storage_guard(pair)
    write_json(pair / "config.json", config)
    for name, arm in ARMS.items():
        arm.mkdir()
        current = {"pair": str(pair), "arm": name, "pair_config_sha256": sha256(pair / "config.json"), "model": base["model"],
                   "confirmation_cases": cases[name], "confirmation_seeds": list(CONFIRMATION_SEEDS[name]), "qualification": QUALIFICATION}
        write_json(arm / "config.json", current)
        write_json(arm / "registration.json", {"config_sha256": sha256(arm / "config.json"), "registered_unix": time.time(), "training_started": False})
        write_json(arm / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})
    write_json(pair / "registration.json", {"config_sha256": sha256(pair / "config.json"), "registered_unix": time.time(),
               "arm_config_sha256": {str(arm / "config.json"): sha256(arm / "config.json") for arm in ARMS.values()},
               "training_started": False, "confirmation_cases_disjoint_from_all_recorded_cases": True})
    shutil.copy2(Path(__file__), pair / Path(__file__).name)
    write_json(pair / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [cpu_copy(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(nested) for nested in value)
    return deepcopy(value)


def update_model(model, optimizer, task_gradients, weights, projected, rate, raw=False):
    before = preflight.flatten(model)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer_before = cpu_copy(optimizer.state_dict()) if raw else None
    optimizer.zero_grad(set_to_none=True)
    combined, combination = projection.combined_gradient(task_gradients, weights)
    combined_before = combined.detach().cpu().clone() if raw else None
    preflight.assign(model, combined, gradients=True)
    norm = optimizer_update(model, optimizer, .5)
    tentative = preflight.flatten(model)
    gram, products, direction_norm = projection.gram_and_products(task_gradients, before.double() - tentative.double())
    solution = None
    if projected:
        solution = projection.solve_projection(gram, products, direction_norm)
        preflight.verify_certificate(gram, products, direction_norm, solution, projection.SPECIFICATION["relative_inward_margin"])
        after = projection.projected_parameters(tentative, task_gradients, solution["raw_gradient_coefficients"])
        preflight.assign(model, after)
        _, actual_products, actual_norm = projection.gram_and_products(task_gradients, before.double() - preflight.flatten(model).double())
        if np.any(actual_products < 0) or not np.any(actual_products > 0):
            raise ValueError("Do not accept an adverse or zero family-protected update")
    else:
        actual_products, actual_norm = products, direction_norm
    record = {"lr": rate, "projected": projected, "combination": combination, "global_norm_before_clip": norm,
              "raw_gram": gram.tolist(), "raw_products_before": products.tolist(), "proposed_step_norm": direction_norm,
              "projection": solution, "actual_products_after": actual_products.tolist(), "actual_step_norm": actual_norm,
              "adverse_tasks_before": int(np.sum(products < 0)), "adverse_tasks_after": int(np.sum(actual_products < 0))}
    snapshot = None
    if raw:
        snapshot = {"before": cpu_copy(before), "tentative": cpu_copy(tentative), "after": cpu_copy(preflight.flatten(model)),
                    "task_gradients": cpu_copy(task_gradients), "combined_gradient": combined_before,
                    "applied_gradient": cpu_copy(preflight.flatten(model, gradients=True)),
                    "optimizer_before": optimizer_before, "optimizer_after": cpu_copy(optimizer.state_dict()), "record": deepcopy(record)}
    optimizer.zero_grad(set_to_none=True)
    return record, snapshot


def checkpoint(model, optimizer, arm, step, best, reference, *, selected=False):
    storage_guard(arm)
    payload = {"model": model.state_dict(), "step": step, "epoch": step, "config_sha256": sha256(arm / "config.json"),
               "bank_sha256": reference["bank_sha256"], "arm": read(arm / "config.json")["arm"],
               "best_step": best["step"], "best_score": best["score"], "preflight_only": False,
               "gradient_updates_performed": step, "independent_confirmation_passed": False}
    if not selected:
        payload.update(optimizer=optimizer.state_dict(), rng=rng_state())
    path = arm / ("selected_checkpoint.pth" if selected else "latest_checkpoint.pth")
    atomic_torch_save(payload, path)
    return sha256(path)


def selected_boundary(best, step, decision):
    if step and decision["passed"] and decision["score"] < best["score"]:
        return {"step": step, "score": decision["score"]}
    return dict(best)


def validate_models(models, optimizers, criterion, config, step, best, bank, reference, baseline):
    for name, model in models.items():
        arm = ARMS[name]
        destination = arm / f"validation_{step:03d}"
        destination.mkdir()
        with mixture.runtime(config["base"], destination):
            metrics, cases = mixture.backend.evaluate_bank(model, criterion, bank)
        if step == 0:
            close_values(metrics, baseline)
        decision = fixed.selection_decision(metrics, baseline, reference["references"], config["base"])
        selected = selected_boundary(best[name], step, decision)
        if selected != best[name]:
            best[name] = selected
            checkpoint(model, optimizers[name], arm, step, best[name], reference, selected=True)
        row = {"step": step, "student": metrics, "cases": cases, "decision": decision, "best_step": best[name]["step"]}
        write_json(arm / f"validation_{step}.json", row)
        append_json(arm / "validation.jsonl", {key: value for key, value in row.items() if key != "cases"})
        print(json.dumps({"arm": name, "step": step, "score": decision["score"], "passed": decision["passed"],
                          "failed_guards": sum(not item["retained"] for item in decision["comparisons"]), "best_step": best[name]["step"]}), flush=True)


def train(pair):
    config = verify_registration(pair)
    wait_for_gpu(pair, True, False)
    bank = torch.load(fixed.PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    reference, baseline = read(fixed.PRODUCER / "reference.json"), read(fixed.PRODUCER / "validation_0.json")["student"]
    initial_score = fixed.selection_decision(baseline, baseline, reference["references"], config["base"])["initialization_score"]
    best = {name: {"step": 0, "score": initial_score} for name in ARMS}
    stop = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda received, frame: stop.append(received))
    with mixture.runtime(config["base"], pair):
        control, control_optimizer, criterion, teachers, sampler = mixture.backend.setup(config["base"])
        models = {"control": control, "projected": deepcopy(control)}
        optimizers = {"control": control_optimizer, "projected": torch.optim.AdamW(
            [{"params": list(models["projected"].parameters()), "lr": config["peak_lr"]}], lr=config["peak_lr"],
            weight_decay=config["base"]["weight_decay"], betas=(.9, .999), eps=1e-8, fused=True)}
        parameters = [(name, list(parameter.shape), parameter.numel()) for name, parameter in control.named_parameters()]
        if json.loads(json.dumps(parameters)) != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json"):
            raise ValueError("Original single-network parameter inventory changed")
        write_json(pair / "parameter_inventory.json", parameters)
        validate_models(models, optimizers, criterion, config, 0, best, bank, reference, baseline)
        for name in ARMS:
            checkpoint(models[name], optimizers[name], ARMS[name], 0, best[name], reference)
        elements = sum(record[2] for record in parameters)
        gradients = {name: torch.empty((14, elements), device="cuda", dtype=torch.float32) for name in ARMS}
        for batch in config["batches"]:
            step, started = batch["step"], time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            directory = pair / "training_batches" / f"step_{step:03d}"
            directory.mkdir(parents=True)
            task_records = []
            for task in batch["tasks"]:
                status = {"state": "training", "step": step - 1, "active_step": step, "task": task["name"],
                          "pid": os.getpid(), "updated_unix": time.time()}
                write_json(pair / "status.json", status)
                for name in ARMS:
                    write_json(ARMS[name] / "status.json", {**status, "best_step": best[name]["step"]})
                episode = sampler.sample(step=1, **task["specification"])
                expected_seed = task["specification"].get("structured", task["specification"])["seed"]
                if episode.metadata.get("seed") != expected_seed or episode.metadata.get("gradient_data") != "synthetic":
                    raise ValueError("Generated training data does not bind its registered synthetic seed")
                cached = mixture.backend.cache_episode(episode)
                teacher, routing = mixture.backend.teacher_target(teachers, criterion, episode)
                path = directory / f'task_{task["index"]:02d}.pt'
                atomic_torch_save({"episode": cached, "teacher": teacher.detach().float().cpu(), "routing": routing, "task": task}, path)
                losses, diagnostics = {}, {}
                del episode
                for name, model in models.items():
                    current = mixture.backend.restore_episode(cached, "cuda")
                    optimizers[name].zero_grad(set_to_none=True)
                    torch.manual_seed(task["gradient_seed"])
                    losses[name] = mixture.backend.backward_episode(model, criterion, current, teacher, 1.)
                    gradients[name][task["index"]].copy_(preflight.flatten(model, gradients=True))
                    diagnostics[name] = deepcopy(current.metadata)
                    optimizers[name].zero_grad(set_to_none=True)
                    del current
                task_records.append({**task, "artifact": str(path), "sha256": sha256(path), "losses": losses,
                                     "diagnostics": diagnostics, "routing": routing})
                del teacher, cached
            write_json(directory / "manifest.json", {"step": step, "tasks": task_records, "same_cached_inputs_and_targets_for_both_arms": True})
            rate = learning_rate(step, STEPS, config["peak_lr"], config["warmup_steps"])
            for name, model in models.items():
                record, snapshot = update_model(model, optimizers[name], gradients[name], [task["weight"] for task in batch["tasks"]],
                                                name == "projected", rate, raw=step in RAW_STEPS)
                if snapshot is not None:
                    path = ARMS[name] / f"raw_update_{step}.pt"
                    atomic_torch_save(snapshot, path)
                    record.update(raw_snapshot=str(path), raw_snapshot_sha256=sha256(path))
                    del snapshot
                record.update(step=step, task_manifest=str(directory / "manifest.json"), task_manifest_sha256=sha256(directory / "manifest.json"),
                              task_losses=[task["losses"][name] for task in task_records])
                append_json(ARMS[name] / "training.jsonl", record)
            elapsed = time.monotonic() - started
            append_json(pair / "training.jsonl", {"step": step, "seconds": elapsed, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "manifest": str(directory / "manifest.json"), "manifest_sha256": sha256(directory / "manifest.json")})
            print(json.dumps({"step": step, "seconds": elapsed, "best": best}), flush=True)
            if step % EVERY == 0:
                validate_models(models, optimizers, criterion, config, step, best, bank, reference, baseline)
            if step % 10 == 0 or stop:
                for name in ARMS:
                    checkpoint(models[name], optimizers[name], ARMS[name], step, best[name], reference)
            if stop:
                break
    verify_registration(pair)
    final = {"state": "interrupted" if stop else "training_complete", "step": step, "pid": os.getpid(), "updated_unix": time.time()}
    for name, arm in ARMS.items():
        write_json(arm / "status.json", {**final, "best_step": best[name]["step"], "best_score": best[name]["score"]})
    write_json(pair / "status.json", {**final, "best": best})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, default=PAIR)
    args = parser.parse_args()
    pair = args.output.resolve()
    if args.stage == "register":
        register(pair)
        return
    with (pair / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(pair / "status.json")["state"] != "registered":
            raise ValueError("Never automatically restart a started paired training run")
        try:
            write_json(pair / "status.json", {"state": "starting", "step": 0, "pid": os.getpid(), "updated_unix": time.time()})
            train(pair)
        except BaseException as error:
            failure = {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()}
            write_json(pair / "status.json", failure)
            for arm in ARMS.values():
                write_json(arm / "status.json", failure)
            raise


if __name__ == "__main__":
    main()
