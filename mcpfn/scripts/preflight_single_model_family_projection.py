"""One fresh fourteen-task actual CUDA AdamW step and its family-protected counterpart."""

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from confirm_single_model_weight_average import inventory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_family_projection as projection
import single_model_fixed_contraction as fixed
import single_model_structured_mixture as mixture
from streamed_large_matrix_training import atomic_torch_save
from train_streamed_large_matrices import optimizer_update, wait_for_gpu


ROOT = fixed.ROOT
DATA_SEED = 121100901
GRADIENT_SEED = 131100901
RATE = 2e-6
POLICY = "Preflight only: one actual fused-CUDA AdamW proposal from selected V8, on14 fresh synthetic training tasks with unchanged scalar losses/teachers. Save its unprojected control and nearest family-protected parameter state; no candidate selection, validation, confirmation or real benchmark."


def read(path):
    return json.loads(Path(path).read_text())


def specifications():
    cases = [{"name": "legacy", "weight": .25, "specification": {"domain": "imputation", "seed": DATA_SEED,
              "shape": [2000, 1000], "component": 0, "transpose": False, "regime": "area_scaled"}}]
    for index, family in enumerate(mixture.FAMILIES):
        shapes = mixture.COMPACT_SHAPES if family in mixture.NEURAL_FAMILIES else mixture.SPECTRAL_SHAPES
        cases.append({"name": family, "weight": .25 / 12, "specification": {"domain": "imputation", "structured": {
            "family": family, "component": mixture.COMPONENTS[index % 4], "seed": DATA_SEED + 100 * len(cases),
            "shape": list(shapes[-1]), "requested_missingness": .6, "regime": "area_scaled"}}})
    cases.append({"name": "recsys", "weight": .5, "specification": {"domain": "recsys", "seed": DATA_SEED + 100 * len(cases),
                                                                              "shape": [2000, 1000]}})
    for index, case in enumerate(cases):
        case.update(index=index, gradient_seed=GRADIENT_SEED + 100 * index)
    return cases


def verify_registration(output):
    config = read(output / "config.json")
    if (read(output / "registration.json")["config_sha256"] != sha256(output / "config.json")
            or config["cases"] != specifications() or config["projection"] != projection.SPECIFICATION
            or config["policy"] != POLICY or config["rate"] != RATE or config["parent"] != str(fixed.PARENT)):
        raise ValueError("Never alter the registered preflight")
    fixed.validate_recipe(config["base"])
    if config["base"] != read(fixed.PRODUCER / "config.json"):
        raise ValueError("Original loss, teacher or initialization recipe changed")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def register(output):
    if output.exists():
        raise FileExistsError("Never overwrite a preflight")
    base = read(fixed.PRODUCER / "config.json")
    fixed.validate_recipe(base)
    mixture.backend.verify_run(base, fixed.PRODUCER)
    if read(ROOT / "displacement_grid_v1_benchmarks/status.json")["state"] != "not_promoted":
        raise ValueError("Do not overlap or restart the preceding campaign")
    evidence = ROOT / "batch_gradient_probe_v1"
    result = read(evidence / "summary.json")
    if not result["complete"] or result["gradient_vectors"] != 128 or result["directional_measurements"] != 768 or result["cuda_initialized"]:
        raise ValueError("Require the completed saved-gradient diagnostic")
    old_configs = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    old_seeds = set().union(*(fixed.seeds_in(read(path)) for path in old_configs))
    cases = specifications()
    seeds = fixed.seeds_in(cases)
    if len(cases) != 14 or len(seeds) != 28 or old_seeds.intersection(seeds):
        raise ValueError("Require fresh distinct task and gradient seeds before any model execution")
    paths = [fixed.PARENT, fixed.PRODUCER / "config.json", ROOT / "displacement_grid_v1_qualification/gate.json"]
    paths.append(ROOT / "domain_gradient_probe_v1/parameter_inventory.json")
    paths += [evidence / name for name in ("config.json", "registration.json", "summary.json", "groups.csv", "directions.csv")]
    inputs = {**base["input_sha256"], **{str(path): sha256(path) for path in paths}}
    sources = {Path(__file__).resolve(), Path(projection.__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"base": base, "model": base["model"], "parent": str(fixed.PARENT), "cases": cases, "rate": RATE,
              "projection": projection.SPECIFICATION, "policy": POLICY, "input_sha256": inputs,
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in old_configs}, "preflight_only": True,
              "candidate_selection_performed": False, "validation_or_confirmation_evaluated": False}
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "model_evaluated": False, "training_updates_performed": 0})
    for path in (Path(__file__), Path(projection.__file__)):
        shutil.copy2(path, output / path.name)
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def flatten(model, gradients=False):
    tensors = [parameter.grad if gradients else parameter.detach() for parameter in model.parameters()]
    if any(tensor is None or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("Require every finite original FP32 parameter or gradient")
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])


@torch.no_grad()
def assign(model, values, gradients=False):
    if values.ndim != 1 or sum(parameter.numel() for parameter in model.parameters()) != values.numel() or not torch.isfinite(values).all():
        raise ValueError("Require the complete finite parameter inventory")
    offset = 0
    for parameter in model.parameters():
        current = values[offset:offset + parameter.numel()].view_as(parameter)
        if gradients:
            parameter.grad = current
        else:
            parameter.copy_(current)
        offset += parameter.numel()


def save_checkpoint(model, output, role):
    if role not in ("unprojected", "projected"):
        raise ValueError("Only the two declared preflight roles are allowed")
    destination = output / f"{role}_step.pth"
    if destination.exists():
        raise FileExistsError("Never replace a preflight state")
    atomic_torch_save({"model": model.state_dict(), "step": 1, "epoch": 1, "config_sha256": sha256(output / "config.json"),
                      "role": role, "preflight_only": True, "selection_qualified": False, "gradient_updates_performed": 1}, destination)
    return {"checkpoint": str(destination), "checkpoint_sha256": sha256(destination)}


def verify_certificate(gram, products, norm, solution, margin):
    if norm <= 0:
        raise ValueError("Require a nonzero actual optimizer proposal")
    coefficients = np.asarray(solution["raw_gradient_coefficients"])
    norms = np.sqrt(np.diag(gram))
    active = norms > 0
    dual = np.asarray(solution["dual_multipliers"])
    expected = np.zeros_like(coefficients)
    expected[active] = dual[active] * norm / norms[active]
    np.testing.assert_allclose(coefficients, expected, rtol=1e-10, atol=1e-14)
    scaled_after = (products + gram @ coefficients)[active] / (norms[active] * norm)
    slack = scaled_after - margin
    if np.any(dual < 0) or slack.min() < -1e-8 or np.max(np.abs(dual[active] * slack)) > 1e-8 * max(1., float(dual.max())):
        raise ValueError("Raw-gradient CPU KKT reconstruction does not certify the projection")


def execute(output):
    config = verify_registration(output)
    tasks = []
    wait_for_gpu(output, True, False)
    (output / "tasks").mkdir()
    with mixture.runtime(config["base"], output):
        student, optimizer, criterion, teachers, sampler = mixture.backend.setup(config["base"])
        if optimizer.state:
            raise ValueError("Require fresh zero AdamW moments")
        before = flatten(student)
        parameter_inventory = [(name, list(parameter.shape), parameter.numel()) for name, parameter in student.named_parameters()]
        if json.loads(json.dumps(parameter_inventory)) != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json"):
            raise ValueError("Original named parameter inventory changed")
        write_json(output / "parameter_inventory.json", parameter_inventory)
        gradients = torch.empty((14, before.numel()), dtype=torch.float32, device=before.device)
        torch.cuda.reset_peak_memory_stats()
        for case in config["cases"]:
            started = time.monotonic()
            write_json(output / "status.json", {"state": "capturing_fresh_family_gradients", "task": case["name"],
                       "completed": len(tasks), "total": 14, "pid": os.getpid(), "updated_unix": time.time()})
            episode = sampler.sample(step=1, **case["specification"])
            expected_seed = case["specification"].get("structured", case["specification"])["seed"]
            if episode.metadata.get("seed") != expected_seed or episode.metadata.get("gradient_data") != "synthetic":
                raise ValueError("Generated episode does not bind its registered synthetic seed")
            teacher, routing = mixture.backend.teacher_target(teachers, criterion, episode)
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(case["gradient_seed"])
            loss = mixture.backend.backward_episode(student, criterion, episode, teacher, 1.)
            gradients[case["index"]].copy_(flatten(student, gradients=True))
            path = output / "tasks" / f'task_{case["index"]:02d}.pt'
            atomic_torch_save({"gradient": gradients[case["index"]].cpu(), "episode": mixture.backend.cache_episode(episode),
                               "teacher": teacher.detach().float().cpu(), "loss": loss, "routing": routing}, path)
            tasks.append({**case, "metadata": episode.metadata, "loss": loss, "routing": routing,
                          "artifact": str(path), "sha256": sha256(path), "seconds": time.monotonic() - started})
            write_json(output / "tasks.json", tasks)
            print(json.dumps({"task": case["name"], "loss": loss, "seconds": tasks[-1]["seconds"]}), flush=True)
            del episode, teacher
        if not torch.equal(flatten(student), before):
            raise ValueError("Gradient capture changed the initialization")
    del teachers, sampler, criterion
    gc.collect()
    torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True)
    combined, combination = projection.combined_gradient(gradients, [case["weight"] for case in config["cases"]])
    assign(student, combined, gradients=True)
    for group in optimizer.param_groups:
        group["lr"] = RATE
    if len(optimizer.param_groups) != 1:
        raise ValueError("Keep the original single AdamW parameter group")
    optimizer_recipe = {key: optimizer.param_groups[0][key] for key in ("lr", "betas", "eps", "weight_decay", "fused")}
    norm = optimizer_update(student, optimizer, .5)
    if len(optimizer.state) != len(parameter_inventory) or any(float(state["step"]) != 1. for state in optimizer.state.values()):
        raise ValueError("Require exactly one actual AdamW step on every parameter")
    tentative = flatten(student)
    applied = flatten(student, gradients=True)
    control = save_checkpoint(student, output, "unprojected")
    direction = before.double() - tentative.double()
    gram, products, direction_norm = projection.gram_and_products(gradients, direction)
    solution = projection.solve_projection(gram, products, direction_norm)
    after = projection.projected_parameters(tentative, gradients, solution["raw_gradient_coefficients"])
    assign(student, after)
    _, realized_products, realized_norm = projection.gram_and_products(gradients, before.double() - flatten(student).double())
    if np.any(realized_products < 0) or not np.any(realized_products > 0):
        raise ValueError("Actual rounded parameter update is not non-adverse for every captured task")
    projected = save_checkpoint(student, output, "projected")
    path = output / "optimizer_step.pt"
    atomic_torch_save({"combined_gradient": combined.cpu(), "applied_gradient": applied.cpu()}, path)
    result = {"complete": True, "tasks": 14, "gradient_elements_per_task": before.numel(), "parameter_tensors": len(parameter_inventory),
              "combination": combination, "global_gradient_norm_before_clip": norm, "projection": solution,
              "actual_products_after": realized_products.tolist(), "actual_direction_norm_after": realized_norm,
              "actual_adverse_tasks_before": int(np.sum(products < 0)), "actual_adverse_tasks_after": int(np.sum(realized_products < 0)),
              "control": control, "projected": projected, "optimizer_step_artifact": str(path), "optimizer_step_sha256": sha256(path),
              "optimizer_recipe": optimizer_recipe,
              "peak_gib": torch.cuda.max_memory_allocated() / 1024**3, "actual_fused_cuda_adamw_steps": 1,
              "preflight_only": True, "checkpoint_selection_performed": False, "validation_or_confirmation_evaluated": False,
              "real_benchmarks_evaluated": False, "goal_success_not_assessed": True}
    verify_registration(output)
    write_json(output / "result.json", result)
    del student, optimizer, gradients, before, tentative, applied, direction, after, combined
    gc.collect()
    torch.cuda.empty_cache()
    write_json(output / "status.json", {"state": "awaiting_independent_cpu_audit", "pid": os.getpid(), "updated_unix": time.time()})
    with (output / "audit.log").open("x") as stream:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "audit", "--output", str(output)],
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=stream, stderr=subprocess.STDOUT, check=True)


def audit(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Audit must be a separate CPU-only process")
    config = verify_registration(output)
    result, tasks, parameters = read(output / "result.json"), read(output / "tasks.json"), read(output / "parameter_inventory.json")
    if (output / "audit.json").exists():
        raise FileExistsError("Never replace an independent audit")
    if len(tasks) != 14 or [{key: task[key] for key in case} for task, case in zip(tasks, config["cases"])] != config["cases"]:
        raise ValueError("Every original registered task must be retained")
    if parameters != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json"):
        raise ValueError("Original named parameter inventory changed")
    paths = [output / name for name in ("config.json", "result.json", "tasks.json", "parameter_inventory.json")]
    hashes = {str(path): sha256(path) for path in paths}
    hashes.update({task["artifact"]: task["sha256"] for task in tasks})
    hashes.update({result[role]["checkpoint"]: result[role]["checkpoint_sha256"] for role in ("control", "projected")})
    hashes[result["optimizer_step_artifact"]] = result["optimizer_step_sha256"]
    verify_hashes(hashes)
    parent = torch.load(config["parent"], map_location="cpu", weights_only=True)["model"]
    states = {role: torch.load(result[role]["checkpoint"], map_location="cpu", weights_only=True) for role in ("control", "projected")}
    inventories = {}
    for role, state in states.items():
        expected_role = "unprojected" if role == "control" else "projected"
        if (not state["preflight_only"] or state["selection_qualified"] or state["step"] != 1 or state["epoch"] != 1
                or state["gradient_updates_performed"] != 1 or state["role"] != expected_role
                or state["config_sha256"] != sha256(output / "config.json")):
            raise ValueError("Preflight checkpoints must never be described as selected candidates")
        inventories[role] = inventory(state["model"], parent)
        parameter_names = {name for name, shape, size in parameters}
        if any(not torch.equal(value, parent[name]) for name, value in state["model"].items() if name not in parameter_names):
            raise ValueError("A nonparameter state tensor changed")
    before = torch.cat([parent[name].flatten() for name, shape, size in parameters])
    tentative = torch.cat([states["control"]["model"][name].flatten() for name, shape, size in parameters])
    after = torch.cat([states["projected"]["model"][name].flatten() for name, shape, size in parameters])
    gradients = torch.empty((14, before.numel()), dtype=torch.float32)
    for index, task in enumerate(tasks):
        payload = torch.load(task["artifact"], map_location="cpu", weights_only=True, mmap=True)
        if payload["gradient"].shape != before.shape or payload["gradient"].dtype != torch.float32:
            raise ValueError("Preserve the entire original parameter-gradient inventory")
        expected_seed = task["specification"].get("structured", task["specification"])["seed"]
        if (payload["episode"]["metadata"]["seed"] != expected_seed or task["metadata"]["seed"] != expected_seed
                or payload["loss"] != task["loss"]):
            raise ValueError("Saved task provenance changed")
        gradients[index].copy_(payload["gradient"])
    step_artifact = torch.load(result["optimizer_step_artifact"], map_location="cpu", weights_only=True)
    combined, combination = projection.combined_gradient(gradients, [case["weight"] for case in config["cases"]])
    torch.testing.assert_close(combined, step_artifact["combined_gradient"], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(combination["weighted_clip_coefficients"], result["combination"]["weighted_clip_coefficients"], rtol=1e-10, atol=1e-12)
    sizes = [size for name, shape, size in parameters]
    global_norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(part) for part in step_artifact["combined_gradient"].split(sizes)])))
    np.testing.assert_allclose(global_norm, result["global_gradient_norm_before_clip"], rtol=1e-6, atol=1e-10)
    expected_applied = step_artifact["combined_gradient"] * min(1., .5 / (result["global_gradient_norm_before_clip"] + 1e-6))
    torch.testing.assert_close(expected_applied, step_artifact["applied_gradient"], rtol=1e-6, atol=1e-10)
    expected_recipe = {"lr": RATE, "betas": [.9, .999], "eps": 1e-8, "weight_decay": config["base"]["weight_decay"], "fused": True}
    if result["optimizer_recipe"] != expected_recipe:
        raise ValueError("Original fused-CUDA AdamW recipe changed")
    applied = step_artifact["applied_gradient"].double()
    analytic_step = RATE * (applied / (applied.abs() + 1e-8) + config["base"]["weight_decay"] * before.double())
    error = (before.double() - analytic_step - tentative.double()).abs()
    bound = 4 * torch.finfo(torch.float32).eps * (before.double().abs() + analytic_step.abs())
    if not (error <= bound).all():
        raise ValueError("Actual fused proposal exceeds the first-step AdamW roundoff envelope")
    gram, products, norm = projection.gram_and_products(gradients, before.double() - tentative.double())
    solution = result["projection"]
    np.testing.assert_allclose(gram, solution["raw_gram"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(products, solution["raw_products_before"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(norm, solution["direction_norm"], rtol=1e-12, atol=1e-12)
    verify_certificate(gram, products, norm, solution, config["projection"]["relative_inward_margin"])
    coefficients = np.asarray(solution["raw_gradient_coefficients"])
    expected = projection.projected_parameters(tentative, gradients, coefficients)
    lower, upper = torch.nextafter(expected, torch.full_like(expected, -torch.inf)), torch.nextafter(expected, torch.full_like(expected, torch.inf))
    if not ((after >= lower) & (after <= upper)).all():
        raise ValueError("Independent projected-parameter arithmetic exceeds one output ULP")
    _, actual_products, actual_norm = projection.gram_and_products(gradients, before.double() - after.double())
    np.testing.assert_allclose(actual_products, result["actual_products_after"], rtol=1e-9, atol=1e-12)
    if np.any(actual_products < 0) or not np.any(actual_products > 0):
        raise ValueError("Actual CPU-reconstructed update is adverse to a captured task")
    verify_registration(output)
    verify_hashes(hashes)
    write_json(output / "audit.json", {"complete": True, "tasks": 14, "gradient_vectors_remeasured": 14,
               "gradient_elements_per_task": before.numel(), "kkt_conditions_verified": True, "inventories": inventories,
               "projection_arithmetic_maximum_difference": float((expected.double() - after.double()).abs().max()),
               "allowed_parameter_arithmetic_ulps": 1, "actual_direction_norm_after": actual_norm,
               "combined_and_applied_gradients_reconstructed": True, "adam_first_step_roundoff_envelope_passed": True,
               "maximum_adam_closed_form_error": float(error.max()),
               "actual_adverse_tasks_before": int(np.sum(products < 0)), "actual_adverse_tasks_after": int(np.sum(actual_products < 0)),
               "cuda_initialized": torch.cuda.is_initialized(), "autograd_replayed": False, "fused_cuda_adamw_replayed": False,
               "original_artifacts_unchanged": True, "artifact_sha256": hashes, "preflight_only": True,
               "checkpoint_selection_performed": False, "goal_success_not_assessed": True})
    (output / "RESULTS.md").write_text("\n".join(["# Fresh-task family-projection feasibility preflight", "", POLICY, "",
        f'Actual original-network parameters: {before.numel()}. Peak training memory: {result["peak_gib"]:.3f} GiB.',
        f'Adverse sampled training-loss directions: {int(np.sum(products < 0))} before; {int(np.sum(actual_products < 0))} after projection.',
        "All fourteen saved full gradient vectors, projection KKT conditions and actual rounded parameter changes are independently CPU-audited.",
        "The audit does not replay autograd or the fused-CUDA optimizer. The recorded optimizer proposal is the audited projection input.",
        "No finite-step loss decrease, validation performance, confirmation, real benchmark, or goal completion is established.",
        "Neither checkpoint is selected or published as a best model; both are preflight-only artifacts.", ""]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
    elif args.stage == "audit":
        audit(output)
    else:
        with (output / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(output / "status.json")["state"] != "registered":
                raise ValueError("Never restart a started preflight")
            try:
                write_json(output / "status.json", {"state": "starting", "pid": os.getpid(), "updated_unix": time.time()})
                execute(output)
                write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
            except BaseException as error:
                write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
                raise


if __name__ == "__main__":
    main()
