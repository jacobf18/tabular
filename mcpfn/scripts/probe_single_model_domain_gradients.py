"""Measure actual V9-objective domain gradients at fixed V8 weights, without updates."""

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import diagnose_single_model_missingness_coverage as coverage
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_structured_mixture as mixture
from streamed_large_matrix_training import atomic_torch_save
from train_single_matrixformer_structured_mixture import ROOT, validate_recipe
from train_streamed_large_matrices import wait_for_gpu


CLIP = .5
POLICY = "Fixed V8 weights; unchanged V9 losses/teachers; isolated unit-weight gradients; current joint clipping versus per-domain clipping; no optimizer updates or candidate selection"


def read(path):
    return json.loads(Path(path).read_text())


def specifications():
    cases = []
    for family_index, family in enumerate(mixture.FAMILIES):
        for component_index, component in enumerate(mixture.COMPONENTS):
            shapes = mixture.COMPACT_SHAPES if family in mixture.NEURAL_FAMILIES else mixture.SPECTRAL_SHAPES
            choice = (family_index + component_index) % 2
            specification = {"family": family, "component": component, "shape": list(shapes[choice]),
                             "regime": ("legacy", "area_scaled")[choice], "requested_missingness": (.2, .6)[choice],
                             "seed": 81100601 + len(cases) * 100}
            cases.append({"group": "structured", "imputation": {"structured": specification}})
    for index in range(16):
        cases.append({"group": "legacy", "imputation": {"seed": 81110601 + index * 100,
            "shape": [256, 128] if index < 8 else [2000, 1000], "component": index % 8,
            "transpose": bool(index % 2), "regime": ("legacy", "area_scaled")[(index // 2) % 2]}})
    for index, case in enumerate(cases):
        case.update(index=index, rating={"seed": 81120601 + index * 100,
                    "shape": [1000, 1000] if index % 2 == 0 else [2000, 1000]}, gradient_seed=82100601 + index * 2)
    return cases


def dot(first, second):
    if first.ndim != 1 or first.shape != second.shape or not first.numel():
        raise ValueError("Require matching nonempty gradient vectors")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise FloatingPointError("Nonfinite gradient vector")
    return sum(float(torch.dot(left.double(), right.double()))
               for left, right in zip(first.split(1_000_000), second.split(1_000_000)))


def coefficient(norm):
    if not math.isfinite(norm) or norm < 0:
        raise ValueError("Require a finite nonnegative norm")
    return min(1., CLIP / (norm + 1e-6))


def geometry(imputation, rating):
    imputation_square, rating_square, product = dot(imputation, imputation), dot(rating, rating), dot(imputation, rating)
    imputation_norm, rating_norm = math.sqrt(imputation_square), math.sqrt(rating_square)
    joint_norm = math.sqrt(max(0., .25 * (imputation_square + rating_square + 2 * product)))
    scales = {"joint_clip": (.5 * coefficient(joint_norm),) * 2,
              "domain_clip_then_mean": (.5 * coefficient(imputation_norm), .5 * coefficient(rating_norm))}
    methods = {}
    for name, (imputation_scale, rating_scale) in scales.items():
        direction_square = (imputation_scale**2 * imputation_square + rating_scale**2 * rating_square
                            + 2 * imputation_scale * rating_scale * product)
        direction_norm = math.sqrt(max(direction_square, 0.))
        imputation_dot = imputation_scale * imputation_square + rating_scale * product
        rating_dot = imputation_scale * product + rating_scale * rating_square
        methods[name] = {"imputation_coefficient": imputation_scale, "rating_coefficient": rating_scale,
            "direction_norm": direction_norm, "imputation_dot": imputation_dot, "rating_dot": rating_dot,
            "imputation_sgd_loss_delta_per_unit_lr": -imputation_dot, "rating_sgd_loss_delta_per_unit_lr": -rating_dot,
            "imputation_cosine": imputation_dot / (direction_norm * imputation_norm) if direction_norm * imputation_norm else None,
            "rating_cosine": rating_dot / (direction_norm * rating_norm) if direction_norm * rating_norm else None}
    return {"imputation_norm": imputation_norm, "rating_norm": rating_norm, "cross_dot": product,
            "cosine": product / (imputation_norm * rating_norm) if imputation_norm * rating_norm else None,
            "imputation_to_rating_norm": imputation_norm / rating_norm if rating_norm else None,
            "joint_norm_before_clip": joint_norm, "methods": methods}


def capture(model):
    values = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"Missing training gradient: {name}")
        values.append(parameter.grad.detach().float().reshape(-1).cpu())
    vector = torch.cat(values)
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Nonfinite captured gradient")
    return vector


def summarize(records):
    summaries = {}
    groups = ("all", "legacy", "structured", *mixture.FAMILIES)
    for group in groups:
        selected = [row for row in records if group == "all" or row["group"] == group
                    or row["metadata"]["imputation"].get("family") == group]
        if not selected:
            raise ValueError("Keep all diagnostic groups and families")
        values = [row["geometry"] for row in selected]
        ratios = [value["imputation_to_rating_norm"] for value in values if value["imputation_to_rating_norm"] is not None]
        cosines = [value["cosine"] for value in values if value["cosine"] is not None]
        summaries[group] = {"cases": len(selected), "defined_norm_ratios": len(ratios),
            "median_imputation_to_rating_norm": float(np.median(ratios)) if ratios else None,
            "defined_cosines": len(cosines), "median_cross_cosine": float(np.median(cosines)) if cosines else None,
            "negative_cross_dot_cases": sum(value["cross_dot"] < 0 for value in values),
            "methods": {method: {"rating_negative_descent_cases": sum(value["methods"][method]["rating_dot"] < 0 for value in values),
                                 "imputation_negative_descent_cases": sum(value["methods"][method]["imputation_dot"] < 0 for value in values),
                                 "median_rating_coefficient": float(np.median([value["methods"][method]["rating_coefficient"] for value in values]))}
                        for method in ("joint_clip", "domain_clip_then_mean")}}
    return summaries


def register(output):
    if output.exists():
        raise FileExistsError("Never replace a registered diagnostic")
    run = ROOT / "structured_mixture_v9"
    base = read(run / "config.json")
    validate_recipe(base)
    mixture.backend.verify_run(base, run)
    status = read(run / "status.json")
    audit_path = ROOT / "structured_mixture_v9_audit_400/audit.json"
    audit = read(audit_path)
    if (status["state"] != "training_complete" or status["best_step"] != 0 or not audit["complete"]
            or not audit["full_400_update_log_and_validation_audited"]):
        raise ValueError("Require completed and fully audited V9, without an eligible trained checkpoint")
    verify_hashes(audit["original_artifact_sha256"])
    cases = specifications()
    seeds = [case["imputation"].get("seed", case["imputation"].get("structured", {}).get("seed")) for case in cases]
    seeds += [case["rating"]["seed"] for case in cases]
    old_seeds = {base["data_seed"] + 16 * step + offset for step in range(1, 401) for offset in (0, 1)}
    for name in ("validation_cases", "confirmation_cases"):
        old_seeds.update(case["structured"]["seed"] if "structured" in case else case["seed"] for case in base[name])
    if len(set(seeds)) != 128 or old_seeds.intersection(seeds):
        raise ValueError("All64 diagnostic pairs need fresh, distinct matrix seeds")
    paths = [run / "config.json", run / "status.json", audit_path, Path(base["parent_checkpoint"])]
    config = {"base": base, "cases": cases, "policy": POLICY, "clip": CLIP, "registered_unix": time.time(),
              "input_sha256": {str(path): sha256(path) for path in paths}, "source_sha256": coverage.source_hashes(),
              "no_optimizer_updates": True, "real_benchmark_inputs_used": False, "checkpoint_selection_performed": False,
              "limitations": "Single fixed checkpoint; one draw per structured family/component; legacy probe is stratified, not a random mixture estimate; SGD geometry is not an AdamW or generalization result"}
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "model_evaluated": False})
    for name in config["source_sha256"]:
        source = Path(name)
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def verify_registration(output):
    config = read(output / "config.json")
    if (read(output / "registration.json")["config_sha256"] != sha256(output / "config.json")
            or config["cases"] != specifications() or config["policy"] != POLICY or config["clip"] != CLIP):
        raise ValueError("Registered cases, policy or clipping changed")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def run_probe(output):
    config = verify_registration(output)
    wait_for_gpu(output, True, False)
    records, original = [], torch.load(config["base"]["parent_checkpoint"], map_location="cpu", weights_only=True)["model"]
    (output / "pairs").mkdir()
    with mixture.runtime(config["base"], output):
        model, optimizer, criterion, teachers, sampler = mixture.backend.setup(config["base"])
        del optimizer
        model.train()
        if any(not torch.equal(value.detach().cpu(), original[name]) for name, value in model.state_dict().items()):
            raise ValueError("Require exact original V8 initialization")
        inventory = [(name, list(parameter.shape), parameter.numel()) for name, parameter in model.named_parameters()]
        write_json(output / "parameter_inventory.json", inventory)
        for specification in config["cases"]:
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            payload, metadata, losses, routes = {}, {}, {}, {}
            for offset, domain in enumerate(("imputation", "recsys")):
                arguments = deepcopy(specification["imputation" if domain == "imputation" else "rating"])
                episode = sampler.sample(domain, 1, **arguments)
                teacher, routing = mixture.backend.teacher_target(teachers, criterion, episode)
                model.zero_grad(set_to_none=True)
                torch.manual_seed(specification["gradient_seed"] + offset)
                loss = mixture.backend.backward_episode(model, criterion, episode, teacher, 1.)
                payload[domain] = capture(model)
                payload[f"{domain}_episode"] = mixture.backend.cache_episode(episode)
                payload[f"{domain}_teacher"] = teacher.detach().float().cpu()
                metadata[domain], losses[domain], routes[domain] = deepcopy(episode.metadata), loss, routing
                model.zero_grad(set_to_none=True)
                del episode, teacher
            statistics = geometry(payload["imputation"], payload["recsys"])
            mixture.backend.storage_guard(output)
            destination = output / "pairs" / f'pair_{specification["index"]:03d}.pt'
            atomic_torch_save(payload, destination)
            record = {**specification, "metadata": metadata, "losses": losses, "teachers": routes, "geometry": statistics,
                      "raw_gradients": str(destination), "raw_sha256": sha256(destination),
                      "seconds": time.monotonic() - started, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
            records.append(record)
            write_json(output / f'pair_{specification["index"]:03d}.json', record)
            write_json(output / "status.json", {"state": "measuring_gradients", "completed": len(records), "expected": 64,
                                               "pid": os.getpid(), "updated_unix": time.time()})
            print(json.dumps({"case": specification["index"], "group": specification["group"],
                              "norm_ratio": statistics["imputation_to_rating_norm"], "cosine": statistics["cosine"],
                              "seconds": record["seconds"], "peak_gib": record["peak_gib"]}), flush=True)
            del payload
        unchanged = all(torch.equal(value.detach().cpu(), original[name]) for name, value in model.state_dict().items())
        if not unchanged:
            raise ValueError("Read-only gradient measurement changed the model state")
    verify_registration(output)
    write_json(output / "summary.json", {"complete": True, "cases": 64, "model_state_unchanged": unchanged,
        "training_performed": False, "checkpoint_selection_performed": False, "goal_success_not_assessed": True,
        "groups": summarize(records), "limitations": config["limitations"],
        "raw_gradient_sha256": {row["raw_gradients"]: row["raw_sha256"] for row in records}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(output / "status.json")["state"] != "registered":
            raise ValueError("Never restart or replace a started diagnostic")
        try:
            write_json(output / "status.json", {"state": "starting", "pid": os.getpid(), "updated_unix": time.time()})
            run_probe(output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
