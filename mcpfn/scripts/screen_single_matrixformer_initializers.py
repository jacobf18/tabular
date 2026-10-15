"""Synthetic-only screening of materialized single-network initializations."""

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, sha256, write_csv, write_json
from single_model_distillation import Episode, episode_metrics, teacher_target
from streamed_large_matrix_training import STREAM_OPTIONS, atomic_torch_save
from tabimpute.streamed_inference import streamed_point_predictions
from train_single_matrixformer import setup
from train_streamed_large_matrices import wait_for_gpu


DEFAULT_PROTOCOL = REPO / "evaluation_protocols/single_model_initializer_screen.json"


def read(path):
    return json.loads(Path(path).read_text())


def bank_specs(protocol):
    bank = protocol["bank"]
    cases = [dict(domain="imputation", shape=tuple(bank["imputation_small_shape"]), component=component,
                  transpose=bool(repeat % 2))
             for repeat in range(bank["imputation_small_repeats"]) for component in bank["imputation_small_components"]]
    cases.extend(dict(domain="imputation", shape=tuple(bank["imputation_large_shape"]), component=component,
                      transpose=bool(component % 2)) for component in bank["imputation_large_components"])
    cases.extend(dict(domain="recsys", shape=tuple(shape)) for shape in bank["ratings_shapes"]
                 for _ in range(bank["ratings_repeats_per_shape"]))
    if len(cases) != bank["expected_cases"]:
        raise ValueError("Synthetic case count differs from registration")
    return [{**case, "seed": bank["seed_start"] + index} for index, case in enumerate(cases)]


def validate_states(states):
    parent = states["general"]
    for role, state in states.items():
        if tuple(state) != tuple(parent):
            raise ValueError(f"Checkpoint keys differ: {role}")
        for name, value in state.items():
            if value.shape != parent[name].shape or value.dtype != parent[name].dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint tensor: {role}/{name}")


@torch.no_grad()
def materialize(candidate, states):
    kind = candidate["kind"]
    if kind == "checkpoint":
        return {name: value.clone() for name, value in states[candidate["checkpoint"]].items()}
    if kind not in ("interpolation", "task_addition", "decoder_interpolation"):
        raise ValueError("Unknown initialization family")
    alpha = candidate["alpha"]
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Require a finite coefficient in [0,1]")
    result = {}
    for name, original in states["general"].items():
        specialist, shared = states["specialist"][name], states["shared"][name]
        if not original.is_floating_point():
            if not torch.equal(original, specialist) or not torch.equal(original, shared):
                raise ValueError("Do not merge inconsistent nonfloating state")
            value = original.clone()
        elif alpha == 0 or (kind == "decoder_interpolation" and not name.startswith("decoder.")):
            value = original.clone()
        elif kind == "task_addition":
            value = original + alpha * (specialist - shared)
        elif alpha == 1:
            value = specialist.clone()
        else:
            value = torch.lerp(original, specialist, alpha)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Initialization produced nonfinite weights")
        result[name] = value
    return result


def cache_episode(episode):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Only synthetic episodes may enter the initialization screen")
    result = {name: getattr(episode, name).detach().cpu().clone() for name in ("source", "target", "mask")}
    result["metadata"] = deepcopy(episode.metadata)
    result["scaling"] = None if episode.ratings is None else {
        name: getattr(episode.ratings, name).detach().cpu().clone()
        for name in ("means", "stds", "row_offsets", "row_scales")}
    return result


def restore_episode(cached, device):
    tensors = {name: cached[name].to(device).clone() for name in ("source", "target", "mask")}
    ratings = None if cached["scaling"] is None else SimpleNamespace(**{
        name: value.to(device).clone() for name, value in cached["scaling"].items()})
    return Episode(**tensors, metadata=deepcopy(cached["metadata"]), ratings=ratings)


def aggregate(cases, field):
    result = {}
    for domain in ("imputation", "recsys"):
        selected = [case[field] for case in cases if case["domain"] == domain]
        if not selected:
            raise ValueError("Both domains must be evaluated")
        result[domain] = {metric: sum(case[metric] for case in selected) / len(selected) for metric in ("rmse", "mae")}
    return result


def worst_score(metrics, references):
    keys = (("imputation", "rmse"), ("recsys", "rmse"), ("recsys", "mae"))
    ratios = {}
    for domain, metric in keys:
        value, baseline = metrics[domain][metric], references[domain][metric]
        if not math.isfinite(value) or value < 0 or not math.isfinite(baseline) or baseline <= 0:
            raise ValueError("Finite nonnegative scores and positive references are required")
        ratios[f"{domain}_{metric}"] = value / baseline
    return max(ratios.values()), ratios


def campaign_ready(campaign):
    status = read(campaign / "status.json")
    if status["state"] == "complete":
        return True
    if status["state"] in ("failed", "not_promoted", "interrupted"):
        raise RuntimeError("Resolve the stopped first-candidate campaign before screening initializations")
    try:
        os.kill(status["pid"], 0)
    except ProcessLookupError:
        if read(campaign / "status.json")["state"] == "complete":
            return True
        raise RuntimeError("Campaign process is missing; a stale status does not establish a live job") from None
    return False


def storage_guard(output):
    if shutil.disk_usage(output).free < 18 * 1024**3:
        raise OSError("Preserve room for the synthetic bank and one selected initialization above the 15 GiB reserve")


def prepare(protocol_path, output):
    protocol = read(protocol_path)
    campaign = REPO / protocol["execution"]["after_campaign"]
    registration = read(campaign.parent / "initializer_screen_registration.json")
    if sha256(protocol_path) != registration["sha256"] or protocol["version"] != 1:
        raise ValueError("Initializer protocol differs from the prospective registration")
    if len({candidate["name"] for candidate in protocol["candidates"]}) != len(protocol["candidates"]):
        raise ValueError("Duplicate candidate names")
    checkpoints = {str(REPO / item["path"]): item["sha256"] for item in protocol["checkpoints"].values()}
    verify_hashes(checkpoints)
    base_config_path = (REPO / protocol["checkpoints"]["pilot"]["path"]).parent / "config.json"
    base_config = read(base_config_path)
    verify_hashes(base_config["source_sha256"])
    verify_hashes(base_config["input_sha256"])
    verify_hashes(base_config["teacher_config_sha256"])
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    config = {"model": base_config["model"], "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
              "base_training_config": str(base_config_path), "base_config_sha256": sha256(base_config_path),
              "input_sha256": checkpoints, "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
              "cases": bank_specs(protocol), "candidates": protocol["candidates"], "campaign_dependency": str(campaign),
              "training_performed": False, "real_benchmark_inputs_used": False,
              "purpose": "Synthetic-only initialization selection; not a trained or benchmark-qualified replacement",
              "cache": "Generate once in GPU/RAM; save the exact CPU tensor bank to disk for auditable numerical reproducibility"}
    write_json(output / "config.json", config)
    shutil.copy2(protocol_path, output / "protocol.json")
    for source in sources:
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return protocol, config, base_config


@torch.no_grad()
def screen(output, protocol, config, base_config):
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    verify_hashes({config["base_training_config"]: config["base_config_sha256"], config["protocol"]: config["protocol_sha256"]})
    states = {role: torch.load(REPO / specification["path"], map_location="cpu", weights_only=True)["model"]
              for role, specification in protocol["checkpoints"].items()}
    validate_states(states)
    model, optimizer, criterion, teachers, sampler = setup(base_config)
    del optimizer
    model.eval().requires_grad_(False)
    bank, reference_cases = [], []
    for index, specification in enumerate(config["cases"]):
        episode = sampler.sample(step=1, **specification)
        frozen, routing = teacher_target(teachers, criterion, episode)
        with compute_context(episode.source.device):
            parent = streamed_point_predictions(teachers[0], episode.source, criterion, **STREAM_OPTIONS)
        reference_cases.append({"index": index, **episode.metadata, "general": episode_metrics(parent, episode),
                                "teacher": episode_metrics(frozen, episode), "routing": routing})
        bank.append(cache_episode(episode))
        write_json(output / "status.json", {"state": "building_synthetic_bank", "completed": len(bank), "expected": len(config["cases"]),
                                            "pid": os.getpid(), "updated_unix": time.time()})
        del episode, frozen, parent
    general_metrics, teacher_metrics = aggregate(reference_cases, "general"), aggregate(reference_cases, "teacher")
    references = {"imputation": {"rmse": min(general_metrics["imputation"]["rmse"], teacher_metrics["imputation"]["rmse"])},
                  "recsys": dict(teacher_metrics["recsys"])}
    storage_guard(output)
    atomic_torch_save(bank, output / "synthetic_bank.pt")
    bank_hash = sha256(output / "synthetic_bank.pt")
    write_json(output / "reference.json", {"general": general_metrics, "teacher": teacher_metrics,
                                           "selection_references": references, "cases": reference_cases, "bank_sha256": bank_hash})
    del teachers, sampler
    gc.collect()
    torch.cuda.empty_cache()
    reports = []
    for candidate in protocol["candidates"]:
        weights = materialize(candidate, states)
        model.load_state_dict(weights, strict=True)
        del weights
        cases = []
        started = time.monotonic()
        for index, cached in enumerate(bank):
            episode = restore_episode(cached, "cuda")
            with compute_context(episode.source.device):
                prediction = streamed_point_predictions(model, episode.source, criterion, **STREAM_OPTIONS)
            metrics = episode_metrics(prediction, episode)
            cases.append({"index": index, **episode.metadata, "student": metrics})
            write_json(output / "status.json", {"state": "screening", "candidate": candidate["name"],
                                                "completed_cases": len(cases), "expected_cases": len(bank),
                                                "pid": os.getpid(), "updated_unix": time.time()})
            del episode, prediction
        metrics = aggregate(cases, "student")
        score, ratios = worst_score(metrics, references)
        if candidate["kind"] == "checkpoint" and candidate["checkpoint"] == "general":
            for domain in metrics:
                for metric in metrics[domain]:
                    if not math.isclose(metrics[domain][metric], general_metrics[domain][metric], rel_tol=1e-6, abs_tol=1e-7):
                        raise ValueError("General control failed paired-cache prediction parity")
        report = {"candidate": candidate, "score": score, "ratios": ratios, "metrics": metrics,
                  "cases": cases, "seconds": time.monotonic() - started, "bank_sha256": bank_hash,
                  "config_sha256": sha256(output / "config.json")}
        write_json(output / "candidates" / f"{candidate['name']}.json", report)
        reports.append(report)
        print(json.dumps({key: report[key] for key in ("candidate", "score", "ratios", "metrics", "seconds")}), flush=True)
    if len(reports) != len(protocol["candidates"]):
        raise ValueError("Never select from an incomplete screen")
    best_index = min(range(len(reports)), key=lambda index: reports[index]["score"])
    best = reports[best_index]
    candidate = best["candidate"]
    write_csv(output / "summary.csv", [{"candidate": row["candidate"]["name"], "worst_ratio": row["score"],
                                        **{f"ratio_{name}": value for name, value in row["ratios"].items()},
                                        **{f"{domain}_{metric}": value for domain, metrics in row["metrics"].items()
                                           for metric, value in metrics.items()}}
                                       for row in reports])
    if candidate["kind"] == "checkpoint":
        checkpoint = REPO / protocol["checkpoints"][candidate["checkpoint"]]["path"]
        new_initialization = False
    else:
        storage_guard(output)
        checkpoint = output / "selected_initialization.pth"
        atomic_torch_save({"model": materialize(candidate, states), "epoch": 0, "step": 0,
                           "initialization_only": True, "initialization_recipe": candidate, "synthetic_score": best["score"],
                           "config_sha256": sha256(output / "config.json"), "bank_sha256": bank_hash}, checkpoint)
        new_initialization = True
    verify_hashes(config["input_sha256"])
    if sha256(output / "synthetic_bank.pt") != bank_hash:
        raise ValueError("Cached synthetic cases changed")
    selection = {"selected": candidate, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                 "new_initialization_materialized": new_initialization, "score": best["score"], "metrics": best["metrics"],
                 "ratios": best["ratios"], "candidates_completed": len(reports), "cases_per_candidate": len(bank),
                 "bank_sha256": bank_hash, "config_sha256": sha256(output / "config.json"),
                 "training_performed": False, "real_benchmark_inputs_used": False, "goal_success_not_assessed": True,
                 "next_step": "Use synthetic evidence to register a follow-on continuation and fresh independent confirmation"}
    write_json(output / "selection.json", selection)
    write_json(output / "status.json", {"state": "complete", "selected": candidate["name"], "pid": os.getpid(), "updated_unix": time.time()})
    print(json.dumps({"selection": selection}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-campaign", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            storage_guard(output)
            protocol, config, base_config = prepare(arguments.protocol.resolve(), output)
            campaign = Path(config["campaign_dependency"])
            while not campaign_ready(campaign):
                if not arguments.wait_for_campaign:
                    raise RuntimeError("The first-candidate campaign is still running")
                write_json(output / "status.json", {"state": "waiting_for_campaign", "pid": os.getpid(), "updated_unix": time.time()})
                time.sleep(30)
            wait_for_gpu(output, arguments.wait_for_campaign, False)
            write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                                                     "precision": "FP32 parameters, BF16 CUDA autocast; cuDNN SDPA disabled"})
            screen(output, protocol, config, base_config)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
