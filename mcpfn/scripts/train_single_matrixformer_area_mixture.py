"""Train a preregistered legacy/area-adjusted prior mixture, keeping V6's objective and gate."""

import argparse
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_area_prior as area
import train_single_matrixformer_column_aligned as previous
from train_single_matrixformer import storage_guard
from train_streamed_large_matrices import wait_for_gpu


balanced, continuation, column, parent = previous.balanced, previous.continuation, previous.column, previous.parent
SEEDS = {"seed": 43100501, "data_seed": 44100501, "validation_seed": 45100501,
         "confirmation_seed": 46100501, "preflight_seed": 47100501, "regime_seed": 48100501}


def prior_evidence(directory):
    config, summary = balanced.read(directory / "config.json"), balanced.read(directory / "summary.json")
    fingerprint = sha256(directory / "config.json")
    if (balanced.read(directory / "status.json")["state"] != "complete" or not summary["complete"]
            or summary["config_sha256"] != fingerprint
            or balanced.read(directory / "registration.json")["config_sha256"] != fingerprint
            or summary["paired_matrices"] != 240 or summary["regime_rows"] != 480 or summary["draws"] != 48
            or not summary["orthonormal_energy_checks_passed"] or not summary["dimension_scaling_checks_passed"]
            or any(summary[field] for field in ("training_performed", "gate_changed", "cuda_initialized", "real_benchmark_success"))
            or any(config[field] for field in ("training_performed", "checkpoint_selection_performed", "real_benchmark_data_used", "gate_changed"))):
        raise ValueError("Require the complete fresh CPU-only paired prior-energy evidence")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    with (directory / "per_shape.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 480 or len(balanced.read(directory / "draws.json")) != 48:
        raise ValueError("Require all paired draws and regimes")
    identities = set()
    for row in rows:
        identity = (int(row["component"]), int(row["replica"]), int(row["rows"]), int(row["columns"]), row["regime"])
        if identity in identities or row["regime"] not in ("legacy", "area_scaled"):
            raise ValueError("Duplicate or unknown paired regime")
        identities.add(identity)
        expected = 1. if row["regime"] == "area_scaled" else config["reference_area"] / (int(row["rows"]) * int(row["columns"]))
        if not math.isclose(float(row["expected_snr_relative_to_reference"]), expected, rel_tol=1e-12):
            raise ValueError("Paired prior-energy relationship does not reproduce")
    expected = {(component, replica, *shape, regime) for component in range(6) for replica in range(8)
                for shape in config["shapes"] for regime in ("legacy", "area_scaled")}
    if identities != expected or config["reference_area"] != area.SPECIFICATION["reference_area"]:
        raise ValueError("Incomplete paired family/shape coverage")
    artifacts = [directory / name for name in ("config.json", "registration.json", "summary.json", "draws.json", "per_shape.csv", "by_family_shape.csv")]
    return {"directory": str(directory), "artifacts_sha256": {str(path): sha256(path) for path in artifacts},
            "interpretation": "Fixed-spectrum dimension mechanism motivates training-only amplitude mixing, not a changed validation target or a guarantee of benchmark improvement"}


def make_config(recipe, diagnostic, after_campaign):
    base = balanced.read(recipe / "config.json")
    continuation.verify_run(base, recipe)
    previous.validate_recipe(base)
    if balanced.read(recipe / "status.json")["state"] != "training_complete":
        raise ValueError("Require completed V6 training")
    predecessor = after_campaign / "cache_launcher_config.json"
    if not balanced.campaign_ready(after_campaign) or balanced.read(predecessor)["run"] != str(recipe):
        raise ValueError("Require the terminal unchanged V6 campaign")
    report = prior_evidence(diagnostic)
    config = deepcopy(base)
    config.update(version=7, **SEEDS, spectral_amplitude_mixture=deepcopy(area.SPECIFICATION), prior_energy_evidence=report,
                  column_aligned_recipe=str(recipe), after_campaign=str(after_campaign),
                  validation_cases=previous.repeated_bank(SEEDS["validation_seed"]),
                  confirmation_cases=previous.repeated_bank(SEEDS["confirmation_seed"], confirmation=True),
                  experiment="From the same confirmed V2, mix legacy and area-adjusted spectral training amplitudes 50/50 by independent seeded coin; preserve replay, latent anchors, ratings, original network, V6 losses and all legacy validation/gate semantics",
                  execution="Process-local training sampler changes clean spectral amplitude before noise/normalization; explicit legacy evaluation sampler, original teachers and gate; callbacks restored on errors")
    config["phases"][0]["name"] = "area_mixture"
    old_seeds = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]}
    new_seeds = {case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]}
    if old_seeds.intersection(new_seeds):
        raise ValueError("Do not reuse the predecessor's selection or confirmation seeds")
    config["input_sha256"].update({str(recipe / "config.json"): sha256(recipe / "config.json"),
                                   str(predecessor): sha256(predecessor), **report["artifacts_sha256"]})
    sources = {Path(__file__).resolve(), REPO / "scripts/confirm_single_matrixformer_area_mixture.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config["source_sha256"].update({str(path): sha256(path) for path in sources})
    validate_recipe(config)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def validate_recipe(config):
    previous.validate_recipe({**config, "version": 6})
    if (config["version"] != 7 or config["spectral_amplitude_mixture"] != area.SPECIFICATION
            or any(config[key] != value for key, value in SEEDS.items())
            or config["steps"] != 400 or config["validate_every"] != 50
            or config["peak_lr"] != 1e-5 or config["warmup_steps"] != 20):
        raise ValueError("Require the registered area-mixture recipe")
    for field, confirmation in (("validation_cases", False), ("confirmation_cases", True)):
        expected = previous.repeated_bank(config["confirmation_seed" if confirmation else "validation_seed"], confirmation=confirmation)
        if json.dumps(config[field], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("Keep every original legacy validation specification with fresh seeds")
    bank_seeds = [case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]]
    if len(set(bank_seeds)) != 148 or config["data_seed"] + 16 * config["steps"] + 1 >= min(bank_seeds):
        raise ValueError("Training and both evaluation banks must have disjoint seeds")


def validate_preflight(config, output):
    report = balanced.read(output / "preflight.json")
    if not report["passed"] or report["config_sha256"] != sha256(output / "config.json") or len(report["cases"]) != 3:
        raise ValueError("Require complete matching full-size preflights")
    for index, case in enumerate(report["cases"]):
        if (case["seed"] != config["preflight_seed"] + index or not math.isfinite(case["grad_norm"])
                or case["teacher"]["target"] != ("initialization" if index < 2 else "specialist")):
            raise ValueError("Invalid preflight teacher, seed or gradient")
        if index < 2:
            if (not case["area_prior"]["applied"] or case["area_prior"]["reason"] != "dimension_scaled"
                    or case["area_prior"]["signal_gain"] != area.signal_gain(2000, 1000)
                    or [case["rows"], case["columns"]] != ([2000, 1000] if index == 0 else [1000, 2000])
                    or case["column_objective"]["weight"] != column.COLUMN_OBJECTIVE["weight"]):
                raise ValueError("Both imputation orientations must exercise the scaled prior and real loss")
        elif "area_prior" in case or "column_objective" in case or [case["rows"], case["columns"]] != [2000, 1000]:
            raise ValueError("Rating preflight must remain unchanged")


def train(config, output, *, resume=False):
    validate_recipe(config)
    validate_preflight(config, output)
    with area.training_prior(), parent.training_teachers(), column.evaluation_metrics(), column.training_objective():
        continuation.train(config, output, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        if not all((arguments.recipe, arguments.diagnostic, arguments.after_campaign)) or arguments.wait or arguments.resume:
            parser.error("Registration requires recipe, diagnostic and predecessor; no wait/resume")
        balanced.register(make_config(arguments.recipe.resolve(), arguments.diagnostic.resolve(), arguments.after_campaign.resolve()), output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = balanced.read(output / "status.json")
        if (status["state"] != "registered" and not arguments.resume) or status["state"] == "training_complete":
            raise ValueError("Never overwrite started training or resume completed training")
        try:
            config = balanced.read(output / "config.json")
            continuation.verify_run(config, output)
            validate_recipe(config)
            if not balanced.campaign_ready(Path(config["after_campaign"])):
                raise RuntimeError("Predecessor campaign must be terminal")
            wait_for_gpu(output, arguments.wait, False)
            storage_guard(output)
            if not arguments.resume:
                write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                                          "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
                write_json(output / "status.json", {"state": "preflight", "pid": os.getpid(), "updated_unix": time.time()})
                with area.training_prior(), parent.training_teachers(), column.training_objective():
                    balanced.preflight(config, output)
            train(config, output, resume=arguments.resume)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
