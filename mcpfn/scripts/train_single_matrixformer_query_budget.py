"""Register and train a query-budget sibling of V7 from the same confirmed V2."""

import argparse
from copy import deepcopy
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
from single_model_query_budget import SPECIFICATION, budget_for_seed
import single_model_query_objective as query
import train_single_matrixformer_area_mixture as previous
from train_single_matrixformer import storage_guard
from train_streamed_large_matrices import wait_for_gpu


balanced, continuation, column, parent, area = (previous.balanced, previous.continuation, previous.column,
                                               previous.parent, previous.area)
SEEDS = {"seed": 51100501, "data_seed": 52100501, "validation_seed": 53100501,
         "confirmation_seed": 54100501, "preflight_seed": 55100501, "regime_seed": 56100501,
         "query_seed_offset": 57100501}


def query_evidence(directory):
    config, summary = balanced.read(directory / "config.json"), balanced.read(directory / "summary.json")
    fingerprint = sha256(directory / "config.json")
    if (balanced.read(directory / "status.json")["state"] != "complete" or not summary["complete"]
            or summary["config_sha256"] != fingerprint
            or balanced.read(directory / "registration.json")["config_sha256"] != fingerprint
            or config["policy"] != SPECIFICATION or config["cases"] != 256
            or not summary["source_unchanged"] or not summary["column_coverage_unchanged"]
            or any(summary[field] for field in ("training_performed", "model_inference_performed", "gate_changed",
                                                "cuda_initialized", "real_benchmark_success"))
            or any(config[field] for field in ("training_performed", "model_inference_performed", "checkpoint_selection_performed",
                                               "gate_changed", "real_benchmark_inputs_used"))):
        raise ValueError("Require the complete fixed-policy synthetic query-coverage probe")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    if sha256(directory / "query_masks.pt") != summary["mask_sha256"]:
        raise ValueError("Registered query masks changed")
    original, subset = summary["arms"]["original"], summary["arms"]["subsampled"]
    if (original["cases"] != 256 or subset["cases"] != 256
            or original["queried_columns"] != subset["queried_columns"]
            or subset["uci_only_columns"] <= original["uci_only_columns"]):
        raise ValueError("Require paired unchanged-column coverage with greater small-variance exposure")
    records = balanced.read(directory / "cases.json")
    if len(records) != 256 or [record["index"] for record in records] != list(range(256)):
        raise ValueError("Incomplete or reordered paired cases")
    for index, record in enumerate(records):
        if (record["seed"] != config["query_seed_start"] + index
                or record["budget"] != budget_for_seed(record["seed"])
                or not 0 < record["subsampled_queries"] <= record["original_queries"]):
            raise ValueError("Query masks do not follow the registered private-seed policy")
    for role, aggregate in (("original", original), ("subsampled", subset)):
        if (sum(record[f"{role}_uci_only_columns"] for record in records) != aggregate["uci_only_columns"]
                or sum(record[f"{role}_uci_only_columns"] > 0 for record in records) != aggregate["cases_with_uci_only_columns"]):
            raise ValueError("Reported rare-query coverage does not reconstruct")
    artifacts = [directory / name for name in ("config.json", "registration.json", "summary.json", "cases.json",
                                               "query_masks.pt", "original_columns.csv", "subsampled_columns.csv")]
    return {"directory": str(directory), "artifacts_sha256": {str(path): sha256(path) for path in artifacts},
            "original_uci_only_columns": original["uci_only_columns"], "subsampled_uci_only_columns": subset["uci_only_columns"],
            "interpretation": "Coverage supports a training-only query-budget experiment, not a changed gate or proof of accuracy"}


def validate_recipe(config):
    previous.previous.validate_recipe({**config, "version": 6})
    if (config["version"] != 8 or config["query_budget_objective"] != SPECIFICATION
            or config["spectral_amplitude_mixture"] != area.SPECIFICATION
            or any(config[key] != value for key, value in SEEDS.items())
            or config["steps"] != 400 or config["validate_every"] != 50
            or config["peak_lr"] != 1e-5 or config["warmup_steps"] != 20
            or config["weight_decay"] != .02 or config["grad_clip"] != .5):
        raise ValueError("Require the registered query-budget recipe")
    for field, confirmation in (("validation_cases", False), ("confirmation_cases", True)):
        seed = config["confirmation_seed" if confirmation else "validation_seed"]
        expected = previous.previous.repeated_bank(seed, confirmation=confirmation)
        if json.dumps(config[field], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("Every legacy selection/confirmation specification must remain unchanged")
    bank_seeds = [case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]]
    if len(set(bank_seeds)) != 148 or config["data_seed"] + 16 * config["steps"] + 1 >= min(bank_seeds):
        raise ValueError("Training and both synthetic banks require disjoint seeds")


def make_config(recipe, diagnostic, after_campaign):
    base = balanced.read(recipe / "config.json")
    continuation.verify_run(base, recipe)
    previous.validate_recipe(base)
    predecessor = after_campaign / "cache_launcher_config.json"
    if (balanced.read(recipe / "status.json")["state"] != "training_complete"
            or not balanced.campaign_ready(after_campaign) or balanced.read(predecessor)["run"] != str(recipe)):
        raise ValueError("Require completed V7 and its terminal original benchmark campaign")
    report = query_evidence(diagnostic)
    config = deepcopy(base)
    config.update(version=8, **SEEDS, query_budget_objective=deepcopy(SPECIFICATION), query_coverage_evidence=report,
                  area_mixture_recipe=str(recipe), after_campaign=str(after_campaign),
                  validation_cases=previous.previous.repeated_bank(SEEDS["validation_seed"]),
                  confirmation_cases=previous.previous.repeated_bank(SEEDS["confirmation_seed"], confirmation=True),
                  experiment="From the same confirmed V2, average full-query and 2/4/8/16-subsampled-query column majorizers; preserve V7 prior mixture, frozen teachers, rating loss, original network and all legacy evaluation/gate semantics",
                  objective="Existing teacher-Huber/NLL/CDF plus 0.5 full-query and 0.5 random-query-budget column majorization for imputation only; exact count rescaling inside the unchanged full-query mean loss",
                  execution="Process-local backward adapter; input matrices and actual validation query masks unchanged; callbacks restored on errors")
    config["phases"][0]["name"] = "query_budget"
    old_seeds = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]}
    if old_seeds.intersection(case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]):
        raise ValueError("Never reuse the predecessor's selection or confirmation seeds")
    config["input_sha256"].update({str(recipe / "config.json"): sha256(recipe / "config.json"),
                                   str(predecessor): sha256(predecessor), **report["artifacts_sha256"]})
    sources = {Path(__file__).resolve(), REPO / "scripts/confirm_single_matrixformer_query_budget.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config["source_sha256"].update({str(path): sha256(path) for path in sources})
    validate_recipe(config)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def validate_preflight(config, output):
    previous.validate_preflight(config, output)
    report = balanced.read(output / "preflight.json")
    for index, case in enumerate(report["cases"]):
        if not math.isfinite(case["loss"]) or case["grad_norm"] <= 0:
            raise ValueError("Preflights require finite real losses and nonzero finite gradients")
        if index == 2:
            if "query_budget_objective" in case:
                raise ValueError("Ratings must not receive query-budget supervision")
            continue
        record = case["query_budget_objective"]
        seed = config["preflight_seed"] + index + config["query_seed_offset"]
        if (record["seed"] != seed or record["budget"] != budget_for_seed(seed)
                or record["full"] != case["column_objective"]
                or record["full_query_weight"] != .5 or record["subsampled_query_weight"] != .5
                or record["subsampled"]["thresholds"] != column.COLUMN_OBJECTIVE["thresholds"]
                or not math.isfinite(record["maximum_cell_coefficient"])
                or not 0 < record["subsampled"]["query_cells"] <= record["full"]["query_cells"]
                or record["full_to_subsampled_query_ratio"] != record["full"]["query_cells"] / record["subsampled"]["query_cells"]):
            raise ValueError("Preflight must exercise the actual correctly normalized query-budget objective")


def train(config, output, *, resume=False):
    validate_recipe(config)
    validate_preflight(config, output)
    with area.training_prior(), parent.training_teachers(), column.evaluation_metrics(), query.training_objective(config["query_seed_offset"]):
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
                with area.training_prior(), parent.training_teachers(), query.training_objective(config["query_seed_offset"]):
                    balanced.preflight(config, output)
            train(config, output, resume=arguments.resume)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
