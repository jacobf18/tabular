"""Preserve the frozen initialization's imputation function while distilling ratings."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from streamed_large_matrix_training import STREAM_OPTIONS
from tabimpute.streamed_inference import streamed_point_predictions
from train_single_matrixformer import storage_guard
import train_single_matrixformer_probability_balance as balanced
from train_streamed_large_matrices import wait_for_gpu


continuation = balanced.continuation
TEACHER_POLICY = {"imputation": "initialization", "recsys": "specialist"}


@torch.no_grad()
def parent_teacher_target(initialization, teachers, criterion, episode, options=STREAM_OPTIONS):
    if (episode.metadata.get("gradient_data") != "synthetic"
            or episode.metadata.get("domain") not in TEACHER_POLICY):
        raise ValueError("Parent teacher requires a declared synthetic episode")
    if (initialization is None or len(teachers) != 2
            or any(model.training or any(parameter.requires_grad for parameter in model.parameters())
                   for model in (initialization, *teachers))):
        raise ValueError("Initialization and original teachers must be frozen in evaluation mode")
    specialist = episode.metadata["domain"] == "recsys"
    model = teachers[1] if specialist else initialization
    with compute_context(episode.source.device):
        target = streamed_point_predictions(model, episode.source, criterion, **options)
    return target.detach(), {"specialist_weight": float(specialist),
                             "target": TEACHER_POLICY[episode.metadata["domain"]],
                             "policy": "frozen initialization retention and rating specialist; training only"}


@contextmanager
def training_teachers():
    original_target = continuation.teacher_target
    original_bank = continuation.prepare_bank
    original_setup = continuation.setup
    original_preflight_target = balanced.teacher_target
    original_preflight_setup = balanced.setup
    parent = []

    def retain_setup(setup_function, config):
        result = setup_function(config)
        parent[:] = [deepcopy(result[0]).eval().requires_grad_(False)]
        return result

    def training_setup(config):
        return retain_setup(original_setup, config)

    def preflight_setup(config):
        return retain_setup(original_preflight_setup, config)

    def target(teachers, criterion, episode, options=STREAM_OPTIONS):
        if not parent:
            raise RuntimeError("Initialize the frozen parent before requesting training targets")
        return parent_teacher_target(parent[0], teachers, criterion, episode, options)

    def reference_bank(*arguments, **keywords):
        continuation.teacher_target = original_target
        try:
            return original_bank(*arguments, **keywords)
        finally:
            continuation.teacher_target = target

    continuation.setup = training_setup
    balanced.setup = preflight_setup
    continuation.teacher_target = target
    balanced.teacher_target = target
    continuation.prepare_bank = reference_bank
    try:
        yield
    finally:
        continuation.setup = original_setup
        balanced.setup = original_preflight_setup
        continuation.teacher_target = original_target
        balanced.teacher_target = original_preflight_target
        continuation.prepare_bank = original_bank
        parent.clear()


def teacher_evidence(directory, checkpoint):
    config = balanced.read(directory / "config.json")
    summary = balanced.read(directory / "summary.json")
    if (balanced.read(directory / "status.json")["state"] != "complete"
            or not summary["complete"] or summary["cases"] != 30
            or summary["small_cases"] != 24 or summary["large_cases"] != 6
            or not summary["model_state_unchanged"] or not summary["pooled_parity_passed"]
            or summary["pooled_parity_checks"] != 120
            or summary["config_sha256"] != sha256(directory / "config.json")
            or config["case_indices"] != list(range(30))
            or Path(config["checkpoints"]["initialization"]).resolve() != Path(checkpoint).resolve()
            or any(config[field] for field in ("training_performed", "checkpoint_selection_performed",
                                               "real_benchmark_data_used", "gate_changed"))
            or any(summary[field] for field in ("training_performed", "checkpoint_selection_performed", "gate_changed"))):
        raise ValueError("Require the completed immutable full-context synthetic diagnostic")
    for field in ("input_sha256", "source_sha256", "checkpoint_sha256"):
        verify_hashes(config[field])
    rows = {row["role"]: row for row in summary["summary"]}
    cases = [balanced.read(directory / "cases" / f"{index:03d}.json") for index in range(30)]
    if (len({case["seed"] for case in cases}) != 30
            or any(case["index"] != index or case["gradient_data"] != "synthetic"
                   or case["domain"] != "imputation" for index, case in enumerate(cases))):
        raise ValueError("Teacher evidence must contain distinct original synthetic cases")
    for role in ("general", "observed_blend", "initialization", "candidate"):
        for metric in ("pooled_rmse", "missbench_nrmse", "uci_nrmse"):
            values = [case["scores"][role][metric] for case in cases]
            if (not all(np.isfinite(value) and value >= 0 for value in values)
                    or not np.isclose(float(np.mean(values)), rows[role][metric], rtol=1e-12, atol=1e-12)):
                raise ValueError("Full-context diagnostic summary does not reproduce")
    if not all(rows["initialization"][metric] < rows["general"][metric]
               for metric in ("pooled_rmse", "missbench_nrmse", "uci_nrmse")):
        raise ValueError("Synthetic evidence does not favor retaining the stronger initialization")
    artifacts = [directory / name for name in ("config.json", "summary.json")]
    artifacts.extend(directory / "cases" / f"{index:03d}.json" for index in range(30))
    return {"directory": str(directory), "cases": 30, "initialization": rows["initialization"],
            "general": rows["general"], "artifacts_sha256": {str(path): sha256(path) for path in artifacts},
            "interpretation": "Reused synthetic evidence motivates a new teacher-policy hypothesis, not independent confirmation or real benchmark success"}


def make_config(domain_run, diagnostic, after_campaign):
    base = balanced.read(domain_run / "config.json")
    continuation.verify_run(base, domain_run)
    if (base["version"] != 4 or base["probability_loss_scales"] != balanced.PROBABILITY_SCALES
            or base["training_teacher_policy"] != {"imputation": "general", "recsys": "specialist"}
            or balanced.read(domain_run / "status.json")["state"] != "training_complete"):
        raise ValueError("Require the completed v4 fixed-domain teacher recipe")
    evidence = teacher_evidence(diagnostic, base["parent_checkpoint"])
    config = deepcopy(base)
    config.update(
        version=5, seed=31100501, data_seed=32100501, validation_seed=33100501,
        confirmation_seed=34100501, preflight_seed=35100501,
        validation_cases=continuation.bank_specs(33100501),
        confirmation_cases=continuation.bank_specs(34100501, confirmation=True),
        training_teacher_policy=dict(TEACHER_POLICY), teacher_policy_evidence=evidence,
        after_campaign=str(after_campaign), domain_recipe=str(domain_run),
        experiment="Sibling from the same confirmed v2 initialization; replace only imputation point teacher with a frozen initialization copy; retain specialist ratings, balanced losses, priors, normalization, shapes, optimizer, selectors and fresh-bank confirmation",
        execution="Process-local setup creates a detached frozen parent outside student state; original two teachers and blended reference-bank callbacks remain unchanged; restore callbacks on exit",
        inference="One unchanged network architecture and state; no teacher, router, extra head or output ensemble",
    )
    config["phases"][0]["name"] = "parent_teacher"
    old_seeds = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]}
    new_seeds = [case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]]
    if (len(set(new_seeds)) != len(new_seeds) or old_seeds.intersection(new_seeds)
            or config["data_seed"] + 16 * config["steps"] + 1 >= config["validation_seed"]):
        raise ValueError("Require disjoint fresh training, validation and confirmation seeds")
    predecessor = after_campaign / "cache_launcher_config.json"
    if balanced.read(predecessor)["run"] != str(domain_run) or not balanced.campaign_ready(after_campaign):
        raise ValueError("Serialize after the completed unchanged v4 campaign")
    config["input_sha256"].update({str(domain_run / "config.json"): sha256(domain_run / "config.json"),
                                    str(predecessor): sha256(predecessor), **evidence["artifacts_sha256"]})
    sources = {Path(__file__).resolve()}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config["source_sha256"].update({str(path): sha256(path) for path in sources})
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def train(config, output, *, resume=False):
    if config["training_teacher_policy"] != TEACHER_POLICY:
        raise ValueError("Require the registered frozen-parent teacher policy")
    with training_teachers():
        balanced.train(config, output, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain-run", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        if (any(value is None for value in (arguments.domain_run, arguments.diagnostic, arguments.after_campaign))
                or arguments.wait or arguments.resume):
            parser.error("Registration requires the domain recipe, diagnostic and predecessor; no wait/resume")
        config = make_config(arguments.domain_run.resolve(), arguments.diagnostic.resolve(),
                             arguments.after_campaign.resolve())
        balanced.register(config, output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = balanced.read(output / "status.json")
        if (status["state"] != "registered" and not arguments.resume) or status["state"] == "training_complete":
            raise ValueError("Never overwrite started training or resume completed training")
        try:
            config = balanced.read(output / "config.json")
            continuation.verify_run(config, output)
            if config["training_teacher_policy"] != TEACHER_POLICY:
                raise ValueError("Require the registered frozen-parent teacher policy")
            if not balanced.campaign_ready(Path(config["after_campaign"])):
                raise RuntimeError("Predecessor campaign must finish before starting")
            wait_for_gpu(output, arguments.wait, False)
            storage_guard(output)
            if not arguments.resume:
                write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                                         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
                write_json(output / "status.json", {"state": "preflight", "pid": os.getpid(), "updated_unix": time.time()})
                with training_teachers():
                    balanced.preflight(config, output)
            train(config, output, resume=arguments.resume)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                                "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
