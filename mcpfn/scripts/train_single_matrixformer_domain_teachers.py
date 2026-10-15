"""Train one student with fixed domain experts, preserving blended evaluation targets."""

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
import train_single_matrixformer_probability_balance as balanced
from train_single_matrixformer import storage_guard
from train_streamed_large_matrices import wait_for_gpu


continuation = balanced.continuation
TEACHER_POLICY = {"imputation": "general", "recsys": "specialist"}


@torch.no_grad()
def domain_teacher_target(teachers, criterion, episode, options=STREAM_OPTIONS):
    if (episode.metadata.get("gradient_data") != "synthetic"
            or episode.metadata.get("domain") not in TEACHER_POLICY):
        raise ValueError("Fixed domain teachers require a declared synthetic episode")
    if len(teachers) != 2 or any(model.training or any(parameter.requires_grad for parameter in model.parameters())
                                 for model in teachers):
        raise ValueError("Both original teachers must be frozen in evaluation mode")
    specialist = episode.metadata["domain"] == "recsys"
    with compute_context(episode.source.device):
        target = streamed_point_predictions(teachers[int(specialist)], episode.source, criterion, **options)
    return target.detach(), {"specialist_weight": float(specialist),
                             "target": TEACHER_POLICY[episode.metadata["domain"]],
                             "policy": "fixed domain expert for synthetic training only"}


@contextmanager
def training_teachers():
    original_target = continuation.teacher_target
    original_bank = continuation.prepare_bank
    original_preflight_target = balanced.teacher_target

    def reference_bank(*arguments, **keywords):
        continuation.teacher_target = original_target
        try:
            return original_bank(*arguments, **keywords)
        finally:
            continuation.teacher_target = domain_teacher_target

    continuation.teacher_target = domain_teacher_target
    continuation.prepare_bank = reference_bank
    balanced.teacher_target = domain_teacher_target
    try:
        yield
    finally:
        continuation.teacher_target = original_target
        continuation.prepare_bank = original_bank
        balanced.teacher_target = original_preflight_target


def teacher_evidence(directory):
    config = balanced.read(directory / "config.json")
    summary = balanced.read(directory / "summary.json")
    if (balanced.read(directory / "status.json")["state"] != "complete"
            or not summary["complete"] or summary["cases"] != 24
            or not summary["model_state_unchanged"] or summary["cuda_initialized"]
            or summary["training_performed"] or summary["checkpoint_selection_performed"]
            or config["training_performed"] or config["selection_performed"]
            or config["real_benchmark_data_used"] or config["case_indices"] != list(range(24))
            or summary["config_sha256"] != sha256(directory / "config.json")):
        raise ValueError("Require the completed immutable synthetic teacher diagnostic")
    for field in ("input_sha256", "source_sha256", "checkpoint_sha256"):
        verify_hashes(config[field])
    rows = {row["role"]: row for row in summary["summary"]}
    cases = [balanced.read(directory / "cases" / f"{index:03d}.json") for index in range(24)]
    if (len({case["seed"] for case in cases}) != 24
            or any(case["index"] != index or case["gradient_data"] != "synthetic"
                   or case["domain"] != "imputation" for index, case in enumerate(cases))):
        raise ValueError("Teacher evidence must contain the original distinct synthetic cases")
    for role in ("general", "observed_blend"):
        for metric in ("pooled_rmse", "missbench_nrmse", "uci_nrmse"):
            values = [case["scores"][role][metric] for case in cases]
            if (not all(np.isfinite(value) and value >= 0 for value in values)
                    or not np.isclose(float(np.mean(values)), rows[role][metric], rtol=1e-12, atol=1e-12)):
                raise ValueError("Teacher diagnostic summary does not reproduce")
    if not all(rows["general"][metric] < rows["observed_blend"][metric]
               for metric in ("pooled_rmse", "missbench_nrmse", "uci_nrmse")):
        raise ValueError("Synthetic evidence does not favor the fixed general imputation teacher")
    artifacts = [directory / name for name in ("config.json", "summary.json")]
    artifacts.extend(directory / "cases" / f"{index:03d}.json" for index in range(24))
    return {"directory": str(directory), "cases": 24, "general": rows["general"],
            "observed_blend": rows["observed_blend"],
            "artifacts_sha256": {str(path): sha256(path) for path in artifacts},
            "interpretation": "Hypothesis derived from a completed reused synthetic bank, not a preregistered diagnostic admission threshold or independent confirmation"}


def make_config(probability_run, diagnostic, after_campaign):
    base = balanced.read(probability_run / "config.json")
    continuation.verify_run(base, probability_run)
    if base["version"] != 3 or base["probability_loss_scales"] != balanced.PROBABILITY_SCALES:
        raise ValueError("Require the registered v3 probability-balanced recipe")
    evidence = teacher_evidence(diagnostic)
    config = deepcopy(base)
    config.update(
        version=4, seed=26100501, data_seed=27100501, validation_seed=28100501,
        confirmation_seed=29100501, preflight_seed=30100501,
        validation_cases=continuation.bank_specs(28100501),
        confirmation_cases=continuation.bank_specs(29100501, confirmation=True),
        training_teacher_policy=dict(TEACHER_POLICY), teacher_policy_evidence=evidence,
        after_campaign=str(after_campaign), probability_recipe=str(probability_run),
        experiment="Sibling of v3 from the same independently confirmed v2 checkpoint; replace only synthetic imputation point teacher with general, retain specialist for ratings, balanced losses, priors, normalization, optimizer, context schedule and selectors; fresh seeds",
        reference_teacher_policy="Original observed blend for imputation, specialist for ratings; unchanged for cached selection and independent confirmation",
        execution="Isolated process-local training/preflight teacher adapter; cached reference-bank generation temporarily restores the original teacher; independent confirmation and real benchmark workers remain unmodified",
    )
    config["phases"][0]["name"] = "domain_teachers"
    old_seeds = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]}
    new_seeds = [case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]]
    if (len(set(new_seeds)) != len(new_seeds) or old_seeds.intersection(new_seeds)
            or config["data_seed"] + 16 * config["steps"] + 1 >= config["validation_seed"]):
        raise ValueError("Require disjoint fresh training, validation and confirmation seeds")
    predecessor = after_campaign / "cache_launcher_config.json"
    if balanced.read(predecessor)["run"] != str(probability_run):
        raise ValueError("Serialize after the unchanged v3 benchmark campaign")
    config["input_sha256"].update({str(probability_run / "config.json"): sha256(probability_run / "config.json"),
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
        raise ValueError("Require the registered fixed domain-teacher policy")
    with training_teachers():
        balanced.train(config, output, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-run", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        if (any(value is None for value in (arguments.probability_run, arguments.diagnostic, arguments.after_campaign))
                or arguments.wait or arguments.resume):
            parser.error("Registration requires the probability recipe, diagnostic and predecessor; no wait/resume")
        config = make_config(arguments.probability_run.resolve(), arguments.diagnostic.resolve(),
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
                raise ValueError("Require the registered fixed domain-teacher policy")
            while not balanced.campaign_ready(Path(config["after_campaign"])):
                if not arguments.wait:
                    raise RuntimeError("Predecessor campaign is still running")
                write_json(output / "status.json", {"state": "waiting_for_campaign", "pid": os.getpid(), "updated_unix": time.time()})
                time.sleep(30)
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
