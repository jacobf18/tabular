"""Prepare a synthetic-only continuation with reduced imputation probability loss."""

import argparse
from copy import deepcopy
import fcntl
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.nn import functional

from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_distillation import distillation_loss, teacher_target
from streamed_large_matrix_training import TRAIN_OPTIONS
from tabimpute.streamed_training import streamed_training_loss
from train_single_matrixformer import setup, storage_guard
import train_single_matrixformer_continuation as continuation
from train_streamed_large_matrices import optimizer_update, wait_for_gpu


PROBABILITY_SCALES = {"imputation": .1, "recsys": 1.}


def read(path):
    return json.loads(Path(path).read_text())


def balanced_loss(criterion, logits, truth, teacher, *, domain, probability_scales):
    if domain not in PROBABILITY_SCALES or set(probability_scales) != set(PROBABILITY_SCALES):
        raise ValueError("Require explicit imputation and rating loss scales")
    scale = probability_scales[domain]
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Probability scale must be finite and in (0, 1]")
    original = distillation_loss(criterion, logits, truth, teacher,
                                 nll_weight=.1 if domain == "recsys" else 1.)
    if scale == 1.:
        return original
    point = 2. * functional.huber_loss(criterion.mean(logits.float()), teacher.detach().float(),
                                      delta=1., reduction="none")
    return scale * original + (1. - scale) * point


def backward_episode(student, criterion, episode, teacher, weight, *, probability_scales):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Neural updates must use synthetic data only")
    with compute_context(episode.source.device):
        loss = streamed_training_loss(
            student, episode.source, episode.target, episode.mask,
            lambda logits, truth, auxiliary: balanced_loss(
                criterion, logits, truth, auxiliary, domain=episode.metadata["domain"],
                probability_scales=probability_scales),
            auxiliary_targets=teacher, **TRAIN_OPTIONS,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite probability-balanced loss")
    (weight * loss).backward()
    return float(loss.detach())


def train(config, output, *, resume=False):
    if config["probability_loss_scales"] != PROBABILITY_SCALES:
        raise ValueError("Require the fixed probability-balance experiment")
    original = continuation.backward_episode
    continuation.backward_episode = partial(backward_episode, probability_scales=config["probability_loss_scales"])
    try:
        continuation.train(config, output, resume=resume)
    finally:
        continuation.backward_episode = original


def diagnostic_evidence(directory):
    if read(directory / "status.json")["state"] != "complete":
        raise ValueError("Require the completed full-context synthetic gradient diagnostic")
    config, summary = read(directory / "config.json"), read(directory / "summary.json")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    if (not summary["complete"] or summary["cases"] != 20 or not summary["model_state_unchanged"]
            or summary["config_sha256"] != sha256(directory / "config.json")
            or config["training_performed"] or config["real_benchmark_inputs_used"]
            or config["case_indices"] != list(range(8)) + list(range(16, 28))
            or "BF16" not in config["precision"]):
        raise ValueError("Require the original immutable full-context synthetic diagnostic")
    ratio = summary["groups"]["imputation"]["mean_gradient"]["probability_to_point_norm"]
    if ratio is None or not math.isfinite(ratio) or ratio <= 2.:
        raise ValueError("Full-context evidence does not support the proposed probability-loss reduction")
    return {"directory": str(directory), "summary_sha256": sha256(directory / "summary.json"),
            "imputation_probability_to_point_norm": ratio,
            "motivation": "Fixed 0.1 imputation probability multiplier from CPU scale imbalance; require full-context ratio >2 before registration",
            "limitation": "Gradient norms at the first pilot are a hypothesis, not proof about AdamW or benchmark gains"}


def make_config(parent, confirmation, diagnostic, after_campaign):
    base = read(parent / "config.json")
    continuation.verify_run(base, parent)
    gate = read(confirmation / "gate.json")
    checkpoint = parent / "best_validation_checkpoint.pth"
    status = read(parent / "status.json")
    if (status["state"] != "training_complete" or read(confirmation / "status.json")["state"] != "complete"
            or not gate["passed"] or not gate["initialization_retained"] or not gate["general_retained"]
            or gate["selected_step"] != status["best_step"] or gate["selected_step"] <= 0
            or gate["checkpoint_sha256"] != sha256(checkpoint)):
        raise ValueError("Require the completed, independently synthetic-confirmed parent checkpoint")
    evidence = diagnostic_evidence(diagnostic)
    config = deepcopy(base)
    for field in ("initializer_screen", "initializer_selection"):
        config.pop(field, None)
    config.update(
        version=3, steps=400, validate_every=50, peak_lr=1e-5, warmup_steps=20,
        seed=21100501, data_seed=22100501, validation_seed=23100501, confirmation_seed=24100501,
        parent_checkpoint=str(checkpoint), parent_confirmation=str(confirmation),
        validation_cases=continuation.bank_specs(23100501),
        confirmation_cases=continuation.bank_specs(24100501, confirmation=True),
        preflight_seed=25100501, probability_loss_scales=dict(PROBABILITY_SCALES),
        probability_balance_evidence=evidence, after_campaign=str(after_campaign),
        objective="2*Huber(student mean, frozen teacher mean) + alpha_domain*(NLL_weight_domain*NLL + .05*CDF); alpha_imputation=.1, alpha_recsys=1; NLL weights 1 and .1",
        experiment="Continue synthetic-confirmed v2 with lower imputation probability loss; unchanged priors, normalization, context sizes, teacher targets, equal-domain mixture and minimax selector",
        inference="One original network state; no teacher parameters, routing or averaging of outputs",
        execution="Isolated worker temporarily injects its registered episode-backward function into the unchanged continuation loop; restores it on exit",
    )
    config["phases"][0].update(name="probability_balance", end_step=config["steps"])
    previous_seeds = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]}
    fresh_seeds = [case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]]
    if len(set(fresh_seeds)) != len(fresh_seeds) or previous_seeds.intersection(fresh_seeds):
        raise ValueError("Require disjoint fresh synthetic banks")
    if config["data_seed"] + 16 * config["steps"] + 1 >= config["validation_seed"]:
        raise ValueError("Training and validation seeds overlap")
    artifacts = [checkpoint, parent / "config.json", confirmation / "gate.json",
                 diagnostic / "config.json", diagnostic / "summary.json", after_campaign / "config.json"]
    config["input_sha256"] = {**base["input_sha256"], **{str(path): sha256(path) for path in artifacts}}
    sources = {Path(__file__).resolve(), REPO / "scripts/confirm_single_matrixformer_continuation.py",
               REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config["source_sha256"] = {str(path): sha256(path) for path in sorted(sources)}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["teacher_config_sha256"])
    return config


def register(config, output):
    output.mkdir(parents=True, exist_ok=False)
    storage_guard(output)
    write_json(output / "config.json", config)
    for source in config["source_sha256"]:
        destination = output / "sources" / Path(source).relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
                                              "registered_unix": time.time(), "training_started": False,
                                              "selected_from_synthetic_evidence_only": True})
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def campaign_ready(directory):
    status = read(directory / "status.json")
    if status["state"] in ("complete", "not_promoted"):
        return True
    if status["state"] in ("failed", "interrupted"):
        raise RuntimeError("Resolve the stopped predecessor campaign before new training")
    try:
        os.kill(status["pid"], 0)
        if Path(f"/proc/{status['pid']}/stat").read_text().split(") ", 1)[1].split()[0] == "Z":
            raise ProcessLookupError()
    except (ProcessLookupError, FileNotFoundError):
        if read(directory / "status.json")["state"] in ("complete", "not_promoted"):
            return True
        raise RuntimeError("Predecessor process is missing or terminal; never restart it here") from None
    return False


def preflight(config, output):
    student, optimizer, criterion, teachers, sampler = setup(config)
    specifications = [dict(domain="imputation", shape=(2000, 1000), component=0, transpose=orientation)
                      for orientation in (False, True)] + [dict(domain="recsys", shape=(2000, 1000))]
    cases = []
    for index, specification in enumerate(specifications):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        episode = sampler.sample(step=1, seed=config["preflight_seed"] + index, **specification)
        target, routing = teacher_target(teachers, criterion, episode)
        loss = backward_episode(student, criterion, episode, target, 1.,
                                probability_scales=config["probability_loss_scales"])
        norm = optimizer_update(student, optimizer, config["grad_clip"])
        case = {**episode.metadata, "loss": loss, "grad_norm": norm, "teacher": routing,
                "seconds": time.monotonic() - started, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
        cases.append(case)
        print(json.dumps({"preflight": case}), flush=True)
        del episode, target
    write_json(output / "preflight.json", {"passed": True, "config_sha256": sha256(output / "config.json"),
                                           "probability_loss_scales": config["probability_loss_scales"], "cases": cases})
    del student, optimizer, teachers, sampler, criterion
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        if (any(value is None for value in (arguments.parent, arguments.confirmation, arguments.diagnostic,
                                            arguments.after_campaign)) or arguments.resume or arguments.wait):
            parser.error("Registration requires parent, confirmation, diagnostic and predecessor campaign; no wait/resume")
        config = make_config(arguments.parent.resolve(), arguments.confirmation.resolve(),
                             arguments.diagnostic.resolve(), arguments.after_campaign.resolve())
        register(config, output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = read(output / "status.json")
        if (status["state"] != "registered" and not arguments.resume) or status["state"] == "training_complete":
            raise ValueError("Never overwrite started training or resume completed training")
        try:
            config = read(output / "config.json")
            continuation.verify_run(config, output)
            while not campaign_ready(Path(config["after_campaign"])):
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
                preflight(config, output)
            train(config, output, resume=arguments.resume)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                                "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
