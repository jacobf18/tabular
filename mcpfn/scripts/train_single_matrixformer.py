"""Train a single checkpoint on synthetic imputation and rating episodes."""

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import torch

from dwip_recsys_provenance import require_synthetic_checkpoint
from dwip_teacher_mean_training import control_recipe, initialize_student, verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_distillation import (
    MixedSampler, distillation_loss, episode_metrics, selection_score, teacher_target, validation_specs,
)
from streamed_large_matrix_training import (
    STREAM_OPTIONS, TRAIN_OPTIONS, atomic_torch_save, learning_rate, restore_rng, rng_state,
)
from tabimpute.streamed_inference import streamed_point_predictions
from tabimpute.streamed_training import streamed_training_loss
from train_streamed_large_matrices import append_json, optimizer_update, wait_for_gpu
from dwip_teacher_mean_training import compute_context


GENERAL = REPO / "runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth"
SPECIALIST = REPO / "runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth"
INPUTS = {
    str(GENERAL): "c0ddf75274c8ff92ca5fd22c1263369ffa7e55eb7cb85fe80961b0afa101f7a9",
    str(SPECIALIST): "1fcadc0c5846f9c35fbbae4e3b519b9244b608272cb1eda68c0850132321f770",
}


def storage_guard(output):
    if shutil.disk_usage(output).free < 18 * 1024**3:
        raise OSError("Require 18 GiB free before writes, preserving at least 15 GiB after checkpoint replacement")


def make_config(arguments):
    if arguments.steps < 2 or not 0 < arguments.validate_every <= arguments.steps:
        raise ValueError("Require at least two updates and a valid validation interval")
    verify_hashes(INPUTS)
    provenance = require_synthetic_checkpoint(SPECIALIST)
    parent_config = json.loads((GENERAL.parent / "config.json").read_text())
    verify_hashes(parent_config["input_sha256"])
    sources = {Path(__file__).resolve()}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    return {
        "version": 1, "model": parent_config["model"], "steps": arguments.steps,
        "validate_every": arguments.validate_every, "seed": 6100501, "data_seed": 7100501,
        "validation_seed": 8100501, "parent_checkpoint": str(GENERAL), "teachers": [str(GENERAL), str(SPECIALIST)],
        "input_sha256": INPUTS, "specialist_provenance": provenance,
        "inherited_synthetic_inputs": parent_config["input_sha256"],
        "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
        "teacher_config_sha256": {str(path.parent / "config.json"): sha256(path.parent / "config.json")
                                  for path in (GENERAL, SPECIALIST)},
        "phases": [dict(name="mixed_pilot", end_step=arguments.steps, rows_low=256, rows_high=2000,
                        cols_low=128, cols_high=1000, batch_size=1)],
        "peak_lr": 5e-6, "warmup_steps": min(20, arguments.steps - 1), "weight_decay": .02,
        "grad_clip": .5, "memory_fraction": .8, "train_options": TRAIN_OPTIONS,
        "training_data": "Synthetic only; no benchmark data, labels, or selection in neural training",
        "mixture": "One imputation and one recommender episode per update; equal matrix-loss weights",
        "objective": "2*Huber(student mean, frozen target, delta=1) + NLL + .05*true CDF MSE; recsys NLL weight .1",
        "teacher_targets": "Imputation: observed-error blend on identical normalized inputs; recsys: specialist",
        "selection": "Minimum mean of domain RMSE / frozen synthetic teacher RMSE; imputation <=1.03x initial, recsys RMSE and MAE <=initial; step zero eligible",
        "inference": "One student checkpoint, no teachers or router; domain-appropriate observed-only preprocessing",
    }


def setup(config):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
    args = control_recipe()
    args.init_model_checkpoint = config["parent_checkpoint"]
    args.seed, args.lr, args.weight_decay = config["seed"], config["peak_lr"], config["weight_decay"]
    student, optimizer, _, criterion = initialize_student(args, torch.device("cuda"))
    teachers = []
    for path in config["teachers"]:
        teacher = deepcopy(student)
        teacher.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
        teachers.append(teacher.eval().requires_grad_(False))
    return student, optimizer, criterion, teachers, MixedSampler(config)


def backward_episode(student, criterion, episode, teacher, weight):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Neural updates must use synthetic data only")
    nll_weight = .1 if episode.metadata["domain"] == "recsys" else 1.
    with compute_context(episode.source.device):
        loss = streamed_training_loss(
            student, episode.source, episode.target, episode.mask,
            lambda logits, truth, auxiliary: distillation_loss(criterion, logits, truth, auxiliary, nll_weight=nll_weight),
            auxiliary_targets=teacher, **TRAIN_OPTIONS,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite distillation loss")
    (weight * loss).backward()
    return float(loss.detach())


def preflight(config, output):
    student, optimizer, criterion, teachers, sampler = setup(config)
    cases = []
    specifications = [dict(domain="imputation", shape=(2000, 1000), component=0, transpose=orientation)
                      for orientation in (False, True)] + [dict(domain="recsys", shape=(1000, 1000))]
    for index, specification in enumerate(specifications):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        episode = sampler.sample(step=1, seed=9100501 + index, **specification)
        target, decision = teacher_target(teachers, criterion, episode)
        loss = backward_episode(student, criterion, episode, target, 1.)
        norm = optimizer_update(student, optimizer, config["grad_clip"])
        case = {**episode.metadata, "loss": loss, "grad_norm": norm, "teacher": decision,
                "seconds": time.monotonic() - started, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
        cases.append(case)
        print(json.dumps({"preflight": case}), flush=True)
        del episode, target
    write_json(output / "preflight.json", {"passed": True, "config_sha256": sha256(output / "config.json"), "cases": cases})
    del student, optimizer, teachers, sampler, criterion
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def validate(student, criterion, teachers, sampler, config, output, step):
    was_training = student.training
    student.eval()
    cases = []
    try:
        for index, specification in enumerate(validation_specs()):
            episode = sampler.sample(step=1, seed=config["validation_seed"] + index, **specification)
            target, decision = teacher_target(teachers, criterion, episode)
            with compute_context(episode.source.device):
                prediction = streamed_point_predictions(student, episode.source, criterion, **STREAM_OPTIONS)
            cases.append({"index": index, **episode.metadata, "student": episode_metrics(prediction, episode),
                          "teacher": episode_metrics(target, episode), "routing": decision,
                          "teacher_disagreement_rmse": float((prediction[episode.mask].double() - target[episode.mask].double()).square().mean().sqrt())})
            del episode, target, prediction
    finally:
        student.train(was_training)
    summaries = {role: {domain: {metric: sum(case[role][metric] for case in cases if case["domain"] == domain)
                                            / sum(case["domain"] == domain for case in cases)
                               for metric in ("rmse", "mae")}
                       for domain in ("imputation", "recsys")} for role in ("student", "teacher")}
    write_json(output / f"validation_{step}.json", {"step": step, "cases": cases, **summaries})
    return summaries


def train(config, output, resume):
    student, optimizer, criterion, teachers, sampler = setup(config)
    fingerprint = sha256(output / "config.json")
    if json.loads((output / "preflight.json").read_text())["config_sha256"] != fingerprint:
        raise ValueError("Preflight/config mismatch")
    step, best_step, best_score, baseline, reference = 0, 0, float("inf"), None, None
    if resume:
        state = torch.load(output / "latest_checkpoint.pth", map_location="cpu", weights_only=True)
        if state["config_sha256"] != fingerprint:
            raise ValueError("Resume config mismatch")
        student.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step, best_step, best_score = state["step"], state["best_step"], state["best_score"]
        baseline, reference = state["baseline"], state["reference"]
        restore_rng(state["rng"])
        del state
    stop = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: stop.append(received))

    def save_latest():
        storage_guard(output)
        atomic_torch_save({"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                           "step": step, "epoch": 102000 + step, "best_step": best_step, "best_score": best_score,
                           "baseline": baseline, "reference": reference, "rng": rng_state(),
                           "config_sha256": fingerprint}, output / "latest_checkpoint.pth")

    def fixed_validation():
        nonlocal baseline, reference, best_score, best_step
        metrics = validate(student, criterion, teachers, sampler, config, output, step)
        if baseline is None:
            baseline, reference = metrics["student"], metrics["teacher"]
        score, eligible = selection_score(metrics["student"], baseline, reference)
        if eligible and score < best_score:
            best_step, best_score = step, score
            storage_guard(output)
            atomic_torch_save({"model": student.state_dict(), "step": step, "epoch": 102000 + step,
                               "metrics": metrics, "config_sha256": fingerprint}, output / "best_validation_checkpoint.pth")
            write_json(output / "best_validation_metrics.json", {"step": step, "score": score, **metrics})
        row = {"step": step, "score": score, "eligible": eligible, "best_step": best_step, **metrics}
        append_json(output / "validation.jsonl", row)
        print(json.dumps({"validation": row}), flush=True)

    if not resume:
        fixed_validation()
        save_latest()
    for step in range(step + 1, config["steps"] + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        rate = learning_rate(step, config["steps"], config["peak_lr"], config["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        microsteps = []
        for domain in ("imputation", "recsys"):
            episode = sampler.sample(domain, step)
            target, decision = teacher_target(teachers, criterion, episode)
            loss = backward_episode(student, criterion, episode, target, .5)
            microsteps.append({**episode.metadata, "loss": loss, "teacher": decision})
            del episode, target
        norm = optimizer_update(student, optimizer, config["grad_clip"])
        row = {"step": step, "lr": rate, "grad_norm": norm, "microsteps": microsteps,
               "seconds": time.monotonic() - started, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
        append_json(output / "training.jsonl", row)
        print(json.dumps({"step": step, "seconds": row["seconds"], "losses": [case["loss"] for case in microsteps]}), flush=True)
        if step % config["validate_every"] == 0 or step == config["steps"]:
            fixed_validation()
        if step % 10 == 0 or step == config["steps"] or stop:
            save_latest()
        write_json(output / "status.json", {"state": "training", "step": step, "best_step": best_step,
                                            "pid": os.getpid(), "updated_unix": time.time()})
        if stop:
            break
    write_json(output / "status.json", {"state": "interrupted" if stop else "training_complete",
                                        "step": step, "best_step": best_step, "best_score": best_score,
                                        "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if not arguments.resume:
        output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            storage_guard(output)
            wait_for_gpu(output, False, False)
            if arguments.resume:
                config = json.loads((output / "config.json").read_text())
                verify_hashes(config["input_sha256"])
                verify_hashes(config["source_sha256"])
                verify_hashes(config["teacher_config_sha256"])
            else:
                config = make_config(arguments)
                write_json(output / "config.json", config)
                for source in config["source_sha256"]:
                    destination = output / "sources" / Path(source).relative_to(REPO)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                                         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
                write_json(output / "status.json", {"state": "preflight", "pid": os.getpid(), "updated_unix": time.time()})
                preflight(config, output)
            train(config, output, arguments.resume)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                                "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
