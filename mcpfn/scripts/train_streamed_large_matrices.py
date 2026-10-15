"""Capacity-test, then fine-tune the fixed 98k checkpoint on a large-matrix curriculum."""

import argparse
from dataclasses import asdict
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

from dwip_teacher_mean_training import (
    DESIGN, FrozenMeanTeachers, control_recipe, initialize_student, read_json, verify_hashes,
)
from recsys_evaluation_common import CHECKPOINT, REPO, sha256, write_json
from streamed_large_matrix_training import (
    CHECKPOINT_SHA256, DEFAULT_PHASES, STREAM_OPTIONS, TRAIN_OPTIONS, CurriculumSampler,
    atomic_torch_save, backward_microstep, learning_rate, parse_curriculum, phase_at,
    restore_rng, rng_state, selection_score, validate,
)


def storage_guard(output, *, launch=False):
    required = (23 if launch else 21) * 1024**3
    free = shutil.disk_usage(output).free
    if free < required:
        raise OSError(f"Require {required / 1024**3:.0f} GiB free for reserve and atomic checkpoints; have {free / 1024**3:.2f}")


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def other_gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip() and int(line.strip()) != os.getpid()]


def wait_for_gpu(output, wait, allow_busy):
    while other_gpu_processes() and not allow_busy:
        if not wait:
            raise RuntimeError("GPU is occupied; use --wait-for-idle, or explicitly --allow-busy-gpu for a capped probe")
        write_json(output / "status.json", {"state": "waiting_for_gpu", "pid": os.getpid(), "updated_unix": time.time()})
        time.sleep(30)


def make_config(arguments):
    phases = parse_curriculum(read_json(arguments.curriculum) if arguments.curriculum else DEFAULT_PHASES)
    if not 0 <= arguments.warmup_steps < phases[-1].end_step:
        raise ValueError("Warmup must be shorter than the curriculum")
    if not 0 < arguments.memory_fraction <= .9:
        raise ValueError("Memory fraction must be in (0, .9]")
    sources = {Path(__file__).resolve(), REPO / "scripts/streamed_large_matrix_training.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    teachers = read_json(DESIGN / "next_single_model_compression_proposal.json")["teachers"]
    checkpoints = {str(CHECKPOINT): CHECKPOINT_SHA256}
    for teacher in teachers:
        checkpoints[str(CHECKPOINT.parent.parent / f"tabimpute-{teacher['queue']}-dwip1e5/best_validation_checkpoint.pth")] = teacher["sha256"]
    verify_hashes(checkpoints)
    parent = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if parent["epoch"] != 98000:
        raise ValueError("Expected the synthetic-selected 98,000-step parent")
    return {
        "version": 1, "model_family": "new_stable", "model": read_json(CHECKPOINT.parent / "config.json")["model"],
        "parent_checkpoint": str(CHECKPOINT), "parent_epoch": 98000, "input_sha256": checkpoints,
        "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
        "phases": [asdict(phase) for phase in phases], "accumulation": 2,
        "seed": 2201, "data_seed": 2150930, "validation_seed": 9150930,
        "spectral_replay": .2, "peak_lr": 5e-6, "warmup_steps": arguments.warmup_steps,
        "min_lr_ratio": .05, "weight_decay": .02, "grad_clip_norm": .5,
        "precision": "FP32 parameters/Adam moments, BF16 autocast", "stream_options": TRAIN_OPTIONS,
        "memory_fraction": arguments.memory_fraction, "validation_every": 250,
        "capacity_mask_fractions": [.6, .99],
        "checkpoint_every": 100, "checkpoint_seconds": 600,
        "selection": "Earliest strict minimum of mean retention/large baseline-relative macro RMSE, with retention <= 1.03 times baseline; include step zero",
        "objective": "true NLL + 2 * Huber(delta=1, equal frozen teacher mean) + .05 * true CDF MSE; FP32 per-cell reductions",
        "batch_weighting": "Mean of two independently sampled matrix-mean losses, accumulated serially",
    }


def setup(config):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
    args = control_recipe()
    args.init_model_checkpoint = config["parent_checkpoint"]
    args.seed = config["seed"]
    args.lr = config["peak_lr"]
    args.weight_decay = config["weight_decay"]
    model, optimizer, _, criterion = initialize_student(args, torch.device("cuda"))
    teachers = FrozenMeanTeachers.load_registered(args, criterion, torch.device("cuda"))
    sampler = CurriculumSampler(parse_curriculum(config["phases"]), data_seed=config["data_seed"],
                                spectral_replay=config["spectral_replay"])
    return model, optimizer, criterion, teachers, sampler


def optimizer_update(model, optimizer, clip):
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    if any(parameter.grad is None for parameter in model.parameters()):
        raise RuntimeError("A trainable parameter has no gradient")
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("Nonfinite updated parameter")
    return float(norm)


def preflight(config, output):
    model, optimizer, criterion, teachers, sampler = setup(config)
    reference = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    shapes = list(dict.fromkeys([(64, 32)] + [
        shape for phase in sampler.phases
        for shape in ((phase.rows_high, phase.cols_high), (phase.cols_high, phase.rows_high))
    ]))
    report = {"passed": False, "config_sha256": sha256(output / "config.json"), "cases": []}
    for rows, columns in shapes:
        model.load_state_dict(reference, strict=True)
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = config["peak_lr"]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses = []
        for microstep in range(config["accumulation"]):
            source, target, _, _ = sampler.sample(
                1, microstep, device="cuda", forced_shape=(rows, columns), component=0,
                seed=8150930 + microstep, support_mode="mixed", transpose=False,
            )
            generator = torch.Generator(device="cuda").manual_seed(7150930 + microstep)
            mask = torch.rand(target.shape, device="cuda", generator=generator) < config["capacity_mask_fractions"][microstep]
            source = target.masked_fill(mask, torch.nan)
            losses.append(backward_microstep(model, teachers, criterion, source, target, mask,
                                            accumulation=config["accumulation"], options=config["stream_options"]))
            del source, target, mask
        norm = optimizer_update(model, optimizer, config["grad_clip_norm"])
        torch.cuda.synchronize()
        changed = sum(not torch.equal(parameter.detach().cpu(), reference[name]) for name, parameter in model.named_parameters())
        if changed == 0:
            raise RuntimeError("Capacity probe failed to update parameters")
        case = {"rows": rows, "columns": columns, "loss": sum(losses) / len(losses), "grad_norm": norm,
                "updated_tensors": changed, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "seconds": time.monotonic() - started, "accumulation": config["accumulation"],
                "mask_fractions": config["capacity_mask_fractions"]}
        report["cases"].append(case)
        write_json(output / "preflight.json", report)
        print(json.dumps(case), flush=True)
    verify_hashes(config["input_sha256"])
    report["passed"] = True
    write_json(output / "preflight.json", report)
    write_json(output / "status.json", {"state": "preflight_passed", "updated_unix": time.time()})


def train(config, output, *, resume=False, max_updates=None):
    report = read_json(output / "preflight.json")
    fingerprint = sha256(output / "config.json")
    if not report["passed"] or report["config_sha256"] != fingerprint:
        raise ValueError("Every declared maximum shape must pass the matching full-objective capacity preflight")
    latest = output / "latest_checkpoint.pth"
    if latest.exists() != resume:
        raise ValueError("Use --resume for an existing rolling checkpoint; never silently restart or overwrite")
    if not resume and (output / "validation.jsonl").exists():
        raise ValueError("Interrupted initialization: use a new output directory rather than overwriting")
    model, optimizer, criterion, teachers, sampler = setup(config)
    start_step, best_step, best_score, baseline = 0, 0, 1., None
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["config_sha256"] != fingerprint:
            raise ValueError("Resume configuration differs from the saved trajectory")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_step, best_score, baseline = (state[key] for key in ("step", "best_step", "best_score", "baseline"))
        restore_rng(state["rng"])
        del state
        best = torch.load(output / "best_validation_checkpoint.pth", map_location="cpu", weights_only=True)
        if best["config_sha256"] != fingerprint:
            raise ValueError("Selected checkpoint does not belong to the resumed run")
        best_step = best["step"]
        best_score, _ = selection_score(best["metrics"], baseline)
        del best
    stop_requested = []
    def request_stop(signum, frame):
        stop_requested.append(signum)
    previous_handlers = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    step = start_step
    started = time.monotonic()
    last_save = started

    def checkpoint():
        storage_guard(output)
        atomic_torch_save({"epoch": config["parent_epoch"] + step, "step": step, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "rng": rng_state(), "config_sha256": fingerprint,
                           "best_step": best_step, "best_score": best_score, "baseline": baseline}, latest)

    def fixed_validation():
        nonlocal best_score, best_step, baseline
        metrics = validate(model, criterion, sampler, validation_seed=config["validation_seed"],
                           options={key: config["stream_options"][key] for key in STREAM_OPTIONS})
        if baseline is None:
            baseline = metrics
        score, eligible = selection_score(metrics, baseline)
        improved = step == 0 or eligible and score < best_score
        if improved:
            best_score, best_step = score, step
            storage_guard(output)
            atomic_torch_save({"epoch": config["parent_epoch"] + step, "step": step, "model": model.state_dict(),
                               "metrics": metrics, "config_sha256": fingerprint}, output / "best_validation_checkpoint.pth")
            write_json(output / "best_validation_metrics.json", {"step": step, "score": score, **metrics})
        append_json(output / "validation.jsonl", {"step": step, "score": score, "eligible": eligible,
                                                  "best_step": best_step, **metrics})
        print(json.dumps({"validation_step": step, "score": score, "eligible": eligible, "best_step": best_step}), flush=True)

    try:
        if not resume:
            fixed_validation()
            checkpoint()
        for step in range(start_step + 1, sampler.phases[-1].end_step + 1):
            storage_guard(output)
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            rate = learning_rate(step, sampler.phases[-1].end_step, config["peak_lr"], config["warmup_steps"], config["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            torch.cuda.reset_peak_memory_stats()
            microsteps = []
            for microstep in range(config["accumulation"]):
                source, target, mask, stats = sampler.sample(step, microstep, device="cuda")
                loss = backward_microstep(model, teachers, criterion, source, target, mask,
                                          accumulation=config["accumulation"], options=config["stream_options"])
                microsteps.append({**stats, "loss": loss})
                del source, target, mask
            norm = optimizer_update(model, optimizer, config["grad_clip_norm"])
            torch.cuda.synchronize()
            record = {"step": step, "parent_epoch": config["parent_epoch"], "lr": rate, "grad_norm": norm,
                      "loss": sum(item["loss"] for item in microsteps) / config["accumulation"],
                      "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
                      "seconds": time.monotonic() - step_started, "microsteps": microsteps}
            append_json(output / "metrics.jsonl", record)
            if step == start_step + 1 or step % 10 == 0:
                print(json.dumps(record), flush=True)
            boundary = step == phase_at(sampler.phases, step).end_step
            if step % config["validation_every"] == 0 or boundary:
                fixed_validation()
            stopping = bool(stop_requested) or max_updates is not None and step - start_step >= max_updates
            if step % config["checkpoint_every"] == 0 or boundary or stopping or time.monotonic() - last_save >= config["checkpoint_seconds"]:
                checkpoint()
                last_save = time.monotonic()
            write_json(output / "status.json", {"state": "training", "step": step, "pid": os.getpid(),
                       "phase": phase_at(sampler.phases, step).name, "best_step": best_step, "best_score": best_score,
                       "updated_unix": time.time()})
            if stopping:
                break
        verify_hashes(config["input_sha256"])
        write_json(output / "status.json", {"state": "complete" if step == sampler.phases[-1].end_step else "paused",
                   "step": step, "best_step": best_step, "best_score": best_score,
                   "elapsed_seconds_this_session": time.monotonic() - started, "updated_unix": time.time()})
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--memory-fraction", type=float, default=.7)
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    arguments = parser.parse_args()
    if arguments.max_updates is not None and arguments.max_updates <= 0:
        parser.error("--max-updates must be positive")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            storage_guard(output, launch=True)
            wait_for_gpu(output, arguments.wait_for_idle, arguments.allow_busy_gpu)
            if arguments.mode == "preflight":
                if (output / "config.json").exists() or arguments.resume:
                    raise ValueError("Preflight requires a fresh output directory")
                config = make_config(arguments)
                write_json(output / "config.json", config)
                for source in config["source_sha256"]:
                    destination = output / "sources" / Path(source).relative_to(REPO)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                preflight(config, output)
            else:
                if arguments.curriculum:
                    raise ValueError("Training uses only the frozen preflight configuration")
                config = read_json(output / "config.json")
                verify_hashes(config["source_sha256"])
                verify_hashes(config["input_sha256"])
                train(config, output, resume=arguments.resume, max_updates=arguments.max_updates)
        except BaseException as error:
            write_json(output / "status.json", {"state": "error", "error": repr(error), "updated_unix": time.time(),
                       "recovery": "Only resume the last atomically saved complete optimizer update; no partial update is saved"})
            raise


if __name__ == "__main__":
    main()
