"""Continue a synthetically selected single-network initialization without benchmark selection."""

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from screen_single_matrixformer_initializers import aggregate, cache_episode, restore_episode, worst_score
from single_model_distillation import episode_metrics, teacher_target
from streamed_large_matrix_training import STREAM_OPTIONS, atomic_torch_save, learning_rate, restore_rng, rng_state
from tabimpute.streamed_inference import streamed_point_predictions
from train_single_matrixformer import backward_episode, preflight as original_preflight, setup, storage_guard
from train_streamed_large_matrices import append_json, optimizer_update, wait_for_gpu


METRICS = (("imputation", "rmse"), ("recsys", "rmse"), ("recsys", "mae"))


def read(path):
    return json.loads(Path(path).read_text())


def verified_selection(screen):
    if read(screen / "status.json")["state"] != "complete":
        raise ValueError("Require a complete synthetic initialization screen")
    config, selected = read(screen / "config.json"), read(screen / "selection.json")
    if selected["training_performed"] or selected["real_benchmark_inputs_used"]:
        raise ValueError("Initialization selection must be synthetic-only and untrained")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    verify_hashes({config["base_training_config"]: config["base_config_sha256"],
                   config["protocol"]: config["protocol_sha256"],
                   str(screen / "config.json"): selected["config_sha256"],
                   str(screen / "synthetic_bank.pt"): selected["bank_sha256"],
                   selected["checkpoint"]: selected["checkpoint_sha256"]})
    reference = read(screen / "reference.json")
    if reference["bank_sha256"] != selected["bank_sha256"]:
        raise ValueError("Synthetic reference bank changed")
    reports = []
    for candidate in config["candidates"]:
        report = read(screen / "candidates" / f"{candidate['name']}.json")
        if (report["candidate"] != candidate or report["config_sha256"] != selected["config_sha256"]
                or report["bank_sha256"] != selected["bank_sha256"] or len(report["cases"]) != len(config["cases"])):
            raise ValueError("Incomplete or mismatched synthetic candidate report")
        for index, (case, specification) in enumerate(zip(report["cases"], config["cases"])):
            if case["index"] != index or any(case[key] != specification[key] for key in ("domain", "seed")):
                raise ValueError("Candidate cases differ from the registered synthetic bank")
        metrics = aggregate(report["cases"], "student")
        score, ratios = worst_score(metrics, reference["selection_references"])
        if metrics != report["metrics"] or ratios != report["ratios"] or score != report["score"]:
            raise ValueError("Synthetic candidate score does not reproduce")
        reports.append(report)
    if (len(reports) != selected["candidates_completed"]
            or selected["cases_per_candidate"] != len(config["cases"])):
        raise ValueError("Incomplete synthetic selection")
    winner = min(reports, key=lambda report: report["score"])
    for key in ("score", "metrics", "ratios"):
        if winner[key] != selected[key]:
            raise ValueError("Selected scores do not match the registered winner")
    if winner["candidate"] != selected["selected"]:
        raise ValueError("Initialization is not the registered synthetic winner")
    return config, selected


def bank_specs(seed_start, *, confirmation=False):
    repeats = 3 if confirmation else 2
    specifications = [dict(domain="imputation", shape=(240, 80), component=component,
                           transpose=bool(repeat % 2)) for repeat in range(repeats) for component in range(8)]
    components = range(6) if confirmation else (0, 1, 4, 5)
    specifications.extend(dict(domain="imputation", shape=(2000, 1000), component=component,
                               transpose=bool(component % 2)) for component in components)
    specifications.extend(dict(domain="recsys", shape=shape) for shape in ((1000, 1000), (2000, 1000))
                          for _ in range(8 if confirmation else 4))
    return [{**specification, "seed": seed_start + index} for index, specification in enumerate(specifications)]


def selection_score(metrics, baseline, reference):
    score, ratios = worst_score(metrics, reference)
    worst_score(baseline, reference)
    eligible = all(metrics[domain][metric] <= baseline[domain][metric] for domain, metric in METRICS)
    return score, ratios, eligible


def training_episode(sampler, config, domain, step):
    if domain == "recsys":
        shapes = config["rating_training_shapes"]
        return sampler.sample(domain, step, shape=tuple(shapes[(step - 1) % len(shapes)]))
    return sampler.sample(domain, step)


def register(screen, output, *, steps, peak_lr, validate_every):
    if steps < 2 or not 0 < validate_every <= steps or not math.isfinite(peak_lr) or peak_lr <= 0:
        raise ValueError("Require positive learning rate, at least two updates, and valid validation interval")
    screen_config, selected = verified_selection(screen)
    config = deepcopy(read(screen_config["base_training_config"]))
    config.update(version=2, steps=steps, peak_lr=peak_lr, validate_every=validate_every,
                  warmup_steps=min(20, steps - 1), seed=13100501, data_seed=14100501,
                  validation_seed=15100501, confirmation_seed=16100501,
                  parent_checkpoint=selected["checkpoint"], initializer_screen=str(screen),
                  initializer_selection=selected, validation_cases=bank_specs(15100501),
                  confirmation_cases=bank_specs(16100501, confirmation=True),
                  rating_training_shapes=[[1000, 1000], [2000, 1000]],
                  selection="Minimum worst imputation RMSE, rating RMSE and rating MAE / frozen synthetic references; every metric <= initialization; step zero eligible",
                  confirmation="Fresh seeds; all three metrics <= initialization and general control, and strict minimax improvement versus initialization",
                  experiment="Continue synthetic-selected initialization with alternating 1000/2000-row ratings, retaining pilot priors, imputation shapes, equal-domain mixture and losses; fresh cached minimax validation",
                  inference="One unchanged network state; no teachers or routing during inference",
                  compatibility_epoch_offset=0)
    config["phases"][0].update(name="mixed_continuation", end_step=steps)
    if config["data_seed"] + 16 * steps + 1 >= config["validation_seed"]:
        raise ValueError("Training seed range must not overlap validation seeds")
    artifacts = [screen / name for name in ("config.json", "selection.json", "reference.json", "synthetic_bank.pt")]
    artifacts.extend(screen / "candidates" / f"{candidate['name']}.json" for candidate in screen_config["candidates"])
    config["input_sha256"] = {**config["input_sha256"], **screen_config["input_sha256"],
                              selected["checkpoint"]: selected["checkpoint_sha256"],
                              screen_config["base_training_config"]: screen_config["base_config_sha256"],
                              screen_config["protocol"]: screen_config["protocol_sha256"],
                              **{str(path): sha256(path) for path in artifacts}}
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt",
               REPO / "scripts/confirm_single_matrixformer_continuation.py"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    config["source_sha256"] = {str(path): sha256(path) for path in sorted(sources)}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["teacher_config_sha256"])
    write_json(output / "config.json", config)
    for source in sources:
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
                                              "registered_unix": time.time(), "training_started": False,
                                              "selected_from_synthetic_evidence_only": True})
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})
    return config


def preflight(config, output):
    original_preflight(config, output)
    student, optimizer, criterion, teachers, sampler = setup(config)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    episode = sampler.sample("recsys", step=1, seed=17100501, shape=(2000, 1000))
    target, routing = teacher_target(teachers, criterion, episode)
    loss = backward_episode(student, criterion, episode, target, 1.)
    norm = optimizer_update(student, optimizer, config["grad_clip"])
    case = {**episode.metadata, "loss": loss, "grad_norm": norm, "teacher": routing,
            "seconds": time.monotonic() - started, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
    manifest = read(output / "preflight.json")
    manifest["cases"].append(case)
    write_json(output / "preflight.json", manifest)
    print(json.dumps({"preflight": case}), flush=True)
    del student, optimizer, teachers, sampler, criterion, episode, target
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def prepare_bank(sampler, teachers, criterion, specifications, output):
    bank, cases = [], []
    for index, specification in enumerate(specifications):
        episode = sampler.sample(step=1, **specification)
        target, routing = teacher_target(teachers, criterion, episode)
        with compute_context(episode.source.device):
            general = streamed_point_predictions(teachers[0], episode.source, criterion, **STREAM_OPTIONS)
        cases.append({"index": index, **episode.metadata, "general": episode_metrics(general, episode),
                      "teacher": episode_metrics(target, episode), "routing": routing})
        bank.append(cache_episode(episode))
        write_json(output / "status.json", {"state": "building_synthetic_bank", "completed": len(bank),
                                            "expected": len(specifications), "pid": os.getpid(), "updated_unix": time.time()})
        del episode, target, general
    general, teacher = aggregate(cases, "general"), aggregate(cases, "teacher")
    references = {"imputation": {"rmse": min(general["imputation"]["rmse"], teacher["imputation"]["rmse"])},
                  "recsys": teacher["recsys"]}
    storage_guard(output)
    atomic_torch_save(bank, output / "synthetic_bank.pt")
    manifest = {"bank_sha256": sha256(output / "synthetic_bank.pt"), "cases": cases,
                "general": general, "teacher": teacher, "references": references}
    write_json(output / "reference.json", manifest)
    return bank, manifest


@torch.no_grad()
def evaluate_bank(model, criterion, bank, *, device="cuda"):
    was_training = model.training
    model.eval()
    cases = []
    try:
        for index, cached in enumerate(bank):
            episode = restore_episode(cached, device)
            with compute_context(episode.source.device):
                prediction = streamed_point_predictions(model, episode.source, criterion, **STREAM_OPTIONS)
            cases.append({"index": index, **episode.metadata, "student": episode_metrics(prediction, episode)})
            del episode, prediction
    finally:
        model.train(was_training)
    return aggregate(cases, "student"), cases


def verify_run(config, output):
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    verify_hashes(config["teacher_config_sha256"])
    if sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]:
        raise ValueError("Continuation configuration changed after registration")


def train(config, output, *, resume=False):
    verify_run(config, output)
    fingerprint = sha256(output / "config.json")
    if read(output / "preflight.json")["config_sha256"] != fingerprint:
        raise ValueError("Preflight/config mismatch")
    student, optimizer, criterion, teachers, sampler = setup(config)
    baseline, best_step, best_score, step = None, 0, float("inf"), 0
    if resume:
        reference = read(output / "reference.json")
        verify_hashes({str(output / "synthetic_bank.pt"): reference["bank_sha256"]})
        bank = torch.load(output / "synthetic_bank.pt", map_location="cpu", weights_only=True)
        state = torch.load(output / "latest_checkpoint.pth", map_location="cpu", weights_only=True)
        if state["config_sha256"] != fingerprint or state["bank_sha256"] != reference["bank_sha256"]:
            raise ValueError("Resume configuration or validation bank changed")
        student.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step, best_step, best_score, baseline = state["step"], state["best_step"], state["best_score"], state["baseline"]
        restore_rng(state["rng"])
        del state
    else:
        bank, reference = prepare_bank(sampler, teachers, criterion, config["validation_cases"], output)
    stop = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: stop.append(received))

    def save_latest():
        storage_guard(output)
        atomic_torch_save({"model": student.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                           "epoch": step, "best_step": best_step, "best_score": best_score, "baseline": baseline,
                           "rng": rng_state(), "config_sha256": fingerprint, "bank_sha256": reference["bank_sha256"]},
                          output / "latest_checkpoint.pth")

    def fixed_validation():
        nonlocal baseline, best_step, best_score
        metrics, cases = evaluate_bank(student, criterion, bank)
        if baseline is None:
            baseline = metrics
        score, ratios, eligible = selection_score(metrics, baseline, reference["references"])
        if eligible and score < best_score:
            best_step, best_score = step, score
            storage_guard(output)
            atomic_torch_save({"model": student.state_dict(), "step": step, "epoch": step, "metrics": metrics,
                               "config_sha256": fingerprint, "bank_sha256": reference["bank_sha256"]},
                              output / "best_validation_checkpoint.pth")
            write_json(output / "best_validation_metrics.json", {"step": step, "score": score, "ratios": ratios, "student": metrics})
        row = {"step": step, "score": score, "ratios": ratios, "eligible": eligible, "best_step": best_step, "student": metrics}
        write_json(output / f"validation_{step}.json", {**row, "cases": cases})
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
            episode = training_episode(sampler, config, domain, step)
            target, routing = teacher_target(teachers, criterion, episode)
            loss = backward_episode(student, criterion, episode, target, .5)
            microsteps.append({**episode.metadata, "loss": loss, "teacher": routing})
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
    verify_run(config, output)
    verify_hashes({str(output / "synthetic_bank.pt"): reference["bank_sha256"]})
    write_json(output / "status.json", {"state": "interrupted" if stop else "training_complete", "step": step,
                                        "best_step": best_step, "best_score": best_score,
                                        "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--peak-lr", type=float)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage == "register" and (arguments.screen is None or arguments.steps is None
                                           or arguments.peak_lr is None or arguments.validate_every is None or arguments.resume):
        parser.error("Registration requires --screen, --steps, --peak-lr and --validate-every; no --resume")
    output = arguments.output.resolve()
    if arguments.stage == "register":
        output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if arguments.stage == "train":
            status = read(output / "status.json")
            if status["state"] != "registered" and not arguments.resume:
                raise ValueError("Never overwrite a started training run without explicit --resume")
            if arguments.resume and status["state"] == "training_complete":
                raise ValueError("Training is already complete")
        try:
            storage_guard(output)
            if arguments.stage == "register":
                register(arguments.screen.resolve(), output, steps=arguments.steps, peak_lr=arguments.peak_lr,
                         validate_every=arguments.validate_every)
                return
            config = read(output / "config.json")
            verify_run(config, output)
            wait_for_gpu(output, False, False)
            if not arguments.resume:
                write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                                         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
                write_json(output / "status.json", {"state": "preflight", "pid": os.getpid(), "updated_unix": time.time()})
                preflight(config, output)
                gc.collect()
                torch.cuda.empty_cache()
            train(config, output, resume=arguments.resume)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                                "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
