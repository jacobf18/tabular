"""Registered128k teacher-mean horizon with unchanged loss and inference."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

import torch

from dwip_distribution_compact_artifacts import DistributionCheckpointer, CompactStorageGuard
from dwip_teacher_compact_artifacts import fixed_improves
from dwip_teacher_mean_horizon import HORIZON_STEPS, PREVIOUS_QUEUE, PROPOSAL, QUEUE, horizon_budget, horizon_recipe
from dwip_teacher_mean_training import (
    CONTROL, FrozenMeanTeachers, base, build_training_prior, compute_context,
    initialize_student, objective_terms, optimizer_step, read_json,
    sample_training_batch, update_diagnostics, verify_hashes,
)
from rowcol_fixed_validation import evaluate_fixed_validation, load_or_build_validation
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json


def validate_registration(registration):
    proposal = read_json(PROPOSAL)
    if registration["kind"] != "dwip_teacher_mean_horizon_v1" or registration["queue"] != QUEUE or QUEUE != proposal["proposed_queue"]:
        raise ValueError("Require the single proposed128k teacher-mean horizon")
    verify_hashes(registration["input_sha256"])
    preflight_path = Path(registration["preflight"])
    verify_hashes({preflight_path: registration["preflight_sha256"]})
    preflight = read_json(preflight_path)
    if preflight.get("training_ready") is not True:
        raise ValueError("Training requires every declared preflight gate, not partial implementation evidence")
    gates = ("core_preflight_passed", "reference_panels_verified", "full_pipeline_and_completion_audit_registered")
    tests = preflight.get("integration_tests", {})
    if any(preflight.get(name) is not True for name in gates) or tests.get("tests", 0) < 500 or any(
            tests.get(name) != 0 for name in ("errors", "failures", "skipped")):
        raise ValueError("Every integration, reference-panel and full-pipeline gate must pass")
    required = {str((REPO / source).resolve()) for source in (
        "scripts/train_dwip_teacher_mean_horizon.py", "scripts/dwip_teacher_mean_training.py",
        "scripts/dwip_distribution_compact_artifacts.py", "scripts/dwip_teacher_mean_horizon.py",
    )}
    required.update(preflight["input_sha256"])
    if not required.issubset(registration["input_sha256"]):
        raise ValueError("The exact trainer and helper sources must be registered")
    for path in required:
        if preflight["input_sha256"].get(path) != registration["input_sha256"][path]:
            raise ValueError("Trainer changed after its successful preflight")
    if registration["budget"] != horizon_budget():
        raise ValueError("Horizon checkpoint budget does not reproduce from completed evidence")
    return proposal


def sampled_update_metrics(model, samples):
    totals = {}
    for name, parameter in model.named_parameters():
        if name not in samples:
            continue
        after = parameter.detach().view(-1)[::max(1, parameter.numel() // 4096)][:4096].float()
        before = samples[name]
        parts = name.split(".")
        group = ".".join(parts[:2]) if len(parts) > 1 and parts[1].isdigit() else parts[0]
        values = totals.setdefault(group, [0., 0.])
        values[0] += float((after - before).square().sum())
        values[1] += float(before.square().sum())
    return {f"sampled_update_ratio/{name}": math.sqrt(change / max(norm, 1e-30))
            for name, (change, norm) in totals.items()}


def train(registration):
    proposal = validate_registration(registration)
    root = REPO / "runs/objective_queues" / registration["queue"]
    work = WORK / f"tabimpute-{registration['queue']}-dwip1e5"
    if not root.is_dir() or work.exists():
        raise ValueError("Require registered queue and an unused work directory; never resume or overwrite")
    if gpu_processes().strip():
        raise RuntimeError("GPU is occupied; preserve other jobs")
    work.mkdir()
    guard = CompactStorageGuard(root, work, registration["budget"])
    guard(launch=True)
    started = time.monotonic()
    args = horizon_recipe()
    args.run_name, args.work_dir = work.name, work
    phases = base.parse_phases(args)
    if len(phases) != 1 or phases[0].end_step != HORIZON_STEPS or args.warmup_steps + args.stable_steps + args.decay_steps != HORIZON_STEPS:
        raise ValueError("Require the exact128k one-phase cosine trajectory")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model, optimizer, scheduler, criterion = initialize_student(args, device)
    teachers = FrozenMeanTeachers.load_registered(args, criterion, device)
    initial_keys = tuple(model.state_dict())
    expected_count = 254
    if len(initial_keys) != expected_count or optimizer.state:
        raise ValueError("Require original254-tensor model and fresh optimizer state")
    checkpointer = DistributionCheckpointer(work, registration["budget"], guard)
    config = {"run_name": work.name, "model_family": "new_stable", "model": json.loads(args.model_overrides_json),
              "prior": base.build_base_prior_config(args), "precision_mode": args.precision_mode,
              "control": PREVIOUS_QUEUE, "control_command": str(REPO / "runs/objective_queues" / CONTROL / "dwip1e5_command.json"),
              "total_steps": HORIZON_STEPS, "phases": json.loads(args.phases_json),
              "warmup_steps": args.warmup_steps, "stable_steps": args.stable_steps, "decay_steps": args.decay_steps,
              "terminal_checkpoint": "latest_checkpoint.pth",
              "teacher_checkpoints": proposal["teachers"], "teacher_serialized_in_student": False,
              "point_target_kind": "detached_equal_frozen_teacher_mean", "nll_loss_weight": 1.,
              "point_loss_weight": 2., "huber_delta": 1., "cdf_loss_weight": .05,
              "true_target_metrics": ["nll_missing", "cdf_loss_missing", "rmse_missing_mean"],
              "teacher_target_metrics": ["teacher_point_loss_missing", "teacher_mean_rmse_missing"],
              "fixed_validation": {"cache": str(args.fixed_validation_cache), "every": 2000,
                                   "selection_metric": "equal_group_mean_rmse_excluding_monitor_groups"},
              "registration": registration}
    write_json(work / "config.json", config)
    cases = load_or_build_validation(args.fixed_validation_cache, base.build_base_prior_config(args),
                                     seed=args.validation_seed, cases_per_group=args.validation_cases_per_group)
    if len(cases) != 72 or case_digest(cases) != read_json(root / "manifest.json")["fixed_case_digest"]:
        raise ValueError("The original fixed selection population changed")
    best_score = float("inf")

    def validate(step):
        nonlocal best_score
        guard()
        metrics = evaluate_fixed_validation(model, criterion, cases, device, args.precision_mode,
                                             prediction_path=work / "validation_predictions_latest.pth")
        metrics["step"] = step
        with (work / "validation_stream.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        if fixed_improves(metrics, best_score):
            best_score = metrics["macro_rmse"]
            checkpointer.save(base.model_checkpoint(model, step=step, metrics=metrics), "best_validation_checkpoint.pth")
            write_json(work / "best_validation_metrics.json", metrics)
            shutil.copyfile(work / "validation_predictions_latest.pth", work / "validation_predictions_best.pth")
        if not model.training or any(teacher.training for teacher in teachers.models):
            raise ValueError("Validation changed student/teacher training modes")
        guard()
        print(json.dumps({"validation_step": step, "macro_rmse": metrics["macro_rmse"]}), flush=True)

    smoothed_loss, best_smoothed_loss = None, float("inf")
    last_metrics = {}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    try:
        validate(0)
        prior = build_training_prior(args, device=device)
        torch.cuda.empty_cache()
        next_guard = time.monotonic()
        for step in range(1, HORIZON_STEPS + 1):
            if time.monotonic() - started >= 24 * 3600:
                raise TimeoutError("The registered24h training cap was reached")
            if time.monotonic() >= next_guard:
                guard()
                next_guard = time.monotonic() + 15
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            source, target, mask, support = sample_training_batch(prior, args, step, device)
            with compute_context(device):
                logits = model(source, decode_mask=mask)
            teacher_target = teachers.target(source, mask)
            terms = objective_terms(model, criterion, logits, target[mask], teacher_target)
            diagnostic, samples = update_diagnostics(model, logits, terms, step, args.update_diagnostics_every)
            gradient_norm = optimizer_step(model, optimizer, scheduler, terms, args.grad_clip_norm)
            updates = sampled_update_metrics(model, samples)
            loss_value = float(terms["loss"].detach())
            smoothed_loss = loss_value if smoothed_loss is None else .995 * smoothed_loss + .005 * loss_value
            best_smoothed_loss = min(best_smoothed_loss, smoothed_loss)
            last_metrics = {**support, **diagnostic, **updates, "step": step,
                            "matrix_rows": source.shape[1], "matrix_cols": source.shape[2], "batch_size": source.shape[0],
                            "loss_missing": loss_value, "nll_missing": float(terms["nll"].detach()),
                            "teacher_point_loss_missing": float(terms["point"].detach()),
                            "cdf_loss_missing": float(terms["cdf"].detach()),
                            "rmse_missing_mean": float(terms["true_rmse"]),
                            "teacher_mean_rmse_missing": float(terms["teacher_rmse"]),
                            "loss_missing_ema": smoothed_loss, "grad_norm": gradient_norm,
                            "lr": float(optimizer.param_groups[0]["lr"]), "time_per_step": time.monotonic() - step_started}
            if step == 1 or step % args.metrics_jsonl_every == 0:
                with (work / "metrics_stream.jsonl").open("a") as stream:
                    stream.write(json.dumps(last_metrics, allow_nan=False) + "\n")
                print(json.dumps(last_metrics, allow_nan=False), flush=True)
            if step % 2000 == 0:
                validate(step)
                if tuple(model.state_dict()) != initial_keys:
                    raise ValueError("Student state keys changed; teachers must stay external")
                checkpointer.save(base.training_checkpoint(model, optimizer, scheduler, scaler, step=step,
                    smoothed_loss=smoothed_loss, best_smoothed_loss=best_smoothed_loss, metrics=last_metrics), "latest_checkpoint.pth")
                write_json(root / "training_status.json", {"state": "training", "step": step,
                           "best_fixed_macro_rmse": best_score, "updated_unix": time.time()})
        write_json(work / "metrics.json", last_metrics)
        verify_hashes(registration["input_sha256"])
        guard()
        write_json(root / "training_status.json", {"state": "complete", "step": HORIZON_STEPS,
                   "best_fixed_macro_rmse": best_score, "elapsed_seconds": time.monotonic() - started,
                   "updated_unix": time.time()})
    except BaseException as error:
        write_json(root / "training_status.json", {"state": "error", "error": repr(error),
                   "step": last_metrics.get("step", 0), "updated_unix": time.time()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    lock_path = REPO / "runs/objective_queues/.dwip_teacher_mean_horizon_training.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(read_json(args.registration))


if __name__ == "__main__":
    main()
