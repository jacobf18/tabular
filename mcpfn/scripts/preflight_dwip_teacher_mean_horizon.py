"""Registered memory-only verification of the unchanged128k mean-Huber recipe."""
import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import shutil
import time

import torch

from audit_dwip_continuous_completion import save_new
from dwip_teacher_mean_horizon import HORIZON_STEPS, PREFLIGHT, PREVIOUS_QUEUE, PROPOSAL, horizon_recipe
from dwip_teacher_mean_training import (
    DESIGN, FrozenMeanTeachers, base, build_training_prior, compute_context, control_recipe,
    initialize_student, objective_terms, optimizer_step, read_json, sample_training_batch, verify_hashes,
)
from preflight_dwip_teacher_mean import reference_batch, reference_terms
from rowcol_fixed_validation import evaluate_fixed_validation, preserved_rng
from run_dwip_budgeted_queue_v1 import immutable_json, RESERVE_BYTES
from run_dwip_continuous_spectrum import case_digest
from run_dwip_teacher_mean_horizon import SOURCES, test_paths
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK
from tabimpute.model.model_new_stable import TabImputeModelStable


NEW_SOURCES = (
    "scripts/preflight_dwip_teacher_mean_horizon.py", "scripts/dwip_teacher_mean_horizon.py",
    "scripts/train_dwip_teacher_mean_horizon.py", "scripts/dwip_distribution_compact_artifacts.py",
)


def verify_teacher_pair(proposed, previous):
    identity = lambda member: (member["queue"], member["sha256"])
    if len(proposed) != 2 or [identity(member) for member in proposed] != [identity(member) for member in previous] or any(
            member.get("weight") != .5 for member in previous):
        raise ValueError("The horizon cannot change the frozen equal-weight teacher pair")


def runtime_projection(max_step_seconds, validation_seconds, previous_followup_seconds):
    values = (max_step_seconds, validation_seconds, previous_followup_seconds)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Require positive finite measured timing inputs")
    training = max_step_seconds * HORIZON_STEPS
    validation = validation_seconds * 65 * 1.25
    saves = 65 * 15.
    followups = max(3600., 2 * previous_followup_seconds)
    setup = 1800.
    overhead = validation + saves + followups + setup
    if training >= 20 * 3600 or overhead >= 4 * 3600 or training + overhead >= 24 * 3600:
        raise ValueError("The registered128k training and full follow-ups do not fit24h")
    return {"training_seconds": training, "fixed_validation_seconds": validation,
            "checkpoint_seconds": saves, "followup_seconds": followups, "setup_and_audit_seconds": setup,
            "overhead_seconds": overhead, "total_seconds": training + overhead,
            "measured_fixed_validation_seconds": validation_seconds,
            "measured_previous_followup_seconds": previous_followup_seconds,
            "basis": "Maximum of all timed95%-missing corners times128k,65 measured fixed validations padded25%,15s per checkpoint event, twice observed full follow-ups with1h floor,30min setup/audit allowance. Require training<20h and overhead<4h."}


def registration():
    args = horizon_recipe()
    previous = REPO / "runs/objective_queues" / PREVIOUS_QUEUE
    work = WORK / f"tabimpute-{PREVIOUS_QUEUE}-dwip1e5"
    manifest = read_json(previous / "manifest.json")
    proposal = read_json(PROPOSAL)
    verify_teacher_pair(proposal["teachers"], read_json(DESIGN / "next_single_model_compression_proposal.json")["teachers"])
    verify_hashes(proposal["input_sha256"])
    verify_hashes(manifest["source_sha256"])
    for name in ("training_stage_audit.json", "teacher_full_comparison.json"):
        verify_hashes(read_json(previous / name)["input_sha256"])
    paths = [REPO / name for name in (*manifest["source_sha256"], *SOURCES, *NEW_SOURCES, *test_paths())]
    paths += [PROPOSAL, args.init_model_checkpoint, args.fixed_validation_cache,
              work / "metrics_stream.jsonl", work / "validation_stream.jsonl"]
    paths += [previous / name for name in ("manifest.json", "training_stage_audit.json", "teacher_full_comparison.json",
                                          "training_status.json", "pipeline_status.json", "completed_stage_audit.json")]
    paths += [WORK / f"tabimpute-{member['queue']}-dwip1e5/best_validation_checkpoint.pth" for member in proposal["teachers"]]
    return {"input_sha256": {str(Path(path).resolve()): sha256(path) for path in paths},
            "steps": list(range(1, 17)), "corner_shapes": [[2, 320, 100], [2, 50, 50], [2, 80, 35]],
            "corner_repeats": 3, "corner_seed": 18092026, "missing_fraction": .95,
            "historical_first_loss": 7.528653621673584, "historical_first_forward_atol": .00001,
            "horizon_steps": HORIZON_STEPS, "validation_records": 65,
            "limits": "Memory-only feasibility:16 matched online batches and12 discarded optimizer updates. No saved weights or validation training. No public benchmark access or selection. One later trial only after complete pipeline admission.",
            "readiness": "Core preflight alone does not admit training; require full tests, input/cache hashes, disk budget and pipeline registration."}


def run(manifest):
    verify_hashes(manifest["input_sha256"])
    registered = read_json(PREFLIGHT / "registration_status.json")
    if registered["manifest_sha256"] != sha256(PREFLIGHT / "manifest.json") or gpu_processes().strip():
        raise ValueError("Require unchanged separate registration and idle GPU")
    if shutil.disk_usage(PREFLIGHT).free < RESERVE_BYTES + 32 * 1024**2:
        raise RuntimeError("Preserve20GiB reserve before the memory-only preflight")
    save_new(PREFLIGHT / "started.json", {"utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
    torch.set_num_threads(4)
    device = torch.device("cuda")
    args, control = horizon_recipe(), control_recipe()
    model, optimizer, scheduler, criterion = initialize_student(args, device)
    teachers = FrozenMeanTeachers.load_registered(args, criterion, device)
    assert not optimizer.state and len(model.state_dict()) == 254
    prior, reference_prior = (build_training_prior(recipe, device=device) for recipe in (args, control))
    previous = REPO / "runs/objective_queues" / PREVIOUS_QUEUE
    work = WORK / f"tabimpute-{PREVIOUS_QUEUE}-dwip1e5"
    with (work / "metrics_stream.jsonl").open() as stream:
        original_first = json.loads(next(stream))
    batches = []
    for step in manifest["steps"]:
        source, target, mask, stats = sample_training_batch(prior, args, step, device)
        expected_source, expected_target, expected_mask, expected_stats = reference_batch(reference_prior, control, step, device)
        torch.testing.assert_close(source, expected_source, rtol=0., atol=0., equal_nan=True)
        assert torch.equal(target, expected_target) and torch.equal(mask, expected_mask) and stats == expected_stats
        with compute_context(device):
            logits = model(source, decode_mask=mask)
        baseline = objective_terms(model, criterion, logits, target[mask])
        reference = reference_terms(model, criterion, logits, target[mask])
        assert all(torch.equal(baseline[key], reference[key]) for key in reference)
        teacher_target = teachers.target(source, mask)
        candidate = objective_terms(model, criterion, logits, target[mask], teacher_target)
        output = base.prediction_distribution(model, criterion)
        expected_point = torch.nn.functional.huber_loss(output.mean(logits.float()).float(), teacher_target.float(), delta=1.)
        assert torch.equal(candidate["point"], expected_point)
        assert torch.equal(candidate["nll"], baseline["nll"]) and torch.equal(candidate["cdf"], baseline["cdf"])
        assert not teacher_target.requires_grad
        if step == 1:
            assert abs(float(baseline["loss"].detach()) - manifest["historical_first_loss"]) <= manifest["historical_first_forward_atol"]
            assert original_first["step"] == 1
            assert abs(float(candidate["loss"].detach()) - original_first["loss_missing"]) <= manifest["historical_first_forward_atol"]
            assert all(original_first[name] == value for name, value in stats.items())
        batches.append({"step": step, "shape": list(source.shape), **stats,
                       "baseline": {key: float(value.detach()) for key, value in reference.items()},
                       "teacher_point_loss": float(candidate["point"].detach()),
                       "content_sha256": case_digest([{"source": source.detach().cpu(), "target": target.cpu(), "mask": mask.cpu()}])})
        del logits, baseline, reference, candidate, expected_point
    cases = torch.load(args.fixed_validation_cache, map_location="cpu", weights_only=True)["cases"]
    torch.cuda.synchronize()
    started = time.monotonic()
    fixed = evaluate_fixed_validation(model, criterion, cases, device, args.precision_mode)
    torch.cuda.synchronize()
    validation_seconds = time.monotonic() - started
    with (work / "validation_stream.jsonl").open() as stream:
        original_fixed = json.loads(next(stream))
    assert original_fixed["step"] == 0 and len(cases) == 72
    assert abs(fixed["macro_rmse"] - original_fixed["macro_rmse"]) <= 1e-12
    assert model.training and all(not teacher.training for teacher in teachers.models)
    timings = []
    for shape in manifest["corner_shapes"]:
        with preserved_rng(manifest["corner_seed"]):
            target = torch.randn(shape, device=device)
            mask = torch.rand(shape, device=device) < manifest["missing_fraction"]
            source = target.masked_fill(mask, float("nan"))
        seconds = []
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(manifest["corner_repeats"] + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            with compute_context(device):
                logits = model(source, decode_mask=mask)
            teacher_target = teachers.target(source, mask)
            terms = objective_terms(model, criterion, logits, target[mask], teacher_target)
            gradient_norm = optimizer_step(model, optimizer, scheduler, terms, args.grad_clip_norm)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            assert math.isfinite(gradient_norm) and all(torch.isfinite(value) for value in terms.values())
            assert all(parameter.grad is None for teacher in teachers.models for parameter in teacher.parameters())
            if repeat:
                seconds.append(elapsed)
        timings.append({"shape": shape, "seconds": seconds, "max_seconds": max(seconds),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "last_gradient_norm": gradient_norm})
    states = list(model.state_dict().values())
    assert len(states) == 254 and all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in states)
    moments = list(optimizer.state.values())
    assert len(moments) == 238 and all(state[key].dtype == torch.float32 and torch.isfinite(state[key]).all()
                                    for state in moments for key in ("exp_avg", "exp_avg_sq"))
    sizes = {}
    for label, payload in (
            ("model", base.model_checkpoint(model, step=12, metrics={})),
            ("optimizer", base.training_checkpoint(model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False),
                step=12, smoothed_loss=0., best_smoothed_loss=0., metrics={}))):
        with io.BytesIO() as buffer:
            torch.save(payload, buffer)
            sizes[label] = buffer.tell()
            if label == "model":
                buffer.seek(0)
                loaded = torch.load(buffer, map_location="cpu", weights_only=True)
                with preserved_rng():
                    public_model = TabImputeModelStable(**json.loads(args.model_overrides_json))
                public_model.load_state_dict(loaded["model"], strict=True)
                assert tuple(public_model.state_dict()) == tuple(model.state_dict())
                del loaded, public_model
    previous_followups = read_json(previous / "pipeline_status.json")["updated_unix"] - read_json(previous / "training_status.json")["updated_unix"]
    projection = runtime_projection(max(item["max_seconds"] for item in timings), validation_seconds, previous_followups)
    verify_hashes(manifest["input_sha256"])
    result = {"utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha256(PREFLIGHT / "manifest.json"),
              "input_sha256": manifest["input_sha256"], "real_batch_equivalence": batches,
              "fixed_original_rmse": fixed["macro_rmse"], "fixed_case_digest": case_digest(cases),
              "timings": timings, "runtime_projection": projection, "serialization_bytes": sizes,
              "finite_fp32_state_tensors": len(states), "finite_fp32_adam_moment_pairs": len(moments),
              "core_preflight_passed": True, "training_ready": False, "remaining_gate": manifest["readiness"],
              "checkpoint_files_created": False}
    save_new(PREFLIGHT / "results.json", result)
    inputs = {**manifest["input_sha256"], **{str(PREFLIGHT / name): sha256(PREFLIGHT / name)
              for name in ("manifest.json", "registration_status.json", "started.json", "results.json")}}
    verify_hashes(inputs)
    save_new(PREFLIGHT / "completion_audit.json", {"state": "complete", "input_sha256": inputs,
             "training_ready": False, "goal_status": "active"})
    print(json.dumps({"core_preflight_passed": True, "runtime_projection": projection, "serialization_bytes": sizes}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    cli = parser.parse_args()
    PREFLIGHT.mkdir(exist_ok=True)
    if (PREFLIGHT / "started.json").exists() or (PREFLIGHT / "results.json").exists():
        raise ValueError("Preserve the existing preflight; no retry")
    if cli.register_only:
        immutable_json(PREFLIGHT / "manifest.json", registration())
        immutable_json(PREFLIGHT / "registration_status.json", {"manifest_sha256": sha256(PREFLIGHT / "manifest.json")})
        print(json.dumps({"registered": str(PREFLIGHT), "training_ready": False}))
    else:
        run(read_json(PREFLIGHT / "manifest.json"))


if __name__ == "__main__":
    main()
