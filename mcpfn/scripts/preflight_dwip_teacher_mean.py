"""Registered real-batch equivalence, serialization and resource preflight."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import torch

from dwip_teacher_compact_artifacts import compact_budget
from dwip_teacher_mean_training import (
    CONTROL, DESIGN, FrozenMeanTeachers, base, build_training_prior, compute_context,
    control_recipe, initialize_student, objective_terms, optimizer_step, read_json,
    sample_training_batch, verify_hashes,
)
from rowcol_fixed_validation import evaluate_fixed_validation, preserved_rng
from run_dwip_budgeted_queue_v1 import immutable_json, RESERVE_BYTES
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK
from spectral_support_normalization import apply_shrunk_support_mask
from tabimpute.model.model_new_stable import TabImputeModelStable
from train_dwip_anchor_mix import ShapeAwareMixedMatrixPrior


QUEUE = "dwip_teacher_mean_preflight_20260917"
SOURCES = (
    "scripts/preflight_dwip_teacher_mean.py", "scripts/dwip_teacher_mean_training.py",
    "scripts/dwip_teacher_compact_artifacts.py", "scripts/train_dwip_teacher_mean.py",
    "tests/test_dwip_teacher_mean_training.py", "tests/test_dwip_teacher_compact_artifacts.py",
    "tests/test_dwip_teacher_mean_registration.py",
)


def reference_batch(prior, args, step, device):
    with preserved_rng(args.data_seed + step):
        (source, target, _, _, _), _ = prior.get_batch()
    source, target = source.to(device=device, dtype=torch.float32), target.to(device=device, dtype=torch.float32)
    source, target, mask, stats = apply_shrunk_support_mask(
        source, target, args.support_mask_mode, seed=args.data_seed + 10000000 + step,
        variance_df=args.support_variance_df,
    )
    return source.float(), target, mask, {**stats, "sampled_prior_index": int(prior.last_prior_index),
                                         "sampled_prior_name": prior.last_prior_name}


def reference_terms(model, criterion, logits, target):
    output = base.prediction_distribution(model, criterion)
    nll = output(logits.float(), target.float()).mean()
    point = torch.nn.functional.huber_loss(output.mean(logits.float()).float(), target.float(), delta=1.)
    cdf = base.cdf_alignment_loss(logits, base.distribution_target(model, target), criterion, chunk_size=2048)
    zero = logits.new_zeros((), dtype=torch.float32)
    loss = 1. * nll + 2. * point + 0. * zero + .05 * cdf + 0. * zero + 0. * zero + 0. * zero + 0. * zero + 0. * zero
    return {"loss": loss, "nll": nll, "point": point, "cdf": cdf}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    cli = parser.parse_args()
    root = REPO / "runs/objective_queues" / QUEUE
    root.mkdir(exist_ok=True)
    if (root / "results.json").exists():
        raise ValueError("Preserve completed preflight evidence")
    args = control_recipe()
    control = root.parent / CONTROL
    proposal = read_json(DESIGN / "next_single_model_compression_proposal.json")
    control_manifest = read_json(control / "manifest.json")
    verify_hashes(proposal["input_sha256"])
    verify_hashes({REPO / path: expected for path, expected in control_manifest["source_sha256"].items()})
    work = WORK / f"tabimpute-{CONTROL}-dwip1e5"
    paths = [REPO / source for source in (*control_manifest["source_sha256"], *SOURCES)]
    paths += [DESIGN / name for name in ("next_single_model_compression_proposal.json", "cdf_retention_amendment.json")]
    paths += [control / "manifest.json", control / "dwip1e5_command.json", args.init_model_checkpoint,
              args.fixed_validation_cache, work / "metrics_stream.jsonl", work / "validation_stream.jsonl"]
    paths += [WORK / f"tabimpute-{member['queue']}-dwip1e5/best_validation_checkpoint.pth" for member in proposal["teachers"]]
    inputs = {str(Path(path).resolve()): sha256(path) for path in paths}
    manifest = {
        "input_sha256": inputs, "steps": list(range(1, 17)),
        "corner_shapes": [[2, 320, 100], [2, 50, 50], [2, 80, 35]],
        "corner_repeats": 3, "corner_seed": 17092026,
        "historical_first_loss": 7.528653621673584,
        "historical_first_forward_atol": 0.00001,
        "budget": compact_budget(work),
        "limits": "Feasibility only:16 new online training batches and three declared shape corners. One warmup plus three timed forward/backward/Adam updates per corner; discard temporary student updates without saving checkpoints. No validation target is used for training, no public benchmark or parameter-average evaluation. Existing files remain unchanged.",
        "readiness": "This core preflight cannot admit training alone. Full dispatcher/follow-up registration and completion-audit coverage must also be implemented and verified.",
    }
    immutable_json(root / "manifest.json", manifest)
    if cli.register_only:
        immutable_json(root / "registration_status.json", {"state": "registered", "evaluation_started": False,
            "utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha256(root / "manifest.json")})
        print(json.dumps({"registered": str(root), "input_count": len(inputs), "budget_gib": manifest["budget"]["required_free_bytes_at_launch"] / 1024**3}))
        return
    registered = read_json(root / "registration_status.json")
    if registered["manifest_sha256"] != sha256(root / "manifest.json") or gpu_processes().strip():
        raise ValueError("Require separately registered unchanged preflight and idle GPU")
    import shutil
    if shutil.disk_usage(root).free < RESERVE_BYTES + 32 * 1024**2:
        raise RuntimeError("Preserve20GiB reserve before the memory-only preflight")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model, optimizer, scheduler, criterion = initialize_student(args, device)
    teachers = FrozenMeanTeachers.load_registered(args, criterion, device)
    assert not optimizer.state and len(model.state_dict()) == 254
    prior = build_training_prior(args, device=device)
    old_prior_class = base.MixedMatrixPrior
    try:
        base.MixedMatrixPrior = ShapeAwareMixedMatrixPrior
        with preserved_rng(args.data_seed):
            reference_prior = base.build_prior(base.parse_phases(args)[0], device=str(device), prior_kind=args.prior_kind,
                generator_type=args.generator_type, missingness_type=args.missingness_type,
                base_config=base.build_base_prior_config(args))
    finally:
        base.MixedMatrixPrior = old_prior_class
    batches = []
    for step in manifest["steps"]:
        source, target, mask, stats = sample_training_batch(prior, args, step, device)
        expected_source, expected_target, expected_mask, expected_stats = reference_batch(reference_prior, args, step, device)
        torch.testing.assert_close(source, expected_source, rtol=0., atol=0., equal_nan=True)
        assert torch.equal(target, expected_target) and torch.equal(mask, expected_mask) and stats == expected_stats
        with compute_context(device):
            logits = model(source, decode_mask=mask)
        baseline = objective_terms(model, criterion, logits, target[mask])
        reference = reference_terms(model, criterion, logits, target[mask])
        for key in reference:
            assert torch.equal(baseline[key], reference[key]), (step, key)
        teacher_target = teachers.target(source, mask)
        candidate = objective_terms(model, criterion, logits, target[mask], teacher_target)
        assert torch.equal(candidate["nll"], baseline["nll"]) and torch.equal(candidate["cdf"], baseline["cdf"])
        assert not teacher_target.requires_grad
        if step == 1:
            assert abs(float(baseline["loss"]) - manifest["historical_first_loss"]) <= manifest["historical_first_forward_atol"]
        batches.append({"step": step, "shape": list(source.shape), **stats,
                        "baseline": {key: float(value.detach()) for key, value in reference.items()},
                        "teacher_point_loss": float(candidate["point"].detach()),
                        "content_sha256": case_digest([{"source": source.detach().cpu(), "target": target.cpu(), "mask": mask.cpu()}])})
        del logits, baseline, reference, candidate
    cases = torch.load(args.fixed_validation_cache, map_location="cpu", weights_only=True)["cases"]
    modes = [teacher.training for teacher in teachers.models]
    fixed = evaluate_fixed_validation(model, criterion, cases, device, args.precision_mode)
    with (work / "validation_stream.jsonl").open() as stream:
        original_fixed = json.loads(next(stream))
    assert original_fixed["step"] == 0
    assert abs(fixed["macro_rmse"] - original_fixed["macro_rmse"]) <= 1e-12
    assert model.training and modes == [teacher.training for teacher in teachers.models] == [False, False]
    timings = []
    for shape in manifest["corner_shapes"]:
        with preserved_rng(manifest["corner_seed"]):
            target = torch.randn(shape, device=device)
            mask = torch.rand(shape, device=device) < .4
            source = target.masked_fill(mask, float("nan"))
        seconds = []
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(manifest["corner_repeats"] + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            with compute_context(device):
                logits = model(source, decode_mask=mask)
            teacher_target = teachers.target(source, mask)
            terms = objective_terms(model, criterion, logits, target[mask], teacher_target)
            gradient_norm = optimizer_step(model, optimizer, scheduler, terms, args.grad_clip_norm)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            assert all(parameter.grad is None for teacher in teachers.models for parameter in teacher.parameters())
            if repeat:
                seconds.append(elapsed)
        timings.append({"shape": shape, "seconds": seconds, "max_seconds": max(seconds),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "last_gradient_norm": gradient_norm})
    states = list(model.state_dict().values())
    assert len(states) == 254 and all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in states)
    moments = [state for state in optimizer.state.values() if "exp_avg" in state]
    assert moments and all(state[key].dtype == torch.float32 and torch.isfinite(state[key]).all()
                           for state in moments for key in ("exp_avg", "exp_avg_sq"))
    sizes = {}
    for label, payload in (
            ("model", base.model_checkpoint(model, step=12, metrics={})),
            ("optimizer", base.training_checkpoint(model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False),
                step=12, smoothed_loss=0., best_smoothed_loss=0., metrics={}))):
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        sizes[label] = buffer.tell()
        ceiling = manifest["budget"]["model_file_ceiling_bytes" if label == "model" else "latest_file_ceiling_bytes"]
        assert sizes[label] <= ceiling
        if label == "model":
            buffer.seek(0)
            loaded = torch.load(buffer, map_location="cpu", weights_only=True)
            with preserved_rng():
                public_model = TabImputeModelStable(**json.loads(args.model_overrides_json))
            public_model.load_state_dict(loaded["model"], strict=True)
            assert tuple(public_model.state_dict()) == tuple(model.state_dict())
            del loaded, public_model
        buffer.close()
    projected_hours = max(item["max_seconds"] for item in timings) * 64000 / 3600
    assert projected_hours < 20., "Leave4h of the24h cap for validation and other training overhead"
    verify_hashes(inputs)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha256(root / "manifest.json"),
              "input_sha256": inputs, "real_batch_equivalence": batches, "fixed_original_rmse": fixed["macro_rmse"],
              "timings": timings, "conservative_max_shape_training_hours": projected_hours,
              "serialization_bytes": sizes, "finite_fp32_state_tensors": len(states),
              "finite_fp32_adam_moment_pairs": len(moments), "core_preflight_passed": True,
              "training_ready": False, "remaining_gate": manifest["readiness"], "checkpoint_files_created": False}
    immutable_json(root / "results.json", result)
    immutable_json(root / "status.json", {"state": "complete", "training_registered_or_launched": False})
    print(json.dumps({key: result[key] for key in ("core_preflight_passed", "fixed_original_rmse", "conservative_max_shape_training_hours", "serialization_bytes", "finite_fp32_adam_moment_pairs", "training_ready")}, indent=2))


if __name__ == "__main__":
    main()
