"""One paired1024-step in-memory Adam pilot, never a checkpoint selector."""
import argparse
from collections import Counter
from copy import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from dwip_teacher_distribution_training import FrozenDistributionTeachers, objective_terms
from dwip_teacher_mean_training import (
    build_training_prior, compute_context, control_recipe, initialize_student,
    objective_terms as mean_objective, optimizer_step, read_json, sample_training_batch, verify_hashes,
)
from preflight_dwip_teacher_distribution import batch_digest, guard
from rowcol_fixed_validation import preserved_rng
from run_dwip_budgeted_queue_v1 import immutable_json
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, write_json


EXPERIMENT = "dwip_teacher_distribution_adam_pilot_20260918"
PREFLIGHT = "dwip_teacher_distribution_preflight_20260918"
ARMS = ("mean_huber_control", "distribution_kl")
EVALUATION_STEPS = (0, 256, 600, 1024)
METRICS = ("true_mse", "teacher_mse", "true_nll", "teacher_kl", "true_cdf")


def summarize_cases(records):
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError("Require nonempty unique generic reference cases")
    if any(not np.isfinite(record[name]) for record in records for name in METRICS):
        raise ValueError("Require finite reference metrics")
    components = sorted({record["component"] for record in records})
    return {"equal_batch_mean": {name: float(np.mean([record[name] for record in records])) for name in METRICS},
            "components": {component: {name: float(np.mean([record[name] for record in records
                                             if record["component"] == component])) for name in METRICS}
                           for component in components}, "cases": records}


def pilot_decision(arms):
    if set(arms) != set(ARMS):
        raise ValueError("Require both complete preregistered pilot arms")
    for arm in ARMS:
        if [record["step"] for record in arms[arm]] != list(EVALUATION_STEPS):
            raise ValueError("Require the full fixed evaluation schedule, not the best pilot step")
    control, candidate = (arms[arm][-1]["equal_batch_mean"] for arm in ARMS)
    if any(not np.isfinite(record[name]) for record in (control, candidate) for name in METRICS):
        raise ValueError("Incomplete final pilot metrics")
    teacher_improves = candidate["teacher_mse"] < control["teacher_mse"]
    truth_improves = candidate["true_mse"] < control["true_mse"]
    return {"teacher_mse_improves_at1024": teacher_improves, "true_mse_improves_at1024": truth_improves,
            "supports_full_trial_admission_review": teacher_improves and truth_improves,
            "training_ready": False, "checkpoint_created_or_selected": False,
            "rule": "Both final1024-step equal-batch teacher-mean MSE and true-target MSE must strictly beat the matched mean-Huber control before recommending full-trial admission review. No intermediate-step selection, seed retry, coefficient search or automatic training launch."}


def register(root):
    prior_root = root.parent / PREFLIGHT
    prior = read_json(prior_root / "manifest.json")
    preflight = read_json(prior_root / "results.json")
    if preflight["core_preflight_passed"] is not True or preflight["training_ready"] is not False:
        raise ValueError("Require a successful core preflight, not existing training admission")
    verify_hashes(prior["input_sha256"])
    paths = [prior_root / "manifest.json", prior_root / "results.json", Path(__file__).resolve(),
             REPO / "tests/test_dwip_teacher_distribution_pilot.py"]
    inputs = {**prior["input_sha256"], **{str(path): sha256(path) for path in paths}}
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "input_sha256": inputs,
        "arms": list(ARMS), "training_steps": 1024, "evaluation_steps": list(EVALUATION_STEPS),
        "reference_data_seed": 2026091803, "reference_steps": list(range(1, 33)),
        "protocol": "Each arm starts at the original17k DWIP checkpoint with fresh FP32 AdamW and the unchanged64k cosine schedule (600 warmup,63400 decay). Execute precisely the same first1024 native training-seed batches. Use BF16 computation and the same external frozen teacher pair. Both arms compute the same probability target for paired arithmetic; control uses its existing equal predictive mean for Huber. Additional control target computation is not a runtime comparison.",
        "evaluation": "Thirty-two fresh generic online reference batches generated once from a disjoint preregistered seed, kept only in memory, never differentiated through or trained on. Evaluate both arms at0,256,600,1024 with equal batch weighting and report all component/case scores. No fixed selector, cached independent, fresh replication or public benchmark data enters the pilot.",
        "decision_rule": "At the final1024 steps only, require strictly lower teacher-mean MSE and true-target MSE than control to recommend full-trial admission review. Finite NLL/KL/CDF required and reported, not optimized by post-hoc choices. No pilot weights are retained or selected. A passing pilot is not evidence of reaching the public target.",
        "limits": "Memory-only learning pilot, no saved checkpoint or model promotion, no coefficient/pair/seed/horizon grid, no automatic full training launch. All existing inputs remain unchanged;20GiB reserve and32MiB owned-output limit. External process cap2h.",
        "storage": guard(root)}
    immutable_json(root / "manifest.json", manifest)
    immutable_json(root / "registration_status.json", {"state": "registered", "gpu_started": False,
                  "manifest_sha256": sha256(root / "manifest.json")})
    return {"state": "registered", "inputs": len(inputs), "steps_per_arm": 1024}


@torch.no_grad()
def evaluate_references(model, criterion, teachers, cases, device):
    was_training = model.training
    records = []
    with preserved_rng():
        model.eval()
        try:
            for case in cases:
                source, target, mask = (case[name].to(device) for name in ("source", "target", "mask"))
                output = teachers.distribution_target(source, mask)
                with compute_context(device):
                    logits = model(source, decode_mask=mask)
                terms = objective_terms(model, criterion, logits, target[mask], output["probabilities"])
                mean = criterion.mean(logits.float())
                records.append({"id": case["id"], "component": case["stats"]["sampled_prior_name"],
                    "true_mse": float((mean - target[mask]).double().square().mean()),
                    "teacher_mse": float((mean - output["mean"]).double().square().mean()),
                    "true_nll": float(terms["nll"]), "teacher_kl": float(terms["kl"]),
                    "true_cdf": float(terms["cdf"])})
        finally:
            model.train(was_training)
    return summarize_cases(records)


def evaluate(root):
    manifest = read_json(root / "manifest.json")
    if sha256(root / "manifest.json") != read_json(root / "registration_status.json")["manifest_sha256"]:
        raise ValueError("Pilot registration changed")
    verify_hashes(manifest["input_sha256"])
    if gpu_processes():
        raise RuntimeError("GPU is busy; preserve existing jobs")
    if any(root.glob("*_evaluation_*.json")):
        raise ValueError("Preserve partial pilot evidence; audit before any resumption")
    device = torch.device("cuda")
    args = control_recipe()
    model, optimizer, scheduler, criterion = initialize_student(args, device)
    teachers = FrozenDistributionTeachers.load_registered(args, criterion, device)
    configured = copy(args)
    configured.data_seed = manifest["reference_data_seed"]
    prior = build_training_prior(configured, device=device)
    cases = []
    for step in manifest["reference_steps"]:
        source, target, mask, stats = sample_training_batch(prior, configured, step, device)
        cases.append({"id": f"online_reference_{step:02d}", "source": source.cpu(), "target": target.cpu(),
                      "mask": mask.cpu(), "stats": stats, "digest": batch_digest(source, target, mask, stats)})
    immutable_json(root / "reference_cases.json", {
        "cases": [{"id": case["id"], "digest": case["digest"], "shape": list(case["source"].shape), "stats": case["stats"]} for case in cases],
        "component_counts": dict(Counter(case["stats"]["sampled_prior_name"] for case in cases))})
    del model, optimizer, scheduler
    results, batches, health = {}, {}, {}
    for arm in ARMS:
        model, optimizer, scheduler, criterion = initialize_student(args, device)
        prior = build_training_prior(args, device=device)
        results[arm], batches[arm] = [], []
        initial = evaluate_references(model, criterion, teachers, cases, device)
        initial["step"] = 0
        results[arm].append(initial)
        immutable_json(root / f"{arm}_evaluation_0000.json", initial)
        arm_started = time.monotonic()
        for step in range(1, manifest["training_steps"] + 1):
            if step == 1 or step % 64 == 0:
                guard(root)
                write_json(root / "status.json", {"state": "in_memory_pilot", "arm": arm, "step": step})
            optimizer.zero_grad(set_to_none=True)
            source, target, mask, stats = sample_training_batch(prior, args, step, device)
            digest = batch_digest(source, target, mask, stats)
            if arm == ARMS[1] and digest != batches[ARMS[0]][step - 1]["digest"]:
                raise ValueError("Paired pilot training batches differ")
            output = teachers.distribution_target(source, mask)
            with compute_context(device):
                logits = model(source, decode_mask=mask)
            terms = mean_objective(model, criterion, logits, target[mask], output["mean"]) if arm == ARMS[0] else objective_terms(model, criterion, logits, target[mask], output["probabilities"])
            gradient_norm = optimizer_step(model, optimizer, scheduler, terms, args.grad_clip_norm)
            batches[arm].append({"step": step, "digest": digest, "shape": list(source.shape), "stats": stats,
                                 "loss": float(terms["loss"].detach()), "gradient_norm": gradient_norm,
                                 "lr": optimizer.param_groups[0]["lr"]})
            del logits, terms, output
            if step in manifest["evaluation_steps"]:
                measured = evaluate_references(model, criterion, teachers, cases, device)
                measured["step"] = step
                results[arm].append(measured)
                immutable_json(root / f"{arm}_evaluation_{step:04d}.json", measured)
                print(json.dumps({"arm": arm, "step": step, **measured["equal_batch_mean"]}), flush=True)
        if not all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Pilot model state is not finite FP32")
        moments = [state for state in optimizer.state.values() if "exp_avg" in state]
        if len(moments) != 238 or any(float(state["step"]) != 1024 or any(
                state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all()
                for key in ("exp_avg", "exp_avg_sq")) for state in moments):
            raise ValueError("Pilot Adam state is not the expected finite FP32 state")
        if any(parameter.grad is not None for teacher in teachers.models for parameter in teacher.parameters()):
            raise ValueError("Pilot updated teacher parameters")
        health[arm] = {"finite_fp32_model_tensors": len(model.state_dict()), "finite_fp32_adam_moment_pairs": len(moments),
                       "elapsed_seconds": time.monotonic() - arm_started, "scheduler_last_epoch": scheduler.last_epoch}
        immutable_json(root / f"{arm}_training_records.json", batches[arm])
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()
    if results[ARMS[0]][0] != results[ARMS[1]][0]:
        raise ValueError("Pilot arms do not reproduce the same original reference scores")
    if [row["lr"] for row in batches[ARMS[0]]] != [row["lr"] for row in batches[ARMS[1]]]:
        raise ValueError("Pilot cosine learning-rate schedules differ")
    verify_hashes(manifest["input_sha256"])
    result = {"utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha256(root / "manifest.json"),
              "arms": results, "health": health, "decision": pilot_decision(results), "storage": guard(root),
              "training_ready": False, "checkpoint_files_created": False}
    immutable_json(root / "results.json", result)
    write_json(root / "status.json", {"state": "complete", "utc": result["utc"], "training_ready": False})
    return {"state": "complete", "decision": result["decision"], "health": health}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = REPO / "runs/objective_queues" / EXPERIMENT
    root.mkdir(exist_ok=True)
    with (root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "results.json").exists():
            raise ValueError("Preserve completed pilot evidence")
        guard(root)
        print(json.dumps(register(root) if args.register_only else evaluate(root), indent=2))


if __name__ == "__main__":
    main()
