"""Trainability smoke test of structured episodes, not a promoted research candidate."""

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import time

import torch

import diagnose_single_model_missingness_coverage as coverage
from dwip_teacher_mean_training import compute_context, initialize_student
from recsys_evaluation_common import atomic_npz, sha256, write_json
from single_model_complete_missingness_prior import CompleteMissingnessSampler, COMPONENTS, NEURAL_FAMILIES
from single_model_distillation import MixedSampler
from single_model_query_objective import backward_episode
from streamed_large_matrix_training import STREAM_OPTIONS, atomic_torch_save
from tabimpute.streamed_inference import streamed_point_predictions
from train_single_matrixformer import SPECIALIST, INPUTS
from train_streamed_large_matrices import optimizer_update, wait_for_gpu


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")


def cases():
    specifications = []
    for index, family in enumerate(coverage.FAMILIES):
        specifications.append({"domain": "imputation", "family": family, "component": COMPONENTS[index % 4],
                               "shape": [24, 12] if family in NEURAL_FAMILIES else [240, 80],
                               "regime": "legacy", "requested_missingness": .6, "seed": 67100601 + index * 100})
    for family in ("mcar", "mnar_censoring", "mnar_polarization", "mnar_soft_polarization"):
        for regime in ("legacy", "area_scaled"):
            index = len(specifications)
            specifications.append({"domain": "imputation", "family": family, "component": COMPONENTS[index % 3],
                                   "shape": [1024, 256], "regime": regime, "requested_missingness": .6,
                                   "seed": 67100601 + index * 100})
    specifications.append({"domain": "recsys", "shape": [1000, 1000], "seed": 67102601})
    return specifications


def register(output):
    if output.exists():
        raise FileExistsError("Never replace a training smoke test")
    prior = ROOT / "complete_missingness_prior_preflight_v1"
    diagnostic = ROOT / "missingness_models_probe_v1_recovery1"
    summary = json.loads((prior / "summary.json").read_text())
    recovered = json.loads((diagnostic / "recovery_audit.json").read_text())
    if not summary["complete"] or summary["usable_training_cases"] != 192 or not recovered["complete"]:
        raise ValueError("Require completed full prior preflight and the preserved model diagnostic")
    checkpoint = ROOT / "query_budget_v8/best_validation_checkpoint.pth"
    confirmation = ROOT / "query_budget_v8_confirmation_defined1/gate.json"
    gate = json.loads(confirmation.read_text())
    if not gate["passed"] or gate["selected_step"] != 250:
        raise ValueError("Require the original actual synthetically selected V8 checkpoint")
    hashes = {str(checkpoint): gate["checkpoint_sha256"], str(SPECIALIST): INPUTS[str(SPECIALIST)],
              str(confirmation): sha256(confirmation), str(prior / "summary.json"): sha256(prior / "summary.json"),
              str(diagnostic / "recovery_audit.json"): sha256(diagnostic / "recovery_audit.json")}
    for directory in (prior, diagnostic):
        config = json.loads((directory / "config.json").read_text())
        coverage.verify_hashes(config["source_sha256"])
        hashes[str(directory / "config.json")] = sha256(directory / "config.json")
    coverage.verify_hashes(hashes)
    config = {"registered_unix": time.time(), "checkpoint": str(checkpoint), "cases": cases(),
              "input_sha256": hashes, "source_sha256": coverage.source_hashes(), "seed": 66100601,
              "query_seed_offset": 68100601, "learning_rate": 1e-6, "weight_decay": .02, "grad_clip": .5,
              "memory_fraction": .8, "episode_gradient_weight": .5,
              "teacher": "frozen V8 initialization for imputation; unchanged specialist for the original rating episode",
              "objective": "unchanged V8 query-budget column-risk plus balanced Huber/NLL/CDF; unchanged rating loss",
              "purpose": "21 sequential isolated smoke-test updates; finite gradients and memory feasibility only",
              "not_a_candidate": True, "checkpoint_selection_performed": False, "real_benchmark_evaluation_authorized": False,
              "inference": "one unchanged original network state, no extra head, teacher or inference ensemble",
              "output_checkpoint": "smoke_test_checkpoint.pth"}
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"state": "registered", "config_sha256": sha256(output / "config.json")})


def finite_training_state(student, *, gradients):
    values = [(name, parameter.grad if gradients else parameter) for name, parameter in student.named_parameters()]
    actual = [(name, value) for name, value in values if value is not None]
    if not actual or any(not torch.isfinite(value).all() for _, value in actual):
        raise FloatingPointError("Nonfinite or entirely absent training state")
    return len(actual)


def run(output):
    config = json.loads((output / "config.json").read_text())
    status = json.loads((output / "status.json").read_text())
    if (status["state"] != "registered" or status["config_sha256"] != sha256(output / "config.json")
            or config["cases"] != cases()):
        raise ValueError("Require unchanged untouched registration")
    coverage.verify_hashes(config["input_sha256"])
    coverage.verify_hashes(config["source_sha256"])
    wait_for_gpu(output, True, False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
    args = coverage.control_recipe()
    args.init_model_checkpoint = config["checkpoint"]
    args.seed, args.lr, args.weight_decay = config["seed"], config["learning_rate"], config["weight_decay"]
    student, optimizer, _, criterion = initialize_student(args, torch.device("cuda"))
    teacher = deepcopy(student).eval().requires_grad_(False)
    sampler = CompleteMissingnessSampler()
    rating_sampler = MixedSampler({"data_seed": 67100601, "phases": [{"name": "smoke", "end_step": 1,
        "rows_low": 256, "rows_high": 2000, "cols_low": 128, "cols_high": 1000, "batch_size": 1}]})
    initial = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
    rows, artifacts = [], {}
    directory = output / "episodes"
    directory.mkdir()
    for index, specification in enumerate(config["cases"]):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        write_json(output / "status.json", {"state": "training_smoke_test", "step": index + 1,
                                            "pid": os.getpid(), "updated_unix": time.time()})
        if specification["domain"] == "imputation":
            episode = sampler.sample(device="cuda", **{key: value for key, value in specification.items() if key != "domain"})
        else:
            episode = rating_sampler.sample("recsys", 1, shape=tuple(specification["shape"]), seed=specification["seed"])
            teacher.load_state_dict(torch.load(SPECIALIST, map_location="cpu", weights_only=True)["model"], strict=True)
        with torch.no_grad(), compute_context(episode.source.device):
            target = streamed_point_predictions(teacher, episode.source, criterion, **STREAM_OPTIONS)
        path = directory / f"case_{index:02d}.npz"
        atomic_npz(path, source=episode.source.cpu().numpy(), target=episode.target.cpu().numpy(),
                   mask=episode.mask.cpu().numpy(), teacher=target.cpu().numpy())
        artifacts[str(path)] = sha256(path)
        loss = backward_episode(student, criterion, episode, target, config["episode_gradient_weight"],
                                 query_seed_offset=config["query_seed_offset"])
        gradient_tensors = finite_training_state(student, gradients=True)
        norm = optimizer_update(student, optimizer, config["grad_clip"])
        finite_training_state(student, gradients=False)
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError("Nonfinite loss or gradient norm")
        row = {"step": index + 1, "specification": specification, "episode": episode.metadata,
               "loss": loss, "grad_norm": norm, "gradient_tensors": gradient_tensors,
               "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "seconds": time.monotonic() - started}
        rows.append(row)
        write_json(output / "steps.json", rows)
        print(json.dumps(row, allow_nan=False), flush=True)
        del episode, target
    changed = sum(not torch.equal(value.detach().cpu(), initial[name]) for name, value in student.state_dict().items())
    if not changed or set(student.state_dict()) != set(initial):
        raise ValueError("Require actual updates in exactly the original single network")
    checkpoint = output / config["output_checkpoint"]
    atomic_torch_save({"model": student.state_dict(), "step": len(rows), "not_a_candidate": True,
                       "config_sha256": sha256(output / "config.json")}, checkpoint)
    coverage.verify_hashes(config["input_sha256"])
    coverage.verify_hashes(config["source_sha256"])
    coverage.verify_hashes(artifacts)
    write_json(output / "summary.json", {"complete": True, "updates": len(rows), "all_gradients_finite": True,
        "changed_tensors": changed, "state_tensors": len(initial), "maximum_peak_gib": max(row["peak_gib"] for row in rows),
        "maximum_grad_norm": max(row["grad_norm"] for row in rows), "clipped_updates": sum(row["grad_norm"] > .5 for row in rows),
        "not_a_candidate": True, "checkpoint_selection_performed": False, "accuracy_improvement_not_assessed": True,
        "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(output / "config.json"), "episode_sha256": artifacts})
    write_json(output / "status.json", {"state": "complete", "updates": len(rows), "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if json.loads((output / "status.json").read_text())["state"] != "registered":
            raise ValueError("Never overwrite a started smoke test")
        try:
            run(output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
