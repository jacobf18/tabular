"""Register and train the complete structured-prior continuation from selected V8."""

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch

import diagnose_single_model_missingness_coverage as coverage
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_structured_mixture as mixture
from single_model_complete_missingness_prior import SPECIFICATION as PRIOR_SPECIFICATION
from train_single_matrixformer_column_aligned import repeated_bank
from train_single_matrixformer import GENERAL, SPECIALIST, storage_guard
from train_streamed_large_matrices import optimizer_update, wait_for_gpu
import train_single_matrixformer_query_budget as previous


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
backend = mixture.backend


def read(path):
    return json.loads(path.read_text())


def validate_recipe(config):
    expected = {"version": 9, "steps": 400, "validate_every": 50, "peak_lr": 1e-5, "warmup_steps": 20,
                "weight_decay": .02, "grad_clip": .5, "memory_fraction": .8, **mixture.SEEDS}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Require the prospectively registered structured-mixture recipe")
    if (config["structured_policy"] != mixture.POLICY or config["structured_prior"] != PRIOR_SPECIFICATION
            or config["training_structured_cases"] != mixture.training_cases()
            or config["query_budget_objective"] != previous.SPECIFICATION
            or config["spectral_amplitude_mixture"] != previous.area.SPECIFICATION
            or config["probability_loss_scales"] != previous.balanced.PROBABILITY_SCALES
            or config["column_objective"] != mixture.column.COLUMN_OBJECTIVE
            or config["training_teacher_policy"] != {"imputation": "initialization", "recsys": "specialist"}
            or config["rating_training_shapes"] != [[1000, 1000], [2000, 1000]]
            or config["teachers"] != [str(GENERAL), str(SPECIALIST)]
            or config["parent_checkpoint"] != str(ROOT / "query_budget_v8/best_validation_checkpoint.pth")
            or config["phases"] != [{"name": "structured_mixture", "end_step": 400, "rows_low": 256,
                 "rows_high": 2000, "cols_low": 128, "cols_high": 1000, "batch_size": 1}]):
        raise ValueError("The original objectives/ratings or new coverage specification changed")
    seeds = []
    for name, confirmation in (("validation_cases", False), ("confirmation_cases", True)):
        legacy_seed = config["confirmation_seed" if confirmation else "validation_seed"]
        structured_seed = config["structured_confirmation_seed" if confirmation else "structured_validation_seed"]
        cases = repeated_bank(legacy_seed, confirmation=confirmation) + mixture.structured_bank(structured_seed, confirmation=confirmation)
        if json.dumps(config[name], sort_keys=True) != json.dumps(cases, sort_keys=True):
            raise ValueError("Keep every original and structured fresh-bank case")
        seeds.extend(case["structured"]["seed"] if "structured" in case else case["seed"] for case in cases)
    train_seeds = {config["data_seed"] + 16 * step + offset for step in range(1, 401) for offset in (0, 1)}
    if len(seeds) != len(set(seeds)) or train_seeds.intersection(seeds):
        raise ValueError("Training, selection and independent confirmation seeds must be disjoint")


def register(output):
    if output.exists():
        raise FileExistsError("Never overwrite a registered training experiment")
    producer = ROOT / "query_budget_v8"
    base = read(producer / "config.json")
    backend.verify_run(base, producer)
    previous.validate_recipe(base)
    gate_path = ROOT / "query_budget_v8_confirmation_defined1/gate.json"
    gate = read(gate_path)
    checkpoint = producer / "best_validation_checkpoint.pth"
    if (read(producer / "status.json")["state"] != "training_complete" or not gate["passed"]
            or gate["selected_step"] != 250 or sha256(checkpoint) != gate["checkpoint_sha256"]):
        raise ValueError("Initialize only from the actual synthetic-selected V8, never the smoke checkpoint")
    after = ROOT / "query_budget_v8_benchmarks_defined1"
    if read(after / "status.json")["state"] != "complete" or not read(after / "campaign_audit.json")["all_evaluations_complete"]:
        raise ValueError("Finish the original full real campaign before starting another experiment")
    smoke = ROOT / "complete_missingness_training_preflight_v1"
    audit, smoke_summary = read(smoke / "training_audit.json"), read(smoke / "summary.json")
    if (not audit["complete"] or audit["updates"] != 21 or not audit["not_a_candidate"]
            or not smoke_summary["all_gradients_finite"]):
        raise ValueError("Require the completed isolated feasibility test, not its checkpoint as initialization")
    verify_hashes(audit["original_artifact_sha256"])
    smoke_config = read(smoke / "config.json")
    verify_hashes(smoke_config["input_sha256"])
    verify_hashes(smoke_config["source_sha256"])
    prior = ROOT / "complete_missingness_prior_preflight_v1"
    prior_summary = read(prior / "summary.json")
    if not prior_summary["complete"] or prior_summary["usable_training_cases"] != 192:
        raise ValueError("Require every structured prior feasibility case")
    verify_hashes(prior_summary["array_sha256"])
    diagnostic = ROOT / "missingness_models_probe_v1_recovery1"
    if not read(diagnostic / "recovery_audit.json")["complete"] or not read(diagnostic / "audit.json")["complete"]:
        raise ValueError("Require the complete preserved all-family model diagnosis")
    config = deepcopy(base)
    config.update(version=9, **mixture.SEEDS, parent_checkpoint=str(checkpoint), structured_policy=deepcopy(mixture.POLICY),
        structured_prior=deepcopy(PRIOR_SPECIFICATION), training_structured_cases=mixture.training_cases(),
        validation_cases=repeated_bank(mixture.SEEDS["validation_seed"]) + mixture.structured_bank(mixture.SEEDS["structured_validation_seed"]),
        confirmation_cases=repeated_bank(mixture.SEEDS["confirmation_seed"], confirmation=True)
            + mixture.structured_bank(mixture.SEEDS["structured_confirmation_seed"], confirmation=True),
        after_campaign=str(after), previous_run=str(producer),
        initialization_provenance={"source_run": str(producer), "selected_step": 250,
            "checkpoint_sha256": gate["checkpoint_sha256"], "confirmation_sha256": sha256(gate_path), "not_smoke_checkpoint": True},
        selection=mixture.POLICY["selection"] + "; " + mixture.POLICY["retention"], confirmation=mixture.POLICY["confirmation"],
        experiment="Continue actual selected V8 with half legacy/area/support imputation and half complete structured imputation; retain every rating episode and unchanged single-network objectives; not an isolated causal ablation",
        inference="One original network and checkpoint; no routing, extra head or inference ensemble",
        synthetic_predictions="Save full general/teacher reference grids and every student validation/confirmation grid for raw-prediction audit")
    config["phases"][0]["name"] = "structured_mixture"
    artifacts = [producer / "config.json", checkpoint, gate_path, after / "config.json", after / "campaign_audit.json"]
    artifacts += [smoke / name for name in ("config.json", "summary.json", "training_audit.json")]
    artifacts += [prior / name for name in ("config.json", "summary.json", "cases.json")]
    artifacts += [diagnostic / name for name in ("config.json", "audit.json", "recovery_audit.json", "per_case.json", "per_family.csv")]
    config["input_sha256"].update({str(path): sha256(path) for path in artifacts})
    config["source_sha256"].update(coverage.source_hashes())
    for name in ("confirm_single_matrixformer_structured_mixture.py", "single_model_structured_mixture.py"):
        path = REPO / "scripts" / name
        config["source_sha256"][str(path)] = sha256(path)
    validate_recipe(config)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    storage_guard(output)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
        "training_started": False, "selected_from_synthetic_evidence_only": True, "real_benchmark_protocols_unchanged": True})
    for name in ("train_single_matrixformer_structured_mixture.py", "confirm_single_matrixformer_structured_mixture.py", "single_model_structured_mixture.py"):
        shutil.copy2(REPO / "scripts" / name, output / name)
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def preflight(config, output):
    student, optimizer, criterion, teachers, sampler = backend.setup(config)
    records = []
    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        specifications = [
            {"domain": "imputation", "shape": (2000, 1000), "component": 0, "regime": "area_scaled",
             "transpose": False, "seed": config["preflight_seed"]},
            {"domain": "imputation", "structured": {"family": "mnar_polarization", "component": "low_rank_mcar",
             "shape": [1024, 256], "regime": "area_scaled", "requested_missingness": .6, "seed": config["preflight_seed"] + 100}},
        ]
        domains = []
        for specification in (specifications[index], {"domain": "recsys", "shape": (2000, 1000), "seed": config["preflight_seed"] + 200 + index}):
            episode = sampler.sample(step=1, **specification)
            teacher, routing = backend.teacher_target(teachers, criterion, episode)
            loss = backend.backward_episode(student, criterion, episode, teacher, .5)
            domains.append({**episode.metadata, "loss": loss, "teacher": routing})
            del episode, teacher
        norm = optimizer_update(student, optimizer, config["grad_clip"])
        records.append({"joint_update": index + 1, "domains": domains, "grad_norm": norm,
                        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3})
        print(json.dumps({"preflight": records[-1]}), flush=True)
    write_json(output / "preflight.json", {"passed": True, "config_sha256": sha256(output / "config.json"), "cases": records,
                                           "scratch_updates_discarded": True, "candidate_training_started": False})
    del student, optimizer, criterion, teachers, sampler
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(output / "status.json")["state"] != "registered":
            raise ValueError("Never overwrite or automatically restart a started experiment")
        try:
            write_json(output / "status.json", {"state": "starting_training", "pid": os.getpid(), "updated_unix": time.time()})
            config = read(output / "config.json")
            validate_recipe(config)
            backend.verify_run(config, output)
            wait_for_gpu(output, True, False)
            with mixture.runtime(config, output):
                write_json(output / "status.json", {"state": "preflighting", "pid": os.getpid(), "updated_unix": time.time()})
                preflight(config, output)
                backend.train(config, output, resume=False)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
