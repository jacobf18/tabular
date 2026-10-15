"""Finish V43's original audit and qualification after its unstarted auditor lost its shell."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import relocate_single_model_v43_recovery_capacity as capacity
import run_single_model_family_balanced_retention_pipeline as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = original.ROOT / "family_balanced_retention_v43_orphan_recovery_v1"
AUDITOR = REPO / "scripts/audit_single_model_family_balanced_retention.py"
TRAINING_CONFIG_SHA256 = "275ef4280315099d0dc66ba8eabcb6056ca02d12f3265b2c5e52dd0daa4b88b5"
PIPELINE_CONFIG_SHA256 = "e08fa0ed1487187c2c94fda5d48380a10b609ebe09f35c54a73cf8660fbbe1c1"
SELECTED_SHA256 = "461664c2bafc8fa89131684b2a19346564cef6e31263c7b5437fb5e63cce1db2"
FAILURE = "RuntimeError('The original chain stopped without a completed independent compact audit; do not retry')"
QUALIFICATION_STAGES = (("replay", False), ("audit-replay", True), ("confirm", False))
POLICY = ("Operational recovery only: V43 completed its original64 attempts with56 accepted updates and selected "
    "step25. Its shell disappeared while the unchanged trainer and coordinator survived; the full auditor was "
    "never launched and the coordinator failed before any endpoint or confirmation execution. Preserve the "
    "original failure and all registered bytes. Require absent original workers, completely unused auditor "
    "outputs and qualification reservations, restored original storage reserves and unchanged selected hash. "
    "Launch the original full auditor once, then require its complete original prerequisite before the sole "
    "endpoint/selection replay, independent audit,284-case confirmation and independent confirmation audit. "
    "Only a passing original gate permits every frozen real benchmark and review. Use original CLIs, locks, "
    "24 BLAS/4 Torch threads, capacity checks and no hidden GPU in GPU stages. No training restart, stage retry, "
    "reselection, consumed-seed reuse, relaxed guard, inference ensemble or scientific success claim.")


def read(path):
    return original.read(path)


def status(state, **fields):
    original.status(OUTPUT, state, **fields)


def validate_terminal(trained, failed):
    if (trained.get("state") != "training_complete" or trained.get("step") != 64
            or trained.get("accepted_updates") != 56 or trained.get("best", {}).get("step") != 25
            or failed.get("state") != "failed" or failed.get("error") != FAILURE):
        raise ValueError("Recover only the exact completed V43 training and pre-audit orchestration failure")


def require_original_failure():
    if (sha256(original.training.OUTPUT / "config.json") != TRAINING_CONFIG_SHA256
            or sha256(original.OUTPUT / "config.json") != PIPELINE_CONFIG_SHA256
            or sha256(original.training.OUTPUT / "selected_checkpoint.pth") != SELECTED_SHA256):
        raise ValueError("Never change V43's original recipe, qualification or selected checkpoint")
    config = original.verify(original.OUTPUT)
    trained, failed = read(original.training.OUTPUT / "status.json"), read(original.OUTPUT / "status.json")
    validate_terminal(trained, failed)
    for pid in (trained["pid"], failed["pid"], config["training_and_audit_chain"]["pid"]):
        if original.process_identity(pid) is not None:
            raise ValueError("Never race an original trainer, coordinator or train/audit shell")
    return config


def require_unstarted():
    for directory in (original.AUDIT, original.AUDIT.with_name(original.AUDIT.name + "_task_replay")):
        if directory.exists() or directory.is_symlink():
            raise FileExistsError("The original full auditor must never have started")
    if (original.ROOT / "family_balanced_retention_v43_train_audit.log").exists():
        raise FileExistsError("Preserve any original auditor launch evidence; do not repeat it")
    for directory in (original.REPLAY, original.CONFIRMATION):
        if (read(directory / "status.json")["state"] != "registered"
                or {path.name for path in directory.iterdir()} != {"config.json", "registration.json", "status.json"}):
            raise ValueError("Neither endpoint replay nor confirmation may have started")
    if (original.CAMPAIGN.exists() or original.REVIEW.exists()
            or (original.training.OUTPUT / "best_validation_checkpoint.pth").exists()):
        raise ValueError("Do not recover over a published model or real campaign")
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(Path(argument.decode()).name == AUDITOR.name for argument in arguments[1:] if argument):
            raise ValueError("An existing original auditor must not be duplicated")


def register():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Never repeat a registered orphan-chain recovery")
    require_original_failure()
    require_unstarted()
    receipt = capacity.recheck()
    independent = read(capacity.OUTPUT / "external_recheck.json")
    if (read(capacity.OUTPUT / "status.json")["state"] != "complete" or receipt["complete"] is not True
            or independent["complete"] is not True or independent["artifact_sha256"] != receipt["artifact_sha256"]
            or independent["config_sha256"] != receipt["config_sha256"]):
        raise ValueError("Require independently restored original qualification headroom")
    paths = [original.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "coordinator.log")]
    paths.extend(original.training.OUTPUT / name for name in
                 ("config.json", "registration.json", "status.json", "training.jsonl", "selected_checkpoint.pth"))
    paths.extend(original.training.OUTPUT / f"validation_{step}.json" for step in (0, 10, 25, 50, 64))
    paths.extend(capacity.OUTPUT / name for name in
                 ("config.json", "registration.json", "independent_recheck.json", "external_recheck.json"))
    sources = [Path(__file__).resolve(), Path(original.__file__).resolve(), AUDITOR,
               Path(capacity.__file__).resolve(), REPO / "tests/test_single_model_family_balanced_orphan_recovery.py"]
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"policy": POLICY, "original_pipeline": str(original.OUTPUT),
               "input_sha256": {str(path): sha256(path) for path in paths},
               "source_sha256": {str(path): sha256(path) for path in sources},
               "qualification_stages": [[name, cpu] for name, cpu in QUALIFICATION_STAGES],
               "original_auditor_never_started": True, "original_failure_preserved": True})
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
               "registered_unix": time.time(), "scientific_stages_started": False, "training_repeated": False})
    status("registered")


def verify():
    config = read(OUTPUT / "config.json")
    if (read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["original_pipeline"] != str(original.OUTPUT)
            or config["qualification_stages"] != [[name, cpu] for name, cpu in QUALIFICATION_STAGES]
            or config["original_auditor_never_started"] is not True or config["original_failure_preserved"] is not True):
        raise ValueError("Never alter the one-shot operational recovery")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    require_original_failure()
    return config


def audit_environment():
    environment = {**os.environ, "PYTHONPATH": f"{REPO / 'scripts'}:{REPO / 'src'}", "TMPDIR": "/run",
                   "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "24", "MKL_NUM_THREADS": "24"}
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        raise ValueError("Do not hide the GPU from the original teacher/reference replay auditor")
    return environment


def run_original_audit():
    require_unstarted()
    original.diagnostics.verify(read(original.training.OUTPUT / "config.json")["diagnostic_storage"], "qualification")
    command = [sys.executable, "-u", str(AUDITOR), "--output", str(original.AUDIT)]
    environment = audit_environment()
    with (OUTPUT / "training-audit.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=REPO)
        status("training-audit", child_pid=process.pid, command=command)
        if process.wait():
            raise RuntimeError("The sole original full audit failed; preserve evidence without a retry")
    original.prerequisite(original.OUTPUT)


def run():
    verify()
    audit_environment()
    run_original_audit()
    for stage, cpu in QUALIFICATION_STAGES:
        verify()
        original.run_child(OUTPUT, stage, Path(original.__file__).resolve(),
                           ["--stage", stage, "--output", str(original.OUTPUT)], cpu=cpu)
    if read(original.CONFIRMATION / "status.json")["state"] == "awaiting_independent_audit":
        original.run_child(OUTPUT, "audit-confirmation", Path(original.__file__).resolve(),
                           ["--stage", "audit-confirmation", "--output", str(original.OUTPUT)], cpu=True)
    gate = read(original.CONFIRMATION / "gate.json")
    if gate["passed"]:
        original.run_child(OUTPUT, "benchmarks", REPO / "scripts/run_single_matrixformer_cached_benchmarks.py",
            ["--run", str(original.training.OUTPUT), "--confirmation", str(original.CONFIRMATION),
             "--source", str(original.PARENT_CACHE), "--output", str(original.CAMPAIGN)])
        original.run_child(OUTPUT, "review", REPO / "scripts/review_single_matrixformer_campaign.py",
                           ["--campaign", str(original.CAMPAIGN), "--output", str(original.REVIEW)], cpu=True)
    verify()
    write_json(OUTPUT / "result.json", {"complete": True, "gate": gate,
               "training_audit_sha256": sha256(original.AUDIT / "audit.json"),
               "endpoint_replay_audit_sha256": sha256(original.REPLAY / "audit.json"),
               "original_failed_pipeline": str(original.OUTPUT),
               "original_failure_sha256": sha256(original.OUTPUT / "status.json"),
               "campaign": str(original.CAMPAIGN) if gate["passed"] else None,
               "review": str(original.REVIEW) if gate["passed"] else None,
               "training_repeated": False, "scientific_stages_repeated": False,
               "no_inference_ensemble": True, "goal_success_not_assessed": True})
    status("complete", confirmed=gate["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    arguments = parser.parse_args()
    with original.arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == "register":
            register()
            return
        with (OUTPUT / "run.lock").open("a") as lock, (original.OUTPUT / "run.lock").open("a") as original_lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(OUTPUT / "status.json")["state"] != "registered":
                raise ValueError("Never restart a begun recovery or any scientific stage")
            try:
                run()
            except BaseException as error:
                status("failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
