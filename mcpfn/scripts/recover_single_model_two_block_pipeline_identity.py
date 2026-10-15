"""Recover V26 orchestration after its shell exec handoff, without repeating science."""

import argparse
import fcntl
from pathlib import Path
import shutil
import time

import run_single_model_top_two_blocks_pipeline as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = original.ROOT / "top_two_blocks_v26_identity_recovery"
AUDITOR_COMMAND = "python scripts/audit_single_model_top_two_blocks.py"
FAILURE = "RuntimeError('The original chain stopped without a completed independent compact audit; do not retry')"
QUALIFICATION_STAGES = (("replay", False), ("audit-replay", True), ("confirm", False))
POLICY = ("Orchestration only: preserve the original failed V26 coordinator/config/status/log bytes. The original "
    "registered bash train/audit chain exec'ed its final auditor with the same PID and start ticks; changed argv "
    "is not a dead process. Bind only that exact original auditor, or its already completed full independent audit. "
    "Never retrain, restart or duplicate the auditor. Require all64 attempts and the original selected step64. "
    "Before registration require both original qualification reservations completely unused. Run each unchanged "
    "original endpoint/selected replay, CPU replay audit, fresh confirmation and CPU confirmation audit exactly "
    "once through its existing CLI and locks. Keep all original sources, inputs, seeds, gates, scope and actual-"
    "volume capacity checks. Only after qualification publish via the original audit and execute the full frozen "
    "tabular/recommender/LLM campaign/review. Write only recovery orchestration logs/status/result in this additive "
    "directory; preserve the original coordinator failure forever. No scientific stage retry, reselection, "
    "relaxed guard, new inference ensemble or goal-completion claim.")


def read(path):
    return original.read(path)


def status(state, **fields):
    original.status(OUTPUT, state, **fields)


def validate_handoff(expected, observed, cwd, audit_state):
    if (observed is None or observed["pid"] != expected["pid"] or observed["start_ticks"] != expected["start_ticks"]
            or observed["command"] != AUDITOR_COMMAND or not expected["command"].endswith("; " + AUDITOR_COMMAND)
            or "train_single_matrixformer_top_two_blocks.py --stage train; " not in expected["command"]
            or Path(cwd) != REPO or audit_state["pid"] != expected["pid"]
            or audit_state["state"] != "initializing_independent_task_replay"):
        raise ValueError("Require the same original process/start ticks and exact final auditor exec handoff")


def audit_state():
    return read(original.AUDIT.with_name(original.AUDIT.name + "_task_replay") / "status.json")


def require_original_failure():
    config = original.verify(original.OUTPUT)
    failed = read(original.OUTPUT / "status.json")
    trained = read(original.training.OUTPUT / "status.json")
    if (failed["state"] != "failed" or failed["error"] != FAILURE or original.process_identity(failed["pid"]) is not None
            or trained["state"] != "training_complete" or trained["step"] != 64 or trained["accepted_updates"] != 64
            or trained["best"]["step"] != 64 or original.process_identity(trained["pid"]) is not None):
        raise ValueError("Preserve the exact stopped coordinator failure and completed64-update V26 training")
    return config


def require_unstarted_reservations():
    for directory in (original.REPLAY, original.CONFIRMATION):
        if (read(directory / "status.json")["state"] != "registered"
                or {path.name for path in directory.iterdir()} != {"config.json", "registration.json", "status.json"}):
            raise ValueError("Recover orchestration only before any reserved scientific stage has started")
    if (original.CAMPAIGN.exists() or original.REVIEW.exists()
            or (original.training.OUTPUT / "best_validation_checkpoint.pth").exists()):
        raise ValueError("Never recover over a published model or begun real campaign")


def register():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Never repeat a begun identity recovery")
    config = require_original_failure()
    require_unstarted_reservations()
    current = audit_state()
    expected = config["training_and_audit_chain"]
    observed = original.process_identity(expected["pid"])
    if current["state"] == "complete":
        if current["pid"] != expected["pid"]:
            raise ValueError("Require the original auditor's complete result")
        original.prerequisite(original.OUTPUT)
    else:
        cwd = Path(f'/proc/{expected["pid"]}/cwd').resolve()
        validate_handoff(expected, observed, cwd, current)
    for volume, path in (("checkpoint", original.ROOT), ("root", Path("/")), ("scratch", original.training.SCRATCH)):
        budget = 16 * 1024**2 if volume != "scratch" else 0
        if shutil.disk_usage(path).free - budget < original.diagnostics.RESERVES[volume]:
            raise OSError("Preserve all original reserves and the small recovery manifest budget")
    paths = [original.OUTPUT / name for name in ("config.json", "registration.json", "status.json")]
    paths.extend(original.training.OUTPUT / name for name in ("config.json", "registration.json", "status.json",
                 "validation.jsonl", "training.jsonl", "selected_checkpoint.pth", "latest_checkpoint.pth"))
    paths.append(original.AUDIT.with_name(original.AUDIT.name + "_task_replay") / "config.json")
    paths.append(original.ROOT / "top_two_blocks_v26_pipeline.log")
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_two_block_identity_recovery.py"]
    recovery = {"policy": POLICY, "original_pipeline": str(original.OUTPUT), "original_chain": expected,
                "observed_auditor": observed, "audit_complete_at_registration": current["state"] == "complete",
                "qualification_stages": [[name, cpu] for name, cpu in QUALIFICATION_STAGES],
                "input_sha256": {str(path): sha256(path) for path in paths},
                "source_sha256": {str(path): sha256(path) for path in sources}}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", recovery)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
               "registered_unix": time.time(), "scientific_stage_started": False, "training_repeated": False})
    status("registered")


def verify():
    config = read(OUTPUT / "config.json")
    if (read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["original_pipeline"] != str(original.OUTPUT)
            or config["qualification_stages"] != [[name, cpu] for name, cpu in QUALIFICATION_STAGES]):
        raise ValueError("Preserve the exact one-shot orchestration-only recovery")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    original_config = require_original_failure()
    if config["original_chain"] != original_config["training_and_audit_chain"]:
        raise ValueError("Never substitute the original registered train/audit chain")
    return config


def wait_for_original_audit(config):
    while True:
        current = audit_state()
        if current["pid"] != config["original_chain"]["pid"]:
            raise RuntimeError("Never accept a replacement auditor")
        if current["state"] == "complete":
            original.prerequisite(original.OUTPUT)
            return
        if current["state"] == "failed":
            raise RuntimeError("The original auditor failed; do not repeat it")
        expected = config["original_chain"]
        observed = original.process_identity(expected["pid"])
        cwd = Path(f'/proc/{expected["pid"]}/cwd').resolve()
        validate_handoff(expected, observed, cwd, current)
        if observed != config["observed_auditor"]:
            raise RuntimeError("The registered original auditor identity changed")
        status("waiting_for_original_audit", auditor_pid=expected["pid"])
        time.sleep(30)


def run():
    config = verify()
    wait_for_original_audit(config)
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
               "original_failed_pipeline": str(original.OUTPUT),
               "original_failure_sha256": sha256(original.OUTPUT / "status.json"),
               "campaign": str(original.CAMPAIGN) if gate["passed"] else None,
               "review": str(original.REVIEW) if gate["passed"] else None,
               "scientific_stages_repeated": False, "no_inference_ensemble": True, "goal_success_not_assessed": True})
    status("complete", confirmed=gate["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register()
        return
    with (OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(OUTPUT / "status.json")["state"] != "registered":
            raise ValueError("Never restart the identity recovery")
        try:
            run()
        except BaseException as error:
            status("failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
