"""Dispatch the unchanged registered teacher replay after its pre-execution CLI failure."""

import argparse
import fcntl
from pathlib import Path

import replay_single_model_point_inclusive_teachers as teacher
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256


RECOVERY = teacher.training.ROOT / "point_inclusive_v21_pipeline_entry_recovery_v1"
POLICY = ("Correct only the missing CLI dispatcher: call the unchanged registered V21 teacher replay and CPU audit "
          "functions once, using their original output reservation. The failed entry point invoked neither function. "
          "Preserve the failed script, pipeline status/logs, training and full audit, all seeds/checkpoints/targets, "
          "source/config hashes and qualification gates. No training restart, repeated prediction or new selection.")


def verify_recovery():
    config = teacher.read(RECOVERY / "recovery_config.json")
    registration = teacher.read(RECOVERY / "recovery_registration.json")
    if (config["policy"] != POLICY or registration["recovery_config_sha256"] != sha256(RECOVERY / "recovery_config.json")
            or config["teacher_output"] != str(teacher.OUTPUT)
            or config["source_sha256"].get(str(Path(__file__).resolve())) != sha256(__file__)
            or not registration["original_failure_preceded_teacher_execution"]):
        raise ValueError("Require the recorded entry-point-only recovery before any teacher computation")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def dispatch(stage):
    verify_recovery()
    expected = {"replay": "registered", "audit": "awaiting_independent_audit"}
    if stage not in expected:
        raise ValueError("Only the original replay and independent audit stages may run")
    with (teacher.OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if teacher.read(teacher.OUTPUT / "status.json")["state"] != expected[stage]:
            raise ValueError("Never restart a teacher stage or overwrite its prior outcome")
        try:
            {"replay": teacher.replay, "audit": teacher.audit}[stage](teacher.OUTPUT)
            verify_recovery()
        except BaseException as error:
            teacher.status(teacher.OUTPUT, "failed", error=repr(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("replay", "audit"), required=True)
    dispatch(parser.parse_args(argv).stage)


if __name__ == "__main__":
    main()
