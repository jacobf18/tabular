"""Preserve a failed pre-initialization waiter and continue the unchanged V15 pipeline."""

import argparse
import os
from pathlib import Path
import shutil
import time

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import run_single_model_targeted_decoder_pipeline as pipeline


OUTPUT = pipeline.ROOT / "targeted_decoder_v15_pipeline_recovery1"


def validate_command(arguments, script):
    if str(script) not in arguments or "--stage" not in arguments:
        raise ValueError("Bind the actual existing registered trainer, not an arbitrary PID")
    if arguments[arguments.index("--stage") + 1:] != ["train"]:
        raise ValueError("Require the exact original default-output training invocation")


def live_handle(pid, expected_start=None):
    os.kill(pid, 0)
    command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().strip("\0").split("\0")
    validate_command(command, Path(pipeline.training.__file__).resolve())
    start = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    if expected_start is not None and start != expected_start:
        raise ValueError("Never confuse a reused PID with the original trainer")
    return {"pid": pid, "command": command, "start_ticks": start}


def recover(pid):
    if OUTPUT.exists():
        raise FileExistsError("Preserve the first recovery attempt")
    original = pipeline.OUTPUT
    failed = pipeline.read(original / "status.json")
    if failed["state"] != "failed" or failed["error"] != "ProcessLookupError(3, 'No such process')":
        raise ValueError("Recover only the observed stale registration-PID failure")
    if (pipeline.AUDIT.exists() or (original / "training_audit.log").exists()
            or pipeline.read(pipeline.REPLAY / "status.json")["state"] != "registered"
            or pipeline.read(pipeline.CONFIRMATION / "status.json")["state"] != "registered"):
        raise ValueError("No audit, neural replay or confirmation may have started")
    handle = live_handle(pid)
    inputs = {str(original / name): sha256(original / name) for name in ("config.json", "registration.json", "status.json", "worker.log")}
    pipeline.verify(original)
    OUTPUT.mkdir()
    for name in ("config.json", "registration.json"):
        shutil.copy2(original / name, OUTPUT / name)
    write_json(OUTPUT / "recovery_registration.json", {"original_pipeline": str(original), "trainer": handle,
               "input_sha256": inputs, "source_sha256": {str(Path(__file__).resolve()): sha256(__file__)},
               "scientific_configuration_changed": False, "training_restarted": False,
               "replay_or_confirmation_repeated": False})
    try:
        while True:
            current = pipeline.read(pipeline.training.OUTPUT / "status.json")
            if current["state"] == "training_complete":
                break
            if current["state"] in {"failed", "interrupted"}:
                raise RuntimeError("The original trainer stopped; preserve its declared outcome")
            live_handle(pid, handle["start_ticks"])
            if current["pid"] == pid and current["state"] != "registered":
                break
            pipeline.status(OUTPUT, "waiting_for_trainer_owned_status", trainer_pid=pid,
                            observed_state=current["state"], observed_status_pid=current["pid"])
            time.sleep(15)
        verify_hashes(inputs)
        pipeline.verify(OUTPUT)
        pipeline.status(OUTPUT, "registered", recovery_of=str(original), trainer_pid=pid)
        pipeline.run(OUTPUT)
        verify_hashes(inputs)
        write_json(OUTPUT / "recovery_audit.json", {"complete": True, "original_failure_preserved": True,
                   "training_restarted": False, "scientific_configuration_changed": False,
                   "result_sha256": sha256(OUTPUT / "result.json")})
    except BaseException as error:
        pipeline.status(OUTPUT, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid", type=int, required=True)
    recover(parser.parse_args().trainer_pid)
