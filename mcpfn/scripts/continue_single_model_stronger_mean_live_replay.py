"""Observe V46's surviving original replay after the idle parent was released."""

import argparse
import fcntl
from pathlib import Path
import time
import types

import recover_single_model_stronger_mean_idle_owner as stopped
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = stopped.original.ROOT / "stronger_mean_distillation_v46_live_replay_continuation_v1"
FAILURE = 'RuntimeError("Require the exact surviving worker and release of its parent\'s CUDA context")'
POLICY = (
    "Continue observing the exact surviving V46 endpoint replay, never restart it. The registered "
    "idle-parent release sent SIGTERM only to its coordinator; its immediate post-release observation "
    "guard failed. Preserve that failure and signal intent. Subsequently verify the original parent "
    "is stopped and absent from GPU owners, and the exact original child survived with the same "
    "PID/start ticks/argv/cwd and is performing its original replay. Do not infer the precise transient "
    "cause from the combined guard's error. Send no signal and perform no second release. Reuse "
    "the previous coordinator's unchanged wait/run code, replacing release with a no-signal check "
    "and writing only to this new registered directory. Preserve all remaining qualification, "
    "complete real-suite/review scope, audit paths, original selection and guards. Keep this "
    "coordinator CUDA-uninitialized, with visible GPU subprocesses. No repeated scientific stage."
)


def require_surviving_chain():
    config = stopped.verify()
    state = stopped.original.read(stopped.OUTPUT / "status.json")
    if (state["state"] != "failed" or state["error"] != FAILURE or stopped.identity(state["pid"]) is not None
            or stopped.identity(config["coordinator"]["pid"]) is not None
            or config["coordinator"]["pid"] in stopped.gpu_owners()):
        raise ValueError("Preserve the stopped failed release and already released original coordinator")
    intent = stopped.original.read(stopped.OUTPUT / "signal_intent.json")
    if intent != {"target": config["coordinator"], "signal": "SIGTERM", "child_signaled": False}:
        raise ValueError("Bind the original parent-only signal without sending another")
    current = stopped.original.read(stopped.original.REPLAY / "status.json")
    observed = stopped.identity(config["child"]["pid"])
    if (current["pid"] != config["child"]["pid"] or current["state"] == "failed"
            or (observed is not None and observed != config["child"])
            or (observed is None and current["state"] != "awaiting_independent_audit")):
        raise ValueError("Continue only the original surviving or completed replay, never a replacement")
    return config


def register():
    if OUTPUT.exists():
        raise FileExistsError("Preserve the first live-replay continuation")
    original = require_surviving_chain()
    directory = stopped.original.CONFIRMATION
    if (set(path.name for path in directory.iterdir()) != {"config.json", "registration.json", "status.json"}
            or stopped.original.read(directory / "status.json")["state"] != "registered"
            or (stopped.original.REPLAY / "audit.json").exists()
            or stopped.original.CAMPAIGN.exists() or stopped.original.REVIEW.exists()):
        raise ValueError("No remaining qualification or benchmark stage may already have run")
    paths = [stopped.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "worker.log", "signal_intent.json")]
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_stronger_mean_live_replay.py"]
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"policy": POLICY, "child": original["child"],
               "input_sha256": {str(path): sha256(path) for path in paths},
               "source_sha256": {str(path): sha256(path) for path in sources}})
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
               "registered_unix": time.time(), "signal_sent": False, "scientific_stage_repeated": False})
    stopped.original.status(OUTPUT, "registered")


def verify():
    config = stopped.original.read(OUTPUT / "config.json")
    if (stopped.original.read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY):
        raise ValueError("Preserve the exact no-signal live-worker continuation")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    original = stopped.verify()
    if original["child"] != config["child"]:
        raise ValueError("Keep the sole original child identity")
    return original


def confirm_release(config):
    current = require_surviving_chain()
    if current != config:
        raise ValueError("Preserve the same registered release and child")
    write_json(OUTPUT / "release_observation.json", {"parent_already_stopped": True,
               "child": config["child"], "signal_sent": False, "scientific_stage_repeated": False})


def execution_namespace():
    namespace = {**vars(stopped), "OUTPUT": OUTPUT, "verify": verify, "release_parent": confirm_release}
    for name in ("run", "wait_for_original_replay"):
        function = getattr(stopped, name)
        if function.__closure__ is not None:
            raise ValueError("Never retain an old orchestration closure")
        namespace[name] = types.FunctionType(function.__code__, namespace, name, function.__defaults__)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    arguments = parser.parse_args()
    with stopped.original.arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == "register":
            register()
            return
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if stopped.original.read(OUTPUT / "status.json")["state"] != "registered":
                raise ValueError("Never restart the live-worker continuation")
            try:
                execution_namespace()["run"]()
            except BaseException as error:
                stopped.original.status(OUTPUT, "failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
