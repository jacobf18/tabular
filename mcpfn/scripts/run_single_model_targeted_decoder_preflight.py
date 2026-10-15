"""Wait for the existing V14 pipeline, then register and audit one fresh targeted preflight."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import preflight_single_model_targeted_decoder as preflight


OUTPUT = preflight.ROOT / "targeted_decoder_preflight_driver_v1"
PREREQUISITE = preflight.ROOT / "decoder_repair_v14_pipeline"


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def run():
    if OUTPUT.exists():
        raise FileExistsError("Never restart a started preflight driver")
    OUTPUT.mkdir()
    paths = [Path(__file__).resolve(), Path(preflight.__file__).resolve(),
             REPO / "tests/test_single_model_targeted_decoder_preflight.py"]
    sources = {str(path): sha256(path) for path in paths}
    write_json(OUTPUT / "config.json", {"prerequisite": str(PREREQUISITE), "preflight": str(preflight.OUTPUT),
               "source_sha256": sources, "training_restart_permitted": False, "research_selection_tolerance": 0})
    with (OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                current = json.loads((PREREQUISITE / "status.json").read_text())
                if current["state"] == "complete":
                    break
                if current["state"] in {"failed", "interrupted"}:
                    raise RuntimeError("The original qualification stopped; never bypass it")
                os.kill(current["pid"], 0)
                status("waiting_for_v14_qualification", prerequisite_pid=current["pid"], prerequisite_state=current["state"])
                time.sleep(30)
            for stage in ("register", "run", "audit"):
                verify_hashes(sources)
                command = [sys.executable, "-u", str(Path(preflight.__file__).resolve()), "--stage", stage]
                environment = dict(os.environ)
                if stage != "run":
                    environment["CUDA_VISIBLE_DEVICES"] = ""
                with (OUTPUT / f"{stage}.log").open("x") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=REPO.parent)
                    status(stage, child_pid=process.pid, command=command)
                    if process.wait():
                        raise RuntimeError(f"Targeted preflight {stage} failed; preserve evidence without retry")
            gate = preflight.read(preflight.OUTPUT / "gate.json")
            verify_hashes(sources)
            write_json(OUTPUT / "result.json", {"complete": True, "gate": gate,
                       "qualification_result_sha256": sha256(PREREQUISITE / "result.json"),
                       "audit_sha256": sha256(preflight.OUTPUT / "audit.json"), "goal_success_not_assessed": True})
            status("complete", accepted=gate["passed"])
        except BaseException as error:
            status("failed", error=repr(error))
            raise


if __name__ == "__main__":
    run()
