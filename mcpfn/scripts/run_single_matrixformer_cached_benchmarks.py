"""Wait for synthetic confirmation, seed fixed parent controls, then run all suites."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import reuse_single_matrixformer_parent_controls as cache
import run_single_matrixformer_benchmarks as campaign


def handoff(arguments, output):
    command = [sys.executable, "-u", str(Path(campaign.__file__).resolve()),
               "--run", str(arguments.run), "--confirmation", str(arguments.confirmation),
               "--output", str(output), "--resume"]
    os.execv(sys.executable, command)


def execute(arguments, output):
    sources = [Path(__file__).resolve(), Path(cache.__file__).resolve(), Path(campaign.__file__).resolve()]
    inputs = [arguments.run / "config.json", arguments.source / "manifest.json", arguments.source / "audit.json"]
    inputs.extend(arguments.run.parent / f"{stage}_protocol_v1/protocol.json" for stage in campaign.STAGES)
    config = {"run": str(arguments.run), "confirmation": str(arguments.confirmation),
              "parent_cache_source": str(arguments.source),
              "input_sha256": {str(path): sha256(path) for path in inputs},
              "source_sha256": {str(path): sha256(path) for path in sources},
              "policy": "After the same independent synthetic gate, reuse only audited unchanged-parent rating controls; original full evaluator and audit scope, fresh student predictions and specialist parity",
              "checkpoint_selection_performed": False, "training_performed": False}
    write_json(output / "cache_launcher_config.json", config)
    for source in sources:
        destination = output / "cache_launcher_sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    gate = campaign.confirmation_state(arguments.confirmation)
    while gate is None:
        write_json(output / "status.json", {"state": "waiting_for_synthetic_confirmation", "pid": os.getpid(), "updated_unix": time.time()})
        time.sleep(30)
        gate = campaign.confirmation_state(arguments.confirmation)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    if gate["passed"]:
        report = cache.prepare(arguments.source, arguments.run.parent / "recsys_protocol_v1",
                               arguments.run / "best_validation_checkpoint.pth", arguments.confirmation, output / "recsys")
        if len(report["controls"]) != 4 or report["student_predictions_copied"] or report["specialist_parity_copied"]:
            raise ValueError("Only the four unchanged-parent controls may be preseeded")
    write_json(output / "status.json", {"state": "delegating_to_unchanged_campaign", "parent_controls_preseeded": gate["passed"],
                                        "pid": os.getpid(), "updated_unix": time.time()})
    handoff(arguments, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.run, arguments.confirmation, arguments.source = arguments.run.resolve(), arguments.confirmation.resolve(), arguments.source.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cache_launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(arguments, output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
