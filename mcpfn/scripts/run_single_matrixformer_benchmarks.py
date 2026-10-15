"""Queue all frozen benchmark suites only after independent synthetic confirmation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from recsys_evaluation_common import REPO, sha256, write_json
from train_streamed_large_matrices import wait_for_gpu


STAGES = ("tabular", "recsys", "llm")


def read(path):
    return json.loads(Path(path).read_text())


def confirmation_state(directory):
    status = read(directory / "status.json")
    if status["state"] == "complete":
        return read(directory / "gate.json")
    if status["state"] in {"failed", "interrupted"}:
        raise RuntimeError(f"Synthetic confirmation stopped: {status}")
    try:
        os.kill(status["pid"], 0)
    except ProcessLookupError:
        status = read(directory / "status.json")
        if status["state"] == "complete":
            return read(directory / "gate.json")
        raise RuntimeError("Confirmation process is missing; do not trust its stale status or restart training") from None
    return None


def require_full_audit(stage, audit):
    if not audit.get("complete"):
        raise ValueError(f"{stage} has not completed its prediction audit")
    if stage == "tabular" and audit.get("cases") != 2262:
        raise ValueError("Tabular audit must include all 2262 frozen cases")
    if stage == "llm" and (audit.get("fold_settings") != 1800 or audit.get("panels") != 180):
        raise ValueError("LLM audit must include all 1800 fold settings and 180 panels")
    if stage == "recsys":
        expected = {(dataset, role) for dataset in ("100K", "netflix", "1M", "10M") for role in ("student", "parent_same_recipe")}
        cases = audit.get("cases", [])
        if len(cases) != 8 or {(case["dataset"], case["role"]) for case in cases} != expected:
            raise ValueError("Ratings audit must include every dataset and matched-preprocessing control")


def execute(arguments, output):
    goal = arguments.run.parent
    protocols = {stage: goal / f"{stage}_protocol_v1" for stage in STAGES}
    config = {"run": str(arguments.run), "confirmation": str(arguments.confirmation), "stages": list(STAGES),
              "protocols": {stage: str(path) for stage, path in protocols.items()},
              "protocol_sha256": {stage: sha256(path / "protocol.json") for stage, path in protocols.items()},
              "source_sha256": sha256(Path(__file__)),
              "policy": "Run the single synthetic-confirmed checkpoint on all suites; never select training updates from benchmark outcomes"}
    config_path = output / "config.json"
    if config_path.exists():
        if read(config_path) != config:
            raise ValueError("Campaign resume configuration changed")
    else:
        write_json(config_path, config)
        shutil.copy2(Path(__file__), output / Path(__file__).name)
    gate = confirmation_state(arguments.confirmation)
    while gate is None:
        write_json(output / "status.json", {"state": "waiting_for_synthetic_confirmation", "pid": os.getpid(), "updated_unix": time.time()})
        time.sleep(30)
        gate = confirmation_state(arguments.confirmation)
    if not gate["passed"]:
        write_json(output / "status.json", {"state": "not_promoted", "reason": "Independent synthetic confirmation did not pass",
                                            "gate": gate, "benchmarks_evaluated": False, "updated_unix": time.time()})
        return
    checkpoint = arguments.run / "best_validation_checkpoint.pth"
    if sha256(checkpoint) != gate["checkpoint_sha256"] or gate["selected_step"] == 0:
        raise ValueError("Synthetic gate does not bind a trained frozen checkpoint")
    registration = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                    "selected_step": gate["selected_step"], "confirmation_sha256": sha256(arguments.confirmation / "gate.json"),
                    "selected_before_benchmark_evaluation": True}
    registration_path = output / "candidate.json"
    if registration_path.exists() and read(registration_path) != registration:
        raise ValueError("Candidate changed during the benchmark campaign")
    if not registration_path.exists():
        write_json(registration_path, registration)
    environment = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}", "OMP_NUM_THREADS": "4",
                   "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    audits = {}
    for stage in STAGES:
        wait_for_gpu(output, True, False)
        script = REPO / "scripts" / f"evaluate_single_matrixformer_{stage}.py"
        command = [sys.executable, "-u", str(script), "--stage", "evaluate", "--protocol", str(protocols[stage]),
                   "--checkpoint", str(checkpoint), "--confirmation", str(arguments.confirmation), "--output", str(output / stage)]
        with (output / f"{stage}.log").open("a") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=REPO.parent)
            write_json(output / "status.json", {"state": "evaluating", "stage": stage, "pid": os.getpid(),
                                                "child_pid": process.pid, "command": command, "updated_unix": time.time()})
            return_code = process.wait()
        if return_code:
            raise RuntimeError(f"{stage} evaluation exited {return_code}; inspect {output / (stage + '.log')}")
        audit_path = output / stage / "audit.json"
        require_full_audit(stage, read(audit_path))
        audits[stage] = {"path": str(audit_path), "sha256": sha256(audit_path)}
    write_json(output / "campaign_audit.json", {"all_evaluations_complete": True, "candidate": registration,
                                               "audits": audits, "goal_success_not_assessed": True,
                                               "required_review": "Compare all suites and per-pattern/panel trade-offs; completed evaluations alone do not establish goal success"})
    write_json(output / "status.json", {"state": "complete", "stages": list(STAGES), "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    arguments.run, arguments.confirmation = arguments.run.resolve(), arguments.confirmation.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=arguments.resume)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(arguments, output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
