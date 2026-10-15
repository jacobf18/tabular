"""Replay both audited decoder proposals after the existing V13 qualification job."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import check_single_model_point_finite_steps as original
from dwip_teacher_mean_training import verify_hashes
import probe_single_model_decoder_subspaces as probe
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_parameter_subspace as subspace
from single_model_point_proposal_grid import finite_decision


OUTPUT = probe.alignment.ROOT / "decoder_finite_steps_v1"
PIPELINE = probe.alignment.ROOT / "finite_guarded_v13_pipeline"
ROLES = ("before", *probe.SCOPES)
POLICY = ("After the existing V13 qualification pipeline terminates, evaluate both predeclared audited "
          "decoder-coordinate proposals at the unchanged V8 state on the same fourteen synthetic training tasks, "
          "with two exact-seed repetitions per role. Preserve every point metric and undefined slot. Independently "
          "CPU-rescore all84 full prediction grids and require strict retention in every repetition for finite "
          "feasibility. No new optimizer replay, model training, checkpoint selection, validation, confirmation, "
          "benchmark, fallback margin or proposal search. Do not preempt or restart V13.")


def source_view():
    config = deepcopy(probe.read(probe.SOURCE / "config.json"))
    config["states"] = {"v8": config["states"]["v8"]}
    return config


def verify(output):
    config = probe.read(output / "config.json")
    if (sha256(output / "config.json") != probe.read(output / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["roles"] != list(ROLES) or config["repeats"] != 2
            or config["source_config"] != source_view()):
        raise ValueError("Never alter the registered decoder finite-step panel")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    probe.verify(probe.OUTPUT)
    return config


def register(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Use the new declared finite-step directory")
    probe.verify(probe.OUTPUT)
    audit = probe.read(probe.OUTPUT / "audit.json")
    if probe.read(probe.OUTPUT / "status.json")["state"] != "complete" or not audit["complete"]:
        raise ValueError("Require the independent restricted-gradient audit first")
    verify_hashes(audit["artifact_sha256"])
    proposals = probe.read(probe.OUTPUT / "result.json")["scopes"]
    if any(not proposals[scope].get("decision", {}).get("admissible", False) for scope in probe.SCOPES):
        raise ValueError("This declared84-grid panel requires both audited admissible proposals")
    paths = [probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "audit.json")]
    paths += [PIPELINE / "config.json", PIPELINE / "registration.json"]
    paths.extend(probe.SOURCE / "tasks" / f'task_{task["index"]:02d}.pt' for task in source_view()["cases"])
    sources = {Path(__file__).resolve(), Path(original.__file__).resolve(), REPO / "tests/test_single_model_decoder_finite_steps.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    output.mkdir(parents=True)
    write_json(output / "config.json", {"policy": POLICY, "roles": list(ROLES), "repeats": 2, "source_config": source_view(),
               "input_sha256": {str(path): sha256(path) for path in paths}, "source_sha256": {str(path): sha256(path) for path in sources}})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "neural_evaluation_performed": False, "checkpoint_selection_performed": False})
    probe.status(output, "registered")


def role_parameters(result, snapshot, role):
    if role == "before":
        return snapshot["before"], None
    if role not in probe.SCOPES:
        raise ValueError("Never introduce an unregistered candidate")
    record = probe.read(probe.OUTPUT / "result.json")["scopes"][role]
    if not record.get("decision", {}).get("admissible", False):
        return None, None
    saved = torch.load(record["artifact"], map_location="cpu", weights_only=True)
    total = record["total_parameters"]
    if not torch.equal(subspace.extract(snapshot["before"], saved["ranges"], total), saved["before"]):
        raise ValueError("Candidate must bind the identical original V8 coordinates")
    return subspace.replace(snapshot["before"], saved["candidate"], saved["ranges"], total), record["actual_products"]


@contextmanager
def runtime():
    callbacks = {"ROLES": ROLES, "POLICY": POLICY, "role_parameters": role_parameters, "verify_registration": verify}
    saved = {name: getattr(original, name) for name in callbacks}
    try:
        for name, value in callbacks.items():
            setattr(original, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(original, name, value)


def run(output):
    verify(output)
    if probe.read(output / "status.json")["state"] not in {"registered", "waiting_for_v13"}:
        raise ValueError("Never repeat a started neural diagnostic")
    terminal = probe.read(PIPELINE / "status.json")
    if terminal["state"] not in {"complete", "failed", "interrupted"}:
        raise ValueError("Do not preempt the live V13 qualification pipeline")
    write_json(output / "predecessor_terminal.json", {"status": terminal, "status_sha256": sha256(PIPELINE / "status.json")})
    probe.status(output, "starting_neural_replay")
    with runtime():
        original.execute(output)
    probe.status(output, "awaiting_independent_audit")


def audit(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Require a separate CPU-only prediction audit")
    config = verify(output)
    if probe.read(output / "status.json")["state"] != "awaiting_independent_audit":
        raise ValueError("Require a fresh complete finite-step replay")
    result = probe.read(output / "result.json")
    rows = [json.loads(line) for line in (output / "v8/cases.jsonl").read_text().splitlines()]
    if len(rows) != 84 or set(result["states"]) != {"v8"}:
        raise ValueError("All84 predeclared prediction grids are mandatory")
    hashes = {str(output / name): sha256(output / name) for name in ("result.json", "v8/cases.jsonl", "v8/summary.json")}
    specifications = {task["name"]: task for task in config["source_config"]["cases"]}
    proposals = probe.read(probe.OUTPUT / "result.json")["scopes"]
    scores, seen = {}, set()
    for row in rows:
        identity = row["role"], row["task"], row["repeat"]
        if identity in seen or row["role"] not in ROLES or row["repeat"] not in (0, 1):
            raise ValueError("Preserve every unique original task/role/repeat")
        seen.add(identity)
        expected_products = None if row["role"] == "before" else proposals[row["role"]]["actual_products"]
        probe.alignment.compare_metric_records(row["products"], expected_products)
        task = specifications[row["task"]]
        cached = torch.load(probe.SOURCE / "tasks" / f'task_{task["index"]:02d}.pt', map_location="cpu", weights_only=False)
        episode = probe.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
        hashes[row["prediction"]] = row["prediction_sha256"]
        if result["prediction_sha256"].get(row["prediction"]) != row["prediction_sha256"]:
            raise ValueError("Every result must bind its saved full prediction grid")
        with np.load(row["prediction"], allow_pickle=False) as saved:
            prediction = torch.from_numpy(saved["prediction"])
        metrics = probe.alignment.metric_summary(prediction, episode, original.point_metric_terms(prediction, episode))
        probe.alignment.compare_metric_records(metrics, row["metrics"])
        current = scores.setdefault(row["role"], {})
        for metric, values in metrics.items():
            current.setdefault(f'{row["task"]}/{metric}', {})[row["repeat"]] = None if values is None else values["value"]
    with runtime():
        summary = original.summarize_state(probe.read(probe.SOURCE / "v8/result.json"), rows)
    probe.alignment.compare_metric_records(summary, result["states"]["v8"])
    values = {role: {slot: [records[repeat] for repeat in (0, 1)] for slot, records in tasks.items()} for role, tasks in scores.items()}
    decisions = {role: finite_decision(values["before"], values[role]) for role in probe.SCOPES}
    verify_hashes(hashes)
    verify(output)
    write_json(output / "audit.json", {"complete": True, "prediction_grids": 84, "artifact_sha256": hashes,
               "cuda_initialized": False, "research_selection_tolerance": 0, "finite_decisions": decisions,
               "trained_checkpoint_created": False, "goal_success_not_assessed": True})
    probe.status(output, "complete", finite_feasible={role: decision["finite_feasible"] for role, decision in decisions.items()})


def wait(output):
    verify(output)
    if probe.read(output / "status.json")["state"] != "registered":
        raise ValueError("Never restart a started waiter")
    while True:
        state = probe.read(PIPELINE / "status.json")
        if state["state"] in {"complete", "failed", "interrupted"}:
            break
        os.kill(state["pid"], 0)
        probe.status(output, "waiting_for_v13", predecessor_pid=state["pid"], predecessor_stage=state["state"])
        time.sleep(30)
    run(output)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with (output / "audit.log").open("x") as stream:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "audit", "--output", str(output)],
                       stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "wait", "run", "audit"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        register(output)
    else:
        lock_name = "wait.lock" if arguments.stage == "wait" else "run.lock"
        with (output / lock_name).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"wait": wait, "run": run, "audit": audit}[arguments.stage](output)
            except BaseException as error:
                probe.status(output, "failed", error=repr(error))
                raise
