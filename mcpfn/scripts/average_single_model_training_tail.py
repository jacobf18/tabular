"""Prospectively average future training states into one ordinary network checkpoint."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from recsys_evaluation_common import REPO, sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def verify(hashes):
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Registered input changed: {path}")


class WeightAverage:
    def __init__(self):
        self.count = 0
        self.mean = {}
        self.dtypes = {}

    def add(self, state):
        if not state or any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Require a finite nonempty state")
        if self.count:
            if tuple(state) != tuple(self.mean):
                raise ValueError("Checkpoint keys differ")
            for name, value in state.items():
                if value.shape != self.mean[name].shape or value.dtype != self.dtypes[name]:
                    raise ValueError("Checkpoint tensor inventory differs")
                if not value.is_floating_point() and not torch.equal(value, self.mean[name]):
                    raise ValueError("Cannot average inconsistent nonfloating state")
        self.count += 1
        for name, value in state.items():
            if self.count == 1:
                self.dtypes[name] = value.dtype
                self.mean[name] = value.double().clone() if value.is_floating_point() else value.clone()
            elif value.is_floating_point():
                self.mean[name].add_(value.double() - self.mean[name], alpha=1. / self.count)

    def state(self):
        if not self.count:
            raise ValueError("No contributing checkpoints")
        return {name: value.to(self.dtypes[name]).clone() for name, value in self.mean.items()}


def load_stable(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    return payload, digest.hexdigest()


def require_live(status):
    if status["state"] in ("failed", "interrupted"):
        raise RuntimeError(f"Producer stopped: {status['state']}; never restart it from this watcher")
    if status["state"] == "training_complete":
        return
    try:
        os.kill(status["pid"], 0)
        process_state = Path(f"/proc/{status['pid']}/stat").read_text().split(") ", 1)[1].split()[0]
        if process_state == "Z":
            raise ProcessLookupError()
    except (ProcessLookupError, FileNotFoundError):
        raise RuntimeError("Producer handle is missing or terminal; do not trust a stale status") from None


def prepare(producer, output):
    base = read(producer / "config.json")
    status = read(producer / "status.json")
    steps = list(range(300, 401, 10))
    if base["steps"] != 400 or status.get("step", 0) >= steps[0] or status["state"] != "training":
        raise ValueError("Register before update 300 of the live 400-update continuation")
    require_live(status)
    verify(base["input_sha256"])
    verify(base["source_sha256"])
    verify(base["teacher_config_sha256"])
    status = read(producer / "status.json")
    require_live(status)
    if status.get("step", 0) >= steps[0] or status["state"] != "training":
        raise ValueError("The prospective registration window closed during verification")
    source = Path(__file__).resolve()
    config = {"model": base["model"], "producer": str(producer), "producer_config_sha256": sha256(producer / "config.json"),
              "producer_inputs": base["input_sha256"], "producer_sources": base["source_sha256"],
              "producer_teacher_configs": base["teacher_config_sha256"],
              "source_sha256": {str(source): sha256(source),
                                 str(REPO / "scripts/recsys_evaluation_common.py"): sha256(REPO / "scripts/recsys_evaluation_common.py")},
              "registered_at_producer_step": status["step"], "registered_unix": time.time(), "steps": steps,
              "rule": "Uniform arithmetic mean of every complete state at updates 300,310,...,400; FP64 accumulator, original output dtypes",
              "training_performed_here": False, "real_benchmark_inputs_used": False,
              "selection_performed": False, "inference": "Exactly one ordinary parameter set, not averaged predictions or multiple models",
              "rationale": "Synthetic validation oscillates between updates 150 and 200; test a prospectively fixed training-tail weight average",
              "evaluation": "Unqualified derived candidate until newly registered synthetic confirmation and full frozen evaluation; never replace the producer selector",
              "step_semantics": "Last contributing producer update, not new optimizer updates in this process",
              "source_retention": "The producer overwrites its latest checkpoint; record full bytes hashes and training provenance, retaining only the average, not eleven full snapshots"}
    write_json(output / "config.json", config)
    shutil.copy2(source, output / source.name)
    shutil.copy2(producer / "config.json", output / "producer_config.json")
    write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    return config


def execute(config, output):
    producer = Path(config["producer"])
    average, receipts = WeightAverage(), []
    for expected in config["steps"]:
        while True:
            status = read(producer / "status.json")
            require_live(status)
            if status.get("step", 0) >= expected:
                break
            if status["state"] == "training_complete":
                raise RuntimeError("Producer finished before a required averaging update")
            write_json(output / "status.json", {"state": "waiting_for_step", "expected_step": expected,
                                                "producer_step": status.get("step"), "collected": len(receipts),
                                                "pid": os.getpid(), "updated_unix": time.time()})
            time.sleep(10)
        payload, digest = load_stable(producer / "latest_checkpoint.pth")
        if payload["step"] != expected:
            raise ValueError(f"Missed required update {expected}; found {payload['step']}; never average a partial window")
        if payload["config_sha256"] != config["producer_config_sha256"]:
            raise ValueError("Producer checkpoint/config mismatch")
        state = payload["model"]
        del payload
        average.add(state)
        del state
        receipt = {"step": expected, "checkpoint_sha256": digest, "captured_unix": time.time()}
        receipts.append(receipt)
        write_json(output / "receipts.json", {"snapshots": receipts, "expected_steps": config["steps"]})
        print(json.dumps(receipt), flush=True)
    while True:
        status = read(producer / "status.json")
        require_live(status)
        if status["state"] == "training_complete":
            break
        time.sleep(10)
    if status["step"] != config["steps"][-1] or average.count != len(config["steps"]):
        raise ValueError("Incomplete registered averaging window")
    verify(config["source_sha256"])
    verify(config["producer_inputs"])
    verify(config["producer_sources"])
    verify(config["producer_teacher_configs"])
    verify({str(producer / "config.json"): config["producer_config_sha256"]})
    if shutil.disk_usage(output).free < 16 * 1024**3:
        raise OSError("Preserve at least 15 GiB after saving the single averaged checkpoint")
    checkpoint = output / "averaged_checkpoint.pth"
    temporary = checkpoint.with_suffix(".pth.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save({"model": average.state(), "step": config["steps"][-1], "epoch": config["steps"][-1],
                        "source_updates": config["steps"], "config_sha256": sha256(output / "config.json"),
                        "receipts_sha256": sha256(output / "receipts.json"), "gradient_updates_performed_here": 0,
                        "derived_checkpoint": True, "step_semantics": config["step_semantics"]}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(output / "result.json", {"complete": True, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                                        "snapshots": len(receipts), "source_updates": config["steps"], "single_parameter_set": True,
                                        "training_performed_here": False, "synthetic_confirmation_performed": False,
                                        "benchmark_evaluation_performed": False, "goal_success_not_assessed": True,
                                        "config_sha256": sha256(output / "config.json"), "receipts_sha256": sha256(output / "receipts.json")})
    write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
                raise RuntimeError("Require CUDA_VISIBLE_DEVICES='' for CPU-only parameter averaging")
            torch.set_num_threads(2)
            config = prepare(arguments.producer.resolve(), output)
            execute(config, output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
