"""Replay V34's failed task at its initializer to diagnose the GPU allocation cap."""

import argparse
from copy import deepcopy
import fcntl
import hashlib
from pathlib import Path
import shutil
import time

import torch

import train_single_matrixformer_full_network as training
from recsys_evaluation_common import sha256


OUTPUT = training.ROOT / "full_network_memory_probe_v1"
MEMORY_FRACTION = .9
POLICY = ("Infrastructure-only replay of the already consumed V34 attempt9 legacy_draw_0 task at the "
          "unchanged V29 initializer, not the lost step8 state. Increase only the allocator fraction from.8 "
          "to.9; keep ordinary setup, teachers, sampling, streamed composite and point backwards, precision "
          "and238 optimizer coordinates unchanged. Capture all three gradients and require exact complete "
          "task/teacher digest equality. Reserve four full GPU parameter vectors as conservative optimizer/"
          "snapshot headroom. No optimizer step, checkpoint write, new synthetic seed, selection, validation, "
          "confirmation or real evaluation. Do not restart V34 or this probe, and do not claim feasibility "
          "of unmeasured tasks or full historical trajectory replay. Preserve every original artifact and "
          "18/6/50 GiB disk reserves. A future scientific trial requires separate registration and fresh seeds.")


def validate_failure(current, coordinator, records, capsule, latest):
    if (current.get("state") != "failed" or coordinator.get("state") != "failed"
            or "OutOfMemoryError" not in current.get("error", "")
            or "111.84 GiB allowed" not in current.get("error", "")
            or len(records) != 8
            or any(record.get("step") != index or record.get("accepted") is not True
                   or record.get("accepted_updates") != index for index, record in enumerate(records, 1))):
        raise ValueError("Require the actual stopped V34 allocation failure after eight recorded accepts")
    training.compact_cache.validate_capsule(capsule)
    expected = training.batches()[8]["tasks"][0]
    if (capsule["task"] != expected or capsule["scaling"] is not None
            or expected["specification"]["shape"] != [2000, 1000]
            or expected["specification"]["seed"] != 811123301
            or expected["gradient_seed"] != 812123301
            or latest.get("step") != 0 or latest.get("gradient_updates_performed") != 0):
        raise ValueError("Replay only the already consumed first task; never invent a persisted step8 model")


def require_capacity():
    for volume, reserve in training.diagnostics.RESERVES.items():
        path = training.ROOT if volume == "checkpoint" else Path("/root") if volume == "root" else Path("/mnt/matrixformer_scratch")
        if shutil.disk_usage(path).free < reserve + 1024**2:
            raise OSError(f"Preserve the full {volume} reserve and one MiB metadata allowance")


def register(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Never replace or relocate a begun memory probe")
    require_capacity()
    previous = training.read(training.OUTPUT / "config.json")
    current = training.read(training.OUTPUT / "status.json")
    coordinator_path = training.ROOT / "full_network_v34_pipeline"
    coordinator = training.read(coordinator_path / "status.json")
    for record in (current, coordinator):
        if Path(f'/proc/{record["pid"]}').exists():
            raise ValueError("Stop the original failed owners before an allocation probe")
    records = [training.json.loads(line) for line in (training.OUTPUT / "training.jsonl").read_text().splitlines()]
    capsule_path = training.OUTPUT / "training_batches/step_009/task_00.pt"
    capsule = torch.load(capsule_path, map_location="cpu", weights_only=False)
    latest_path = training.OUTPUT / "latest_checkpoint.pth"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False, mmap=True)
    validate_failure(current, coordinator, records, capsule, latest)
    forbidden = [training.OUTPUT / name for name in ("selected_checkpoint.pth", "best_validation_checkpoint.pth", "validation_10.json")]
    forbidden.extend(training.ROOT / name for name in ("full_network_v34_benchmarks", "full_network_v34_review"))
    if any(path.exists() for path in forbidden):
        raise ValueError("Preserve the actual absent trained boundary, selection and real evaluation")
    if (previous["base"]["memory_fraction"] != .8 or sha256(training.PARENT) != training.PARENT_SHA256
            or sha256(training.OUTPUT / "config.json") != training.read(training.OUTPUT / "registration.json")["config_sha256"]):
        raise ValueError("Bind the original memory limit and unchanged initializer")
    paths = [capsule_path, latest_path]
    paths.extend(training.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "training.jsonl", "train_and_audit.log"))
    paths.extend(coordinator_path / name for name in ("config.json", "registration.json", "status.json"))
    sources = [Path(__file__).resolve(), training.REPO / "tests/test_single_model_full_network_memory.py"]
    config = {"policy": POLICY, "memory_fraction": MEMORY_FRACTION,
              "parent_checkpoint": str(training.PARENT), "parent_sha256": training.PARENT_SHA256,
              "historical_task_replay": capsule, "task_capsule": str(capsule_path),
              "failed_training_config": str(training.OUTPUT / "config.json"),
              "input_sha256": {**previous["input_sha256"], **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**previous["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "gpu_headroom_vectors": 4, "training_updates_performed": 0,
              "new_seed_consumption": False, "candidate_selection_performed": False}
    training.verify_hashes(config["input_sha256"])
    training.verify_hashes(config["source_sha256"])
    output.mkdir()
    training.write_json(output / "config.json", config)
    training.write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time()})
    training.status(output, "registered")


def execution_config(config):
    if (config["policy"] != POLICY or config["memory_fraction"] != MEMORY_FRACTION
            or config["gpu_headroom_vectors"] != 4 or config["parent_sha256"] != training.PARENT_SHA256
            or config["parent_checkpoint"] != str(training.PARENT)
            or config["training_updates_performed"] != 0
            or config["new_seed_consumption"] is not False or config["candidate_selection_performed"] is not False):
        raise ValueError("Keep the memory-only probe separate from scientific training and selection")
    previous = training.read(Path(config["failed_training_config"]))
    updated = deepcopy(previous)
    if updated["base"]["memory_fraction"] != .8:
        raise ValueError("Only the registered original.8 allocation limit may be changed")
    updated["base"]["memory_fraction"] = config["memory_fraction"]
    return updated


def gradient_records(gradients, entries):
    if (gradients.shape != (3, training.TRAINABLE_PARAMETERS)
            or [entry["kind"] for entry in entries] != ["composite", "rmse", "common_nrmse"]):
        raise ValueError("Require every original full-coordinate composite and defined point gradient")
    records = []
    for index, entry in enumerate(entries):
        gradient = gradients[index]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Every full-network gradient coordinate must be finite")
        records.append({**entry, "parameters": gradient.numel(),
                        "sha256": hashlib.sha256(gradient.numpy().tobytes()).hexdigest(),
                        "norm": float(torch.linalg.vector_norm(gradient.double()))})
    return records


def capture(output, config, updated):
    capsule = config["historical_task_replay"]
    with training.teacher_policy.runtime(updated["base"], output, updated["teacher_identities"]), training.compact_cache.runtime():
        model, view, optimizer, criterion, teachers, sampler, initial = training.setup(updated)
        if len(list(view.parameters())) != 238 or sum(parameter.numel() for parameter in view.parameters()) != 77998496:
            raise ValueError("Keep every original trainable tensor")
        headroom = torch.zeros((config["gpu_headroom_vectors"], training.TRAINABLE_PARAMETERS), device="cuda")
        gradients = torch.empty((3, training.TRAINABLE_PARAMETERS), dtype=torch.float32, device="cpu")
        torch.cuda.reset_peak_memory_stats()
        directory = output / "capture"
        directory.mkdir()
        started = time.monotonic()
        with training.full_network.capture_coordinates(model, view):
            entries, records, values = training.compact_cache.capture_batch(output, directory,
                {"step": 9, "tasks": [deepcopy(capsule["task"])]}, model, optimizer, criterion, teachers, sampler, gradients)
        torch.cuda.synchronize()
        generated = torch.load(directory / "task_00.pt", map_location="cpu", weights_only=False)
        if generated != capsule or records[0]["name"] != "legacy_draw_0":
            raise ValueError("The complete historical task and teacher digest must replay exactly")
        frozen = training.full_network.frozen_state_equal(initial, model, view)
        if frozen["changed_trainable_tensors"] or frozen["frozen_tensors"] != 16 or optimizer.state:
            raise ValueError("The allocation diagnostic must not train or alter any original state")
        return {"complete": True, "policy": POLICY, "memory_fraction": MEMORY_FRACTION,
                "parent_sha256": config["parent_sha256"], "task_content_sha256": capsule["content_sha256"],
                "historical_task_and_teacher_digest_equal": True, "reconstructed_step8_state": False,
                "gradient_records": gradient_records(gradients, entries), "point_before": values,
                "task_metadata": records[0]["metadata"],
                "gpu_name": torch.cuda.get_device_properties(0).name,
                "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "seconds": time.monotonic() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "headroom_bytes": headroom.numel() * headroom.element_size(), "frozen": frozen,
                "training_updates_performed": 0, "optimizer_stepped": False,
                "checkpoint_written": False, "validation_evaluated": False, "real_benchmarks_evaluated": False,
                "all_training_tasks_memory_qualified": False,
                "config_sha256": sha256(output / "config.json")}


def run(output):
    if output != OUTPUT or training.read(output / "status.json")["state"] != "registered":
        raise ValueError("Never restart a begun allocation diagnostic")
    config = training.read(output / "config.json")
    if sha256(output / "config.json") != training.read(output / "registration.json")["config_sha256"]:
        raise ValueError("The registered memory diagnostic changed")
    training.verify_hashes(config["input_sha256"])
    training.verify_hashes(config["source_sha256"])
    updated = execution_config(config)
    require_capacity()
    training.status(output, "starting_memory_probe")
    training.wait_for_gpu(output, True, False)
    result = capture(output, config, updated)
    training.write_json(output / "result.json", result)
    training.status(output, "complete", result_sha256=sha256(output / "result.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("register", "run"))
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if training.read(OUTPUT / "status.json")["state"] != "registered":
                raise ValueError("Never change the terminal status of a begun diagnostic")
            try:
                run(OUTPUT)
            except BaseException as error:
                training.status(OUTPUT, "failed", error=repr(error))
                raise
