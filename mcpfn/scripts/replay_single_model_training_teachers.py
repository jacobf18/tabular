"""Independently replay every training teacher grid without duplicating identical cached bytes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import sha256, write_json
import single_model_structured_teacher as policy
from tabimpute.streamed_inference import streamed_point_predictions
import train_single_matrixformer_structured_teacher as training
from train_streamed_large_matrices import append_json, wait_for_gpu


OUTPUT = training.ROOT / "structured_teacher_v20_teacher_replay"
POLICY = ("Independently load all three original frozen checkpoints and replay every registered training teacher "
          "grid from its cached synthetic input. Require full-grid float32 bitwise equality, never an error "
          "tolerance or routing-label-only test. Preserve the identical replay bytes in the original cached "
          "teacher tensor rather than writing redundant full grids. Record each independently computed grid "
          "digest, teacher identity and task hash; independently CPU-check all1792 records against those cached "
          "bytes. No training, candidate/validation evaluation, selection, confirmation or real benchmark.")


def read(path):
    return json.loads(Path(path).read_text())


def status(output, state, **fields):
    write_json(output / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def tensor_digest(tensor):
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError("Require the complete finite CPU float32 teacher grid")
    digest = hashlib.sha256(json.dumps({"dtype": "float32", "shape": list(tensor.shape)}, sort_keys=True).encode())
    digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def register(output, pipeline_config):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Reserve the independent training-teacher replay once")
    training.verify_registration(training.OUTPUT)
    inputs = [training.OUTPUT / "config.json", training.OUTPUT / "registration.json", pipeline_config]
    output.mkdir()
    write_json(output / "config.json", {"policy": POLICY, "run": str(training.OUTPUT), "tasks": 1792,
               "input_sha256": {str(path): sha256(path) for path in inputs},
               "source_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                   str(Path(policy.__file__).resolve()): sha256(policy.__file__)}})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "predictions_performed": False})
    status(output, "registered")


def verify(output):
    config = read(output / "config.json")
    if (config["policy"] != POLICY or config["run"] != str(training.OUTPUT) or config["tasks"] != 1792
            or read(output / "registration.json")["config_sha256"] != sha256(output / "config.json")):
        raise ValueError("Preserve the registered full-training teacher replay")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return training.verify_registration(training.OUTPUT)


def require_finished_training(config):
    terminal = read(training.OUTPUT / "status.json")
    if (terminal["state"] != "training_complete" or terminal["step"] != training.STEPS
            or Path(f'/proc/{terminal["pid"]}').exists() or terminal["pid"] == os.getpid()):
        raise ValueError("Teacher replay requires all64 attempts complete and a separate process")
    if sum(len(batch["tasks"]) for batch in config["batches"]) != 1792:
        raise ValueError("Never replace the full1792-target replay with a sample")
    return terminal


def manifests(config):
    for batch in config["batches"]:
        path = training.OUTPUT / "training_batches" / f'step_{batch["step"]:03d}' / "manifest.json"
        manifest = read(path)
        if manifest["step"] != batch["step"] or len(manifest["tasks"]) != 28:
            raise ValueError("Require every original full training batch")
        for task, logged in zip(batch["tasks"], manifest["tasks"]):
            expected = path.parent / f'task_{task["index"]:02d}.pt'
            if logged["artifact"] != str(expected) or sha256(expected) != logged["sha256"]:
                raise ValueError("Original cached task identity or bytes changed")
            cached = torch.load(expected, map_location="cpu", weights_only=False, mmap=True)
            training.preflight.task_audit.check_task(cached, task, logged)
            role = training.preflight.check_routing(cached, task, config)
            yield batch["step"], path, task, logged, cached, role


@torch.no_grad()
def replay(output):
    config = verify(output)
    terminal = require_finished_training(config)
    if read(output / "status.json")["state"] != "registered" or (output / "grids.jsonl").exists():
        raise ValueError("Never restart a teacher replay")
    wait_for_gpu(output, True, False)
    status(output, "loading_original_teachers")
    model, optimizer, criterion, teachers, sampler = training.original.alignment.mixture.original_setup(config["base"])
    del optimizer, sampler
    models = {"initialization": model.eval().requires_grad_(False), "general": teachers[0], "specialist": teachers[1]}
    states = {role: policy.verify_model(current, config["teacher_identities"][role]) for role, current in models.items()}
    count = 0
    for step, manifest, task, logged, cached, role in manifests(config):
        episode = training.original.alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
        with compute_context(episode.source.device):
            prediction = streamed_point_predictions(models[role], episode.source, criterion, **policy.STREAM_OPTIONS)
        target = prediction.detach().float().cpu()
        if not torch.equal(target, cached["teacher"]):
            raise ValueError("Independent original-checkpoint prediction differs from cached teacher targets")
        append_json(output / "grids.jsonl", {"step": step, "index": task["index"], "task": task["name"],
            "role": role, "identity": config["teacher_identities"][role], "manifest": str(manifest),
            "manifest_sha256": sha256(manifest), "artifact": logged["artifact"], "artifact_sha256": logged["sha256"],
            "shape": list(target.shape), "dtype": "float32", "prediction_sha256": tensor_digest(target),
            "bitwise_equal": True, "identical_replay_bytes_preserved_in": logged["artifact"], "tensor_key": "teacher"})
        count += 1
        status(output, "replaying", completed=count, expected=1792, step=step)
    if count != 1792:
        raise ValueError("Replay omitted training grids")
    write_json(output / "result.json", {"complete": True, "full_grids": count, "model_states": states,
               "training_pid": terminal["pid"], "replay_pid": os.getpid(), "grids_sha256": sha256(output / "grids.jsonl"),
               "config_sha256": sha256(output / "config.json"), "full_grid_bitwise_equal": True,
               "training_or_candidate_evaluation_performed": False, "goal_success_not_assessed": True})
    verify(output)
    status(output, "awaiting_independent_audit")


def check_row(row, step, manifest, task, logged, cached, role, config):
    expected = {"step": step, "index": task["index"], "task": task["name"], "role": role,
                "identity": config["teacher_identities"][role], "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                "artifact": logged["artifact"], "artifact_sha256": logged["sha256"], "shape": list(cached["teacher"].shape),
                "dtype": "float32", "prediction_sha256": tensor_digest(cached["teacher"]), "bitwise_equal": True,
                "identical_replay_bytes_preserved_in": logged["artifact"], "tensor_key": "teacher"}
    if row != expected:
        raise ValueError("Every independently replayed grid must bind its original task, teacher and exact tensor bytes")


def audit(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Teacher replay audit must be CPU only")
    config = verify(output)
    terminal = require_finished_training(config)
    if read(output / "status.json")["state"] != "awaiting_independent_audit" or (output / "audit.json").exists():
        raise ValueError("Audit the complete original teacher replay once")
    report = read(output / "result.json")
    if (not report["complete"] or report["full_grids"] != 1792 or not report["full_grid_bitwise_equal"]
            or report["training_pid"] != terminal["pid"] or report["replay_pid"] == terminal["pid"]
            or report["training_or_candidate_evaluation_performed"]
            or report["config_sha256"] != sha256(output / "config.json")):
        raise ValueError("Require complete separate-process teacher replay evidence")
    expected_states = {}
    for role, identity in config["teacher_identities"].items():
        original = torch.load(identity["checkpoint"], map_location="cpu", weights_only=False, mmap=True)["model"]
        expected_states[role] = {**identity, "bitwise_equal": True, "state_tensors": len(original)}
    if report["model_states"] != expected_states:
        raise ValueError("Teacher replay did not bind all original checkpoint states")
    rows = [json.loads(line) for line in (output / "grids.jsonl").read_text().splitlines()]
    if len(rows) != 1792:
        raise ValueError("Require every original grid record, including all training attempts")
    hashes = {str(output / name): sha256(output / name) for name in ("config.json", "registration.json", "result.json")}
    hashes[str(output / "grids.jsonl")] = report["grids_sha256"]
    counts = {role: 0 for role in config["teacher_identities"]}
    for row, (step, manifest, task, logged, cached, role) in zip(rows, manifests(config)):
        check_row(row, step, manifest, task, logged, cached, role, config)
        hashes[str(manifest)] = row["manifest_sha256"]
        hashes[logged["artifact"]] = logged["sha256"]
        counts[role] += 1
    if counts != {"initialization": 128, "general": 1536, "specialist": 128}:
        raise ValueError("Preserve all legacy, structured and rating teacher targets")
    hashes.update({value["checkpoint"]: value["sha256"] for value in config["teacher_identities"].values()})
    verify_hashes(hashes)
    write_json(output / "audit.json", {"complete": True, "full_grids": 1792, "role_counts": counts,
               "full_grid_bitwise_equal": True, "artifact_sha256": hashes, "cuda_initialized": False,
               "duplicate_target_bytes_written": False, "goal_success_not_assessed": True})
    status(output, "complete")


def verify_audit(output):
    verify(output)
    audit = read(output / "audit.json")
    if (read(output / "status.json")["state"] != "complete" or not audit["complete"] or audit["full_grids"] != 1792
            or audit["role_counts"] != {"initialization": 128, "general": 1536, "specialist": 128}
            or not audit["full_grid_bitwise_equal"] or audit["cuda_initialized"]):
        raise ValueError("Require independently audited full-training teacher provenance before qualification")
    verify_hashes(audit["artifact_sha256"])
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("replay", "audit"), required=True)
    arguments = parser.parse_args()
    try:
        {"replay": replay, "audit": audit}[arguments.stage](OUTPUT)
    except BaseException as error:
        status(OUTPUT, "failed", error=repr(error))
        raise
