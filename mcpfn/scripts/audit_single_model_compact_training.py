"""Independently check compact replay records and export small regeneration capsules."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import probe_single_model_compact_training as probe


OUTPUT = probe.training.ROOT / "teacher_continuation_v22_compact_replay_probe_v1_audit"
POLICY = ("CPU-only independent check of every successful compact replay record against all original "
    "1792 cached tasks and teacher tensors. Recompute typed content fingerprints and verify exactly128 "
    "replayed rating states. Export only task specifications, expected complete-task digests and the four "
    "small normalization vectors; never copy source/target/mask/teacher grids into capsules. Preserve all "
    "old evidence. This certifies the recorded diagnostic and capsule contents, not student autograd, "
    "a future optimizer trajectory, model qualification or a newly registered training procedure.")


def make_capsule(step, task, digest, scaling):
    ratings = task["specification"]["domain"] == "recsys"
    if ratings != (scaling is not None):
        raise ValueError("Only rating specifications carry compact normalization state")
    if scaling is not None:
        probe.compact.validate_scaling(scaling)
    return {"step": step, "task": deepcopy(task), "content_sha256": digest,
            "scaling": None if scaling is None else {name: scaling[name].clone() for name in probe.compact.NAMES}}


def check_record(row, step, task, path, cached, role):
    scaling = cached["episode"]["scaling"]
    capsule = make_capsule(step, task, probe.content_digest(cached), scaling)
    size = 0 if scaling is None else sum(value.numel() * value.element_size() for value in scaling.values())
    expected = {"step": step, "index": task["index"], "name": task["name"], "role": role,
        "artifact": str(path), "artifact_sha256": sha256(path), "exact_episode_and_teacher": True,
        "content_sha256": capsule["content_sha256"], "rating_state_replayed": scaling is not None,
        "scaling_bytes": size, "scaling_content_sha256": probe.content_digest(scaling)}
    probe.require_equal(row, expected, "independently checked compact replay record")
    return capsule, size


def audit(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Audit compact replay with CPU-only access to saved evidence")
    if output.exists():
        raise FileExistsError("Never overwrite or repeat this audit")
    original_config = probe.verify()
    status = probe.read(probe.OUTPUT / "status.json")
    result = probe.read(probe.OUTPUT / "result.json")
    if (status["state"] != "complete" or Path(f'/proc/{status["pid"]}').exists()
            or result["complete"] is not True or result["tasks"] != 1792
            or result["role_counts"] != {"initialization": 128, "general": 1536, "specialist": 128}
            or result["all_episode_and_teacher_bytes_match"] is not True or result["rating_contexts_replayed"] != 128
            or result["seed_only_regeneration"] is not False or result["prior_seed_only_failure_preserved"] is not True
            or any(result[key] is not False for key in ("student_gradients_replayed", "training_performed", "model_saved",
                "cached_artifacts_deleted", "confirmation_repeated", "real_benchmarks_evaluated", "independent_audit_of_this_diagnostic"))
            or result["records_sha256"] != sha256(probe.OUTPUT / "tasks.jsonl")):
        raise ValueError("Require the complete stopped exact compact-state diagnostic, not a prefix")
    rows = [json.loads(line) for line in (probe.OUTPUT / "tasks.jsonl").read_text().splitlines()]
    if len(rows) != 1792:
        raise ValueError("Every original task must have exactly one replay record")
    paths = [probe.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "result.json", "tasks.jsonl")]
    hashes = {str(path): sha256(path) for path in paths}
    capsules, vector_bytes, task_bytes, rating_contexts = [], 0, 0, 0
    for batch in original_config["batches"]:
        manifest_path = probe.training.original_manifest(batch["step"])
        manifest = probe.read(manifest_path)
        hashes[str(manifest_path)] = sha256(manifest_path)
        for task, logged in zip(batch["tasks"], manifest["tasks"]):
            path = manifest_path.parent / f'task_{task["index"]:02d}.pt'
            if logged["artifact"] != str(path) or logged["sha256"] != sha256(path):
                raise ValueError("Preserve every original cached artifact and its audited identity")
            cached = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            probe.training.preflight.task_audit.check_task(cached, task, logged)
            role = probe.training.preflight.check_routing(cached, task, original_config)
            capsule, size = check_record(rows[len(capsules)], batch["step"], task, path, cached, role)
            capsules.append(capsule)
            vector_bytes += size
            rating_contexts += int(capsule["scaling"] is not None)
            task_bytes += path.stat().st_size
            hashes[str(path)] = logged["sha256"]
    if len(capsules) != 1792 or rating_contexts != 128 or vector_bytes != result["normalization_vector_bytes"]:
        raise ValueError("Retain every task and account for all compact-state bytes")
    sources = [Path(__file__).resolve(), Path(probe.__file__).resolve(), Path(probe.compact.__file__).resolve(),
               REPO / "tests/test_single_model_compact_training_audit.py"]
    config = {"policy": POLICY, "input_sha256": hashes, "source_sha256": {str(path): sha256(path) for path in sources}}
    verify_hashes(hashes)
    output.mkdir()
    write_json(output / "config.json", config)
    payload = {"capsules": capsules, "teacher_identities": original_config["teacher_identities"],
               "probe_config_sha256": sha256(probe.OUTPUT / "config.json"), "policy": POLICY}
    digest = probe.training.original.save_tensor(payload, output / "capsules.pt")
    restored = torch.load(output / "capsules.pt", map_location="cpu", weights_only=False, mmap=True)
    probe.require_equal(restored, payload, "exact capsule serialization roundtrip")
    verify_hashes(hashes)
    verify_hashes(config["source_sha256"])
    write_json(output / "audit.json", {"complete": True, "tasks": len(capsules), "rating_contexts": rating_contexts,
        "normalization_vector_bytes": vector_bytes, "capsule_file_bytes": (output / "capsules.pt").stat().st_size,
        "original_task_artifact_bytes": task_bytes, "capsules_sha256": digest,
        "artifact_sha256": {**hashes, str(output / "capsules.pt"): digest}, "config_sha256": sha256(output / "config.json"),
        "cuda_initialized": False, "old_artifacts_deleted": False, "full_matrix_grids_in_capsules": False,
        "student_autograd_replayed": False, "future_training_registered": False, "goal_success_not_assessed": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    audit(arguments.output.resolve())
