"""Preserve two audited raw endpoints while restoring durable capacity for the full paired trial."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import FunctionType

import torch

import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = archive.ROOT
ROOT_ARCHIVE = REPO / "runs/single_model_goal_20261005/paired_preflight_archive_v1"
CHECKPOINT_RESERVE = 18 * 1024**3
ROOT_RESERVE = 3 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def plan(kind):
    if kind == "paired-preflight":
        source = ROOT / "paired_decoder_preflight_v1/raw_update.pt"
        audit_path = source.parent / "audit.json"
        terminal, audit = read(source.parent / "status.json"), read(audit_path)
        if (terminal["state"] != "complete" or not audit["complete"] or audit["training_tasks"] != 28
                or audit["prediction_grids"] != 56 or audit["raw_endpoint"]["gradient_vectors"] != 84):
            raise ValueError("Require the completed28-task paired preflight and independent audit")
        inputs, output, group = [audit_path, source.parent / "status.json"], ROOT / "paired_preflight_capacity_archive_v1", ROOT_ARCHIVE / "group_00"
    elif kind == "finite-endpoint":
        source = ROOT / "finite_guarded_v13/raw_final_update.pt"
        audit_path = ROOT / "finite_guarded_v13_audit/audit.json"
        terminal, audit = read(source.parent / "status.json"), read(audit_path)
        replay_path = ROOT / "finite_guarded_v13_endpoint_replay/audit.json"
        replay = read(replay_path)
        if (terminal["state"] != "training_complete" or not audit["complete_training_audit"] or audit["through_step"] != 50
                or audit["training_episodes"] != 700 or not replay["complete"] or replay["endpoint_grids"] != 28):
            raise ValueError("Require the original complete V13 training and independent replay")
        verify_hashes(replay["artifact_sha256"])
        output = ROOT / "finite_endpoint_capacity_archive_v1"
        inputs, group = [audit_path, source.parent / "status.json", replay_path], output / "group_00"
    else:
        raise ValueError("Only the two explicit completed raw endpoints may be archived")
    if Path(f'/proc/{terminal["pid"]}').exists():
        raise ValueError("Never archive a live writer")
    expected = audit["artifact_sha256"][str(source)]
    if source.is_symlink() or not source.is_file() or sha256(source) != expected:
        raise ValueError("Require the audited regular original with identical bytes")
    return source, {"source": str(source), "sha256": expected}, inputs, output, group


def publish_staged(directory, record, staging):
    if staging.is_symlink() or not staging.is_file() or not staging.is_relative_to(Path("/dev/shm")):
        raise ValueError("Require the complete owned RAM-staged squashfs image")
    compressed, maximum = staging.stat().st_size, Path(record["source"]).stat().st_size
    if compressed >= maximum or shutil.disk_usage(ROOT).free - compressed < CHECKPOINT_RESERVE:
        raise OSError("Require net savings and18 GiB free while the durable image and original coexist")
    directory.mkdir()
    image, mountpoint = directory / "gradients.squashfs", directory / "files"
    mountpoint.mkdir()
    digest = sha256(staging)
    shutil.copyfile(staging, image)
    with image.open("rb") as stream:
        os.fsync(stream.fileno())
    if sha256(image) != digest or sha256(staging) != digest:
        raise ValueError("Complete staged and durable compressed bytes must match")
    archive.sync_directory(directory)
    mounted = archive.mount_unit(image, mountpoint)
    destination = mountpoint / Path(record["source"]).name
    verify_hashes({str(destination): record["sha256"], record["source"]: record["sha256"]})
    payload = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("The raw optimizer endpoint must retain mmap-readable serialization")
    del payload
    verified = {"image": str(image), "image_sha256": digest, "image_bytes": compressed,
                "uncompressed_bytes": maximum, "all_original_hashes_verified": True,
                "torch_mmap_readable": True, "records": [record], **mounted}
    write_json(directory / "verified_before_replacement.json", verified)
    with (directory / "verified_before_replacement.json").open("rb") as stream:
        os.fsync(stream.fileno())
    archive.sync_directory(directory)
    replacement = archive.replace_with_verified_link(record["source"], destination, record["sha256"])
    write_json(directory / "audit.json", {**verified, "complete": True, "replacements": [replacement],
               "net_bytes_reclaimed": maximum - compressed})
    archive.sync_directory(directory)
    return {"group": str(directory), "audit_sha256": sha256(directory / "audit.json"),
            "net_bytes_reclaimed": maximum - compressed}


def run(kind, staging=None):
    source, record, inputs, output, group = plan(kind)
    if output.exists() or group.exists():
        raise FileExistsError("Preserve every earlier archive attempt")
    if (kind == "finite-endpoint") != (staging is not None):
        raise ValueError("Only V13 consumes the already completed RAM-staged image")
    if shutil.disk_usage(ROOT).free < CHECKPOINT_RESERVE:
        raise OSError("Preserve the unchanged checkpoint write reserve")
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(),
               REPO / "tests/test_single_model_paired_capacity.py"]
    config = {"kind": kind, "record": record, "durable_group": str(group),
              "checkpoint_minimum_free_bytes": CHECKPOINT_RESERVE, "root_minimum_free_bytes": ROOT_RESERVE,
              "input_sha256": {str(path): sha256(path) for path in inputs},
              "source_sha256": {str(path): sha256(path) for path in sources},
              "staged_image": None if staging is None else str(staging),
              "staged_image_sha256": None if staging is None else sha256(staging),
              "policy": "Lossless archival of completed audited raw evidence; paired preflight image on root retains3 GiB there, V13 image on checkpoint volume retains18 GiB there; no older archive, checkpoint or scientific protocol changes"}
    output.mkdir()
    write_json(output / "config.json", config)
    group.parent.mkdir(parents=True, exist_ok=True)
    if kind == "paired-preflight":
        compressor = FunctionType(archive.compress_group.__code__,
                                  {**archive.__dict__, "ROOT": Path("/root"), "MINIMUM_FREE": ROOT_RESERVE},
                                  "compress_on_root", archive.compress_group.__defaults__, archive.compress_group.__closure__)
        result = compressor(group, [record])
    else:
        result = publish_staged(group, record, staging)
    metadata = read(group / "audit.json")
    if not source.is_symlink() or source.resolve() != group / "files" / source.name:
        raise ValueError("Canonical bytes must resolve to their durable archive")
    hashes = {str(group / "audit.json"): result["audit_sha256"], str(source): record["sha256"],
              metadata["image"]: metadata["image_sha256"], metadata["unit"]: metadata["unit_sha256"]}
    for action in ("is-active", "is-enabled"):
        subprocess.run(["systemctl", action, "--quiet", Path(metadata["unit"]).name], check=True)
    options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(group / "files")], text=True).strip().split(",")
    if "ro" not in options:
        raise ValueError("Require the persistent read-only mount")
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    if shutil.disk_usage(ROOT).free < CHECKPOINT_RESERVE or shutil.disk_usage("/root").free < ROOT_RESERVE:
        raise OSError("Both explicitly declared durable reserves must remain intact")
    write_json(output / "independent_recheck.json", {"complete": True, "canonical_paths_and_hashes_checked": 1,
               "persistent_read_only_mounts_checked": 1, "artifact_sha256": hashes,
               "net_bytes_reclaimed_across_volumes": result["net_bytes_reclaimed"],
               "config_sha256": sha256(output / "config.json"), "registered_scientific_evidence_unchanged": True,
               "no_baseline_or_checkpoint_modified": True})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", choices=("paired-preflight", "finite-endpoint"), required=True)
    parser.add_argument("--staged-image", type=Path)
    arguments = parser.parse_args()
    run(arguments.artifact, arguments.staged_image)
