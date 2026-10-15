"""Losslessly archive V18's completed audited task tree without changing canonical paths."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = archive.ROOT
SOURCE = ROOT / "replicated_decoder_v18/training_batches"
OUTPUT = ROOT / "replicated_task_archive_v1"
MINIMUM_FREE = 18 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def tree_inventory(source, audited):
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Require the original unarchived directory")
    paths = list(source.rglob("*"))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in paths):
        raise ValueError("Never archive linked or unknown filesystem objects")
    actual = {str(path) for path in paths if path.is_file()}
    expected = {name for name in audited if Path(name).is_relative_to(source)}
    if not actual or actual != expected:
        raise ValueError("Every file must be independently audited with no missing or extra artifacts")
    return {name: audited[name] for name in sorted(actual)}


def plan():
    inputs = [ROOT / name for name in (
        "replicated_decoder_v18/status.json", "replicated_decoder_v18_audit/audit.json",
        "replicated_decoder_v18_endpoint_replay/audit.json", "replicated_decoder_v18_pipeline/status.json",
        "replicated_decoder_v18_pipeline/result.json", "replicated_decoder_v18_confirmation/audit.json")]
    terminal, audit, replay, pipeline, result, confirmation = map(read, inputs)
    if (terminal["state"] != "training_complete" or pipeline["state"] != "complete"
            or any(Path(f'/proc/{current["pid"]}').exists() for current in (terminal, pipeline))
            or not result["complete"] or result["gate"]["passed"] or not result["gate"]["confirmation_evaluated"]
            or result["campaign"] is not None or result["review"] is not None):
        raise ValueError("Require the actual completed unsuccessful V18 lifecycle, never active training")
    if (not audit["complete_training_audit"] or audit["through_step"] != 64 or audit["training_episodes"] != 1792
            or audit["prediction_grids"] != 1216 or not replay["complete"] or replay["endpoint_grids"] != 56
            or not confirmation["complete"] or confirmation["prediction_grids"] != 1136):
        raise ValueError("Require every completed independent V18 audit")
    records = tree_inventory(SOURCE, audit["artifact_sha256"])
    if (len(records) != 1856 or sum(Path(name).suffix == ".pt" for name in records) != 1792
            or sum(Path(name).name == "manifest.json" for name in records) != 64):
        raise ValueError("Preserve all1792 tasks and64 manifests")
    verify_hashes(records)
    return records, inputs


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart an archive attempt")
    records, inputs = plan()
    backup = SOURCE.with_name("training_batches.uncompressed_archive_backup")
    if backup.exists():
        raise FileExistsError("Preserve any earlier backup for explicit recovery")
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(),
               REPO / "tests/test_single_model_replicated_task_archive.py"]
    config = {"source": str(SOURCE), "records": records, "minimum_free_bytes": MINIMUM_FREE,
              "input_sha256": {str(path): sha256(path) for path in inputs},
              "source_sha256": {str(path): sha256(path) for path in sources},
              "policy": "Lossless task-tree archival after terminal training, replay and confirmation; preserve every canonical path and byte, with no checkpoint or scientific policy change"}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    image, mountpoint = OUTPUT / "tasks.squashfs", OUTPUT / "files"
    mountpoint.mkdir()
    maximum = sum(Path(name).stat().st_size for name in records)
    if shutil.disk_usage("/run").free < maximum + 4 * 1024**3:
        raise OSError("Preserve4 GiB of RAM-backed staging headroom")
    with tempfile.TemporaryDirectory(prefix="single-model-task-archive-", dir="/run") as scratch:
        staging = Path(scratch) / image.name
        with (OUTPUT / "build.log").open("x") as stream:
            subprocess.run(["mksquashfs", str(SOURCE), str(staging), "-comp", "zstd", "-Xcompression-level", "8",
                            "-b", "1048576", "-processors", "4", "-noappend", "-no-progress"],
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        compressed = staging.stat().st_size
        if compressed >= maximum or shutil.disk_usage(ROOT).free - compressed < MINIMUM_FREE:
            raise OSError("Require net savings and the unchanged18 GiB durable staging reserve")
        shutil.copyfile(staging, image)
        with image.open("rb") as stream:
            os.fsync(stream.fileno())
        if sha256(image) != sha256(staging):
            raise ValueError("Durable image must match staging bytes")
    archive.sync_directory(OUTPUT)
    mounted = archive.mount_unit(image, mountpoint)
    archived = {str(mountpoint / Path(name).relative_to(SOURCE)): digest for name, digest in records.items()}
    verify_hashes(archived)
    for name in archived:
        if Path(name).suffix == ".pt":
            payload = torch.load(name, map_location="cpu", weights_only=False, mmap=True)
            if not isinstance(payload, dict):
                raise ValueError("Every cached task must remain mmap-readable")
    del payload
    verify_hashes(records)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    verified = {"complete_before_replacement": True, "files": len(records), "image": str(image),
                "image_sha256": sha256(image), "image_bytes": compressed, "uncompressed_bytes": maximum,
                "all_original_hashes_verified": True, "all_task_payloads_mmap_readable": True, **mounted}
    write_json(OUTPUT / "verified_before_replacement.json", verified)
    with (OUTPUT / "verified_before_replacement.json").open("rb") as stream:
        os.fsync(stream.fileno())
    archive.sync_directory(OUTPUT)
    SOURCE.rename(backup)
    archive.sync_directory(SOURCE.parent)
    SOURCE.symlink_to(mountpoint, target_is_directory=True)
    archive.sync_directory(SOURCE.parent)
    verify_hashes(records)
    if not SOURCE.is_symlink() or SOURCE.resolve() != mountpoint:
        raise ValueError("Canonical task tree must resolve to the verified read-only mount")
    for action in ("is-active", "is-enabled"):
        subprocess.run(["systemctl", action, "--quiet", Path(mounted["unit"]).name], check=True)
    options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(mountpoint)], text=True).strip().split(",")
    if "ro" not in options:
        raise ValueError("Require a durable read-only archive")
    backup_records = {str(backup / Path(name).relative_to(SOURCE)): digest for name, digest in records.items()}
    if tree_inventory(backup, backup_records) != backup_records:
        raise ValueError("Never remove an unaccounted backup artifact")
    verify_hashes(backup_records)
    shutil.rmtree(backup)
    archive.sync_directory(SOURCE.parent)
    verify_hashes(records)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(OUTPUT / "independent_recheck.json", {**verified, "complete": True,
               "canonical_paths_and_hashes_checked": len(records), "persistent_read_only_mounts_checked": 1,
               "net_bytes_reclaimed": maximum - compressed, "config_sha256": sha256(OUTPUT / "config.json"),
               "registered_scientific_evidence_unchanged": True, "no_baseline_or_checkpoint_modified": True})
    print(json.dumps({"complete": True, "net_bytes_reclaimed": maximum - compressed,
                      "recheck_sha256": sha256(OUTPUT / "independent_recheck.json")}), flush=True)


if __name__ == "__main__":
    main()
