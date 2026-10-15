"""Preserve six closed diagnostics on scratch to restore V43 qualification headroom."""

import os
from pathlib import Path
import shutil
import time

import torch

import relocate_single_model_idle_diagnostics_to_root as previous
import relocate_single_model_idle_diagnostics_to_scratch as copying
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from relocate_single_model_gradient_archive import sync_directory


ROOT = previous.ROOT
OUTPUT = ROOT / "family_balanced_recovery_capacity_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/family_balanced_recovery_capacity_v1"
POLICY = ("Storage only: move the six already audited, idle root-resident nonselected diagnostics from the "
    "original idle_diagnostic_root_relocation_v1 to UUID-bound local scratch. Preserve all original serialized "
    "bytes, canonical paths, selected checkpoints, sources and adverse results. Require stopped owners and "
    "no open readers; copy/fsync/hash/mmap every file before atomic symlink publication. Restore the original "
    "6 GiB root reserve plus16 MiB qualification headroom without weakening18 GiB checkpoint or50 GiB scratch "
    "reserves. The starting root shortfall is the reason for relocation, not permission to run science below "
    "its reserve. No training, confirmation, benchmark or scientific retry. Scratch is not an external backup.")


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def check_capacity(total):
    if (shutil.disk_usage("/").free < 3 * 1024**3
            or shutil.disk_usage("/").free + total < 6 * 1024**3 + 16 * 1024**2
            or shutil.disk_usage(ROOT).free < 18 * 1024**3 + 16 * 1024**2
            or shutil.disk_usage(copying.SCRATCH).free - total < 50 * 1024**3):
        raise OSError("Restore the original reserves by preservation, never by relaxing admission")


def inventory():
    copying.require_mount()
    previous.require_stopped_owners()
    old_config_path = previous.OUTPUT / "config.json"
    old_receipt_path = previous.OUTPUT / "independent_recheck.json"
    old_config, receipt = previous.read(old_config_path), previous.read(old_receipt_path)
    if (receipt["complete"] is not True or receipt["canonical_artifacts_checked"] != 6
            or receipt["config_sha256"] != sha256(old_config_path)
            or {row["name"] for row in old_config["records"]} != set(previous.ARTIFACTS)):
        raise ValueError("Bind all six original completed diagnostic preservation records")
    verify_hashes(old_config["input_sha256"])
    verify_hashes(old_config["source_sha256"])
    inputs = {**old_config["input_sha256"], str(old_config_path): sha256(old_config_path),
              str(old_receipt_path): sha256(old_receipt_path)}
    records = []
    for old in old_config["records"]:
        canonical, source = Path(old["source"]), Path(old["destination"])
        if (not canonical.is_symlink() or canonical.resolve() != source or source.is_symlink()
                or not source.is_file() or source.stat().st_nlink != 1
                or source.stat().st_dev != Path("/").stat().st_dev
                or source.stat().st_size != old["bytes"] or sha256(source) != old["sha256"]
                or receipt["artifact_sha256"].get(str(source)) != old["sha256"]):
            raise ValueError("Preserve exactly the original single-link root-resident diagnostic bytes")
        payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        previous.check_payload(payload, old["name"])
        del payload
        records.append({"name": old["name"], "canonical": str(canonical), "source": str(source),
                        "destination": str(DESTINATION / old["name"]), "sha256": old["sha256"], "bytes": old["bytes"]})
    previous.require_no_artifact_readers([row["source"] for row in records])
    check_capacity(sum(row["bytes"] for row in records))
    return records, inputs


def copy_and_verify(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    sync_directory(destination.parent)
    if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
        raise ValueError("Reject any inexact diagnostic copy")
    payload = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    previous.check_payload(payload, row["name"])
    del payload


def recheck():
    config = previous.read(OUTPUT / "config.json")
    if (config["policy"] != POLICY or config["scratch_uuid"] != copying.UUID
            or previous.read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")):
        raise ValueError("Preserve the registered storage-only operation")
    copying.require_mount()
    previous.require_stopped_owners()
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    hashes = {}
    for row in config["records"]:
        canonical, source, destination = (Path(row[key]) for key in ("canonical", "source", "destination"))
        if not source.is_symlink() or source.resolve() != destination or canonical.resolve() != destination:
            raise ValueError("Every original path must still resolve to its exact preserved bytes")
        payload = torch.load(canonical, map_location="cpu", weights_only=False, mmap=True)
        previous.check_payload(payload, row["name"])
        del payload
        hashes.update({str(path): row["sha256"] for path in (canonical, source, destination)})
    verify_hashes(hashes)
    if (shutil.disk_usage("/").free < 6 * 1024**3 + 16 * 1024**2
            or shutil.disk_usage(ROOT).free < 18 * 1024**3
            or shutil.disk_usage(copying.SCRATCH).free < 50 * 1024**3):
        raise OSError("The original qualification reserves must actually be restored")
    return {"complete": True, "canonical_artifacts_checked": 6, "artifact_sha256": hashes,
            "all_original_serialized_bytes_and_paths_preserved": True, "provider_persistent_backup": False,
            "bytes_relocated": sum(row["bytes"] for row in config["records"]),
            "config_sha256": sha256(OUTPUT / "config.json"), "root_available_bytes": shutil.disk_usage("/").free}


def main():
    if OUTPUT.exists() or DESTINATION.exists():
        raise FileExistsError("Never restart or overwrite a begun storage operation")
    records, inputs = inventory()
    sources = [Path(__file__).resolve(), Path(previous.__file__).resolve(), Path(copying.__file__).resolve(),
               REPO / "tests/test_single_model_v43_recovery_capacity.py"]
    OUTPUT.mkdir()
    DESTINATION.mkdir(parents=True)
    write_json(OUTPUT / "config.json", {"policy": POLICY, "records": records, "input_sha256": inputs,
               "source_sha256": {str(path): sha256(path) for path in sources}, "scratch_uuid": copying.UUID})
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"), "registered_unix": time.time()})
    write_json(DESTINATION / "OWNERSHIP.json", {"registered_at": str(OUTPUT), "config_sha256": sha256(OUTPUT / "config.json")})
    status("copying_closed_diagnostics")
    try:
        for row in records:
            copy_and_verify(row)
        previous.require_stopped_owners()
        previous.require_no_artifact_readers([row["source"] for row in records])
        for row in records:
            copying.publish(row)
        write_json(OUTPUT / "independent_recheck.json", recheck())
        status("complete", bytes_relocated=sum(row["bytes"] for row in records))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
