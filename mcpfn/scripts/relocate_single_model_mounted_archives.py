"""Move three terminal archive images to root without changing canonical evidence bytes."""

from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

import relocate_single_model_gradient_archive as durable
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = durable.ROOT
CAPACITY = REPO / "runs/single_model_goal_20261005/legacy_snapshot_capacity_v1"
OUTPUT = REPO / "runs/single_model_goal_20261005/mounted_archive_relocation_v1"
NAMES = ("finite_endpoint_capacity_archive_v1", "decoder_checkpoint_coarchive_v1", "paired_endpoint_archive_v1")
RESERVE = 6 * 1024**3
POLICY = ("Storage only: copy, fsync and hash three complete immutable archive images onto root; retain their "
          "canonical image paths with durable symlinks. Restart existing unchanged enabled mount units to release "
          "old loop backing inodes. Preserve every checkpoint, raw tensor, original audit and mount-unit byte. "
          "Require terminal absent producers and no live scientific workers or open readers. Retain6 GiB root "
          "reserve. Never erase evidence or change training recipes, scores, selections or scientific claims.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def terminal(path):
    report = read(path)
    if (report.get("state") not in {"complete", "training_complete"}
            or not isinstance(report.get("pid"), int) or report["pid"] <= 0
            or Path(f'/proc/{report["pid"]}').exists()):
        raise ValueError("Require terminal producers with no live original process")
    return sha256(path)


def scientific_arguments(arguments):
    own = Path(__file__).name
    return any(Path(argument).suffix == ".py" and Path(argument).name != own
               and ("single_model" in Path(argument).name or "single_matrixformer" in Path(argument).name)
               for argument in arguments)


def require_idle(protected):
    protected = {str(Path(path)) for path in protected}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if scientific_arguments(arguments):
                raise ValueError(f"A scientific worker is live: {process.name}")
            mappings = (process / "maps").read_text()
            descriptors = [os.readlink(entry) for entry in (process / "fd").iterdir()]
            if any(path in mappings or any(target == path or target.startswith(path + "/")
                                          for target in descriptors) for path in protected):
                raise ValueError(f"An archive reader is live: {process.name}")
        except (FileNotFoundError, ProcessLookupError):
            continue


def bind_archive(name):
    if name not in NAMES:
        raise ValueError("Only the three fixed terminal archive images may move")
    folder = ROOT / name
    independent = folder / "independent_recheck.json"
    audit_path = folder / "group_00/audit.json"
    checked, audit = read(independent), read(audit_path)
    image = folder / "group_00/gradients.squashfs"
    mount = folder / "group_00/files"
    if (not checked.get("complete") or not audit.get("complete")
            or not audit.get("torch_mmap_readable") or not audit.get("all_original_hashes_verified")
            or not audit.get("persistent_mount_enabled") or audit.get("image") != str(image)
            or image.is_symlink() or not image.is_file() or image.stat().st_nlink != 1
            or image.stat().st_size != audit["image_bytes"]):
        raise ValueError("Require complete untouched original archive images")
    unit = Path(audit["unit"])
    expected_unit = subprocess.check_output(["systemd-escape", "--path", "--suffix=mount", str(mount)], text=True).strip()
    if unit != Path("/etc/systemd/system") / expected_unit:
        raise ValueError("Require the original persistent mount unit")
    required = {str(audit_path): sha256(audit_path), str(image): audit["image_sha256"], str(unit): audit["unit_sha256"]}
    records = audit["records"]
    expected_count = 12 if name == "decoder_checkpoint_coarchive_v1" else 1
    if len(records) != expected_count or len({row["source"] for row in records}) != expected_count:
        raise ValueError("Require the exact original canonical record inventory")
    protected = [str(image), str(mount)]
    for row in records:
        source = Path(row["source"])
        if not source.is_symlink() or not source.resolve().is_relative_to(mount):
            raise ValueError("Every canonical source must still resolve into the original mount")
        required[str(source)] = row["sha256"]
        protected.append(str(source))
    if any(checked["artifact_sha256"].get(path) != expected for path, expected in required.items()):
        raise ValueError("Original independent audit must bind all original artifacts")
    bindings = {str(independent): sha256(independent), **required}
    for parent in sorted({str(Path(row["source"]).parent) for row in records}):
        state = Path(parent) / "status.json"
        bindings[str(state)] = terminal(state)
    config = folder / "config.json"
    bindings[str(config)] = checked["config_sha256"]
    require_idle(protected)
    verify_hashes(bindings)
    return {"name": name, "image": str(image), "destination": str(OUTPUT / name / image.name),
            "image_bytes": audit["image_bytes"], "image_sha256": audit["image_sha256"],
            "unit": str(unit), "mount": str(mount), "records": records,
            "protected": protected, "input_sha256": bindings}


def check_mount(record, destination=None):
    unit = Path(record["unit"]).name
    for action in ("is-active", "is-enabled"):
        subprocess.run(["systemctl", action, "--quiet", unit], check=True)
    report = json.loads(subprocess.check_output(
        ["findmnt", "--json", "--mountpoint", record["mount"], "-o", "SOURCE,TARGET,FSTYPE,OPTIONS"], text=True))["filesystems"]
    if (len(report) != 1 or report[0]["target"] != record["mount"] or report[0]["fstype"] != "squashfs"
            or "ro" not in report[0]["options"].split(",")):
        raise ValueError("Require the unchanged read-only SquashFS mount")
    device = report[0]["source"]
    backing = subprocess.check_output(["losetup", "--noheadings", "--output", "BACK-FILE", device], text=True).strip()
    expected = Path(destination or record["image"]).resolve()
    if backing.endswith(" (deleted)") or Path(backing).resolve() != expected:
        raise ValueError("Loop device must release the old inode and use the exact durable image")
    return {"device": device, "backing_file": backing, "read_only": True, "persistent": True}


def move_one(record):
    require_idle(record["protected"])
    verify_hashes(record["input_sha256"])
    before = check_mount(record)
    result = durable.relocate(record["image"], record["destination"], record["image_sha256"], minimum_free=RESERVE)
    write_json(OUTPUT / record["name"] / "durable_copy.json", result)
    require_idle(record["protected"])
    subprocess.run(["systemctl", "restart", Path(record["unit"]).name], check=True)
    after = check_mount(record, record["destination"])
    verify_hashes(record["input_sha256"])
    if sha256(record["destination"]) != record["image_sha256"]:
        raise ValueError("Durable archive bytes must remain identical after remount")
    for row in record["records"]:
        payload = torch.load(row["source"], map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise ValueError("Every original checkpoint or tensor payload must remain mmap readable")
        del payload
    return {**result, "mount_before": before, "mount_after": after, "mmap_records_checked": len(record["records"]),
            "all_original_artifact_hashes_unchanged": True}


def capacity_ready():
    terminal(CAPACITY / "status.json")
    report = read(CAPACITY / "independent_recheck.json")
    if not report.get("complete") or not report.get("capacity_target_reached") or not report.get("all_original_bytes_and_paths_preserved"):
        raise ValueError("Require completed successful lossless root capacity recovery")
    if sha256(CAPACITY / "config.json") != report["config_sha256"]:
        raise ValueError("Capacity registration changed")
    return {str(CAPACITY / name): sha256(CAPACITY / name)
            for name in ("status.json", "independent_recheck.json", "config.json")}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never restart or overwrite an archive relocation")
    inputs = capacity_ready()
    sizes = [Path(ROOT / name / "group_00/gradients.squashfs").stat().st_size for name in NAMES]
    if shutil.disk_usage(OUTPUT.parent).free - sum(sizes) < RESERVE:
        raise OSError("Require enough measured root capacity for all three images and the6 GiB reserve")
    with ExitStack() as locks:
        parents = set()
        for name in NAMES:
            report = read(ROOT / name / "group_00/audit.json")
            parents.update(str(Path(row["source"]).parent) for row in report["records"])
        for parent in sorted(parents):
            path = Path(parent) / "run.lock"
            if path.exists():
                stream = locks.enter_context(path.open("r"))
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = [bind_archive(name) for name in NAMES]
        sources = [Path(__file__).resolve(), Path(durable.__file__).resolve(),
                   REPO / "tests/test_single_model_mounted_archive_relocation.py"]
        OUTPUT.mkdir()
        config = {"policy": POLICY, "records": records, "input_sha256": inputs, "reserve_bytes": RESERVE,
                  "source_sha256": {str(path): sha256(path) for path in sources},
                  "root_free_before": shutil.disk_usage(OUTPUT).free, "checkpoint_free_before": shutil.disk_usage(ROOT).free}
        write_json(OUTPUT / "config.json", config)
        status("relocating")
        results = []
        for record in records:
            results.append(move_one(record))
            write_json(OUTPUT / "progress.json", {"results": results})
            print(json.dumps(results[-1]), flush=True)
        verify_hashes(inputs)
        verify_hashes(config["source_sha256"])
        for record in records:
            verify_hashes(record["input_sha256"])
            check_mount(record, record["destination"])
        if shutil.disk_usage(OUTPUT).free < RESERVE:
            raise OSError("Root reserve must remain intact")
        write_json(OUTPUT / "independent_recheck.json", {"complete": True, "results": results,
                   "config_sha256": sha256(OUTPUT / "config.json"), "root_free_bytes": shutil.disk_usage(OUTPUT).free,
                   "checkpoint_free_bytes": shutil.disk_usage(ROOT).free, "canonical_paths_and_bytes_unchanged": True,
                   "scientific_training_or_evaluation_performed": False})
        status("complete", archives=len(results))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            status("failed", error=repr(error))
        raise
