"""Finish verified archive remounts after the relocation stopped for a concurrent test process."""

from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import shutil
import subprocess

import torch

import relocate_single_model_continuation_archives as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = original.OUTPUT.with_name("continuation_archive_relocation_finish_v1")
ERROR = "ValueError('A scientific worker is live: 2980107')"


def validate_copies(config, failed, progress, copies):
    records = config["records"]
    if (failed["state"] != "failed" or failed["error"] != ERROR or len(records) != 2
            or len(progress["results"]) != 1 or len(copies) != 2
            or [record["name"] for record in records] != list(original.NAMES)
            or not progress["results"][0]["all_original_artifact_hashes_unchanged"]
            or progress["results"][0]["destination"] != records[0]["destination"]):
        raise ValueError("Finish only the documented post-copy pre-remount idle-guard interruption")
    for record, copied in zip(records, copies):
        if (copied["source"] != record["image"] or copied["destination"] != record["destination"]
                or copied["sha256"] != record["image_sha256"] or copied["bytes_relocated"] != record["image_bytes"]
                or not copied["canonical_path_preserved"] or not copied["durable_copy_fsynced"]):
            raise ValueError("Require both original durable copy receipts; never copy or overwrite again")


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never restart this additive remount finalization")
    paths = [original.OUTPUT / name for name in ("config.json", "status.json", "progress.json")]
    config, failed, progress = map(original.read, paths)
    if Path(f'/proc/{failed["pid"]}').exists():
        raise ValueError("The interrupted relocation must have exited")
    receipts = [original.OUTPUT / record["name"] / "durable_copy.json" for record in config["records"]]
    copies = [original.read(path) for path in receipts]
    validate_copies(config, failed, progress, copies)
    protected = [path for record in config["records"] for path in record["protected"]]
    original.require_idle(protected)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    if shutil.disk_usage(original.OUTPUT).free < original.RESERVE:
        raise OSError("Retain the original6 GiB root reserve")
    with ExitStack() as locks:
        for folder in sorted({str(Path(row["source"]).parent) for record in config["records"] for row in record["records"]}):
            lock = Path(folder) / "run.lock"
            if lock.exists():
                fcntl.flock(locks.enter_context(lock.open("r")), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for record in config["records"]:
            source, destination = Path(record["image"]), Path(record["destination"])
            if not source.is_symlink() or source.resolve() != destination or sha256(destination) != record["image_sha256"]:
                raise ValueError("Both existing durable images must retain their exact registered bytes and canonical links")
            verify_hashes(record["input_sha256"])
        original.check_mount(config["records"][0], config["records"][0]["destination"])
        OUTPUT.mkdir()
        inputs = {str(path): sha256(path) for path in [*paths, *receipts]}
        sources = [Path(__file__).resolve(), Path(original.__file__).resolve(),
                   REPO / "tests/test_single_model_continuation_relocation_finish.py"]
        write_json(OUTPUT / "config.json", {"input_sha256": inputs,
            "source_sha256": {str(path): sha256(path) for path in sources}, "original_failure_preserved": True,
            "no_archive_copy_repeated": True, "reserve_bytes": original.RESERVE})
        pending = config["records"][1]
        original.require_idle(protected)
        subprocess.run(["systemctl", "restart", Path(pending["unit"]).name], check=True)
        mounts = []
        for record in config["records"]:
            mounts.append(original.check_mount(record, record["destination"]))
            verify_hashes(record["input_sha256"])
            for row in record["records"]:
                payload = torch.load(row["source"], map_location="cpu", weights_only=False, mmap=True)
                if not isinstance(payload, dict):
                    raise ValueError("Retain both original mmap-readable endpoint payloads")
                del payload
        verify_hashes(inputs)
        verify_hashes(original.read(OUTPUT / "config.json")["source_sha256"])
        if shutil.disk_usage(original.OUTPUT).free < original.RESERVE:
            raise OSError("Never consume the root reserve")
        write_json(OUTPUT / "independent_recheck.json", {"complete": True, "pid": os.getpid(),
            "config_sha256": sha256(OUTPUT / "config.json"), "original_failure_preserved": True,
            "copies_repeated": 0, "remounts_finished": 1, "mmap_records_checked": 2, "mounts": mounts,
            "root_free_bytes": shutil.disk_usage(original.OUTPUT).free,
            "checkpoint_free_bytes": shutil.disk_usage(original.ROOT).free,
            "all_original_artifacts_and_units_unchanged": True})


if __name__ == "__main__":
    main()
