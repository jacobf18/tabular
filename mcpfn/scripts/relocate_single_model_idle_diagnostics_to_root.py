"""Restore checkpoint headroom without touching live or selected model artifacts."""

import json
import os
from pathlib import Path
import shutil
import time

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from relocate_single_model_gradient_archive import sync_directory


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
OUTPUT = ROOT / "idle_diagnostic_root_relocation_v1"
DESTINATION = REPO / "runs/single_model_goal_20261005/idle_diagnostic_root_relocation_v1/files"
ARTIFACTS = ("top_block_preflight_v1/preflight_checkpoint.pth",
             *(f"top_block_rate_probe_v1/rate_{index}/candidate_parameters.pt" for index in range(4)),
             "compact_continuation_v23/latest_checkpoint.pth")
OWNERS = {"top_block_preflight_v1": "complete", "top_block_rate_probe_v1": "failed",
          "top_block_rate_probe_v1_audit_recovery1": "complete", "compact_continuation_v23": "training_complete",
          "compact_continuation_v23_pipeline": "complete"}
AUDITS = ("top_block_preflight_v1/audit.json", "top_block_rate_probe_v1_audit_recovery1/audit.json",
          "compact_continuation_v23_audit/audit.json")
POLICY = ("Relocate exactly six completed, idle, nonselected diagnostics from checkpoint storage to root: the "
    "nonpromotable top-block preflight, four audited training-rate candidate tensors, and V23's unqualified "
    "step96 optimizer tail, never its selected step32. Preserve all original serialized bytes, canonical paths "
    "as atomic symlinks, mmap access, registered sources/configurations/statuses and audit hashes. Require "
    "stopped exact owners, independently completed original audits and no open descriptors or mappings of "
    "these specific inodes, including aliases. Unrelated live training is permitted and never interrupted. "
    "Hash and fsync complete copies before publication and independently recheck afterward. Preserve6 GiB root, "
    "18 GiB checkpoint and50 GiB scratch. No active, selected, published or teacher artifact moves; no evidence "
    "is deleted, no scientific evaluation runs, and no claim of an external backup is made.")


def read(path):
    return json.loads(Path(path).read_text())


def require_stopped_owners():
    for name, expected in OWNERS.items():
        current = read(ROOT / name / "status.json")
        if (current["state"] != expected or type(current["pid"]) is not int or current["pid"] <= 0
                or Path(f'/proc/{current["pid"]}').exists()):
            raise ValueError("Every exact diagnostic owner must be stopped in its original audited state")


def identity(path):
    current = Path(path).stat()
    return current.st_dev, current.st_ino


def require_no_artifact_readers(paths, proc=Path("/proc")):
    protected = {identity(path) for path in paths}
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    if identity(descriptor) in protected:
                        raise ValueError(f"Diagnostic descriptor still open in process {process.name}")
                except FileNotFoundError:
                    continue
            for line in (process / "maps").read_text().splitlines():
                fields = line.split(maxsplit=5)
                major, minor = fields[3].split(":")
                if (os.makedev(int(major, 16), int(minor, 16)), int(fields[4])) in protected:
                    raise ValueError(f"Diagnostic mmap still open in process {process.name}")
        except (FileNotFoundError, ProcessLookupError):
            continue


def check_payload(payload, name):
    if name not in ARTIFACTS:
        raise ValueError("Only the six exact nonselected diagnostic paths may move")
    if name.startswith("top_block_rate_probe_v1/"):
        if (not isinstance(payload, torch.Tensor) or payload.shape != (19496843,)
                or payload.dtype != torch.float32 or not torch.isfinite(payload).all()):
            raise ValueError("Preserve the complete audited nonpromotable rate candidate")
    elif name.startswith("top_block_preflight_v1/"):
        if (payload["preflight_only"] is not True or payload["selection_qualified"] is not False
                or payload["gradient_updates_performed"] != 0 or not payload["model"]):
            raise ValueError("Preflight checkpoint must remain explicitly nonpromotable")
    elif (payload["step"] != 96 or payload["best_step"] != 32 or payload["gradient_updates_performed"] != 67
          or payload["independent_confirmation_passed"] is not False or not payload["model"] or not payload["optimizer"]):
        raise ValueError("Require V23's audited unqualified terminal tail, never the selected checkpoint")


def inventory():
    require_stopped_owners()
    records, inputs, audited = [], {}, {}
    for name in AUDITS:
        path = ROOT / name
        audit = read(path)
        if audit["complete"] is not True:
            raise ValueError("Require original independently completed diagnostic audits")
        audited.update(audit["artifact_sha256"])
        inputs[str(path)] = sha256(path)
    selected = ROOT / "compact_continuation_v23/selected_checkpoint.pth"
    inputs[str(selected)] = audited[str(selected)]
    for owner in OWNERS:
        for filename in ("status.json", "config.json"):
            path = ROOT / owner / filename
            inputs[str(path)] = sha256(path)
    for name in ARTIFACTS:
        path = ROOT / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_dev != ROOT.stat().st_dev or path.samefile(selected)):
            raise ValueError("Only distinct single-link regular diagnostics on checkpoint storage may move")
        digest = sha256(path)
        if digest != audited[str(path)]:
            raise ValueError("Never relocate a diagnostic differing from its original independent audit")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        check_payload(payload, name)
        del payload
        records.append({"name": name, "source": str(path), "destination": str(DESTINATION / name),
                        "sha256": digest, "bytes": path.stat().st_size})
    require_no_artifact_readers([row["source"] for row in records])
    verify_hashes(inputs)
    return records, inputs


def require_capacity(total):
    if (shutil.disk_usage("/").free - total - 1024**2 < 6 * 1024**3
            or shutil.disk_usage(ROOT).free < 18 * 1024**3
            or shutil.disk_usage("/mnt/matrixformer_scratch").free < 50 * 1024**3):
        raise OSError("Preserve all three filesystem reserves plus the complete root-copy budget")


def copy_and_verify(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    sync_directory(destination.parent)
    if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
        raise ValueError("Require the complete byte-identical fsynced destination")
    payload = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    check_payload(payload, row["name"])
    del payload


def publish(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    temporary = source.with_suffix(source.suffix + ".root-relocation-link")
    if (source.is_symlink() or source.stat().st_nlink != 1 or sha256(source) != row["sha256"]
            or sha256(destination) != row["sha256"] or temporary.exists() or temporary.is_symlink()):
        raise ValueError("Publish only an untouched original and its verified complete copy")
    require_no_artifact_readers([source, destination])
    temporary.symlink_to(destination)
    temporary.replace(source)
    sync_directory(source.parent)


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def main():
    if OUTPUT.exists() or DESTINATION.exists():
        raise FileExistsError("Never repeat a started diagnostic relocation")
    records, inputs = inventory()
    total = sum(row["bytes"] for row in records)
    require_capacity(total)
    sources = [Path(__file__).resolve(), REPO / "scripts/relocate_single_model_gradient_archive.py",
               REPO / "tests/test_single_model_idle_diagnostic_root_relocation.py"]
    config = {"policy": POLICY, "records": records, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources}, "serialized_bytes": total,
              "root_free_before": shutil.disk_usage("/").free, "checkpoint_free_before": shutil.disk_usage(ROOT).free}
    OUTPUT.mkdir()
    DESTINATION.mkdir(parents=True)
    if DESTINATION.stat().st_dev != Path("/").stat().st_dev:
        raise ValueError("Use the declared root filesystem, never scratch or checkpoint storage")
    write_json(OUTPUT / "config.json", config)
    write_json(DESTINATION / "OWNERSHIP.json", {"policy": POLICY, "registered_at": str(OUTPUT),
               "config_sha256": sha256(OUTPUT / "config.json")})
    status("copying_idle_diagnostics")
    try:
        for index, row in enumerate(records):
            require_stopped_owners()
            require_no_artifact_readers([row["source"]])
            copy_and_verify(row)
            write_json(OUTPUT / f"verified_{index:02d}.json", row)
            publish(row)
        hashes = {}
        for row in records:
            source, destination = Path(row["source"]), Path(row["destination"])
            if not source.is_symlink() or source.resolve() != destination:
                raise ValueError("Preserve each exact canonical path through its verified root copy")
            hashes[str(source)] = hashes[str(destination)] = row["sha256"]
            payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
            check_payload(payload, row["name"])
            del payload
        require_stopped_owners()
        verify_hashes(hashes)
        verify_hashes(inputs)
        verify_hashes(config["source_sha256"])
        require_capacity(0)
        write_json(OUTPUT / "independent_recheck.json", {"complete": True, "canonical_artifacts_checked": len(records),
                   "all_original_serialized_bytes_and_paths_preserved": True, "canonical_torch_mmaps_readable": True,
                   "selected_checkpoints_and_registered_sources_unchanged": True, "artifact_sha256": hashes,
                   "bytes_relocated_to_root": total, "scientific_training_or_evaluation_performed": False,
                   "external_backup": False, "root_free_after": shutil.disk_usage("/").free,
                   "checkpoint_free_after": shutil.disk_usage(ROOT).free,
                   "config_sha256": sha256(OUTPUT / "config.json")})
        status("complete", bytes_relocated_to_root=total)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
