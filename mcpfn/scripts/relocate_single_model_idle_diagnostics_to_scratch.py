"""Preserve thirteen idle diagnostic checkpoints on the existing local scratch disk."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from relocate_single_model_gradient_archive import sync_directory


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
OUTPUT = ROOT / "local_scratch_diagnostic_relocation_v1"
SCRATCH = Path("/mnt/matrixformer_scratch")
DESTINATION = SCRATCH / "single_model_goal_20261005_owned/idle_diagnostics_v1"
UNIT = Path("/etc/systemd/system/mnt-matrixformer_scratch.mount")
UUID = "811c4abc-237f-407a-9a87-4c2928292b66"
OWNERS = {"family_projection_v12_control": "failed", "family_projection_v12_projected": "failed",
          "query_budget_v8_tail_average": "complete"}
POLICY = ("Use the pre-existing empty DOSCRATCH ext4 filesystem, mounted by its UUID with an enabled systemd "
          "unit, without formatting, resizing or changing filesystem reserved blocks. Copy and fsync exactly "
          "the two failed V12 optimizer-bearing tails and eleven completed V8 tail-average contributing states. "
          "Require their original registered audit/retention hashes and idle source owners, verify exact copied "
          "bytes and torch mmap before publishing canonical symlinks, then independently recheck all thirteen. "
          "Keep every published baseline, original teacher, selected model, active V23 output and existing "
          "archive on its current storage. Preserve all original failures and limited audit claims. This is "
          "machine-local scratch, not provider-persistent block storage or an external backup; it may not "
          "survive VM replacement. Preserve the18 GiB checkpoint-volume and6 GiB root reserves, plus50 GiB "
          "available on scratch. No scientific, source, precision, selection or qualification change.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def require_mount():
    subprocess.run(["mountpoint", "--quiet", str(SCRATCH)], check=True)
    actual = subprocess.check_output(["findmnt", "-n", "-o", "UUID,FSTYPE", "--mountpoint", str(SCRATCH)], text=True).split()
    options = subprocess.check_output(["findmnt", "-n", "-o", "OPTIONS", "--mountpoint", str(SCRATCH)], text=True).strip().split(",")
    if actual != [UUID, "ext4"] or "rw" not in options or "ro" in options:
        raise ValueError("Use only the identified existing local scratch filesystem, never the root or checkpoint disk")
    for action in ("is-active", "is-enabled"):
        subprocess.run(["systemctl", action, "--quiet", UNIT.name], check=True)


def require_idle():
    for name, expected in OWNERS.items():
        current = read(ROOT / name / "status.json")
        if current["state"] != expected or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Relocate only the stopped original diagnostic owners, preserving both failures")


def inventory():
    require_mount()
    require_idle()
    audit_path = ROOT / "family_projection_v12_terminal_audit/audit.json"
    retained_path = ROOT / "query_budget_v8_tail_average/retained_states.json"
    report, retained = read(audit_path), read(retained_path)
    if (report["complete"] is not True or report["full_100_update_audit_complete"] is not False
            or report["training_restarted"] is not False or report["checkpoint_promoted"] is not False
            or retained["expected_steps"] != list(range(300, 401, 10))
            or [row["step"] for row in retained["states"]] != list(range(300, 401, 10))):
        raise ValueError("Preserve exact failed-V12 evidence and all eleven original retained V8 states")
    sources = {str(ROOT / name / "latest_checkpoint.pth"):
               report["input_sha256"][str(ROOT / name / "latest_checkpoint.pth")]
               for name in OWNERS if name.startswith("family_projection_")}
    for row in retained["states"]:
        expected = ROOT / "query_budget_v8_tail_average/contributing_states" / f'step_{row["step"]}.pth'
        if row["path"] != str(expected):
            raise ValueError("Retain exact registered contributing-state paths")
        sources[str(expected)] = row["sha256"]
    records = []
    for filename, digest in sorted(sources.items()):
        source = Path(filename)
        if source.is_symlink() or source.stat().st_nlink != 1 or sha256(source) != digest:
            raise ValueError("Copy only original unlinked diagnostic bytes with existing independent hashes")
        records.append({"source": str(source), "destination": str(DESTINATION / source.relative_to(ROOT)),
                        "sha256": digest, "bytes": source.stat().st_size})
    if len(records) != 13:
        raise ValueError("Never narrow the thirteen-artifact relocation inventory")
    inputs = [audit_path, retained_path, UNIT]
    inputs.extend(ROOT / name / filename for name in OWNERS for filename in ("status.json", "config.json", "registration.json"))
    return records, {str(path): sha256(path) for path in inputs}


def copy_and_verify(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    sync_directory(destination.parent)
    if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
        raise ValueError("Do not publish a scratch copy without exact original serialized bytes")
    payload = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Each copied checkpoint must retain ordinary torch mmap model access")
    del payload


def publish(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    temporary = source.with_suffix(source.suffix + ".scratch-link")
    if (source.is_symlink() or source.stat().st_nlink != 1 or sha256(source) != row["sha256"]
            or sha256(destination) != row["sha256"] or temporary.exists() or temporary.is_symlink()):
        raise ValueError("Require exact copied bytes and preserve any prior incomplete replacement")
    temporary.symlink_to(destination)
    temporary.replace(source)
    sync_directory(source.parent)


def main():
    if OUTPUT.exists() or DESTINATION.exists():
        raise FileExistsError("Never overwrite or restart a local-scratch relocation")
    records, inputs = inventory()
    total = sum(row["bytes"] for row in records)
    if (shutil.disk_usage(SCRATCH).free - total < 50 * 1024**3
            or shutil.disk_usage(ROOT).free < 18 * 1024**3 or shutil.disk_usage("/").free < 6 * 1024**3):
        raise OSError("Keep all declared filesystem reserves before copying any diagnostic evidence")
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_scratch_diagnostic_relocation.py"]
    OUTPUT.mkdir()
    DESTINATION.mkdir(parents=True)
    config = {"policy": POLICY, "records": records, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources}, "scratch_uuid": UUID,
              "original_serialized_bytes": total, "provider_persistent_backup": False}
    write_json(OUTPUT / "config.json", config)
    write_json(DESTINATION / "OWNERSHIP.json", {"policy": POLICY, "registered_at": str(OUTPUT),
                                               "config_sha256": sha256(OUTPUT / "config.json")})
    status("copying_idle_diagnostics")
    try:
        for row in records:
            require_idle()
            require_mount()
            copy_and_verify(row)
            write_json(OUTPUT / f'verified_{records.index(row):02d}.json', row)
            publish(row)
            print(json.dumps({"relocated": row["source"], "bytes": row["bytes"], "sha256": row["sha256"]}), flush=True)
        require_idle()
        require_mount()
        hashes = {}
        for row in records:
            source, destination = Path(row["source"]), Path(row["destination"])
            if not source.is_symlink() or source.resolve() != destination:
                raise ValueError("Every original diagnostic path must resolve to its verified scratch bytes")
            hashes[str(source)] = hashes[str(destination)] = row["sha256"]
            payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
            if not isinstance(payload, dict) or "model" not in payload:
                raise ValueError("Canonical checkpoint mmap reads must survive relocation")
            del payload
        verify_hashes(hashes)
        verify_hashes(inputs)
        verify_hashes(config["source_sha256"])
        write_json(OUTPUT / "independent_recheck.json", {"complete": True, "canonical_artifacts_checked": 13,
            "all_original_serialized_bytes_and_paths_preserved": True, "canonical_torch_mmaps_readable": True,
            "filesystem_formatted_or_resized": False, "published_baselines_and_active_outputs_untouched": True,
            "provider_persistent_backup": False, "artifact_sha256": hashes,
            "bytes_relocated_to_local_scratch": total, "checkpoint_available_bytes": shutil.disk_usage(ROOT).free,
            "config_sha256": sha256(OUTPUT / "config.json"), "goal_or_scientific_qualification_changed": False})
        status("complete", bytes_relocated_to_local_scratch=total)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
