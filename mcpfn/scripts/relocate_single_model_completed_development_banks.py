"""Preserve completed V20/V21 development grids on local scratch without reevaluation."""

import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

import relocate_single_model_idle_diagnostics_to_scratch as storage
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = storage.ROOT
OUTPUT = ROOT / "completed_v20_v21_development_relocation_v1"
DESTINATION = storage.SCRATCH / "single_model_goal_20261005_owned/completed_v20_v21_development_v1"
OWNERS = ("structured_teacher_v20", "point_inclusive_v21")
BOUNDARIES = (0, 10, 25, 50, 64)
POLICY = ("Storage only: copy/fsync/hash all1520 already audited V20/V21 development NPZ grids before any "
          "canonical replacement, then publish exact per-file symlinks. Preserve every selected model, original "
          "teacher, baseline, archive, active output, config, status, audit and failed result. No reevaluation, "
          "regeneration, source change or scientific seed. Keep18/6/50 GiB checkpoint/root/scratch reserves. "
          "Scratch is machine-local, not provider-persistent storage or an external backup.")


def read(path):
    return json.loads(Path(path).read_text())


def require_idle():
    for owner in OWNERS:
        for suffix in ("", "_pipeline"):
            current = read(ROOT / (owner + suffix) / "status.json")
            if current["state"] not in {"training_complete", "complete", "failed"} or Path(f'/proc/{current["pid"]}').exists():
                raise ValueError("Preserve terminal, stopped original owners and their failures")


def inventory():
    storage.require_mount()
    require_idle()
    records, inputs = [], []
    for owner in OWNERS:
        audit_path = ROOT / (owner + "_audit") / "audit.json"
        audit = read(audit_path)
        if not audit["complete"] or not audit["complete_training_audit"] or audit["through_step"] != 64:
            raise ValueError("Require the original completed independent training audit")
        inputs.append(audit_path)
        inputs.extend(ROOT / (owner + suffix) / name for suffix in ("", "_pipeline")
                      for name in ("config.json", "status.json", "registration.json"))
        expected = [ROOT / owner / f"validation_{step:03d}" / "student_predictions_000" / f"case_{index:03d}.npz"
                    for step in BOUNDARIES for index in range(152)]
        actual = sorted((ROOT / owner).glob("validation_*/student_predictions_*/*.npz"))
        if sorted(expected) != actual:
            raise ValueError("Preserve every original development grid, never a subset")
        for source in expected:
            digest = audit["artifact_sha256"][str(source)]
            if source.is_symlink() or source.stat().st_nlink != 1 or sha256(source) != digest:
                raise ValueError("Only original singly linked independently audited grid bytes may move")
            records.append({"source": str(source), "destination": str(DESTINATION / source.relative_to(ROOT)),
                            "sha256": digest, "bytes": source.stat().st_size})
    return records, {str(path): sha256(path) for path in inputs}


def copy_and_verify(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    storage.sync_directory(destination.parent)
    if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
        raise ValueError("Require an exact serialized copy before publishing")
    with np.load(destination, allow_pickle=False) as payload:
        if not payload.files:
            raise ValueError("Require a readable nonempty original prediction grid")
        for name in payload.files:
            payload[name]


def recheck(config):
    storage.require_mount()
    require_idle()
    hashes = {}
    for row in config["records"]:
        source, destination = Path(row["source"]), Path(row["destination"])
        if not source.is_symlink() or source.readlink() != destination or source.resolve() != destination:
            raise ValueError("Preserve exact canonical symlinks and original serialized bytes")
        if destination.stat().st_size != row["bytes"]:
            raise ValueError("Preserve the entire original grid")
        hashes[str(source)] = hashes[str(destination)] = row["sha256"]
        with np.load(source, allow_pickle=False) as payload:
            for name in payload.files:
                payload[name]
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return {"complete": True, "canonical_artifacts_checked": len(config["records"]),
            "bytes_relocated": sum(row["bytes"] for row in config["records"]),
            "artifact_sha256": hashes, "qualification_changed": False, "provider_persistent_backup": False}


def main():
    if OUTPUT.exists() or DESTINATION.exists():
        raise FileExistsError("Never overwrite or retry a begun relocation")
    records, inputs = inventory()
    total = sum(row["bytes"] for row in records)
    for path, reserve, budget in ((ROOT, 18, 0), (Path("/"), 6, 0), (storage.SCRATCH, 50, total)):
        if shutil.disk_usage(path).free - budget < reserve * 1024**3:
            raise OSError("Preserve all original storage reserves")
    sources = (Path(__file__).resolve(), Path(storage.__file__).resolve(),
               REPO / "tests/test_single_model_completed_development_relocation.py")
    config = {"policy": POLICY, "records": records, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources}}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "status.json", {"state": "copying", "pid": os.getpid(), "updated_unix": time.time()})
    try:
        for row in records:
            copy_and_verify(row)
        require_idle()
        verify_hashes(inputs)
        for row in records:
            storage.publish(row)
        report = recheck(config)
        write_json(OUTPUT / "audit.json", report)
        write_json(OUTPUT / "status.json", {"state": "complete", "pid": os.getpid(),
                   "updated_unix": time.time(), "bytes_relocated": total})
        print(json.dumps({key: value for key, value in report.items() if key != "artifact_sha256"}), flush=True)
    except BaseException as error:
        write_json(OUTPUT / "status.json", {"state": "failed", "pid": os.getpid(),
                   "updated_unix": time.time(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
