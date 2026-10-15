"""Preserve the consumed, failed V23 confirmation bank on declared local scratch."""

import os
from pathlib import Path
import shutil

import torch

import relocate_single_model_idle_diagnostics_to_root as bounded
import relocate_single_model_idle_diagnostics_to_scratch as scratch
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from relocate_single_model_gradient_archive import sync_directory


ROOT = bounded.ROOT
OWNER = ROOT / "compact_continuation_v23_confirmation"
OUTPUT = ROOT / "v23_failed_confirmation_bank_relocation_v1"
DESTINATION = scratch.SCRATCH / "single_model_goal_20261005_owned/v23_failed_confirmation_bank_v1/synthetic_bank.pt"
POLICY = ("Storage only: move the exact original, already consumed and independently audited284-case failed V23 "
    "confirmation bank to UUID-bound machine-local scratch. Copy, fsync, hash and mmap-verify before publishing "
    "an atomic canonical symlink. Preserve the failed gate, selected checkpoint, every prediction and original "
    "source/config/status/audit byte. Require stopped V23 owners and no readers/mappings of this specific inode, "
    "while unrelated V24 training continues untouched. Keep18 GiB checkpoint,6 GiB root and50 GiB scratch plus "
    "the full copy budget. Never regenerate, reevaluate, reuse as fresh, delete evidence or claim qualification. "
    "Scratch is machine-local, not provider-persistent storage or an external backup.")


def check_bank(payload):
    if (not isinstance(payload, list) or len(payload) != 284
            or any(not isinstance(episode, dict) or set(episode) != {"source", "target", "mask", "metadata", "scaling"}
                   for episode in payload)):
        raise ValueError("Preserve the complete284-case original consumed bank")


def require_owner():
    for directory in (OWNER, ROOT / "compact_continuation_v23_pipeline"):
        current = bounded.read(directory / "status.json")
        if current["state"] != "complete" or current["pid"] <= 0 or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Require both completed, absent original V23 owners")
    gate, audit = bounded.read(OWNER / "gate.json"), bounded.read(OWNER / "audit.json")
    if (gate["passed"] is not False or gate["confirmation_evaluated"] is not True or audit["complete"] is not True
            or audit["cases"] != 284 or audit["prediction_grids"] != 1136
            or gate["audit_sha256"] != sha256(OWNER / "audit.json")):
        raise ValueError("Require the unchanged independently audited failed V23 confirmation")
    return gate, audit


def require_capacity(total):
    if (shutil.disk_usage(ROOT).free < 18 * 1024**3 or shutil.disk_usage("/").free < 6 * 1024**3
            or shutil.disk_usage(scratch.SCRATCH).free - total - 1024**2 < 50 * 1024**3):
        raise OSError("Preserve all original reserves and the complete scratch-copy budget")


def main():
    if OUTPUT.exists() or DESTINATION.parent.exists():
        raise FileExistsError("Never repeat a started bank relocation")
    scratch.require_mount()
    gate, audit = require_owner()
    source = OWNER / "synthetic_bank.pt"
    if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1 or source.stat().st_dev != ROOT.stat().st_dev:
        raise ValueError("Require the original single-link regular bank on checkpoint storage")
    digest = sha256(source)
    if digest != gate["bank_sha256"] or digest != audit["artifact_sha256"][str(source)]:
        raise ValueError("The bank must retain every original audited serialized byte")
    row = {"source": str(source), "destination": str(DESTINATION), "bytes": source.stat().st_size, "sha256": digest}
    require_capacity(row["bytes"])
    bounded.require_no_artifact_readers([source])
    inputs = {str(OWNER / filename): sha256(OWNER / filename)
              for filename in ("config.json", "registration.json", "status.json", "gate.json", "audit.json", "manifest.json")}
    manifest = bounded.read(OWNER / "manifest.json")
    inputs[manifest["checkpoint"]] = manifest["checkpoint_sha256"]
    inputs[str(scratch.UNIT)] = sha256(scratch.UNIT)
    sources = [Path(__file__).resolve(), Path(bounded.__file__).resolve(), Path(scratch.__file__).resolve(),
               REPO / "scripts/relocate_single_model_gradient_archive.py", REPO / "tests/test_single_model_v23_bank_relocation.py"]
    config = {"policy": POLICY, "record": row, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources}, "scratch_is_provider_persistent": False}
    OUTPUT.mkdir()
    DESTINATION.parent.mkdir(parents=True)
    write_json(OUTPUT / "config.json", config)
    write_json(DESTINATION.parent / "OWNERSHIP.json", {"policy": POLICY, "config": str(OUTPUT / "config.json"),
               "config_sha256": sha256(OUTPUT / "config.json")})
    write_json(OUTPUT / "status.json", {"state": "copying", "pid": os.getpid()})
    try:
        with source.open("rb") as reader, DESTINATION.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024**2)
            writer.flush()
            os.fsync(writer.fileno())
        sync_directory(DESTINATION.parent)
        if DESTINATION.stat().st_size != row["bytes"] or sha256(DESTINATION) != digest:
            raise ValueError("Require a complete byte-identical fsynced copy before publication")
        payload = torch.load(DESTINATION, map_location="cpu", weights_only=True, mmap=True)
        check_bank(payload)
        del payload
        require_owner()
        scratch.require_mount()
        bounded.publish(row)
        hashes = {str(source): digest, str(DESTINATION): digest}
        verify_hashes(hashes)
        verify_hashes(inputs)
        verify_hashes(config["source_sha256"])
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        check_bank(payload)
        del payload
        require_capacity(0)
        write_json(OUTPUT / "audit.json", {"complete": True, "cases": 284, "original_bytes_and_canonical_path_preserved": True,
                   "original_failed_confirmation_unchanged": True, "canonical_torch_mmap_readable": True,
                   "bytes_relocated": row["bytes"], "artifact_sha256": hashes, "config_sha256": sha256(OUTPUT / "config.json"),
                   "scratch_is_provider_persistent": False, "scientific_evaluation_performed": False})
        write_json(OUTPUT / "status.json", {"state": "complete", "pid": os.getpid(), "bytes_relocated": row["bytes"]})
    except BaseException as error:
        write_json(OUTPUT / "status.json", {"state": "failed", "pid": os.getpid(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
