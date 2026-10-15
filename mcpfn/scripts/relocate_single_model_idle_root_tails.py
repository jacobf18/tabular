"""Preserve two old nonselected optimizer tails while restoring root headroom."""

import os
from pathlib import Path
import shutil

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import relocate_single_model_idle_diagnostics_to_scratch as copying
import relocate_single_model_mounted_archives as readers


ROOT = copying.ROOT
OUTPUT = ROOT / "idle_root_optimizer_tail_relocation_v1"
SOURCE = REPO / "runs/single_model_goal_20261005"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/idle_root_optimizer_tails_v1"
OWNERS = {"mixed_distill_v1": (200, 150), "mixed_distill_v2": (400, 350)}
POLICY = ("Storage-only relocation of exactly the terminal mixed_distill_v1 and mixed_distill_v2 nonselected "
    "latest optimizer-bearing checkpoints from root to the existing UUID-bound machine-local scratch disk. "
    "Keep both best-validation checkpoints, all teachers, every other selected/published baseline, all archives, "
    "active or recent training evidence and original configuration/status bytes untouched. Require stopped "
    "owners, no scientific workers or open readers, exact registered owner-config identity and the declared "
    "nonselected tail steps. Hash original bytes before copying; fsync, hash and mmap-check each complete copy "
    "before atomic canonical symlink publication. Independently recheck both canonical hashes and mmap access. "
    "Preserve18 GiB checkpoint,6 GiB root and50 GiB scratch reserves. No formatting, resizing, source, scientific, "
    "selection or qualification change. Local scratch is not provider-persistent storage or an external backup "
    "and may not survive VM replacement. No published or selected model moves.")


def checkpoint_identity(payload, current, expected, config_digest):
    step, best_step = expected
    if (current["state"] != "training_complete" or current["step"] != step or current["best_step"] != best_step
            or payload["step"] != step or payload["best_step"] != best_step or best_step >= step
            or payload["best_score"] != current["best_score"] or payload["config_sha256"] != config_digest
            or not isinstance(payload["model"], dict) or not payload["model"]
            or not isinstance(payload["optimizer"], dict) or not payload["optimizer"]):
        raise ValueError("Require the exact terminal nonselected optimizer tail and original owner configuration")


def require_idle():
    for name in OWNERS:
        readers.terminal(SOURCE / name / "status.json")
    readers.require_idle([str(SOURCE / name / "latest_checkpoint.pth") for name in OWNERS])


def inventory():
    copying.require_mount()
    require_idle()
    records, inputs = [], {str(copying.UNIT): sha256(copying.UNIT)}
    for name, expected in OWNERS.items():
        directory = SOURCE / name
        path, selected = directory / "latest_checkpoint.pth", directory / "best_validation_checkpoint.pth"
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_dev != Path("/").stat().st_dev or path.samefile(selected)):
            raise ValueError("Move only separate single-link regular nonselected files physically on root")
        current = copying.read(directory / "status.json")
        completed = copying.read(directory / "training_completion_audit.json")
        if (completed["training_complete"] is not True or completed["updates"] != expected[0]
                or completed["selected_step"] != expected[1] or completed["checkpoint_sha256"] != sha256(selected)
                or completed["config_sha256"] != sha256(directory / "config.json")):
            raise ValueError("Preserve the original completed audit and selected checkpoint identity")
        digest = sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        checkpoint_identity(payload, current, expected, sha256(directory / "config.json"))
        del payload
        records.append({"source": str(path), "destination": str(DESTINATION / name / path.name),
                        "sha256": digest, "bytes": path.stat().st_size})
        for filename in ("status.json", "config.json", "training_completion_audit.json", "best_validation_checkpoint.pth"):
            source = directory / filename
            inputs[str(source)] = sha256(source)
        if name == "mixed_distill_v2":
            path = directory / "registration.json"
            inputs[str(path)] = sha256(path)
    return records, inputs


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), **fields})


def main():
    if OUTPUT.exists() or DESTINATION.exists():
        raise FileExistsError("Never overwrite or restart an optimizer-tail relocation")
    records, inputs = inventory()
    total = sum(row["bytes"] for row in records)
    if (len(records) != 2 or shutil.disk_usage(copying.SCRATCH).free - total < 50 * 1024**3
            or shutil.disk_usage(ROOT).free < 18 * 1024**3 or shutil.disk_usage("/").free < 6 * 1024**3):
        raise OSError("Preserve exactly two tails and every original filesystem reserve")
    sources = [Path(__file__).resolve(), Path(copying.__file__).resolve(), Path(readers.__file__).resolve(),
               REPO / "scripts/relocate_single_model_gradient_archive.py",
               REPO / "tests/test_single_model_idle_root_tails.py",
               REPO / "tests/test_single_model_scratch_diagnostic_relocation.py"]
    config = {"policy": POLICY, "records": records, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources},
              "original_serialized_bytes": total, "provider_persistent_backup": False,
              "root_free_before": shutil.disk_usage("/").free}
    OUTPUT.mkdir()
    DESTINATION.mkdir(parents=True)
    write_json(OUTPUT / "config.json", config)
    write_json(DESTINATION / "OWNERSHIP.json", {"policy": POLICY, "registered_at": str(OUTPUT),
               "config_sha256": sha256(OUTPUT / "config.json")})
    status("copying_idle_nonselected_tails")
    try:
        for index, row in enumerate(records):
            require_idle()
            copying.require_mount()
            copying.copy_and_verify(row)
            write_json(OUTPUT / f"verified_{index:02d}.json", row)
            copying.publish(row)
        require_idle()
        copying.require_mount()
        hashes = {}
        for row in records:
            source, destination = Path(row["source"]), Path(row["destination"])
            if not source.is_symlink() or source.resolve() != destination:
                raise ValueError("Every original diagnostic path must retain its exact copied bytes")
            hashes[str(source)] = hashes[str(destination)] = row["sha256"]
            payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
            name = source.parent.name
            checkpoint_identity(payload, copying.read(source.parent / "status.json"), OWNERS[name],
                                inputs[str(source.parent / "config.json")])
            del payload
        verify_hashes(hashes)
        verify_hashes(inputs)
        verify_hashes(config["source_sha256"])
        write_json(OUTPUT / "independent_recheck.json", {"complete": True, "canonical_artifacts_checked": 2,
                   "all_original_serialized_bytes_and_paths_preserved": True, "canonical_torch_mmaps_readable": True,
                   "selected_checkpoints_and_registered_sources_unchanged": True, "provider_persistent_backup": False,
                   "artifact_sha256": hashes, "bytes_relocated_to_local_scratch": total,
                   "root_free_after": shutil.disk_usage("/").free, "config_sha256": sha256(OUTPUT / "config.json"),
                   "scientific_training_or_evaluation_performed": False})
        status("complete", bytes_relocated_to_local_scratch=total)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
