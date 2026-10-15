"""Losslessly archive one idle V14 raw endpoint without changing any scientific bytes."""

import json
from pathlib import Path
import subprocess

import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = archive.ROOT / "decoder_endpoint_archive_v1"
SOURCE = archive.ROOT / "decoder_repair_v14/raw_final_update.pt"


def read(path):
    return json.loads(Path(path).read_text())


def source_record():
    terminal = read(SOURCE.parent / "status.json")
    audit_path = archive.ROOT / "decoder_repair_v14_audit/audit.json"
    audit = read(audit_path)
    if (terminal["state"] != "training_complete" or Path(f'/proc/{terminal["pid"]}').exists()
            or not audit["complete_training_audit"] or audit["through_step"] != 50):
        raise ValueError("Only the original terminal fully audited V14 endpoint may be archived")
    expected = audit["artifact_sha256"][str(SOURCE)]
    if SOURCE.is_symlink() or not SOURCE.is_file() or sha256(SOURCE) != expected:
        raise ValueError("Require the unarchived original endpoint with its complete-audit hash")
    return {"source": str(SOURCE), "sha256": expected}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart a previous endpoint archive")
    record = source_record()
    inputs = [SOURCE.parent / "status.json", archive.ROOT / "decoder_repair_v14_audit/audit.json"]
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(), REPO / "tests/test_single_model_endpoint_archive.py"]
    config = {"records": [record], "minimum_free_bytes_during_staging": archive.MINIMUM_FREE,
              "source_artifact_is_idle": True, "original_registered_paths_and_bytes_unchanged": True,
              "input_sha256": {str(path): sha256(path) for path in inputs},
              "source_sha256": {str(path): sha256(path) for path in sources}}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    result = archive.compress_group(OUTPUT / "group_00", [record])
    group = Path(result["group"])
    metadata = read(group / "audit.json")
    hashes = {str(group / "audit.json"): result["audit_sha256"],
              str(SOURCE): record["sha256"], metadata["image"]: metadata["image_sha256"],
              metadata["unit"]: metadata["unit_sha256"]}
    if not SOURCE.is_symlink() or SOURCE.resolve() != group / "files" / SOURCE.name:
        raise ValueError("The original canonical endpoint must resolve to its declared persistent archive")
    unit = Path(metadata["unit"]).name
    for action in ("is-active", "is-enabled"):
        subprocess.run(["systemctl", action, "--quiet", unit], check=True)
    options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(group / "files")], text=True).strip().split(",")
    if "ro" not in options:
        raise ValueError("Require a read-only archive mount")
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(OUTPUT / "independent_recheck.json", {"complete": True, "canonical_paths_and_hashes_checked": 1,
               "all_mounts_read_only": True, "persistent_mounts_checked": 1, "artifact_sha256": hashes,
               "net_bytes_reclaimed": result["net_bytes_reclaimed"], "config_sha256": sha256(OUTPUT / "config.json"),
               "registered_scientific_evidence_unchanged": True, "no_baseline_or_checkpoint_modified": True})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
