"""Preserve completed doubled-draw optimizer evidence while reserving room for qualification."""

import argparse
import json
from pathlib import Path
import subprocess

import archive_single_model_idle_evidence as idle
import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = archive.ROOT


def read(path):
    return json.loads(Path(path).read_text())


def plan(kind):
    if kind == "preflight":
        source = ROOT / "replicated_decoder_preflight_v1/raw_update.pt"
        audit_path = source.parent / "audit.json"
        terminal_state, minimum = "complete", 21 * 1024**3
    elif kind == "endpoint":
        source = ROOT / "replicated_decoder_v18/raw_final_update.pt"
        audit_path = ROOT / "replicated_decoder_v18_audit/audit.json"
        terminal_state, minimum = "training_complete", 18 * 1024**3
    else:
        raise ValueError("Only the two explicitly declared terminal artifacts may be archived")
    terminal, audit = read(source.parent / "status.json"), read(audit_path)
    if terminal["state"] != terminal_state or Path(f'/proc/{terminal["pid"]}').exists() or not audit["complete"]:
        raise ValueError("Never archive live, incomplete or unaudited optimizer evidence")
    inputs = [source.parent / "status.json", audit_path]
    if kind == "preflight":
        if audit["training_tasks"] != 28 or audit["prediction_grids"] != 56 or audit["raw_endpoint"]["gradient_vectors"] != 84:
            raise ValueError("Require the complete28-task/84-gradient preflight audit")
    else:
        if (not audit["complete_training_audit"] or audit["through_step"] != 64 or audit["training_episodes"] != 1792
                or audit["prediction_grids"] != 1216):
            raise ValueError("Require the complete registered V18 training audit")
        replay = ROOT / "replicated_decoder_v18_endpoint_replay"
        replay_audit = read(replay / "audit.json")
        if read(replay / "status.json")["state"] != "complete" or not replay_audit["complete"] or replay_audit["endpoint_grids"] not in (28, 56):
            raise ValueError("Finish independent neural endpoint replay before archival")
        verify_hashes(replay_audit["artifact_sha256"])
        inputs.extend([replay / "audit.json", replay / "status.json"])
    expected = audit["artifact_sha256"][str(source)]
    if source.is_symlink() or not source.is_file() or sha256(source) != expected:
        raise ValueError("Require the unarchived regular original with its independently audited hash")
    return source, {"source": str(source), "sha256": expected}, inputs, minimum


def run(kind):
    output = ROOT / f"replicated_{kind}_archive_v1"
    if output.exists():
        raise FileExistsError("Never restart or overwrite an archive attempt")
    source, record, inputs, minimum = plan(kind)
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(), Path(idle.__file__).resolve(),
               REPO / "tests/test_single_model_replicated_evidence_archive.py"]
    config = {"artifact": kind, "records": [record], "minimum_free_bytes_during_staging": minimum,
              "input_sha256": {str(path): sha256(path) for path in inputs},
              "source_sha256": {str(path): sha256(path) for path in sources},
              "policy": "Lossless archival of audited terminal raw optimizer evidence only; no checkpoint, scientific bytes, live artifact, registered source or qualification change"}
    output.mkdir()
    write_json(output / "config.json", config)
    compressor = archive.compress_group if kind == "preflight" else idle.idle_compressor()
    result = compressor(output / "group_00", [record])
    group = Path(result["group"])
    metadata = read(group / "audit.json")
    hashes = {str(group / "audit.json"): result["audit_sha256"], str(source): record["sha256"],
              metadata["image"]: metadata["image_sha256"], metadata["unit"]: metadata["unit_sha256"]}
    if not source.is_symlink() or source.resolve() != group / "files" / source.name:
        raise ValueError("The canonical artifact must resolve to its durable compressed image")
    for action in ("is-active", "is-enabled"):
        subprocess.run(["systemctl", action, "--quiet", Path(metadata["unit"]).name], check=True)
    options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(group / "files")], text=True).strip().split(",")
    if "ro" not in options:
        raise ValueError("The archive must remain read-only")
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(output / "independent_recheck.json", {"complete": True, "canonical_paths_and_hashes_checked": 1,
               "persistent_read_only_mounts_checked": 1, "artifact_sha256": hashes,
               "net_bytes_reclaimed": result["net_bytes_reclaimed"], "config_sha256": sha256(output / "config.json"),
               "registered_scientific_evidence_unchanged": True, "no_baseline_or_checkpoint_modified": True})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", choices=("preflight", "endpoint"), required=True)
    arguments = parser.parse_args()
    run(arguments.artifact)
