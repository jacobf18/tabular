"""Preserve completed point-gradient and decoder evidence in read-only compressed archives."""

import json
import os
from pathlib import Path
import subprocess
import time
from types import FunctionType

import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = archive.ROOT
OUTPUT = ROOT / "idle_evidence_archive_v1"
MINIMUM_FREE = 18 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def require_idle():
    for name, expected in (("point_alignment_v1", "complete"), ("targeted_decoder_v15", "training_complete"),
                           ("stronger_decoder_v16", "training_complete"), ("stronger_decoder_v16_pipeline", "complete"),
                           ("v8_decoder_preflight_v1", "complete")):
        current = read(ROOT / name / "status.json")
        if current["state"] != expected or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Require the completed idle scientific sources before staging archives")


def original_record(path, expected):
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise ValueError("Require a regular original artifact with its registered hash")
    return {"source": str(path), "sha256": expected}


def inventory():
    records, proposals, preserved, inputs = [], [], [], {}
    source = ROOT / "point_alignment_v1"
    for state in ("projected80", "v8"):
        path = source / state / "result.json"
        report = read(path)
        inputs[str(path)] = sha256(path)
        if len(report["tasks"]) != 14 or not report["parameters_and_optimizer_restored"]:
            raise ValueError("Require the complete restored diagnostic, not training evidence in progress")
        for index, task in enumerate(report["tasks"]):
            path = source / state / f"task_{index:02d}.pt"
            if task["artifact"] != str(path):
                raise ValueError("Preserve the original diagnostic task identities")
            if state == "v8" and index < 11:
                if not path.is_symlink() or sha256(path) != task["sha256"]:
                    raise ValueError("Preserve the earlier durable root archive without replacing its links")
                preserved.append({"source": str(path), "destination": str(path.resolve()), "sha256": task["sha256"]})
            else:
                records.append(original_record(path, task["sha256"]))
        path = source / state / "proposal.pt"
        if report["proposal"] != str(path):
            raise ValueError("Preserve the original proposal identity")
        proposals.append(original_record(path, report["proposal_sha256"]))
    endpoints = []
    for name, count in (("targeted_decoder_v15", 100), ("stronger_decoder_v16", 64)):
        path = ROOT / f"{name}_audit" / "audit.json"
        report = read(path)
        if not report["complete_training_audit"] or report["through_step"] != count:
            raise ValueError("Only complete independently audited terminal decoder endpoints may move")
        inputs[str(path)] = sha256(path)
        endpoint = ROOT / name / "raw_final_update.pt"
        endpoints.append(original_record(endpoint, report["artifact_sha256"][str(endpoint)]))
    if len(records) != 17 or len(preserved) != 11 or len(proposals) != 2 or len(endpoints) != 2:
        raise ValueError("Retain the exact declared artifact inventory")
    groups = [records[start:start + 2] for start in range(0, len(records), 2)]
    groups.extend([record] for record in [*proposals, *endpoints])
    return groups, preserved, inputs


def idle_compressor():
    return FunctionType(archive.compress_group.__code__, {**archive.__dict__, "MINIMUM_FREE": MINIMUM_FREE},
                        "compress_idle_group", archive.compress_group.__defaults__, archive.compress_group.__closure__)


def recheck(config, groups):
    hashes, actual = {}, []
    for group in groups:
        directory = Path(group["group"])
        path = directory / "audit.json"
        metadata = read(path)
        hashes[str(path)] = group["audit_sha256"]
        hashes[metadata["image"]] = metadata["image_sha256"]
        hashes[metadata["unit"]] = metadata["unit_sha256"]
        unit = Path(metadata["unit"]).name
        for action in ("is-active", "is-enabled"):
            subprocess.run(["systemctl", action, "--quiet", unit], check=True)
        options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(directory / "files")], text=True).strip().split(",")
        if "ro" not in options:
            raise ValueError("Require each persistent scientific archive to be read-only")
        for record in metadata["records"]:
            source = Path(record["source"])
            if not source.is_symlink() or source.resolve() != directory / "files" / source.name:
                raise ValueError("Every canonical path must resolve to its declared durable image")
            hashes[str(source)] = record["sha256"]
            actual.append(record)
    if actual != [record for group in config["groups"] for record in group] or len(actual) != 21:
        raise ValueError("Never omit or substitute an original scientific artifact")
    for record in config["preserved_root_links"]:
        source = Path(record["source"])
        if not source.is_symlink() or str(source.resolve()) != record["destination"]:
            raise ValueError("The earlier root archive must stay untouched")
        hashes[str(source)] = record["sha256"]
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(OUTPUT / "independent_recheck.json", {"complete": True, "canonical_artifacts_checked": 21,
               "preserved_root_links_checked": 11, "persistent_read_only_mounts_checked": len(groups),
               "artifact_sha256": hashes, "config_sha256": sha256(OUTPUT / "config.json")})


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never restart or replace an archive attempt")
    require_idle()
    groups, preserved, inputs = inventory()
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(), REPO / "tests/test_single_model_idle_evidence_archive.py"]
    config = {"groups": groups, "preserved_root_links": preserved, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources}, "minimum_free_bytes": MINIMUM_FREE,
              "policy": "Lossless idle-evidence archival; no checkpoint, baseline, scientific source, registered path/hash or existing archive changes"}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    completed, compressor = [], idle_compressor()
    try:
        for index, records in enumerate(groups):
            require_idle()
            status("archiving", group=index)
            result = compressor(OUTPUT / f"group_{index:02d}", records)
            completed.append(result)
            print(json.dumps(result), flush=True)
        status("independent_recheck")
        recheck(config, completed)
        write_json(OUTPUT / "audit.json", {"complete": True, "groups": completed, "canonical_artifacts": 21,
                   "net_bytes_reclaimed": sum(group["net_bytes_reclaimed"] for group in completed),
                   "config_sha256": sha256(OUTPUT / "config.json"),
                   "independent_recheck_sha256": sha256(OUTPUT / "independent_recheck.json"),
                   "original_information_preserved_losslessly": True, "no_scientific_protocol_changes": True})
        status("complete", net_bytes_reclaimed=sum(group["net_bytes_reclaimed"] for group in completed))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
