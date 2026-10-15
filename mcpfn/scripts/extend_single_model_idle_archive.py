"""Archive the final idle diagnostic gradients while preserving an eighteen-GiB reserve."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import FunctionType

import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = archive.ROOT / "compressed_gradient_archive_v3"
MINIMUM_FREE = 18 * 1024**3
GROUP_SIZE = 2
POLICY = ("Losslessly archive idle completed diagnostic pairs40 through63 in two-file groups. No trainer, "
          "qualification or diagnostic writer may be active. Preserve at least18 GiB throughout staging, "
          "all root reserve, original canonical paths/hashes and existing archives. Keep every original "
          "until its durable compressed image is hashed, persistently mounted read-only, decompressed, "
          "and torch-mmap checked. Independently recheck all new images, units and canonical bytes. "
          "No scientific evidence, baseline, checkpoint, source or registration is deleted or modified.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def idle_group_compressor():
    return FunctionType(archive.compress_group.__code__, {**archive.__dict__, "MINIMUM_FREE": MINIMUM_FREE},
                        "compress_idle_group", archive.compress_group.__defaults__, archive.compress_group.__closure__)


def require_idle():
    jobs = {
        "domain_gradient_probe_v1": "complete",
        "targeted_decoder_v15": "training_complete",
        "targeted_decoder_v15_pipeline_recovery1": "complete",
        "decoder_rate_probe_v1": "failed",
        "decoder_rate_probe_v1_audit_recovery1": "complete",
    }
    for name, expected in jobs.items():
        terminal = read(archive.ROOT / name / "status.json")
        if terminal["state"] != expected or Path(f'/proc/{terminal["pid"]}').exists():
            raise ValueError("Require all related scientific writers to be terminal and their handles absent")


def recheck(output):
    config, audit = read(output / "config.json"), read(output / "audit.json")
    if not audit["complete"] or audit["config_sha256"] != sha256(output / "config.json"):
        raise ValueError("Require the complete unchanged archive manifest")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    hashes, count = {}, 0
    for group in audit["groups"]:
        group_path = Path(group["group"])
        path = group_path / "audit.json"
        hashes[str(path)] = group["audit_sha256"]
        metadata = read(path)
        hashes[metadata["image"]] = metadata["image_sha256"]
        hashes[metadata["unit"]] = metadata["unit_sha256"]
        unit = Path(metadata["unit"]).name
        subprocess.run(["systemctl", "is-active", "--quiet", unit], check=True)
        subprocess.run(["systemctl", "is-enabled", "--quiet", unit], check=True)
        options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(group_path / "files")], text=True).strip().split(",")
        if "ro" not in options:
            raise ValueError("Every persistent scientific archive must be mounted read-only")
        for row in metadata["records"]:
            source = Path(row["source"])
            if not source.is_symlink() or source.resolve() != group_path / "files" / source.name:
                raise ValueError("Canonical paths must resolve to their declared durable archive")
            hashes[str(source)] = row["sha256"]
            count += 1
    if count != 24 or count != audit["canonical_artifacts"]:
        raise ValueError("Retain all twenty-four final original gradient artifacts")
    verify_hashes(hashes)
    write_json(output / "independent_recheck.json", {"complete": True, "canonical_paths_and_hashes_checked": count,
               "persistent_mounts_checked": len(audit["groups"]), "all_mounts_read_only": True,
               "artifact_sha256": hashes, "audit_sha256": sha256(output / "audit.json")})


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve every previous archive attempt")
    require_idle()
    inputs = {}
    for name, expected in (("compressed_gradient_archive_v1", 24), ("compressed_gradient_archive_v2", 16)):
        path = archive.ROOT / name / "independent_recheck.json"
        prior = read(path)
        if not prior["complete"] or prior["canonical_paths_and_hashes_checked"] != expected:
            raise ValueError("Preserve both completed independently checked earlier archives")
        inputs[str(path)] = sha256(path)
    records = []
    for index in range(40, 64):
        source = archive.SOURCE / "pairs" / f"pair_{index:03d}.pt"
        path = archive.SOURCE / f"pair_{index:03d}.json"
        row = read(path)
        if source.is_symlink() or row["raw_gradients"] != str(source) or sha256(source) != row["raw_sha256"]:
            raise ValueError("Each new original gradient must match its registered identity and hash")
        inputs[str(path)] = sha256(path)
        records.append({"source": str(source), "sha256": row["raw_sha256"]})
    if shutil.disk_usage(archive.ROOT).free < MINIMUM_FREE + 1024**3:
        raise OSError("Reserve18 GiB plus staging headroom before archival")
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(), REPO / "tests/test_single_model_idle_archive.py"]
    OUTPUT.mkdir()
    config = {"policy": POLICY, "records": records, "group_size": GROUP_SIZE, "minimum_free_bytes": MINIMUM_FREE,
              "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sources}}
    write_json(OUTPUT / "config.json", config)
    compressor, groups = idle_group_compressor(), []
    try:
        for start in range(0, len(records), GROUP_SIZE):
            require_idle()
            status("archiving", group=len(groups))
            group = compressor(OUTPUT / f"group_{len(groups):02d}", records[start:start + GROUP_SIZE])
            groups.append(group)
            print(json.dumps(group), flush=True)
        write_json(OUTPUT / "audit.json", {"complete": True, "groups": groups, "canonical_artifacts": len(records),
                   "net_bytes_reclaimed": sum(row["net_bytes_reclaimed"] for row in groups),
                   "config_sha256": sha256(OUTPUT / "config.json"), "registered_hashes_and_paths_unchanged": True,
                   "training_or_benchmark_modified": False, "original_information_preserved_losslessly": True})
        status("independent_recheck")
        recheck(OUTPUT)
        status("complete", net_bytes_reclaimed=sum(row["net_bytes_reclaimed"] for row in groups))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
