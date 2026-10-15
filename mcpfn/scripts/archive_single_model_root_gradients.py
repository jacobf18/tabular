"""Losslessly compress eleven already relocated idle gradient artifacts on root."""

import json
import os
from pathlib import Path
import shutil
import time

import archive_single_model_legacy_snapshots as legacy
import relocate_single_model_mounted_archives as relocation
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = legacy.ROOT
OUTPUT = ROOT / "root_gradient_capacity_v1"
SOURCE = relocation.ROOT / "point_alignment_v1"
DURABLE = ROOT / "durable_gradient_archive/point_alignment_v1/v8"
POLICY = ("Storage only: compress the eleven previously durably relocated original V8 diagnostic gradient files. "
          "Require completed absent producers, no readers, exact relocation/source hashes and single-link regular "
          "durable originals. Preserve all canonical paths and original serialized bytes, original metadata and "
          "read-only persistent mmap access. Use root only with the existing3 GiB staging reserve; leave the "
          "checkpoint volume's18 GiB floor untouched. Never modify checkpoint weights, recipes, scores or selections.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def inventory():
    relocation.terminal(SOURCE / "status.json")
    relocation.terminal(legacy.OUTPUT / "status.json")
    report = read(DURABLE / "audit.json")
    original = read(SOURCE / "v8/result.json")
    if (not report["complete"] or not report["canonical_paths_and_registered_hashes_unchanged"]
            or report["experiment_code_or_results_changed"] or len(report["artifacts"]) != 11
            or not original["parameters_and_optimizer_restored"]):
        raise ValueError("Require the complete original restored diagnostic and durable eleven-file relocation")
    inputs = {str(path): sha256(path) for path in (DURABLE / "audit.json", SOURCE / "status.json", SOURCE / "v8/result.json",
                                                 legacy.OUTPUT / "status.json", legacy.OUTPUT / "independent_recheck.json")}
    if not read(legacy.OUTPUT / "independent_recheck.json")["complete"]:
        raise ValueError("The original legacy capacity archive must finish before this supplement")
    relocation_source = REPO / "scripts/relocate_single_model_gradient_archive.py"
    inputs[str(relocation_source)] = report["script_sha256"]
    records, canonical = [], {}
    for index, row in enumerate(report["artifacts"]):
        source, destination = SOURCE / f"v8/task_{index:02d}.pt", DURABLE / f"task_{index:02d}.pt"
        task = original["tasks"][index]
        if (row["source"] != str(source) or row["destination"] != str(destination) or row["sha256"] != task["sha256"]
                or task["artifact"] != str(source) or not source.is_symlink() or source.resolve() != destination
                or destination.is_symlink() or not destination.is_file() or destination.stat().st_nlink != 1):
            raise ValueError("Every durable original must bind its unchanged canonical diagnostic record")
        records.append({"source": str(destination), "sha256": row["sha256"]})
        canonical[str(source)] = row["sha256"]
    relocation.require_idle([str(SOURCE), str(DURABLE)])
    verify_hashes(inputs)
    verify_hashes(canonical)
    return records, canonical, inputs


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never restart this lossless root gradient archive")
    records, canonical, inputs = inventory()
    sources = [Path(__file__).resolve(), Path(legacy.__file__).resolve(), Path(legacy.archive.__file__).resolve(),
               Path(relocation.__file__).resolve(), REPO / "scripts/single_model_registered_adaptation.py",
               REPO / "tests/test_single_model_root_capacity_supplement.py"]
    OUTPUT.mkdir()
    config = {"policy": POLICY, "records": records, "canonical_sha256": canonical, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources}, "minimum_root_free": legacy.MINIMUM_ROOT_FREE,
              "compressor_source_sha256": legacy.COMPRESSOR_SOURCE_SHA256}
    write_json(OUTPUT / "config.json", config)
    status("archiving")
    result = legacy.compressor()(OUTPUT / "group_00", records)
    group = Path(result["group"])
    report = read(group / "audit.json")
    if (not report["complete"] or report["records"] != records or not report["torch_mmap_readable"]
            or not report["all_original_hashes_verified"] or not report["persistent_mount_enabled"]):
        raise ValueError("Require every exact original gradient byte in the complete persistent archive")
    mount_record = {"image": report["image"], "unit": report["unit"], "mount": str(group / "files")}
    relocation.check_mount(mount_record)
    hashes = {**canonical, **{row["source"]: row["sha256"] for row in records},
              str(group / "audit.json"): result["audit_sha256"], report["image"]: report["image_sha256"],
              report["unit"]: report["unit_sha256"]}
    for row in records:
        path = Path(row["source"])
        if not path.is_symlink() or path.resolve() != group / "files" / path.name:
            raise ValueError("Every durable path must resolve into its verified read-only archive")
    verify_hashes(hashes)
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])
    if shutil.disk_usage(ROOT).free < legacy.MINIMUM_ROOT_FREE:
        raise OSError("Never spend the original root staging reserve")
    write_json(OUTPUT / "independent_recheck.json", {"complete": True, "records": 11, "artifact_sha256": hashes,
               "config_sha256": sha256(OUTPUT / "config.json"), "net_bytes_reclaimed": result["net_bytes_reclaimed"],
               "root_free_bytes": shutil.disk_usage(ROOT).free, "all_original_bytes_and_paths_preserved": True,
               "scientific_training_or_evaluation_performed": False})
    status("complete", records=11)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            status("failed", error=repr(error))
        raise
