"""Archive terminal audited training tasks in bounded groups without consuming the write reserve."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

import compress_single_model_gradient_archive as archive
from archive_single_model_replicated_tasks import tree_inventory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = archive.ROOT
OUTPUT = ROOT / "completed_task_archive_v1"
MAXIMUM_GROUP = 384 * 1024**2
MINIMUM_FREE = 18 * 1024**3
RUNS = (("v8_decoder_v17", 64, 1216, "v8_decoder_v17_pipeline"),
        ("stronger_decoder_v16", 64, 1368, "stronger_decoder_v16_pipeline"),
        ("decoder_repair_v14", 50, 1216, "decoder_repair_v14_pipeline"),
        ("finite_guarded_v13", 50, 1064, "finite_guarded_v13_pipeline"),
        ("targeted_decoder_v15", 100, 1368, "targeted_decoder_v15_pipeline_recovery1"))


def read(path):
    return json.loads(Path(path).read_text())


def partition(records, limit=MAXIMUM_GROUP):
    by_batch = {}
    for name, digest in sorted(records.items()):
        by_batch.setdefault(Path(name).parent, []).append({"source": name, "sha256": digest})
    groups, current, size = [], [], 0
    for directory, members in by_batch.items():
        required = sum(Path(row["source"]).stat().st_size for row in members)
        if required > limit:
            raise ValueError("A complete audited batch exceeds the staging bound")
        if current and size + required > limit:
            groups.append(current)
            current, size = [], 0
        current.extend(members)
        size += required
    if current:
        groups.append(current)
    return groups


def plan():
    groups, inputs = [], []
    for name, steps, grids, pipeline_name in RUNS:
        run, pipeline = ROOT / name, ROOT / pipeline_name
        paths = [run / "status.json", ROOT / f"{name}_audit/audit.json", pipeline / "status.json",
                 pipeline / "result.json", ROOT / f"{name}_endpoint_replay/audit.json"]
        terminal, audit, state, result, replay = map(read, paths)
        if (terminal["state"] != "training_complete" or terminal["step"] != steps or state["state"] != "complete"
                or any(Path(f'/proc/{entry["pid"]}').exists() for entry in (terminal, state))
                or not result["complete"] or result["gate"]["passed"] or result["campaign"] is not None
                or result["review"] is not None):
            raise ValueError("Archive only the declared terminal unsuccessful training lifecycles")
        if (not audit["complete_training_audit"] or audit["through_step"] != steps
                or audit["training_episodes"] != steps * 14 or audit["prediction_grids"] != grids
                or not replay["complete"] or replay["endpoint_grids"] != 28):
            raise ValueError("Require the complete original training and endpoint audits")
        records = tree_inventory(run / "training_batches", audit["artifact_sha256"])
        if (len(records) != steps * 15 or sum(Path(path).name == "manifest.json" for path in records) != steps
                or sum(Path(path).suffix == ".pt" for path in records) != steps * 14):
            raise ValueError("Retain every original task and manifest")
        verify_hashes(records)
        groups.extend(partition(records))
        inputs.extend(paths)
    return groups, inputs


def compress(directory, records):
    directory.mkdir()
    image, mountpoint = directory / "tasks.squashfs", directory / "files"
    mountpoint.mkdir()
    sources = sorted({Path(row["source"]).parent for row in records})
    if len({path.name for path in sources}) != len(sources):
        raise ValueError("Never collide batch directory names inside an archive")
    original = {row["source"]: row["sha256"] for row in records}
    expected = {str(path) for source in sources for path in source.rglob("*") if path.is_file()}
    if expected != set(original) or any(Path(name).is_symlink() for name in original):
        raise ValueError("Each staged directory must contain exactly the audited original files")
    maximum = sum(Path(name).stat().st_size for name in original)
    if maximum > MAXIMUM_GROUP or shutil.disk_usage("/dev/shm").free < maximum + 4 * 1024**3:
        raise OSError("Preserve bounded RAM staging and4 GiB of RAM headroom")
    with tempfile.TemporaryDirectory(prefix="single-model-task-group-", dir="/dev/shm") as scratch:
        staging = Path(scratch) / image.name
        with (directory / "build.log").open("x") as stream:
            subprocess.run(["mksquashfs", *map(str, sources), str(staging), "-keep-as-directory", "-comp", "zstd",
                            "-Xcompression-level", "8", "-b", "1048576", "-processors", "4", "-noappend", "-no-progress"],
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        compressed = staging.stat().st_size
        if compressed >= maximum or shutil.disk_usage(ROOT).free - compressed < MINIMUM_FREE:
            raise OSError("Require savings and the unchanged18 GiB durable write reserve")
        shutil.copyfile(staging, image)
        with image.open("rb") as stream:
            os.fsync(stream.fileno())
        if sha256(image) != sha256(staging):
            raise ValueError("Durable image must match the complete staged image")
    archive.sync_directory(directory)
    mounted = archive.mount_unit(image, mountpoint)
    destinations = {name: mountpoint / Path(name).parent.name / Path(name).name for name in original}
    verify_hashes({str(destinations[name]): digest for name, digest in original.items()})
    for name, destination in destinations.items():
        if destination.suffix == ".pt":
            payload = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
            if not isinstance(payload, dict):
                raise ValueError("Every original training task must remain mmap-readable")
    verified = {"image": str(image), "image_sha256": sha256(image), "records": records,
                "image_bytes": compressed, "uncompressed_bytes": maximum,
                "all_original_hashes_verified": True, "all_task_payloads_mmap_readable": True, **mounted}
    write_json(directory / "verified_before_replacement.json", verified)
    with (directory / "verified_before_replacement.json").open("rb") as stream:
        os.fsync(stream.fileno())
    archive.sync_directory(directory)
    for name, destination in destinations.items():
        replacement = archive.replace_with_verified_link(name, destination, original[name])
        with (directory / "replacement.jsonl").open("a") as stream:
            stream.write(json.dumps(replacement) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    verify_hashes(original)
    write_json(directory / "audit.json", {**verified, "complete": True, "net_bytes_reclaimed": maximum - compressed})
    archive.sync_directory(directory)
    return {"directory": str(directory), "audit_sha256": sha256(directory / "audit.json"),
            "net_bytes_reclaimed": maximum - compressed}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart an archive attempt")
    groups, inputs = plan()
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(),
               REPO / "scripts/archive_single_model_replicated_tasks.py", REPO / "tests/test_single_model_completed_task_groups.py"]
    config = {"groups": groups, "maximum_group_bytes": MAXIMUM_GROUP, "minimum_free_bytes": MINIMUM_FREE,
              "input_sha256": {str(path): sha256(path) for path in inputs},
              "source_sha256": {str(path): sha256(path) for path in sources},
              "policy": "Lossless archival of five completed audited task trees; no checkpoint, canonical bytes or scientific policy changes"}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    results = []
    for index, records in enumerate(groups):
        result = compress(OUTPUT / f"group_{index:03d}", records)
        results.append(result)
        write_json(OUTPUT / "progress.json", {"complete": False, "groups_complete": len(results), "groups_expected": len(groups),
                                               "pid": os.getpid(), "results": results})
        print(json.dumps(result), flush=True)
    hashes = {row["source"]: row["sha256"] for group in groups for row in group}
    for result in results:
        directory = Path(result["directory"])
        audit = read(directory / "audit.json")
        hashes.update({str(directory / "audit.json"): result["audit_sha256"],
                       audit["image"]: audit["image_sha256"], audit["unit"]: audit["unit_sha256"]})
        for action in ("is-active", "is-enabled"):
            subprocess.run(["systemctl", action, "--quiet", Path(audit["unit"]).name], check=True)
        options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(directory / "files")], text=True).strip().split(",")
        if "ro" not in options:
            raise ValueError("Every persistent archive must remain read-only")
        for row in audit["records"]:
            source = Path(row["source"])
            if not source.is_symlink() or source.resolve() != directory / "files" / source.parent.name / source.name:
                raise ValueError("Every original path must resolve to its declared archive")
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(OUTPUT / "independent_recheck.json", {"complete": True, "canonical_paths_and_hashes_checked": sum(map(len, groups)),
               "persistent_read_only_mounts_checked": len(groups), "net_bytes_reclaimed": sum(row["net_bytes_reclaimed"] for row in results),
               "artifact_sha256": hashes, "config_sha256": sha256(OUTPUT / "config.json"),
               "registered_scientific_evidence_unchanged": True, "no_baseline_or_checkpoint_modified": True})


if __name__ == "__main__":
    main()
