"""Co-archive completed decoder checkpoints losslessly while retaining every canonical byte."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = archive.ROOT
OUTPUT = ROOT / "decoder_checkpoint_coarchive_v1"
TRAINING_RUNS = (("decoder_repair_v14", 50), ("targeted_decoder_v15", 100), ("stronger_decoder_v16", 64),
                 ("v8_decoder_v17", 64), ("replicated_decoder_v18", 64), ("paired_decoder_v19", 64))
PREFLIGHT_RUNS = ("decoder_training_preflight_v1", "targeted_decoder_preflight_v1", "v8_decoder_preflight_v1",
                  "replicated_decoder_preflight_v1", "paired_decoder_preflight_v1", "structured_teacher_preflight_v1")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def require_terminal():
    for name, expected in [(name, "training_complete") for name, count in TRAINING_RUNS] + [
            (name, "complete") for name in PREFLIGHT_RUNS]:
        current = read(ROOT / name / "status.json")
        if current["state"] != expected or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Co-archive only completed idle checkpoints, never V20 or an active writer")


def inventory():
    require_terminal()
    records, inputs = [], {}
    for name, count in (*TRAINING_RUNS, *((name, None) for name in PREFLIGHT_RUNS)):
        directory = ROOT / name
        path = ROOT / f"{name}_audit/audit.json" if count is not None else directory / "audit.json"
        audit = read(path)
        if (not audit["complete"] or (count is not None and
                (not audit["complete_training_audit"] or audit["through_step"] != count))):
            raise ValueError("Require each original complete independent checkpoint audit")
        inputs[str(path)] = sha256(path)
        inputs[str(directory / "status.json")] = sha256(directory / "status.json")
        source = directory / ("latest_checkpoint.pth" if count is not None else "preflight_checkpoint.pth")
        expected = audit["artifact_sha256"][str(source)]
        if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1 or sha256(source) != expected:
            raise ValueError("Require each original unlinked checkpoint with its independent audit hash")
        records.append({"source": str(source), "sha256": expected, "archive_name": f"{name}__{source.name}"})
    if len(records) != 12:
        raise ValueError("Preserve the exact twelve-checkpoint inventory")
    return records, inputs


def stage_inputs(directory, records):
    names = [row["archive_name"] for row in records]
    if not names or len(set(names)) != len(names) or any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("Require unique safe checkpoint archive names")
    for row in records:
        source = Path(row["source"])
        if source.is_symlink() or source.stat().st_nlink != 1 or sha256(source) != row["sha256"]:
            raise ValueError("Never stage changed or already linked checkpoint bytes")
    inputs = directory / "named_inputs"
    inputs.mkdir()
    for row in records:
        os.link(row["source"], inputs / row["archive_name"])
    archive.sync_directory(inputs)
    return inputs


def remove_staging_links(directory, records):
    inputs = directory / "named_inputs"
    expected = {inputs / row["archive_name"] for row in records}
    if set(inputs.iterdir()) != expected:
        raise ValueError("Preserve any unaccounted staging artifact")
    for row in records:
        source, staged = Path(row["source"]), inputs / row["archive_name"]
        archived = directory / "files" / row["archive_name"]
        if (not source.is_symlink() or source.resolve() != archived or staged.is_symlink()
                or sha256(staged) != row["sha256"] or sha256(source) != row["sha256"]):
            raise ValueError("Never release staging links before durable identical canonical replacements")
    for staged in expected:
        staged.unlink()
    inputs.rmdir()
    archive.sync_directory(directory)


COMPRESSOR_SOURCE = inspect.getsource(archive.compress_group)
COMPRESSOR_SOURCE = replace_once(COMPRESSOR_SOURCE, "    directory.mkdir()\n", "    directory.mkdir()\n    inputs = stage_inputs(directory, records)\n")
COMPRESSOR_SOURCE = replace_once(COMPRESSOR_SOURCE, '*[row["source"] for row in records]', 'str(inputs)')
if COMPRESSOR_SOURCE.count('mountpoint / Path(row["source"]).name') != 2:
    raise ValueError("Require both original archive lookup and canonical replacement locations")
COMPRESSOR_SOURCE = COMPRESSOR_SOURCE.replace('mountpoint / Path(row["source"]).name', 'mountpoint / row["archive_name"]')
COMPRESSOR_SOURCE = replace_once(COMPRESSOR_SOURCE, '    return {"group": str(directory),',
                                '    remove_staging_links(directory, records)\n    return {"group": str(directory),')
COMPRESSOR_SOURCE_SHA256 = hashlib.sha256(COMPRESSOR_SOURCE.encode()).hexdigest()


def compressor():
    namespace = {**archive.__dict__, "stage_inputs": stage_inputs, "remove_staging_links": remove_staging_links}
    exec(compile(COMPRESSOR_SOURCE, __file__, "exec"), namespace)
    return namespace["compress_group"]


def recheck(config, result):
    group = Path(result["group"])
    metadata = read(group / "audit.json")
    if metadata["records"] != config["records"] or not metadata["complete"]:
        raise ValueError("Preserve every original checkpoint in the co-archive")
    hashes = {str(group / "audit.json"): result["audit_sha256"], metadata["image"]: metadata["image_sha256"],
              metadata["unit"]: metadata["unit_sha256"]}
    for action in ("is-active", "is-enabled"):
        subprocess.run(["systemctl", action, "--quiet", Path(metadata["unit"]).name], check=True)
    options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(group / "files")], text=True).strip().split(",")
    if "ro" not in options or (group / "named_inputs").exists():
        raise ValueError("Require the durable read-only archive with verified staging links released")
    for row in config["records"]:
        source = Path(row["source"])
        if not source.is_symlink() or source.resolve() != group / "files" / row["archive_name"]:
            raise ValueError("Every original canonical checkpoint must resolve to its declared archived bytes")
        hashes[str(source)] = row["sha256"]
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(OUTPUT / "independent_recheck.json", {"complete": True, "canonical_checkpoints_checked": 12,
               "all_model_optimizer_and_metadata_bytes_unchanged": True, "all_checkpoints_torch_mmap_readable": True,
               "persistent_read_only_mounts_checked": 1, "artifact_sha256": hashes,
               "config_sha256": sha256(OUTPUT / "config.json"), "net_bytes_reclaimed": result["net_bytes_reclaimed"],
               "selected_alias_or_active_training_artifact_modified": False})


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart a co-archive attempt")
    records, inputs = inventory()
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(),
               REPO / "scripts/single_model_registered_adaptation.py", REPO / "tests/test_single_model_decoder_checkpoint_coarchive.py"]
    config = {"records": records, "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sources},
              "compressor_source_sha256": COMPRESSOR_SOURCE_SHA256, "minimum_free_bytes": archive.MINIMUM_FREE,
              "policy": "Lossless completed-checkpoint co-archive; preserve every model/optimizer/metadata byte and canonical read. No selected alias, active V20 artifact, scientific source or prior archive changes. Retain the original21 GiB concurrent-writer staging floor."}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    status("building_coarchive")
    try:
        result = compressor()(OUTPUT / "group_00", records)
        require_terminal()
        status("independent_recheck")
        recheck(config, result)
        status("complete", net_bytes_reclaimed=result["net_bytes_reclaimed"])
        print(json.dumps(result), flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
