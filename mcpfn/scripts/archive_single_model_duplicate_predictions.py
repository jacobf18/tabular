"""Coalesce exactly identical completed prediction files without changing any canonical bytes."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

import archive_single_model_decoder_checkpoints as checkpoints
import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = archive.ROOT
OUTPUT = ROOT / "duplicate_prediction_archive_v1"
RUNS = ("structured_mixture_v9", "domain_clipping_v10", "family_rehearsal_v11")
DIRECTORIES = ("reference_predictions", "student_predictions_000")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def require_terminal():
    for name in RUNS:
        current = read(ROOT / name / "status.json")
        if current["state"] != "training_complete" or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Only the three completed idle structured-prior sources may be coalesced")


def duplicate_records(paths):
    sizes, groups = defaultdict(list), defaultdict(list)
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("Require regular unlinked original prediction files")
        sizes[path.stat().st_size].append(path)
    for candidates in sizes.values():
        if len(candidates) < 2:
            continue
        for path in candidates:
            groups[sha256(path)].append(path)
    return [{"source": str(path), "sha256": digest, "archive_name": f"{digest}.npz"}
            for digest, duplicates in sorted(groups.items()) if len(duplicates) > 1 for path in sorted(duplicates)]


def inventory():
    require_terminal()
    paths, inputs = [], {}
    for name in RUNS:
        directory = ROOT / name
        for filename in ("config.json", "registration.json", "status.json", "reference.json", "validation_0.json"):
            path = directory / filename
            inputs[str(path)] = sha256(path)
        for folder in DIRECTORIES:
            predictions = sorted((directory / folder).glob("*.npz"))
            if len(predictions) != 152:
                raise ValueError("Preserve each full original152-case directory without substituting another pass")
            paths.extend(predictions)
    records = duplicate_records(paths)
    if not records:
        raise ValueError("Require verified identical payloads, not merely similar predictions")
    verify_hashes(inputs)
    return records, inputs


def stage_inputs(directory, records):
    owners = {}
    for row in records:
        source = Path(row["source"])
        if (row["archive_name"] != f'{row["sha256"]}.npz' or len(row["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in row["sha256"])
                or source.is_symlink() or source.stat().st_nlink != 1 or sha256(source) != row["sha256"]):
            raise ValueError("Only exactly identical original prediction bytes may share an archive entry")
        owners.setdefault(row["archive_name"], source)
    if not owners or len(owners) >= len(records):
        raise ValueError("Require actual duplicate files, never lossy prediction quantization")
    inputs = directory / "named_inputs"
    inputs.mkdir()
    for name, source in owners.items():
        os.link(source, inputs / name)
    archive.sync_directory(inputs)
    return inputs


def validate_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        if not arrays.files:
            raise ValueError("Retain the original nonempty numeric prediction payload")
        for name in arrays.files:
            if arrays[name].dtype.hasobject:
                raise ValueError("Prediction archive may not introduce object arrays")


COMPRESSOR_SOURCE = replace_once(checkpoints.COMPRESSOR_SOURCE,
    '        payload = torch.load(archived, map_location="cpu", weights_only=False, mmap=True)\n'
    '        if not isinstance(payload, dict):\n'
    '            raise ValueError("Archived gradient payload must retain readable torch mmap serialization")\n'
    '        del payload\n', '        validate_npz(archived)\n')
COMPRESSOR_SOURCE = replace_once(COMPRESSOR_SOURCE, '"torch_mmap_readable": True', '"all_npz_payloads_readable": True')
COMPRESSOR_SOURCE_SHA256 = hashlib.sha256(COMPRESSOR_SOURCE.encode()).hexdigest()


def compressor():
    namespace = {**archive.__dict__, "stage_inputs": stage_inputs,
                 "remove_staging_links": checkpoints.remove_staging_links, "validate_npz": validate_npz}
    exec(compile(COMPRESSOR_SOURCE, __file__, "exec"), namespace)
    return namespace["compress_group"]


def recheck(config, result):
    group = Path(result["group"])
    report = read(group / "audit.json")
    if report["records"] != config["records"] or not report["all_npz_payloads_readable"]:
        raise ValueError("Every original prediction file must remain accounted for")
    hashes = {str(group / "audit.json"): result["audit_sha256"], report["image"]: report["image_sha256"],
              report["unit"]: report["unit_sha256"]}
    for action in ("is-active", "is-enabled"):
        subprocess.run(["systemctl", action, "--quiet", Path(report["unit"]).name], check=True)
    options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(group / "files")], text=True).strip().split(",")
    if "ro" not in options or (group / "named_inputs").exists():
        raise ValueError("Require durable read-only predictions with verified staging links released")
    for row in config["records"]:
        source = Path(row["source"])
        if not source.is_symlink() or source.resolve() != group / "files" / row["archive_name"]:
            raise ValueError("Every canonical original prediction file must retain its exact bytes")
        hashes[str(source)] = row["sha256"]
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(OUTPUT / "independent_recheck.json", {"complete": True,
               "canonical_prediction_files_checked": len(config["records"]),
               "distinct_identical_payloads": len({row["sha256"] for row in config["records"]}),
               "all_prediction_bytes_and_paths_unchanged": True, "no_rounding_or_recomputation": True,
               "artifact_sha256": hashes, "config_sha256": sha256(OUTPUT / "config.json"),
               "net_bytes_reclaimed": result["net_bytes_reclaimed"], "scientific_evaluation_or_selection_performed": False})


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart a prediction coalescing attempt")
    records, inputs = inventory()
    sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(), Path(checkpoints.__file__).resolve(),
               REPO / "scripts/single_model_registered_adaptation.py", REPO / "tests/test_single_model_duplicate_prediction_archive.py"]
    config = {"records": records, "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sources},
              "compressor_source_sha256": COMPRESSOR_SOURCE_SHA256, "minimum_free_bytes": archive.MINIMUM_FREE,
              "policy": "Coalesce only byte-identical completed prediction files, retaining every canonical path and full original numeric array. No rounding, omission, new predictions, scientific source or prior archive changes. Retain21 GiB staging floor with the active V20 writer."}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    status("archiving_identical_predictions")
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
