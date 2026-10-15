"""Preserve another sixteen idle diagnostic gradient files for the next training budget."""

import json
import os
from pathlib import Path

import compress_single_model_gradient_archive as archive
from recsys_evaluation_common import sha256, write_json


OUTPUT = archive.ROOT / "compressed_gradient_archive_v2"


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never restart or replace an archive extension")
    previous = json.loads((archive.OUTPUT / "independent_recheck.json").read_text())
    terminal = json.loads((archive.SOURCE / "status.json").read_text())
    if (not previous["complete"] or previous["canonical_paths_and_hashes_checked"] != 24
            or terminal["state"] != "complete" or Path(f'/proc/{terminal["pid"]}').exists()):
        raise ValueError("Require the completed independently verified first archive and idle source")
    records = []
    for index in range(24, 40):
        source = archive.SOURCE / "pairs" / f"pair_{index:03d}.pt"
        row = json.loads((archive.SOURCE / f"pair_{index:03d}.json").read_text())
        if source.is_symlink() or row["raw_gradients"] != str(source) or sha256(source) != row["raw_sha256"]:
            raise ValueError("Each new original gradient must match its registered identity and hash")
        records.append({"source": str(source), "sha256": row["raw_sha256"]})
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"records": records, "first_index": 24, "last_index": 39,
               "previous_recheck_sha256": sha256(archive.OUTPUT / "independent_recheck.json"),
               "source_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                                   str(Path(archive.__file__).resolve()): sha256(archive.__file__)},
               "registered_scientific_paths_and_bytes_unchanged": True})
    groups = []
    for start in range(0, len(records), archive.GROUP_SIZE):
        result = archive.compress_group(OUTPUT / f"group_{start // archive.GROUP_SIZE:02d}", records[start:start + archive.GROUP_SIZE])
        groups.append(result)
        print(json.dumps(result), flush=True)
    for row in records:
        if sha256(row["source"]) != row["sha256"]:
            raise ValueError("Canonical evidence changed during compression")
    write_json(OUTPUT / "audit.json", {"complete": True, "groups": groups, "canonical_artifacts": len(records),
               "net_bytes_reclaimed": sum(row["net_bytes_reclaimed"] for row in groups),
               "config_sha256": sha256(OUTPUT / "config.json"), "registered_hashes_and_paths_unchanged": True,
               "training_or_benchmark_modified": False, "original_information_preserved_losslessly": True})
    with (OUTPUT / "audit.json").open("rb") as stream:
        os.fsync(stream.fileno())
    archive.sync_directory(OUTPUT)


if __name__ == "__main__":
    main()
