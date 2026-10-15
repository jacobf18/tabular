"""Move completed diagnostic bytes durably while preserving every canonical path and hash."""

import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
DESTINATION = Path("/root/tabular/mcpfn/runs/single_model_goal_20261005/durable_gradient_archive/point_alignment_v1/v8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def relocate(source, destination, expected, *, minimum_free=6 * 1024**3):
    source, destination = Path(source), Path(destination)
    if sha256(source) != expected:
        raise ValueError("Never relocate an artifact with changed contents")
    if source.is_symlink():
        if source.resolve() != destination.resolve() or sha256(destination) != expected:
            raise ValueError("An existing canonical link must bind the same durable archive")
        return {"source": str(source), "destination": str(destination), "sha256": expected, "already_linked": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists():
        if sha256(destination) != expected:
            raise FileExistsError("Never replace unrelated archive contents")
    else:
        if temporary.exists():
            raise FileExistsError("Preserve an incomplete copy for explicit inspection")
        if shutil.disk_usage(destination.parent).free - size < minimum_free:
            raise OSError("Retain the destination filesystem's free-space reserve")
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if sha256(temporary) != expected:
            raise ValueError("The durable copy must verify before replacing any source entry")
        temporary.replace(destination)
        sync_directory(destination.parent)
    link = source.with_suffix(source.suffix + ".archive-link")
    if link.exists() or link.is_symlink():
        raise FileExistsError("Never replace an existing staging link")
    link.symlink_to(destination)
    link.replace(source)
    sync_directory(source.parent)
    if not source.is_symlink() or sha256(source) != expected or sha256(destination) != expected:
        raise ValueError("Canonical bytes must remain identical after atomic relocation")
    return {"source": str(source), "destination": str(destination), "sha256": expected, "bytes_relocated": size,
            "canonical_path_preserved": True, "durable_copy_fsynced": True}


def main():
    source = ROOT / "point_alignment_v1"
    status = json.loads((source / "status.json").read_text())
    if status["state"] != "complete" or Path(f'/proc/{status["pid"]}').exists():
        raise ValueError("Relocate only completed, idle diagnostic artifacts")
    result = json.loads((source / "v8/result.json").read_text())
    records = []
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for task in result["tasks"][:11]:
        path = Path(task["artifact"])
        if path.parent != source / "v8" or path.name != f'task_{len(records):02d}.pt':
            raise ValueError("Only the first eleven owned V8 gradient artifacts may move")
        record = relocate(path, DESTINATION / path.name, task["sha256"])
        records.append(record)
        with (DESTINATION / "relocation.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
    report = {"complete": True, "artifacts": records, "script_sha256": sha256(__file__),
              "canonical_paths_and_registered_hashes_unchanged": True, "experiment_code_or_results_changed": False}
    with (DESTINATION / "audit.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(DESTINATION)


if __name__ == "__main__":
    main()
