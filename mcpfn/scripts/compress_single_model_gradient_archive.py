"""Losslessly compress completed gradient evidence while preserving canonical reads."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from recsys_evaluation_common import REPO, write_json
from relocate_single_model_gradient_archive import sha256, sync_directory


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
SOURCE = ROOT / "domain_gradient_probe_v1"
OUTPUT = ROOT / "compressed_gradient_archive_v1"
COUNT, GROUP_SIZE = 24, 4
MINIMUM_FREE = 21 * 1024**3


def replace_with_verified_link(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file() or not destination.is_file():
        raise ValueError("Require a regular original and an independently readable archive")
    if sha256(source) != expected or sha256(destination) != expected:
        raise ValueError("Never replace an original without identical archived bytes")
    size = source.stat().st_size
    temporary = source.with_suffix(source.suffix + ".compressed-link")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("Preserve any earlier incomplete link")
    temporary.symlink_to(destination)
    temporary.replace(source)
    sync_directory(source.parent)
    if not source.is_symlink() or sha256(source) != expected:
        raise ValueError("Canonical bytes must remain unchanged after atomic replacement")
    return {"source": str(source), "destination": str(destination), "sha256": expected,
            "bytes_archived": size, "canonical_path_preserved": True}


def mount_unit(image, mountpoint):
    name = subprocess.check_output(["systemd-escape", "--path", "--suffix=mount", str(mountpoint)], text=True).strip()
    unit = Path("/etc/systemd/system") / name
    if unit.exists():
        raise FileExistsError("Never replace an existing systemd mount unit")
    contents = (f"[Unit]\nDescription=Read-only single-model gradient archive\nRequiresMountsFor={image.parent}\n\n"
                f"[Mount]\nWhat={image}\nWhere={mountpoint}\nType=squashfs\nOptions=loop,ro\n\n"
                "[Install]\nWantedBy=multi-user.target\n")
    patch = f"*** Begin Patch\n*** Add File: {unit}\n" + "".join("+" + line + "\n" for line in contents.splitlines()) + "*** End Patch\n"
    subprocess.run(["apply_patch", patch], check=True)
    with unit.open("rb") as stream:
        os.fsync(stream.fileno())
    sync_directory(unit.parent)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "--now", name], check=True)
    subprocess.run(["mountpoint", "--quiet", str(mountpoint)], check=True)
    subprocess.run(["systemctl", "is-enabled", "--quiet", name], check=True)
    return {"unit": str(unit), "unit_sha256": sha256(unit), "persistent_mount_enabled": True}


def compress_group(directory, records):
    directory.mkdir()
    image, mountpoint = directory / "gradients.squashfs", directory / "files"
    mountpoint.mkdir()
    maximum_bytes = sum(Path(row["source"]).stat().st_size for row in records)
    if shutil.disk_usage("/run").free < maximum_bytes + 4 * 1024**3:
        raise OSError("Keep ample RAM-backed build space without consuming root reserve")
    with tempfile.TemporaryDirectory(prefix="single-model-squash-", dir="/run") as scratch:
        staging = Path(scratch) / image.name
        with (directory / "build.log").open("x") as stream:
            subprocess.run(["mksquashfs", *[row["source"] for row in records], str(staging), "-comp", "zstd",
                            "-Xcompression-level", "8", "-b", "1048576", "-processors", "4", "-noappend", "-no-progress"],
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        compressed_bytes = staging.stat().st_size
        if compressed_bytes >= maximum_bytes or shutil.disk_usage(ROOT).free - compressed_bytes < MINIMUM_FREE:
            raise OSError("Require net savings and21 GiB free before replacing any original")
        shutil.copyfile(staging, image)
        with image.open("rb") as stream:
            os.fsync(stream.fileno())
        if sha256(image) != sha256(staging):
            raise ValueError("Durable compressed image must match the completed build")
    sync_directory(directory)
    mounted = mount_unit(image, mountpoint)
    for row in records:
        archived = mountpoint / Path(row["source"]).name
        if sha256(archived) != row["sha256"]:
            raise ValueError("Verify every decompressed byte before replacing any canonical file")
        payload = torch.load(archived, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise ValueError("Archived gradient payload must retain readable torch mmap serialization")
        del payload
    verified = {"image": str(image), "image_sha256": sha256(image), "image_bytes": compressed_bytes,
                "uncompressed_bytes": maximum_bytes, "all_original_hashes_verified": True,
                "torch_mmap_readable": True, **mounted, "records": records}
    write_json(directory / "verified_before_replacement.json", verified)
    with (directory / "verified_before_replacement.json").open("rb") as stream:
        os.fsync(stream.fileno())
    sync_directory(directory)
    replacements = []
    for row in records:
        replacements.append(replace_with_verified_link(row["source"], mountpoint / Path(row["source"]).name, row["sha256"]))
        with (directory / "replacement.jsonl").open("a") as stream:
            stream.write(json.dumps(replacements[-1]) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    write_json(directory / "audit.json", {**verified, "complete": True, "replacements": replacements,
               "net_bytes_reclaimed": maximum_bytes - compressed_bytes})
    sync_directory(directory)
    return {"group": str(directory), "audit_sha256": sha256(directory / "audit.json"),
            "net_bytes_reclaimed": maximum_bytes - compressed_bytes}


def main():
    status = json.loads((SOURCE / "status.json").read_text())
    if status["state"] != "complete" or Path(f'/proc/{status["pid"]}').exists() or OUTPUT.exists():
        raise ValueError("Require idle completed owned evidence and a fresh archive")
    summary = json.loads((SOURCE / "summary.json").read_text())
    if not summary["complete"] or summary["training_performed"] or summary["checkpoint_selection_performed"]:
        raise ValueError("Archive only the completed diagnostic, never live training or a baseline checkpoint")
    records = []
    for index in range(COUNT):
        path = SOURCE / "pairs" / f"pair_{index:03d}.pt"
        record = json.loads((SOURCE / f"pair_{index:03d}.json").read_text())
        if path.is_symlink() or record["raw_gradients"] != str(path) or sha256(path) != record["raw_sha256"]:
            raise ValueError("Original paired-gradient identity/hash changed")
        records.append({"source": str(path), "sha256": record["raw_sha256"]})
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"source": str(SOURCE), "records": records, "group_size": GROUP_SIZE,
               "minimum_free_bytes_before_replacement": MINIMUM_FREE, "read_only_persistent_mounts": True,
               "source_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                   str(REPO / "tests/test_single_model_compressed_archive.py"): sha256(REPO / "tests/test_single_model_compressed_archive.py")},
               "scientific_bytes_or_registered_paths_changed": False})
    groups = []
    for start in range(0, COUNT, GROUP_SIZE):
        result = compress_group(OUTPUT / f"group_{start // GROUP_SIZE:02d}", records[start:start + GROUP_SIZE])
        groups.append(result)
        print(json.dumps(result), flush=True)
    for row in records:
        if sha256(row["source"]) != row["sha256"]:
            raise ValueError("Every canonical gradient must remain readable and identical")
    write_json(OUTPUT / "audit.json", {"complete": True, "groups": groups, "canonical_artifacts": COUNT,
               "net_bytes_reclaimed": sum(row["net_bytes_reclaimed"] for row in groups),
               "config_sha256": sha256(OUTPUT / "config.json"), "registered_hashes_and_paths_unchanged": True,
               "training_or_benchmark_modified": False, "original_information_preserved_losslessly": True})
    sync_directory(OUTPUT)


if __name__ == "__main__":
    main()
