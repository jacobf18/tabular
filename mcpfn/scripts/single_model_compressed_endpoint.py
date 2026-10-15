"""Write exact optimizer evidence through RAM to a persistent read-only archive."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from compress_single_model_gradient_archive import mount_unit
from recsys_evaluation_common import sha256, write_json
from relocate_single_model_gradient_archive import sync_directory
import single_model_composite_gradient_storage as composite
import single_model_shared_tensor_storage as accounting


MINIMUM_FREE = 18 * 1024**3
ROOT_RESERVE = 6 * 1024**3
RAM_RESERVE = 4 * 1024**3
ALLOWANCE = accounting.METADATA_ALLOWANCE
POLICY = ("Serialize ordinary torch CPU evidence in RAM, preserving every protected gradient and the "
          "unchanged shorter composite proposal alias. Compress the exact serialized bytes in RAM with "
          "zstd squashfs before any durable image write. Check actual image size against the18 GiB "
          "durable reserve; never stage raw endpoint bytes on root or the checkpoint disk. Persist and "
          "fsync the image, enable a read-only persistent mount, verify exact serialized SHA256 and "
          "torch mmap gradient aliases, then publish a fresh canonical symlink and a distinct receipt. "
          "No overwrite, lossy encoding, omitted tensors, or claim that this is an uncompressed write.")


def exists(path):
    return path.exists() or path.is_symlink()


def filesystem(path):
    return subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(path)], text=True).strip()


def verify_mount(directory, mounted):
    mountpoint = directory / "files"
    subprocess.run(["mountpoint", "--quiet", str(mountpoint)], check=True)
    options = subprocess.check_output(["findmnt", "-n", "-o", "OPTIONS", "-T", str(mountpoint)], text=True).strip().split(",")
    unit = Path(mounted["unit"])
    expected_name = subprocess.check_output(["systemd-escape", "--path", "--suffix=mount", str(mountpoint)], text=True).strip()
    if (filesystem(mountpoint) != "squashfs" or "ro" not in options or "rw" in options
            or unit != Path("/etc/systemd/system") / expected_name
            or not mounted["persistent_mount_enabled"] or sha256(unit) != mounted["unit_sha256"]):
        raise ValueError("Require the exact persistent read-only squashfs mount")
    subprocess.run(["systemctl", "is-enabled", "--quiet", unit.name], check=True)
    return {str(unit): sha256(unit)}


def compress(source, image, log):
    with log.open("x") as stream:
        subprocess.run(["mksquashfs", str(source), str(image), "-comp", "zstd", "-Xcompression-level", "8",
                        "-b", "1048576", "-processors", "4", "-noappend", "-no-progress"],
                       check=True, stdout=stream, stderr=subprocess.STDOUT)


def copy_exclusive(source, destination):
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())


def save(payload, path):
    path = Path(path).absolute()
    directory = path.with_suffix(".compressed")
    receipt = path.with_suffix(".compressed-write.json")
    if any(exists(current) for current in (path, directory, receipt)):
        raise FileExistsError("Never replace endpoint evidence, an unfinished archive, or its receipt")
    composite.check_alias(payload)
    tensor_bytes = accounting.storage_bytes(payload)
    planned = tensor_bytes + ALLOWANCE
    if filesystem("/run") != "tmpfs":
        raise OSError("Stage endpoint bytes only on the RAM-backed /run filesystem")
    if shutil.disk_usage("/run").free < 2 * planned + RAM_RESERVE:
        raise OSError("Reserve RAM for both ordinary serialized bytes and the compressed image")
    if shutil.disk_usage("/").free < ROOT_RESERVE + ALLOWANCE:
        raise OSError("Preserve the6 GiB root reserve for the small persistent mount unit")
    if shutil.disk_usage(path.parent).free < MINIMUM_FREE + ALLOWANCE:
        raise OSError("Preserve the18 GiB durable reserve before staging endpoint evidence")
    directory.mkdir()
    with tempfile.TemporaryDirectory(prefix="single-model-endpoint-", dir="/run") as scratch_name:
        scratch = Path(scratch_name)
        raw, staging, log = scratch / path.name, scratch / "endpoint.squashfs", scratch / "build.log"
        torch.save(payload, raw)
        size, digest = raw.stat().st_size, sha256(raw)
        if size > planned:
            raise OSError("The registered serialization allowance was exceeded")
        restored = torch.load(raw, map_location="cpu", weights_only=False, mmap=True)
        composite.check_alias(restored)
        if accounting.storage_bytes(restored) != tensor_bytes:
            raise ValueError("Ordinary serialization must preserve every distinct storage")
        del restored
        compress(raw, staging, log)
        compressed_bytes = staging.stat().st_size
        before = shutil.disk_usage(path.parent).free
        if before - compressed_bytes - ALLOWANCE < MINIMUM_FREE:
            raise OSError("Actual compressed image would violate the18 GiB durable reserve")
        image = directory / "endpoint.squashfs"
        copy_exclusive(staging, image)
        image_digest = sha256(staging)
        if sha256(image) != image_digest:
            raise ValueError("Durable image must match all RAM-built compressed bytes")
        copy_exclusive(log, directory / "build.log")
    sync_directory(directory)
    mountpoint = directory / "files"
    mountpoint.mkdir()
    mounted = mount_unit(image, mountpoint)
    verify_mount(directory, mounted)
    archived = mountpoint / path.name
    if sha256(archived) != digest or archived.stat().st_size != size:
        raise ValueError("Read-only archive must reproduce every original serialized byte")
    restored = torch.load(archived, map_location="cpu", weights_only=False, mmap=True)
    composite.check_alias(restored)
    if accounting.storage_bytes(restored) != tensor_bytes:
        raise ValueError("Mounted evidence must retain every distinct tensor storage")
    del restored
    after = shutil.disk_usage(path.parent).free
    if after < MINIMUM_FREE or shutil.disk_usage("/").free < ROOT_RESERVE:
        raise OSError("Do not publish endpoint evidence after a concurrent reserve violation")
    report = {"format": "ram-serialized-squashfs-composite-v1", "policy": POLICY, "path": str(path),
              "sha256": digest, "file_bytes": size, "tensor_storage_bytes": tensor_bytes,
              "metadata_allowance_bytes": ALLOWANCE, "planned_raw_bytes": planned,
              "image": str(image), "image_sha256": image_digest, "image_bytes": compressed_bytes,
              "archive_path": str(archived), "free_before_image": before, "free_after": after,
              "minimum_free_bytes": MINIMUM_FREE, "raw_staging_filesystem": "tmpfs",
              "all_tensor_bytes_retained": True, "composite_proposal_alias_preserved": True,
              "composite_tasks": payload["record"]["composite_tasks"],
              "protected_tasks": payload["record"]["protected_tasks"], **mounted}
    write_json(directory / "verified_before_publication.json", report)
    with (directory / "verified_before_publication.json").open("rb") as stream:
        os.fsync(stream.fileno())
    sync_directory(directory)
    path.symlink_to(archived)
    sync_directory(path.parent)
    write_json(receipt, report)
    with receipt.open("rb") as stream:
        os.fsync(stream.fileno())
    sync_directory(path.parent)
    return digest


def audit(payload, path):
    path = Path(path).absolute()
    directory, receipt = path.with_suffix(".compressed"), path.with_suffix(".compressed-write.json")
    report = json.loads(receipt.read_text())
    expected = accounting.storage_bytes(payload)
    image = directory / "endpoint.squashfs"
    archived = directory / "files" / path.name
    composite.check_alias(payload)
    if (report["format"] != "ram-serialized-squashfs-composite-v1" or report["policy"] != POLICY
            or report["path"] != str(path) or report["sha256"] != sha256(path)
            or report["tensor_storage_bytes"] != expected or report["metadata_allowance_bytes"] != ALLOWANCE
            or report["planned_raw_bytes"] != expected + ALLOWANCE
            or report["file_bytes"] != path.stat().st_size or report["file_bytes"] > report["planned_raw_bytes"]
            or report["image"] != str(image) or report["image_sha256"] != sha256(image)
            or report["image_bytes"] != image.stat().st_size or report["archive_path"] != str(archived)
            or not path.is_symlink() or path.readlink() != archived
            or report["minimum_free_bytes"] != MINIMUM_FREE
            or report["free_before_image"] - report["image_bytes"] - ALLOWANCE < MINIMUM_FREE
            or report["free_after"] < MINIMUM_FREE or report["raw_staging_filesystem"] != "tmpfs"
            or report["all_tensor_bytes_retained"] is not True
            or report["composite_proposal_alias_preserved"] is not True
            or report["composite_tasks"] != payload["record"]["composite_tasks"]
            or report["protected_tasks"] != payload["record"]["protected_tasks"]
            or report != json.loads((directory / "verified_before_publication.json").read_text())):
        raise ValueError("Compressed endpoint receipt must reproduce exact bytes, aliases, and storage reserves")
    restored = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    composite.check_alias(restored)
    if accounting.storage_bytes(restored) != expected:
        raise ValueError("Audit the actual mounted tensor storages, not only a supplied payload")
    del restored
    return {str(receipt): sha256(receipt), str(image): sha256(image), **verify_mount(directory, report)}
