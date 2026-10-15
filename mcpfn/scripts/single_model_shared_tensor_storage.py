"""Account for exact serialized CPU tensor storages without counting aliases twice."""

import json
import shutil

import torch

from recsys_evaluation_common import sha256, write_json
from streamed_large_matrix_training import atomic_torch_save


MINIMUM_FREE = 18 * 1024**3
METADATA_ALLOWANCE = 4 * 1024**2


def storage_bytes(payload):
    storages = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError("Account only for explicit CPU strided tensor storages")
            storage = value.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, dict):
            for key, nested in value.items():
                visit(key)
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError("Never silently omit an unsupported serialized object")

    visit(payload)
    return sum(storages.values())


def save(payload, path):
    receipt = path.with_suffix(".write.json")
    if path.exists() or receipt.exists():
        raise FileExistsError("Never overwrite raw optimizer evidence or its storage receipt")
    tensor_bytes = storage_bytes(payload)
    required = tensor_bytes + METADATA_ALLOWANCE
    before = shutil.disk_usage(path.parent).free
    if before - required < MINIMUM_FREE:
        raise OSError("Preserve the18 GiB write reserve including every unique serialized storage")
    atomic_torch_save(payload, path)
    size = path.stat().st_size
    after = shutil.disk_usage(path.parent).free
    if size > required or after < MINIMUM_FREE:
        raise OSError("Preserve unexpected output for investigation; serialized allowance or reserve exceeded")
    digest = sha256(path)
    write_json(receipt, {"path": str(path), "sha256": digest, "tensor_storage_bytes": tensor_bytes,
                        "metadata_allowance_bytes": METADATA_ALLOWANCE, "planned_bytes": required,
                        "file_bytes": size, "free_before": before, "free_after": after,
                        "minimum_free_bytes": MINIMUM_FREE, "all_tensor_bytes_retained": True})
    return digest


def audit(payload, path):
    receipt = path.with_suffix(".write.json")
    report = json.loads(receipt.read_text())
    expected = storage_bytes(payload)
    if (report["path"] != str(path) or report["sha256"] != sha256(path)
            or report["tensor_storage_bytes"] != expected or report["metadata_allowance_bytes"] != METADATA_ALLOWANCE
            or report["planned_bytes"] != expected + METADATA_ALLOWANCE or report["file_bytes"] != path.stat().st_size
            or report["file_bytes"] > report["planned_bytes"] or report["minimum_free_bytes"] != MINIMUM_FREE
            or report["free_before"] - report["planned_bytes"] < MINIMUM_FREE or report["free_after"] < MINIMUM_FREE
            or not report["all_tensor_bytes_retained"]):
        raise ValueError("Shared-storage receipt must reproduce without reducing the write reserve")
    return {str(receipt): sha256(receipt)}
