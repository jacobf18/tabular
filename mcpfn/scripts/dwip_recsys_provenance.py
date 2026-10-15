"""Fail-closed checkpoint lineage for synthetic-only DWIP experiments."""
import hashlib
import json
from pathlib import Path


POLICY = "dwip-recsys-synthetic-only-v1"
TRAINING_DATA = "Synthetic only; no real ratings in gradient updates"
AUDITED_ROOTS = {
    "1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d": "original synthetic teacher-mean checkpoint",
    "c9041c7a3bde914a4c2cfb051fd2c9f1619099ade264815d658d89b92cc7a6a9": "full_context_expected_mae_250; synthetic ancestry audited",
}
INELIGIBLE_HASHES = {
    "b52d16c10ca9e450d27307118edfbb8838f4e5d07c81a1b37ce1dea5fe868469",
    "2c28b60781e6ce4a3a9389029046a499771595ecb51aa371cfa16b7ea82a528b",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_path(checkpoint):
    return Path(str(checkpoint) + ".provenance.json")


def require_synthetic_checkpoint(checkpoint, _visited=None):
    checkpoint = Path(checkpoint).resolve()
    visited = set() if _visited is None else set(_visited)
    if checkpoint in visited:
        raise ValueError("Cyclic checkpoint lineage")
    visited.add(checkpoint)
    digest = file_sha256(checkpoint)
    if digest in INELIGIBLE_HASHES:
        raise ValueError("Ineligible real-trained checkpoint; synthetic fine-tuning cannot clean its lineage")
    if digest in AUDITED_ROOTS:
        return {"policy": POLICY, "checkpoint_sha256": digest, "audited_root": digest,
                "eligible": True}
    sidecar = receipt_path(checkpoint)
    if not sidecar.is_file():
        raise ValueError("Unknown checkpoint provenance; require an audited synthetic root or verified lineage receipt")
    receipt = json.loads(sidecar.read_text())
    if receipt.get("policy") != POLICY or receipt.get("checkpoint_sha256") != digest:
        raise ValueError("Checkpoint provenance policy/hash mismatch")
    config_path = checkpoint.parent / "config.json"
    if file_sha256(config_path) != receipt.get("config_sha256"):
        raise ValueError("Training config provenance hash mismatch")
    config = json.loads(config_path.read_text())
    if (config.get("training_policy") != POLICY or config.get("training_data") != TRAINING_DATA
            or config.get("arguments", {}).get("real_adaptation")
            or config.get("adaptation_provenance") or config.get("adaptation_schedule")):
        raise ValueError("Ineligible non-synthetic training manifest")
    parent_path = Path(config["initial_checkpoint"])
    if not parent_path.is_absolute():
        raise ValueError("Lineage requires an absolute parent checkpoint path")
    parent = require_synthetic_checkpoint(parent_path, visited)
    if parent["checkpoint_sha256"] != config.get("initial_sha256"):
        raise ValueError("Parent checkpoint provenance hash mismatch")
    return {"policy": POLICY, "checkpoint_sha256": digest, "audited_root": parent["audited_root"],
            "eligible": True}


def write_synthetic_receipt(checkpoint):
    checkpoint = Path(checkpoint)
    receipt = {"policy": POLICY, "checkpoint_sha256": file_sha256(checkpoint),
               "config_sha256": file_sha256(checkpoint.parent / "config.json")}
    destination = receipt_path(checkpoint)
    temporary = destination.with_suffix(".tmp")
    if destination.is_symlink() or temporary.exists():
        raise ValueError("Preserve unexpected provenance receipt path")
    with temporary.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    temporary.replace(destination)


def require_synthetic_episode(episode):
    if episode.metadata.get("gradient_data") != "synthetic":
        raise ValueError("Only explicitly synthetic episodes may enter a gradient update")
