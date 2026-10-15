"""Replay every registered synthetic mask and audit saved clean/noisy arrays on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_single_model_missingness_coverage as probe
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json


def audit_normalization(source, target, stored):
    query = stored["query"]
    if query.dtype != np.bool_ or query.shape != source.shape or target.shape != source.shape:
        raise ValueError("Invalid query mask or raw pair shape")
    keep = ~query
    counts = keep.sum(axis=0, keepdims=True)
    observed = np.where(keep, source.astype(np.float64), 0.)
    mean = observed.sum(axis=0, keepdims=True) / np.maximum(counts, 1)
    centered = np.where(keep, observed - mean, 0.)
    squared = (centered ** 2).sum(axis=0, keepdims=True)
    pool = squared.sum() / max(int(counts.sum()), 1)
    pool = pool if pool > 1e-6 else 1.
    scale = np.sqrt(np.maximum((squared + pool) / (counts + 1), 1e-6))
    normalized = np.where(keep, (source - mean) / scale, np.nan)
    truth = (target - mean) / scale
    for name, expected in (("observed_mean", mean), ("observed_scale", scale),
                           ("model_source", normalized), ("model_target", truth)):
        np.testing.assert_allclose(stored[name], expected, rtol=2e-5, atol=2e-5, equal_nan=True)
    if not np.isfinite(stored["model_target"]).all() or not np.array_equal(np.isnan(stored["model_source"]), query):
        raise ValueError("Normalization lost query or clean-target coverage")


def audit(directory):
    config = json.loads((directory / "config.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    records = json.loads((directory / "per_case.json").read_text())
    if status["state"] != "complete" or status["cases"] != 192 or len(records) != 192:
        raise ValueError("Require every original paired case")
    digest = sha256(directory / "config.json")
    if manifest["config_sha256"] != digest or status["config_sha256"] != digest:
        raise ValueError("Registration changed")
    original_hashes = {str(directory / filename): sha256(directory / filename) for filename in
                       ("config.json", "status.json", "manifest.json", "per_case.json", "per_case.csv", "per_family.csv", "RESULTS.md")}
    if len(manifest["files"]) != 208 or len({record["path"] for record in manifest["files"]}) != 208:
        raise ValueError("Incomplete saved raw pairs and masks")
    expected_paths = {str(directory / f"context_{context['id']:02d}" / filename)
                      for context in config["contexts"] for filename in
                      ("raw_pair.npz", *[f"{family}.npz" for family in probe.FAMILIES])}
    if {record["path"] for record in manifest["files"]} != expected_paths:
        raise ValueError("Unexpected artifact path coverage")
    original_hashes.update({record["path"]: record["sha256"] for record in manifest["files"]})
    verify_hashes(config["source_hashes"])
    verify_hashes(original_hashes)
    keys = {(record["context"], record["family"]) for record in records}
    if len(keys) != 192 or keys != {(context["id"], family) for context in config["contexts"] for family in probe.FAMILIES}:
        raise ValueError("Duplicate or missing paired cases")
    indexed = {(record["context"], record["family"]): record for record in records}
    distinct_spectral_pairs = 0
    for context in config["contexts"]:
        folder = directory / f"context_{context['id']:02d}"
        source, target, mechanism_source = probe.generate_pair(context)
        with np.load(folder / "raw_pair.npz", allow_pickle=False) as raw:
            for name, regenerated in (("source", source), ("target", target), ("mechanism_source", mechanism_source)):
                if not np.array_equal(raw[name], regenerated.numpy()):
                    raise ValueError("Raw pair does not reproduce from its registered seed")
        if context["kind"] == "spectrum_decay":
            if torch.equal(source, target):
                raise ValueError("Spectral target unexpectedly replaced by noisy source")
            distinct_spectral_pairs += 1
        for pattern in context["patterns"]:
            family = pattern["family"]
            query = probe.apply_family(mechanism_source, pattern["config"], family, pattern["seed"])
            with np.load(folder / f"{family}.npz", allow_pickle=False) as stored:
                if not np.array_equal(stored["query"], query.numpy()):
                    raise ValueError("Stored mask does not reproduce")
                audit_normalization(source.numpy(), target.numpy(), stored)
            actual = probe.describe_mask(source, target, query)
            row = indexed[context["id"], family]
            if any(row[key] != value for key, value in actual.items()):
                raise ValueError("Saved mask statistics changed")
    verify_hashes(original_hashes)
    verify_hashes(config["source_hashes"])
    return {"complete": True, "config_sha256": digest, "cases": len(records), "raw_pairs": len(config["contexts"]),
            "noisy_clean_spectral_pairs": distinct_spectral_pairs, "all_masks_replayed": True,
            "normalization_independently_recomputed_numpy_float64": True, "atol": 2e-5, "rtol": 2e-5,
            "model_inference_performed": False, "original_artifacts_unchanged": True,
            "original_artifact_sha256": original_hashes, "auditor_sha256": sha256(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never replace an audit")
    torch.set_num_threads(2)
    result = audit(args.probe.resolve())
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "original_artifact_sha256"}))


if __name__ == "__main__":
    main()
