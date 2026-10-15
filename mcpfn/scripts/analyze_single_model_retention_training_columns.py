"""Inspect all consumed native-anchor retention columns without model inference."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_admission_attempts as attempts
import single_model_column_anchor_retention as retention
from single_model_compact_training_cache import validate_capsule
from single_model_composite_gradient_storage import byte_equal
from single_model_structured_mixture import StructuredSampler


POLICY = ("CPU-only descriptive replay of every actually eligible retention task in a completed registered "
    "training prefix, including optimizer-rejected proposals and preserving data-only rejections separately. "
    "Use consumed seeds only, original generator sources and masks, exact archived input/mask/metadata "
    "identity, and all columns including query-empty, single-query and constant columns. Independently "
    "recompute the prospective equal-column inverse-variance coefficients and compare their float32 "
    "implementation. Report hypothetical unit-prediction-drift penalty weights, not realized student "
    "losses, gradients, dominance or causal effects. No model loading, teacher or student inference, CUDA "
    "initialization, new training data, selection, gate change or real-data evaluation. This source/mask "
    "replay does not replace the mandatory full task/teacher digest and training audit.")


def read(path):
    return json.loads(Path(path).read_text())


def prefix(path, boundary):
    lines = Path(path).read_text().splitlines(keepends=True)[:boundary]
    records = [json.loads(line) for line in lines]
    if len(records) != boundary or [row["step"] for row in records] != list(range(1, boundary + 1)):
        raise ValueError("Require the entire completed contiguous training prefix")
    return "".join(lines), records


def specification_identity(task):
    specification = task["specification"]
    structured = specification.get("structured")
    if structured is not None:
        if specification["domain"] != "imputation":
            raise ValueError("Never relabel ratings as structured imputation")
        return structured["seed"], structured["component"] == "anchor_latent_mcar"
    if specification["domain"] not in {"imputation", "recsys"}:
        raise ValueError("Require an original synthetic domain")
    return specification["seed"], specification["domain"] == "imputation" and specification["component"] in (6, 7)


def column_rows(source, target, mask):
    retention.validate_columns(target, mask)
    if source.shape != target.shape or source.device.type != "cpu" or target.device.type != "cpu":
        raise ValueError("Inspect only the full original CPU episode")
    coefficients = retention.column_coefficients(target, mask)[0].numpy()
    labels, selected = target[0].double().numpy(), mask[0].numpy()
    total_queries = int(selected.sum())
    queried_columns = int(selected.any(axis=0).sum())
    rows = []
    for column_index in range(labels.shape[1]):
        queries = selected[:, column_index]
        count = int(queries.sum())
        variance = float(np.var(labels[queries, column_index], ddof=0)) if count else None
        denominator = max(variance, retention.SPECIFICATION["variance_floor"]) if count else None
        expected = total_queries / (queried_columns * count * denominator) if count else 0.
        actual = coefficients[queries, column_index]
        if count and not np.allclose(actual, np.float32(expected), rtol=1e-6, atol=0.):
            raise ValueError("Independent column coefficient must match the streamed float32 implementation")
        if np.any(coefficients[~queries, column_index] != 0):
            raise ValueError("Nonqueries must not receive retention coefficients")
        rows.append({"column": column_index, "queries": count,
            "observations": int(torch.isfinite(source[0, :, column_index]).sum()),
            "variance": variance, "target_std": math.sqrt(variance) if count else None,
            "variance_floor_active": count > 0 and variance < retention.SPECIFICATION["variance_floor"],
            "single_query": count == 1, "constant_multi_query": count > 1 and variance == 0.,
            "float32_coefficient": float(actual[0]) if count else 0.,
            "unit_drift_penalty": retention.SPECIFICATION["weight"] / (queried_columns * denominator) if count else 0.})
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        for grouping, value in (("all", "ALL"), ("cohort", row["cohort"]),
                ("family", row["family"]), ("query_class", "empty" if row["queries"] == 0
                    else "single" if row["single_query"] else "constant_multi" if row["constant_multi_query"]
                    else "nonconstant_floor" if row["variance_floor_active"] else "nonconstant_unfloored")):
            groups[grouping, value].append(row)
    result = []
    for (grouping, value), members in sorted(groups.items()):
        result.append({"grouping": grouping, "group": value, "columns": len(members),
            "tasks": len({(row["step"], row["task_index"]) for row in members}),
            "queries": sum(row["queries"] for row in members),
            "floor_columns": sum(row["variance_floor_active"] for row in members),
            "unit_drift_penalty": math.fsum(row["unit_drift_penalty"] for row in members),
            "task_weighted_unit_drift_penalty": math.fsum(row["task_weight"] * row["unit_drift_penalty"] for row in members)})
    return result


def save_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run, boundary, output):
    if torch.cuda.is_initialized() or output.exists() or output.is_symlink():
        raise ValueError("Never initialize CUDA or overwrite an existing diagnosis")
    config = read(run / "config.json")
    if (sha256(run / "config.json") != read(run / "registration.json")["config_sha256"]
            or config["anchor_retention"] != retention.SPECIFICATION
            or boundary not in config["boundaries"] or boundary <= 0
            or read(run / f"validation_{boundary}.json")["step"] != boundary):
        raise ValueError("Bind the original registration and a completed trained boundary")
    if shutil.disk_usage(output.parent).free < 50 * 1024**3 + 64 * 1024**2:
        raise OSError("Preserve50 GiB scratch reserve plus64 MiB diagnostic capacity")
    verify_hashes(config["source_sha256"])
    snapshot, records = prefix(run / "training.jsonl", boundary)
    inputs = {str(run / name): sha256(run / name)
        for name in ("config.json", "registration.json", f"validation_{boundary}.json")}

    def bind(path, expected=None):
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("A registered or training-log-bound artifact changed")
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError("One immutable input cannot have conflicting identities")
        inputs[str(path)] = digest

    admissions = []
    for record in records:
        bind(record["admission_manifest"], record["admission_sha256"])
        admissions.append(read(record["admission_manifest"]))
    counts = attempts.inventory(config["batches"][:boundary], records, admissions)
    tasks = []
    decisions = 0
    for batch, record in zip(config["batches"][:boundary], records):
        if record["attempt_kind"] == "data_admission_rejected":
            continue
        bind(record["task_manifest"], record["task_manifest_sha256"])
        bind(record["retention_manifest"], record["retention_manifest_sha256"])
        captured, manifest = read(record["task_manifest"]), read(record["retention_manifest"])
        if (captured["step"] != batch["step"] or len(captured["tasks"]) != len(batch["tasks"])
                or manifest["complete"] is not True or manifest["specification"] != retention.SPECIFICATION
                or manifest["backward_source_sha256"] != retention.BACKWARD_SOURCE_SHA256
                or manifest["reference"]["sha256"] != retention.REFERENCE_SHA256
                or manifest["inference_ensemble"] is not False
                or len(manifest["records"]) != len(batch["tasks"])):
            raise ValueError("Retain all actual task slots and original retention decisions")
        by_seed = {item["seed"]: item for item in manifest["records"]}
        if len(by_seed) != len(batch["tasks"]):
            raise ValueError("Each consumed task must have a distinct original seed")
        for registered, task in zip(batch["tasks"], captured["tasks"]):
            if any(task[name] != value for name, value in registered.items()):
                raise ValueError("Captured task differs from the registered generator specification")
            seed, selected = specification_identity(registered)
            decision = by_seed[seed]
            if decision["eligible"] is not selected or retention.eligible(decision["metadata"]) is not selected:
                raise ValueError("Check every original eligible and ineligible task")
            decisions += 1
            bind(task["artifact"], task["sha256"])
            if not selected:
                if "artifact" in decision or "sha256" in decision:
                    raise ValueError("Ineligible tasks must not gain a retention reference")
                continue
            bind(decision["artifact"], decision["sha256"])
            tasks.append((batch["step"], registered, task, decision))
    if not tasks or decisions != counts["training_tasks"]:
        raise ValueError("Require complete training coverage and at least one actual retention task")
    verify_hashes(inputs)
    sources = {str(Path(module.__file__).resolve()): sha256(module.__file__) for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file()}
    sources[str(REPO / "tests/test_single_model_retention_training_columns.py")] = sha256(
        REPO / "tests/test_single_model_retention_training_columns.py")
    output.mkdir()
    write_json(output / "config.json", {"policy": POLICY, "run": str(run), "boundary": boundary,
        "training_prefix_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
        "input_sha256": inputs, "source_sha256": sources, "registered_generator_source_sha256": config["source_sha256"]})
    with (output / "training_prefix.jsonl").open("x") as stream:
        stream.write(snapshot)
    write_json(output / "status.json", {"state": "analyzing", "boundary": boundary})
    sampler = StructuredSampler(config["base"], device="cpu")
    rows, task_rows = [], []
    for step, registered, task, decision in tasks:
        capsule = torch.load(task["artifact"], map_location="cpu", weights_only=False)
        validate_capsule(capsule)
        if capsule["task"] != registered or capsule["scaling"] is not None:
            raise ValueError("Use the original native-anchor capsule, never rating scaling or modified tasks")
        episode = sampler.sample(step=1, **registered["specification"])
        cached = torch.load(decision["artifact"], map_location="cpu", weights_only=False)
        if (not retention.eligible(episode.metadata) or cached["metadata"] != episode.metadata
                or decision["metadata"] != episode.metadata or cached["specification"] != retention.SPECIFICATION
                or cached["identity"]["sha256"] != retention.REFERENCE_SHA256
                or not byte_equal(episode.source, cached["source"]) or not byte_equal(episode.mask, cached["mask"])
                or list(episode.target.shape) != decision["shape"] or int(episode.mask.sum()) != decision["queries"]):
            raise ValueError("CPU regeneration must preserve the complete archived source, mask, metadata and reference identity")
        details = {"step": step, "task_index": registered["index"], "name": registered["name"],
            "seed": episode.metadata["seed"], "cohort": "structured" if episode.metadata.get("structured", False) else "legacy",
            "family": episode.metadata.get("family", "legacy"), "task_weight": registered["weight"]}
        current = [{**details, **row} for row in column_rows(episode.source, episode.target, episode.mask)]
        rows.extend(current)
        task_rows.append({**details, "rows": episode.target.shape[1], "columns": len(current),
            "queries": int(episode.mask.sum()), "queried_columns": sum(row["queries"] > 0 for row in current),
            "floor_columns": sum(row["variance_floor_active"] for row in current),
            "unit_drift_penalty": math.fsum(row["unit_drift_penalty"] for row in current),
            "source_mask_metadata_exact": True})
    verify_hashes(inputs)
    verify_hashes(sources)
    verify_hashes(config["source_sha256"])
    if prefix(run / "training.jsonl", boundary)[0] != snapshot or torch.cuda.is_initialized():
        raise ValueError("The completed prefix must remain unchanged and CUDA uninitialized")
    save_csv(output / "columns.csv", rows)
    save_csv(output / "tasks.csv", task_rows)
    result = {"complete": True, "policy": POLICY, "attempts": counts, "eligibility_decisions": decisions,
        "retention_tasks": len(tasks), "columns": len(rows), "groups": aggregate(rows),
        "cuda_initialized": False, "model_inference_performed": False, "training_performed": False,
        "selection_changed": False, "gate_changed": False, "real_data_used": False,
        "full_task_teacher_digest_replayed": False, "realized_loss_or_gradient_measured": False,
        "config_sha256": sha256(output / "config.json")}
    write_json(output / "summary.json", result)
    write_json(output / "status.json", {"state": "complete", "summary_sha256": sha256(output / "summary.json")})
    print(json.dumps({name: value for name, value in result.items() if name not in {"policy", "groups", "attempts"}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.run.resolve(), arguments.boundary, arguments.output.absolute())
