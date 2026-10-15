"""Decompose completed synthetic validation grids on CPU without touching a live training run."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

import diagnose_single_model_column_failures as columns
from recsys_evaluation_common import REPO, sha256, write_json
from dwip_teacher_mean_training import verify_hashes


POLICY = ("Read-only CPU analysis of all152 original synthetic development cases at one completed boundary. "
    "Decompose every queried column of all136 imputation cases using the existing target-dtype standard "
    "deviation, thresholds and per-case/equal-column definitions. Preserve undefined columns, role-specific "
    "canonical coverage and original weights; reproduce all stored per-case metrics before summarizing. "
    "Keep all16 rating cases in a separate stored-metric paired summary without pretending their user-macro "
    "errors are column NRMSE. Bind original general, initializer and candidate prediction hashes and cached "
    "truth. No model inference, CUDA initialization, new seed, training, checkpoint selection, gate change, "
    "real-data evaluation or live-run artifact modification. Diagnostic strata are descriptive, not evidence "
    "that a changed prior or optimizer would causally improve real benchmarks.")
METRICS = {"missbench": "missbench_nrmse", "uci": "uci_nrmse", "common": "common_nrmse"}


def read(path):
    return json.loads(Path(path).read_text())


def paired_records(reference, initial, candidate):
    records = list(zip(reference, initial, candidate))
    if len(reference) != 152 or len(initial) != 152 or len(candidate) != 152:
        raise ValueError("Retain all152 original development cases")
    counts = defaultdict(int)
    for index, group in enumerate(records):
        first = group[0]
        fields = ("seed", "domain", "rows", "columns", "gradient_data")
        if (first["index"] != index or first["gradient_data"] != "synthetic"
                or first["domain"] not in {"imputation", "recsys"}
                or any(record["index"] != index or any(record[field] != first[field] for field in fields)
                       or record.get("structured", False) != first.get("structured", False) for record in group)):
            raise ValueError("Every role must retain the same ordered synthetic case, seed, domain and full shape")
        counts["ratings" if first["domain"] == "recsys" else "structured" if first.get("structured", False) else "legacy"] += 1
    if dict(counts) != {"legacy": 40, "ratings": 16, "structured": 96}:
        raise ValueError("Preserve all40 legacy,96 structured and16 rating cases")
    return records


def verify_metrics(rows, truth, mask, arrays, expected):
    reconstructed = {}
    for role in columns.ROLES:
        residual = arrays[role][mask].astype(np.float64) - truth[mask].astype(np.float64)
        scores = {"rmse": float(np.sqrt(np.mean(residual**2)))}
        for suite, metric in METRICS.items():
            scores[metric] = math.fsum(row[f"{role}_{suite}_contribution"] for row in rows)
            count = sum(row[f"{role}_{suite}_nrmse"] is not None for row in rows)
            coverage_key = {"common": "common_nonconstant_columns", "missbench": "missbench_scored_columns", "uci": "uci_scored_columns"}[suite]
            if count != expected[role]["column_coverage"][coverage_key]:
                raise ValueError("Never change any role's original defined-column coverage")
        for name, value in scores.items():
            if not math.isclose(value, expected[role][name], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Saved-grid {role}/{name} does not reproduce the stored metric")
        reconstructed[role] = scores
    return reconstructed


def grouped_contributions(rows, case_counts):
    groups = defaultdict(list)
    for row in rows:
        for grouping in ("all", "prior", "scale_band", "support_band", "pattern"):
            groups[row["cohort"], grouping, "ALL" if grouping == "all" else str(row[grouping])].append(row)
    result = []
    for (cohort, grouping, group), selected in sorted(groups.items()):
        denominator = case_counts[cohort]
        for suite in METRICS:
            values = {role: math.fsum(row[f"{role}_{suite}_contribution"] for row in selected) / denominator for role in columns.ROLES}
            result.append({"cohort": cohort, "grouping": grouping, "group": group, "metric": suite,
                "columns": len(selected), "represented_cases": len({row["index"] for row in selected}),
                "cohort_cases": denominator, **values,
                "candidate_minus_initialization": values["candidate"] - values["initialization"],
                "candidate_minus_general": values["candidate"] - values["general"]})
    return result


def save_csv(path, rows):
    if not rows:
        raise ValueError("Never publish an empty diagnostic table")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def registered_bank(config, bank_path):
    reference_path = bank_path.parent / "reference.json"
    if (bank_path.name != "synthetic_bank.pt" or str(bank_path) not in config["input_sha256"]
            or str(reference_path) not in config["input_sha256"]):
        raise ValueError("Explicitly identify a bank and reference already bound by the training registration")
    return config["input_sha256"][str(bank_path)], config["input_sha256"][str(reference_path)]


def analyze(run, boundary, output, bank_path):
    if output.exists() or boundary <= 0 or torch.cuda.is_initialized():
        raise ValueError("Use a new CPU-only diagnostic and an already completed trained boundary")
    if shutil.disk_usage(run).free < 18 * 1024**3 + 64 * 1024**2:
        raise OSError("Preserve18 GiB plus64 MiB bounded diagnostic output capacity")
    config = read(run / "config.json")
    if boundary not in config["boundaries"] or sha256(run / "config.json") != read(run / "registration.json")["config_sha256"]:
        raise ValueError("Use a boundary from the original registered training schedule")
    bank_digest, reference_digest = registered_bank(config, bank_path)
    reference_path = bank_path.parent / "reference.json"
    if sha256(reference_path) != reference_digest:
        raise ValueError("The original registered development reference changed")
    reference, initial, candidate = read(reference_path), read(run / "validation_0.json"), read(run / f"validation_{boundary}.json")
    if initial["step"] != 0 or candidate["step"] != boundary or reference["bank_sha256"] != bank_digest:
        raise ValueError("Keep the actual initializer, completed candidate boundary and original bank")
    records = paired_records(reference["cases"], initial["cases"], candidate["cases"])
    paths = [run / "config.json", run / "registration.json", run / "validation_0.json", run / f"validation_{boundary}.json", reference_path]
    inputs = {str(path): sha256(path) for path in paths}
    inputs[str(bank_path)] = reference["bank_sha256"]
    for group in records:
        if group[0]["domain"] == "imputation":
            for record in group:
                path, digest = record["prediction_file"], record["prediction_sha256"]
                if path in inputs and inputs[path] != digest:
                    raise ValueError("A prediction path cannot carry contradictory identities")
                inputs[path] = digest
    verify_hashes(inputs)
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_development_columns.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    source_hashes = {str(path): sha256(path) for path in sources}
    output.mkdir()
    write_json(output / "config.json", {"policy": POLICY, "run": str(run), "bank": str(bank_path), "boundary": boundary,
        "input_sha256": inputs, "source_sha256": source_hashes, "cuda_initialized": False})
    write_json(output / "status.json", {"state": "analyzing", "boundary": boundary})
    bank = torch.load(bank_path, map_location="cpu", weights_only=True, mmap=True)
    if len(bank) != 152:
        raise ValueError("Do not truncate the original truth bank")
    all_rows, case_rows, ratings = [], [], []
    for index, group in enumerate(records):
        original, before, after = group
        if original["domain"] == "recsys":
            for metric in ("rmse", "mae"):
                ratings.append({"index": index, "metric": metric, "initialization": before["student"][metric],
                    "candidate": after["student"][metric], "delta": after["student"][metric] - before["student"][metric]})
            continue
        cached = bank[index]
        if (cached["metadata"]["seed"] != original["seed"] or cached["metadata"]["gradient_data"] != "synthetic"
                or list(cached["target"].shape) != [1, original["rows"], original["columns"]]):
            raise ValueError("Bind the full cached synthetic truth grid and original seed")
        truth, mask, source = (cached[name][0].numpy() for name in ("target", "mask", "source"))
        arrays = {}
        for role, record in zip(columns.ROLES, group):
            with np.load(record["prediction_file"], allow_pickle=False) as archive:
                arrays[role] = archive["general" if role == "general" else "student"][0]
        rows = columns.column_rows(truth, mask, source, arrays)
        expected = {"general": original["general"], "initialization": before["student"], "candidate": after["student"]}
        scores = verify_metrics(rows, truth, mask, arrays, expected)
        cohort = "structured" if original.get("structured", False) else "legacy"
        details = {"index": index, "seed": original["seed"], "cohort": cohort,
            "prior": original.get("prior", original.get("component", "unknown")),
            "pattern": original.get("family", "legacy"), "rows": original["rows"], "columns": original["columns"]}
        all_rows.extend({**details, **row} for row in rows)
        for role, values in scores.items():
            case_rows.append({**details, "role": role, **values})
    summary_rows = grouped_contributions(all_rows, {"legacy": 40, "structured": 96})
    verify_hashes(inputs)
    verify_hashes(source_hashes)
    if torch.cuda.is_initialized():
        raise ValueError("The descriptive analysis must never initialize CUDA")
    save_csv(output / "columns.csv", all_rows)
    save_csv(output / "cases.csv", case_rows)
    save_csv(output / "contributions.csv", summary_rows)
    save_csv(output / "ratings.csv", ratings)
    result = {"complete": True, "policy": POLICY, "imputation_cases": 136, "rating_cases": 16,
        "imputation_columns": len(all_rows), "stored_metric_checks": 136 * 3 * 4,
        "cuda_initialized": False, "model_inference_performed": False, "training_performed": False,
        "checkpoint_selection_performed": False, "gate_changed": False, "real_data_used": False,
        "config_sha256": sha256(output / "config.json"), "contributions": summary_rows}
    write_json(output / "summary.json", result)
    write_json(output / "status.json", {"state": "complete", "summary_sha256": sha256(output / "summary.json")})
    print(json.dumps({key: value for key, value in result.items() if key not in ("policy", "contributions")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.run.resolve(), arguments.boundary, arguments.output.resolve(), arguments.bank.absolute())
