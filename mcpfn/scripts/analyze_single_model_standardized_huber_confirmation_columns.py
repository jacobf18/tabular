"""Decompose every V54 consumed-confirmation column using only audited saved grids."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

import analyze_single_model_development_columns as development
import diagnose_single_model_standardized_huber_initializer as diagnosis
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROLES = ("general", "teacher", "baseline", "parent", "candidate")
ROLE_RECORDS = {"general": ("reference", "general"), "teacher": ("reference", "teacher"),
    "baseline": ("initialization", "student"), "parent": ("parent", "student"), "candidate": ("student", "student")}
BUNDLES = (("general", "parent", "candidate"), ("teacher", "baseline", "candidate"))
COHORT_COUNTS = {"native": 60, "structured": 192, "ratings": 32}
FAMILIES = ("mcar", "mar", "mnar", "mnar_censoring", "mnar_two_phase_subset", "mnar_skip_logic",
    "mnar_cluster_level", "mar_neural", "mar_block_neural", "mnar_latent_factor", "mnar_polarization", "mnar_soft_polarization")
BASE_FIELDS = ("column", "queries", "observations", "target_std", "scale_band", "support_band", "common_eligible")
WRITE_BUDGET = 256 * 1024**2
POLICY = (
    "Read-only CPU decomposition of every queried column in all252 imputation cases and all32 rating cases "
    "from V54's fully audited failed284-case confirmation and completed actual-V29 attribution. Preserve "
    "general, observed-only blended teacher, frozen V8 retention baseline, actual V29 parent and V54 step50 "
    "candidate as distinct roles. Reuse original target-dtype scales, canonical thresholds, role-specific "
    "defined-column coverage and common truth-defined columns. Keep all cases, orientations, priors and "
    "adverse directions; equal-case/equal-column weighting remains unchanged. Reproduce every stored "
    "imputation metric and all32 original attribution comparisons before publishing summaries. Ratings "
    "retain original stored macro metrics and are not reinterpreted as column NRMSE. No model inference, "
    "CUDA initialization, new episode/seed, training, selection, confirmation repeat, metric/gate change "
    "or real-data evaluation. Consumed-bank strata are descriptive, not fresh qualification, permission "
    "to target a case or proof of a causal training mechanism. All drift RMSEs are relative to actual V29."
)


def paired_records(reports):
    if set(reports) != {"reference", "initialization", "parent", "student"} or any(
            len(report["cases"]) != 284 for report in reports.values()):
        raise ValueError("Retain every original284 case in all four saved reports")
    paired = []
    counts = Counter()
    fields = ("seed", "domain", "rows", "columns", "gradient_data", "structured", "family", "prior",
        "component", "transposed", "targets")
    for index in range(284):
        group = {name: report["cases"][index] for name, report in reports.items()}
        reference = group["reference"]
        if (reference.get("gradient_data") != "synthetic" or reference["domain"] not in {"imputation", "recsys"}
                or any(record["index"] != index or any(record.get(field) != reference.get(field) for field in fields)
                    for record in group.values())):
            raise ValueError("Preserve ordered case identity, full shapes, cohorts and original synthetic queries")
        cohort = "ratings" if reference["domain"] == "recsys" else "structured" if reference.get("structured", False) else "native"
        counts[cohort] += 1
        paired.append((cohort, group))
    if dict(counts) != COHORT_COUNTS:
        raise ValueError("Preserve all60 native,192 structured and32 rating cases")
    families = Counter(group["reference"].get("family") for cohort, group in paired if cohort == "structured")
    if families != Counter({family: 16 for family in FAMILIES}):
        raise ValueError("Preserve all twelve original structured families with sixteen cases each")
    return paired


def checked_columns(truth, mask, source, arrays, expected):
    if set(arrays) != set(ROLES) or set(expected) != set(ROLES):
        raise ValueError("Retain all five prediction roles without conflating V8 and V29")
    combined, scores = [], {}
    for bundle in BUNDLES:
        aliases = dict(zip(development.columns.ROLES, bundle))
        predictions = {alias: arrays[role] for alias, role in aliases.items()}
        rows = development.columns.column_rows(truth, mask, source, predictions)
        checked = development.verify_metrics(rows, truth, mask, predictions,
            {alias: expected[role] for alias, role in aliases.items()})
        if not combined:
            combined = [{key: row[key] for key in BASE_FIELDS} for row in rows]
        if len(rows) != len(combined):
            raise ValueError("Every role must retain the same original queried columns")
        for destination, row in zip(combined, rows):
            if any(destination[key] != row[key] for key in BASE_FIELDS):
                raise ValueError("Never change truth scales, query counts or observed support by role")
            for alias, role in aliases.items():
                for key, value in row.items():
                    if key.startswith(alias + "_") and key != alias + "_drift_rmse":
                        renamed = role + key[len(alias):]
                        if renamed in destination and destination[renamed] != value:
                            raise ValueError("Duplicated candidate arithmetic must agree exactly")
                        destination[renamed] = value
                if role in scores and scores[role] != checked[alias]:
                    raise ValueError("Duplicated candidate metrics must agree exactly")
                scores[role] = checked[alias]
    for row in combined:
        column = row["column"]
        selected = mask[:, column]
        parent = arrays["parent"][selected, column].astype(np.float64)
        row["query_band"] = development.columns.band(row["queries"], development.columns.SUPPORT_EDGES)
        for role in ROLES:
            values = arrays[role][selected, column].astype(np.float64)
            row[f"{role}_drift_rmse"] = float(np.sqrt(np.mean((values - parent)**2)))
    return combined, scores


def grouped_contributions(rows, case_counts):
    groups = defaultdict(list)
    for row in rows:
        for grouping in ("all", "prior", "pattern", "scale_band", "support_band", "query_band", "shape"):
            groups[row["cohort"], grouping, "ALL" if grouping == "all" else str(row[grouping])].append(row)
    result = []
    for (cohort, grouping, group), selected in sorted(groups.items()):
        denominator = case_counts[cohort]
        for suite in development.METRICS:
            values = {role: math.fsum(row[f"{role}_{suite}_contribution"] for row in selected) / denominator for role in ROLES}
            result.append({"cohort": cohort, "grouping": grouping, "group": group, "metric": suite,
                "columns": len(selected), "represented_cases": len({row["index"] for row in selected}),
                "cohort_cases": denominator, **values, "candidate_minus_parent": values["candidate"] - values["parent"],
                "candidate_minus_baseline": values["candidate"] - values["baseline"],
                "candidate_minus_general": values["candidate"] - values["general"]})
    return result


def attribution_keys():
    base = {(group, metric) for group in ("imputation", "structured_overall")
        for metric in ("missbench_nrmse", "uci_nrmse", "common_nrmse", "rmse")}
    base.update(("recsys", metric) for metric in ("rmse", "mae"))
    return {(reference, group, metric) for reference in ("V8", "general") for group, metric in base} | {
        ("V8", family, "common_nrmse") for family in FAMILIES}


def verify_attribution(cases, ratings, comparisons):
    if len(comparisons) != 32 or {(row["reference"], row["group"], row["metric"]) for row in comparisons} != attribution_keys():
        raise ValueError("Reconstruct all32 distinct original attribution comparisons")
    for comparison in comparisons:
        group, metric = comparison["group"], comparison["metric"]
        for role, key in (("parent", "V29_initializer"), ("candidate", "V54_step50"),
                ("baseline" if comparison["reference"] == "V8" else "general", "reference_value")):
            if group == "recsys":
                values = [row[role] for row in ratings if row["metric"] == metric]
                expected_count = 32
            else:
                selected = [row for row in cases if row["role"] == role]
                if group == "imputation":
                    selected = [row for row in selected if row["cohort"] == "native"]
                    expected_count = 60
                elif group == "structured_overall":
                    selected = [row for row in selected if row["cohort"] == "structured"]
                    expected_count = 192
                else:
                    selected = [row for row in selected if row["cohort"] == "structured" and row["pattern"] == group]
                    expected_count = 16
                values = [row[metric] for row in selected]
            if len(values) != expected_count or not math.isclose(
                    math.fsum(values) / len(values), comparison[key], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("Complete saved-grid aggregates must reproduce every original comparison")


def analyze(output):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized() or output.exists()
            or output.is_symlink() or output.parent != diagnosis.original.ROOT):
        raise ValueError("Use one new output directory with CUDA hidden and uninitialized")
    diagnosis.require_closed_failure()
    config = diagnosis.verify()
    diagnostic_audit = development.read(diagnosis.OUTPUT / "audit.json")
    if (development.read(diagnosis.OUTPUT / "status.json")["state"] != "complete"
            or not diagnostic_audit["complete"] or diagnostic_audit["prediction_grids"] != 284
            or diagnostic_audit["cuda_initialized"] or not diagnostic_audit["post_hoc_consumed_bank"]):
        raise ValueError("Require the completed original actual-V29 attribution and independent CPU audit")
    for mount, reserve in ((diagnosis.original.ROOT, 18 * 1024**3 + WRITE_BUDGET),
            (REPO, 6 * 1024**3), (diagnosis.training.SCRATCH, 50 * 1024**3)):
        if shutil.disk_usage(mount).free < reserve:
            raise OSError("Preserve original reserves plus the bounded256 MiB diagnostic allowance")
    confirmation = diagnosis.original.CONFIRMATION
    reports = {name: development.read(confirmation / f"{name}.json") for name in ("reference", "initialization", "student")}
    reports["parent"] = development.read(diagnosis.OUTPUT / "initializer.json")
    paired = paired_records(reports)
    inputs = {}
    for inventory in (diagnostic_audit["artifact_sha256"], development.read(confirmation / "audit.json")["artifact_sha256"]):
        for path, digest in inventory.items():
            if path in inputs and inputs[path] != digest:
                raise ValueError("Original audits must agree on every shared input identity")
            inputs[path] = digest
    inputs[str(diagnosis.OUTPUT / "audit.json")] = sha256(diagnosis.OUTPUT / "audit.json")
    for _, records in paired:
        for record in records.values():
            if inputs.get(record["prediction_file"]) != record["prediction_sha256"]:
                raise ValueError("Every full prediction grid must already belong to the original audits")
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_standardized_huber_confirmation_columns.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    source_hashes = {str(path): sha256(path) for path in sources}
    verify_hashes(inputs)
    output.mkdir()
    write_json(output / "config.json", {"policy": POLICY, "roles": list(ROLES), "cohort_counts": COHORT_COUNTS,
        "bank": config["bank"], "input_sha256": inputs, "source_sha256": source_hashes,
        "write_budget_bytes": WRITE_BUDGET, "cuda_initialized": False})
    write_json(output / "status.json", {"state": "analyzing", "pid": os.getpid()})
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True, mmap=True)
    if len(bank) != 284:
        raise ValueError("Never truncate the original284-case truth bank")
    all_rows, case_rows, ratings = [], [], []
    for index, (cohort, records) in enumerate(paired):
        reference = records["reference"]
        expected = {role: records[record_name][metric_name] for role, (record_name, metric_name) in ROLE_RECORDS.items()}
        if cohort == "ratings":
            for metric in ("rmse", "mae"):
                values = {role: expected[role][metric] for role in ROLES}
                if any(not math.isfinite(value) for value in values.values()):
                    raise ValueError("Keep all original finite rating macro scores")
                ratings.append({"index": index, "metric": metric, **values,
                    "candidate_minus_parent": values["candidate"] - values["parent"]})
            continue
        cached = bank[index]
        if (cached["metadata"]["seed"] != reference["seed"] or cached["metadata"]["gradient_data"] != "synthetic"
                or list(cached["target"].shape) != [1, reference["rows"], reference["columns"]]):
            raise ValueError("Bind each full original truth grid and seed without cropping or refitting")
        truth, mask, source = (cached[name][0].numpy() for name in ("target", "mask", "source"))
        arrays = {}
        for record_name, record in records.items():
            with np.load(record["prediction_file"], allow_pickle=False) as archive:
                for role, (owner, metric_name) in ROLE_RECORDS.items():
                    if owner == record_name:
                        arrays[role] = archive[metric_name][0]
        rows, scores = checked_columns(truth, mask, source, arrays, expected)
        details = {"index": index, "seed": reference["seed"], "cohort": cohort,
            "prior": reference.get("prior", reference.get("component", "unknown")),
            "pattern": reference.get("family", "native"), "shape": f'{reference["rows"]}x{reference["columns"]}',
            "rows": reference["rows"], "columns": reference["columns"]}
        all_rows.extend({**details, **row} for row in rows)
        case_rows.extend({**details, "role": role, **values} for role, values in scores.items())
    verify_attribution(case_rows, ratings, diagnostic_audit["comparisons"])
    contributions = grouped_contributions(all_rows, COHORT_COUNTS)
    verify_hashes(inputs)
    verify_hashes(source_hashes)
    if torch.cuda.is_initialized():
        raise ValueError("Never initialize CUDA during this saved-grid diagnosis")
    for name, rows in (("columns", all_rows), ("cases", case_rows), ("contributions", contributions), ("ratings", ratings)):
        development.save_csv(output / f"{name}.csv", rows)
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > WRITE_BUDGET:
        raise OSError("Diagnostic output exceeds its prospectively declared write budget")
    summary = {"complete": True, "policy": POLICY, "cases": 284, "imputation_cases": 252, "rating_cases": 32,
        "roles": list(ROLES), "queried_columns": len(all_rows), "unique_imputation_metric_checks": len(case_rows) * 4,
        "attribution_comparisons": 32, "rating_case_metric_rows": len(ratings),
        "common_eligible_columns": sum(row["common_eligible"] for row in all_rows), "contributions": contributions,
        "cuda_initialized": False, "model_inference_performed": False, "training_performed": False,
        "checkpoint_selection_performed": False, "confirmation_repeated": False, "gate_changed": False,
        "real_data_used": False, "config_sha256": sha256(output / "config.json")}
    write_json(output / "summary.json", summary)
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) + 4096 > WRITE_BUDGET:
        raise OSError("Preserve the complete diagnostic write budget through final publication")
    write_json(output / "status.json", {"state": "complete", "summary_sha256": sha256(output / "summary.json")})
    print(json.dumps({key: value for key, value in summary.items() if key not in ("policy", "contributions")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.output.absolute())
