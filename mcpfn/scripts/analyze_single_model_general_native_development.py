"""Describe every saved V47 development boundary without inference or reselection."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

import analyze_single_model_development_columns as previous
import summarize_single_model_general_native_outcome as outcome
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = outcome.ROOT
RUN = ROOT / "general_native_retention_v47"
OUTPUT = ROOT / "general_native_retention_v47_development_diagnosis_v1"
BOUNDARIES = outcome.BOUNDARIES
ROLES = previous.columns.ROLES
COUNTS = {"native": 40, "structured": 96, "ratings": 16}
POLICY = (
    "CPU-only description of all five original V47 synthetic development boundaries, including "
    "the actual V29 initializer. Preserve all152 cases, every imputation column including empty "
    "and undefined columns, all priors, recorded transpose and support-transpose metadata, and "
    "both adverse and improved strata. Reproduce original canonical per-case metrics and coverage "
    "before decomposition. Column contributions use original within-case denominators followed "
    "by the full cohort count, not subgroup renormalization. RMSE is not additive: column squared "
    "errors decompose MSE, while case RMSE changes are reported separately. Ratings retain saved "
    "original-unit user-macro RMSE/MAE; no column NRMSE is assigned to ratings. V8 guard limits "
    "are not actual-initializer metrics. This consumed-bank diagnosis is descriptive, not causal "
    "evidence, fresh confirmation, or real-benchmark performance. No inference, CUDA, new seeds, "
    "training, checkpoint selection, modified gates, or confirmation access."
)
NON_METADATA = {"general", "teacher", "student", "routing", "prediction_file", "prediction_sha256"}
GROUPINGS = ("all", "prior", "component", "pattern", "transposed", "support_transposed", "prior_orientation")


def metadata(record):
    return {key: value for key, value in record.items() if key not in NON_METADATA}


def validate_pairing(reference, reports):
    if [report["step"] for report in reports] != list(BOUNDARIES):
        raise ValueError("Require all five original boundaries in order")
    for report in reports:
        previous.paired_records(reference, reports[0]["cases"], report["cases"])
        for original, candidate in zip(reference, report["cases"]):
            if metadata(original) != metadata(candidate):
                raise ValueError("Preserve all recorded metadata, including both transpose fields")


def direction(delta):
    return "worse" if delta > 0 else "better" if delta < 0 else "unchanged"


def details(record, step):
    cohort = "ratings" if record["domain"] == "recsys" else "structured" if record.get("structured", False) else "native"
    prior = record.get("prior", record.get("preference_family", str(record.get("component", "not_recorded"))))
    transposed = str(record.get("transposed", "not_recorded"))
    support = str(record.get("support_transposed", "not_recorded"))
    return {"step": step, "index": record["index"], "seed": record["seed"], "cohort": cohort,
        "prior": prior, "component": str(record.get("component", "not_recorded")),
        "pattern": record.get("family", "native" if cohort == "native" else "ratings"),
        "transposed": transposed, "support_transposed": support,
        "prior_orientation": f"{prior}|transpose={transposed}|support_transpose={support}",
        "rows": record["rows"], "columns": record["columns"]}


def complete_columns(truth, mask, source, arrays):
    queried = previous.columns.column_rows(truth, mask, source, arrays)
    by_column = {row["column"]: row for row in queried}
    total_queries = int(mask.sum())
    result = []
    for column in range(truth.shape[1]):
        row = by_column.get(column)
        if row is None:
            row = {key: None for key in queried[0]}
            row.update(column=column, queries=0, observations=int(np.isfinite(source[:, column]).sum()),
                scale_band="no_query", support_band=previous.columns.band(
                    int(np.isfinite(source[:, column]).sum()), previous.columns.SUPPORT_EDGES), common_eligible=False)
            for role in ROLES:
                for suite in previous.METRICS:
                    row[f"{role}_{suite}_contribution"] = 0.
        for role in ROLES:
            row[f"{role}_mse_contribution"] = row[f"{role}_rmse"] ** 2 * row["queries"] / total_queries if row["queries"] else 0.
        row["common_direction"] = direction(row["candidate_common_nrmse"] - row["initialization_common_nrmse"]) if row["common_eligible"] else "undefined"
        row["mse_direction"] = direction(row["candidate_mse_contribution"] - row["initialization_mse_contribution"]) if row["queries"] else "no_query"
        result.append(row)
    return result


def grouped_columns(rows):
    groups = defaultdict(list)
    for row in rows:
        for grouping in (*GROUPINGS, "scale_band", "support_band", "common_direction", "mse_direction"):
            groups[row["step"], row["cohort"], grouping, "ALL" if grouping == "all" else row[grouping]].append(row)
    result = []
    for (step, cohort, grouping, group), members in sorted(groups.items()):
        for metric in (*previous.METRICS, "mse"):
            values = {role: math.fsum(row[f"{role}_{metric}_contribution"] for row in members) / COUNTS[cohort] for role in ROLES}
            result.append({"step": step, "cohort": cohort, "grouping": grouping, "group": group, "metric": metric,
                "columns": len(members), "queried_columns": sum(row["queries"] > 0 for row in members),
                "represented_cases": len({row["index"] for row in members}), "cohort_cases": COUNTS[cohort],
                **values, "delta_vs_actual_V29": values["candidate"] - values["initialization"],
                "delta_vs_general": values["candidate"] - values["general"]})
    return result


def grouped_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        for grouping in (*GROUPINGS, "direction"):
            groups[row["step"], row["cohort"], row["metric"], grouping, "ALL" if grouping == "all" else row[grouping]].append(row)
    result = []
    for (step, cohort, metric, grouping, group), members in sorted(groups.items()):
        values = {role: math.fsum(row[role] for row in members) / len(members) for role in ROLES}
        result.append({"step": step, "cohort": cohort, "metric": metric, "grouping": grouping, "group": group,
            "cases": len(members), "cohort_cases": COUNTS[cohort], **values,
            "delta_vs_actual_V29": values["candidate"] - values["initialization"],
            "cohort_delta_contribution": math.fsum(row["delta_vs_actual_V29"] for row in members) / COUNTS[cohort],
            "better_cases": sum(row["direction"] == "better" for row in members),
            "worse_cases": sum(row["direction"] == "worse" for row in members),
            "unchanged_cases": sum(row["direction"] == "unchanged" for row in members)})
    return result


def render(summary, case_groups, column_groups):
    lines = ["# V47 complete synthetic development diagnosis", "", POLICY, "",
        "V47 remains an audited negative: no trained boundary qualifies. No confirmation cases were used.", "",
        "## Complete Cohort Trajectory", "", "Differences below compare against actual V29 step0, not the V8 guard limit.", "",
        "| Step | Cohort | Metric | V29 | Candidate | Change | Better / worse cases |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in case_groups:
        if row["grouping"] == "all" and row["metric"] in ("common_nrmse", "rmse", "mae"):
            lines.append(f'| {row["step"]} | {row["cohort"]} | {row["metric"]} | {row["initialization"]:.10g} | '
                f'{row["candidate"]:.10g} | {row["delta_vs_actual_V29"]:+.10g} | {row["better_cases"]} / {row["worse_cases"]} |')
    for cohort, grouping in (("native", "prior"), ("native", "transposed"), ("native", "support_transposed"),
            ("structured", "pattern"), ("native", "scale_band"), ("structured", "scale_band")):
        lines.extend(["", f"## Final64 {cohort}: {grouping}", "",
            "All groups are retained. NRMSE contributions keep the full cohort denominator; they are not subgroup means.", "",
            "| Group | Cases | Columns | V29 contribution | Candidate contribution | Change |", "|---|---:|---:|---:|---:|---:|"])
        for row in column_groups:
            if (row["step"], row["cohort"], row["grouping"], row["metric"]) == (64, cohort, grouping, "common"):
                lines.append(f'| {row["group"]} | {row["represented_cases"]} | {row["columns"]} | '
                    f'{row["initialization"]:.10g} | {row["candidate"]:.10g} | {row["delta_vs_actual_V29"]:+.10g} |')
    lines.extend(["", "## Files and Limitations", "",
        "`columns.csv` preserves all columns at all boundaries, including zero-query and undefined columns; "
        "`cases.csv` preserves all canonical case metrics and original-unit rating scores. "
        "`column_groups.csv` and `case_groups.csv` preserve every grouping, including adverse and improved groups. "
        "`original_comparisons.csv` retains all110 original guard comparisons without reselection.", "",
        "Missing transpose fields mean not recorded, not false. Support augmentation transpose and final matrix transpose "
        "are separate metadata fields. RMSE case means and NRMSE equal-column means weight errors differently; "
        "column MSE contributions cannot be added to obtain mean case RMSE. Directional groups are retrospective "
        "descriptions only, not selectors or prospective evidence for another training recipe.", "",
        f'Coverage: {summary["cases"]} case-boundaries; {summary["imputation_columns"]} column-boundaries; '
        f'{summary["stored_metric_checks"]} canonical metric/role checks. No real data or model inference.', ""])
    return "\n".join(lines)


def analyze():
    if OUTPUT.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Use a new output and a CUDA-hidden CPU-only process")
    if shutil.disk_usage(ROOT).free < 18 * 1024**3 + 256 * 1024**2:
        raise OSError("Preserve the18 GiB reserve plus256 MiB diagnostic budget")
    config = previous.read(RUN / "config.json")
    if sha256(RUN / "config.json") != outcome.RUN_SHA256 or previous.read(RUN / "registration.json")["config_sha256"] != outcome.RUN_SHA256:
        raise ValueError("Bind the immutable V47 registration")
    closed = outcome.OUTPUT / "summary.json"
    closed_audit = outcome.OUTPUT / "audit.json"
    verify_hashes({str(closed): "0105a19f6187c5afb9dbcaa6bb8c9b00ca8191e6c6674c08c0b4f98753aff808",
        str(closed_audit): "f104d735ee9f9221bcfb6c92d69a2846dbd366820fc594e58d99afcbb391c22c"})
    bank_path = outcome.pipeline.training.original.alignment.fixed.PRODUCER / "synthetic_bank.pt"
    bank_digest, reference_digest = previous.registered_bank(config, bank_path)
    reference_path = bank_path.parent / "reference.json"
    reference = previous.read(reference_path)
    reports = [previous.read(RUN / f"validation_{step}.json") for step in BOUNDARIES]
    validate_pairing(reference["cases"], reports)
    comparisons = outcome.comparison_rows(reports)
    inputs = {str(path): sha256(path) for path in [RUN / "config.json", RUN / "registration.json", closed, closed_audit,
        *(RUN / f"validation_{step}.json" for step in BOUNDARIES)]}
    inputs.update({str(bank_path): bank_digest, str(reference_path): reference_digest})
    if reference["bank_sha256"] != bank_digest:
        raise ValueError("Require the original registered synthetic truth bank")
    for record in [*reference["cases"], *(case for report in reports for case in report["cases"])]:
        if record["domain"] == "imputation":
            path, digest = record["prediction_file"], record["prediction_sha256"]
            if path in inputs and inputs[path] != digest:
                raise ValueError("A prediction cannot have inconsistent hash identities")
            inputs[path] = digest
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])
    sources = {str(Path(module.__file__).resolve()): sha256(module.__file__) for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file()}
    sources[str(REPO / "tests/test_single_model_general_native_development.py")] = sha256(REPO / "tests/test_single_model_general_native_development.py")
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"policy": POLICY, "input_sha256": inputs, "source_sha256": sources,
        "boundaries": BOUNDARIES, "actual_initializer": "V29", "guard_reference": "V8", "output_budget_bytes": 256 * 1024**2})
    write_json(OUTPUT / "status.json", {"state": "analyzing"})
    bank = torch.load(bank_path, map_location="cpu", weights_only=True, mmap=True)
    if len(bank) != 152:
        raise ValueError("Keep every original case")
    all_columns, cases = [], []
    checks = 0
    for report in reports:
        for index, original in enumerate(reference["cases"]):
            before, after, cached = reports[0]["cases"][index], report["cases"][index], bank[index]
            if (any(original[key] != value for key, value in cached["metadata"].items())
                    or list(cached["target"].shape) != [1, original["rows"], original["columns"]]):
                raise ValueError("Bind the full original truth shape and every metadata field")
            identity = details(original, report["step"])
            expected = {"general": original["general"], "initialization": before["student"], "candidate": after["student"]}
            if identity["cohort"] == "ratings":
                scores = {role: {metric: values[metric] for metric in ("rmse", "mae")} for role, values in expected.items()}
            else:
                truth, mask, source = (cached[name][0].numpy() for name in ("target", "mask", "source"))
                arrays = {}
                for role, record in zip(ROLES, (original, before, after)):
                    with np.load(record["prediction_file"], allow_pickle=False) as archive:
                        arrays[role] = archive["general" if role == "general" else "student"][0]
                rows = complete_columns(truth, mask, source, arrays)
                scores = previous.verify_metrics(rows, truth, mask, arrays, expected)
                checks += 12
                all_columns.extend({**identity, **row} for row in rows)
            for metric in scores["candidate"]:
                values = {role: scores[role][metric] for role in ROLES}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError("Keep only the originally finite case scores, never silently omit failures")
                delta = values["candidate"] - values["initialization"]
                cases.append({**identity, "metric": metric, **values, "delta_vs_actual_V29": delta, "direction": direction(delta)})
    column_groups, case_groups = grouped_columns(all_columns), grouped_cases(cases)
    verify_hashes(inputs)
    verify_hashes(sources)
    if torch.cuda.is_initialized():
        raise ValueError("No CUDA initialization is allowed")
    for name, rows in (("columns", all_columns), ("cases", cases), ("column_groups", column_groups),
            ("case_groups", case_groups), ("original_comparisons", comparisons)):
        previous.save_csv(OUTPUT / f"{name}.csv", rows)
    summary = {"complete": True, "policy": POLICY, "cases": 152 * len(BOUNDARIES),
        "imputation_columns": len(all_columns), "empty_query_columns": sum(row["queries"] == 0 for row in all_columns),
        "stored_metric_checks": checks, "boundaries": list(BOUNDARIES), "cuda_initialized": False,
        "inference_performed": False, "training_performed": False, "confirmation_evaluated": False,
        "checkpoint_selection_performed": False, "gate_changed": False, "real_data_used": False,
        "config_sha256": sha256(OUTPUT / "config.json")}
    (OUTPUT / "RESULTS.md").write_text(render(summary, case_groups, column_groups))
    summary["artifact_sha256"] = {str(path): sha256(path) for path in OUTPUT.iterdir() if path.suffix in (".csv", ".md")}
    write_json(OUTPUT / "summary.json", summary)
    write_json(OUTPUT / "status.json", {"state": "complete", "summary_sha256": sha256(OUTPUT / "summary.json")})
    print(json.dumps({key: value for key, value in summary.items() if key not in ("policy", "artifact_sha256")}))


def audit():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized() or (OUTPUT / "audit.json").exists():
        raise ValueError("Run a separate read-only CPU recheck once")
    summary, config = previous.read(OUTPUT / "summary.json"), previous.read(OUTPUT / "config.json")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    verify_hashes(summary["artifact_sha256"])
    if summary["config_sha256"] != sha256(OUTPUT / "config.json"):
        raise ValueError("Diagnostic registration changed")
    with (OUTPUT / "columns.csv").open() as stream:
        columns = list(csv.DictReader(stream))
    with (OUTPUT / "cases.csv").open() as stream:
        cases = list(csv.DictReader(stream))
    by_case = defaultdict(list)
    for row in columns:
        by_case[int(row["step"]), int(row["index"])].append(row)
    checked = 0
    for step in BOUNDARIES:
        report = previous.read(RUN / f"validation_{step}.json")
        initial = previous.read(RUN / "validation_0.json")
        for record in report["cases"]:
            index = record["index"]
            selected = [row for row in cases if int(row["step"]) == step and int(row["index"]) == index]
            expected_metrics = {"rmse", "mae"} if record["domain"] == "recsys" else {"rmse", *previous.METRICS.values()}
            if len(selected) != len(expected_metrics) or {row["metric"] for row in selected} != expected_metrics:
                raise ValueError("Every case must keep each original metric exactly once")
            for row in selected:
                if not math.isclose(float(row["candidate"]), record["student"][row["metric"]], rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("A CSV metric differs from the original stored prediction metric")
                if not math.isclose(float(row["initialization"]), initial["cases"][index]["student"][row["metric"]], rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("The CSV initializer is not actual V29")
                checked += 1
            if record["domain"] == "recsys":
                if (step, index) in by_case:
                    raise ValueError("Do not relabel ratings as imputation")
                continue
            members = by_case[step, index]
            if Counter(int(row["column"]) for row in members) != Counter(range(record["columns"])):
                raise ValueError("Keep every original column exactly once, including query-empty columns")
            for role, expected in (("candidate", record["student"]), ("initialization", initial["cases"][index]["student"])):
                for suite, metric in previous.METRICS.items():
                    value = math.fsum(float(row[f"{role}_{suite}_contribution"]) for row in members)
                    if not math.isclose(value, expected[metric], rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError("CSV contributions do not recover original case NRMSE")
                mse = math.fsum(float(row[f"{role}_mse_contribution"]) for row in members)
                if not math.isclose(math.sqrt(mse), expected["rmse"], rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("Column MSE must recover original query-pooled case RMSE")
    for filename, members in (("column_groups", columns), ("case_groups", cases)):
        with (OUTPUT / f"{filename}.csv").open() as stream:
            saved = list(csv.DictReader(stream))
        indexed = defaultdict(list)
        groupings = (*GROUPINGS, "scale_band", "support_band", "common_direction", "mse_direction") if filename == "column_groups" else (*GROUPINGS, "direction")
        for member in members:
            for grouping in groupings:
                key = (member["step"], member["cohort"], grouping, "ALL" if grouping == "all" else member[grouping],
                    "columns" if filename == "column_groups" else member["metric"])
                indexed[key].append(member)
        for row in saved:
            chosen = indexed[row["step"], row["cohort"], row["grouping"], row["group"],
                "columns" if filename == "column_groups" else row["metric"]]
            if len(chosen) != int(row["columns" if filename == "column_groups" else "cases"]):
                raise ValueError("Group coverage changed")
            denominator = COUNTS[row["cohort"]] if filename == "column_groups" else len(chosen)
            for role in ROLES:
                field = f'{role}_{row["metric"]}_contribution' if filename == "column_groups" else role
                value = math.fsum(float(member[field]) for member in chosen) / denominator
                if not math.isclose(value, float(row[role]), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Independently regrouped CSV values differ")
    if len(columns) != summary["imputation_columns"] or len(by_case) != 136 * 5 or torch.cuda.is_initialized():
        raise ValueError("Require complete five-boundary CPU coverage")
    write_json(OUTPUT / "audit.json", {"complete": True, "summary_sha256": sha256(OUTPUT / "summary.json"),
        "case_metric_checks": checked, "imputation_case_boundaries": len(by_case), "column_boundaries": len(columns),
        "csv_regrouping_recomputed": True, "input_and_source_hashes_rechecked": True, "cuda_initialized": False,
        "model_inference_performed": False, "raw_grid_replay_performed": False})
    print(json.dumps(previous.read(OUTPUT / "audit.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    audit() if arguments.audit_only else analyze()
