"""Decompose already-audited synthetic guards without selecting or evaluating models."""

import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import train_single_matrixformer_finite_guarded as training


def metadata(row):
    return {key: value for key, value in row.items() if key not in {"student", "prediction_file", "prediction_sha256"}}


def matches(row, group):
    if group == "imputation":
        return row["domain"] == "imputation" and not row.get("structured", False)
    if group == "recsys":
        return row["domain"] == "recsys"
    if group == "structured_overall":
        return row.get("structured", False)
    return row.get("structured", False) and row["family"] == group


def component(row):
    if row["domain"] == "recsys":
        return row["preference_family"]
    return row["component"] if row.get("structured", False) else row["prior"]


def paired_values(initial, current):
    if len(initial) != len(current) or not initial:
        raise ValueError("Retain every paired development case")
    seen = set()
    for before, after in zip(initial, current):
        identity = before["index"], before["seed"]
        if metadata(before) != metadata(after) or identity in seen:
            raise ValueError("Development identity, ordering and metadata must match exactly")
        seen.add(identity)
        yield before, after


def decompose(initial, current, comparisons):
    pairs = list(paired_values(initial, current))
    summaries, contributions, cases, identities = [], [], [], set()
    for guard in comparisons:
        identity = guard["group"], guard["metric"]
        if identity in identities:
            raise ValueError("Duplicate retention guard")
        identities.add(identity)
        group, metric = identity
        selected = [(before, after) for before, after in pairs if matches(before, group)]
        if not selected:
            raise ValueError("Never drop a retention cohort")
        defined = []
        for before, after in selected:
            baseline, candidate = before["student"][metric], after["student"][metric]
            if baseline is None or candidate is None:
                if baseline is not None or candidate is not None or metric != "common_nrmse":
                    raise ValueError("Undefined truth-only coverage must remain identical")
                continue
            if not all(math.isfinite(value) and value >= 0 for value in (baseline, candidate)):
                raise ValueError("Require finite nonnegative defined scores")
            defined.append((before, baseline, candidate))
        if not defined:
            raise ValueError("Cannot report an all-undefined retention cohort")
        baseline_mean = math.fsum(before for row, before, after in defined) / len(defined)
        candidate_mean = math.fsum(after for row, before, after in defined) / len(defined)
        if (not math.isclose(baseline_mean, guard["initialization"], rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(candidate_mean, guard["candidate"], rel_tol=1e-12, abs_tol=1e-12)
                or guard["retained"] != (guard["candidate"] <= guard["initialization"])):
            raise ValueError("Audited guard aggregates and strict decisions must reproduce")
        base = {"group": group, "metric": metric}
        summaries.append({**base, "cases": len(selected), "defined_cases": len(defined),
                          "undefined_cases": len(selected) - len(defined), "initialization": guard["initialization"],
                          "candidate": guard["candidate"], "retained": guard["retained"],
                          "improved_cases": sum(after < before for row, before, after in defined),
                          "worsened_cases": sum(after > before for row, before, after in defined)})
        buckets = defaultdict(list)
        for row, before, after in defined:
            label = component(row)
            buckets[label].append((before, after))
            cases.append({**base, "index": row["index"], "seed": row["seed"], "component": label,
                          "rows": row["rows"], "columns": row["columns"], "initialization": before,
                          "candidate": after, "delta": after - before,
                          "guard_delta_contribution": (after - before) / len(defined)})
        for label, values in sorted(buckets.items()):
            baseline = math.fsum(before for before, after in values) / len(values)
            candidate = math.fsum(after for before, after in values) / len(values)
            contribution = math.fsum(after - before for before, after in values) / len(defined)
            contributions.append({**base, "component": label, "cases": len(values), "cohort_defined_cases": len(defined),
                                  "initialization": baseline, "candidate": candidate,
                                  "change_percent": 100 * (candidate / baseline - 1) if baseline else None,
                                  "worsened_cases": sum(after > before for before, after in values),
                                  "guard_delta_contribution": contribution})
    return {"guards": summaries, "components": contributions, "cases": cases}


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(audit_path, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("This is a CPU-only analysis of audited saved scores")
    if output.exists():
        raise FileExistsError("Never overwrite an earlier diagnostic")
    audit = training.read(audit_path)
    if not audit["complete"] or audit["research_selection_tolerance"] != 0:
        raise ValueError("Require the unchanged completed audit, not unaudited live logs")
    training.verify_registration(training.OUTPUT)
    verify_hashes(audit["artifact_sha256"])
    verify_hashes(audit["source_sha256"])
    current_path = training.OUTPUT / f'validation_{audit["through_step"]}.json'
    if str(current_path) not in audit["artifact_sha256"]:
        raise ValueError("The requested boundary must have been independently audited")
    initial_path = training.alignment.fixed.PRODUCER / "validation_0.json"
    current, initial = training.read(current_path), training.read(initial_path)
    if len(current["cases"]) != 152 or len(current["decision"]["comparisons"]) != 22:
        raise ValueError("Preserve every original case and retention guard")
    result = decompose(initial["cases"], current["cases"], current["decision"]["comparisons"])
    output.mkdir(parents=True)
    source_paths = [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "tests/test_single_model_guard_contributions.py"]
    config = {"boundary": audit["through_step"], "source_sha256": {str(path): sha256(path) for path in source_paths},
              "input_sha256": {str(path): sha256(path) for path in (audit_path, current_path, initial_path)},
              "scope": "Descriptive paired decomposition of all22 audited synthetic guards; no new forward, selection, confirmation or real evaluation"}
    write_json(output / "config.json", config)
    for name, rows in result.items():
        write_csv(output / f"{name}.csv", rows)
    write_json(output / "result.json", {**result, "complete": True, "goal_success_not_assessed": True,
                                        "config_sha256": sha256(output / "config.json")})
    lines = ["# Audited synthetic retention contributions", "", config["scope"], "",
             "Positive contributions worsen the cohort guard; negative contributions improve it.",
             "These are development diagnostics, not independent evidence or real benchmark gains.", "",
             "| Guard | Initial | Candidate | Worse / defined cases | Retained |",
             "|---|---:|---:|---:|---|"]
    for row in result["guards"]:
        lines.append(f'| {row["group"]}/{row["metric"]} | {row["initialization"]:.10f} | {row["candidate"]:.10f} | '
                     f'{row["worsened_cases"]}/{row["defined_cases"]} | {row["retained"]} |')
    lines.extend(["", "## Component decomposition", "",
                  "| Cohort | Metric | Component | Cases | Change (%) | Guard contribution |",
                  "|---|---|---|---:|---:|---:|"])
    for row in result["components"]:
        if row["metric"] not in {"common_nrmse", "mae", "rmse"} or row["group"] == "structured_overall":
            continue
        relative = "undefined" if row["change_percent"] is None else f'{row["change_percent"]:+.6f}'
        lines.append(f'| {row["group"]} | {row["metric"]} | {row["component"]} | {row["cases"]} | '
                     f'{relative} | {row["guard_delta_contribution"]:+.10f} |')
    (output / "RESULTS.md").write_text("\n".join([*lines, ""]))
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.audit.resolve(), arguments.output.resolve())
