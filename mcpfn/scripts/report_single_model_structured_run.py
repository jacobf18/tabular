"""Report every registered V9 validation boundary after its complete CPU audit."""

import argparse
import csv
import json
import math
from pathlib import Path

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
from single_model_structured_mixture import FAMILIES, STRUCTURED_METRICS


def read(path):
    return json.loads(Path(path).read_text())


def retention_values(metrics, config):
    values = [(domain, name, metrics[domain][name]) for domain, name in config["retention_metrics"]]
    values.extend(("structured_overall", name, metrics["structured"]["overall"][name]) for name in STRUCTURED_METRICS)
    if set(metrics["structured"]["families"]) != set(FAMILIES):
        raise ValueError("Keep all twelve structured families")
    values.extend((family, "common_nrmse", metrics["structured"]["families"][family]["common_nrmse"]) for family in FAMILIES)
    if len(values) != 22 or len({(group, metric) for group, metric, value in values}) != 22:
        raise ValueError("Require exactly the original22 distinct retention comparisons")
    if any(value is None or not math.isfinite(value) or value < 0 for group, metric, value in values):
        raise ValueError("Do not omit undefined or nonfinite retention metrics")
    return values


def summarize(reports, config, audit, status):
    if (status.get("state") != "training_complete" or status.get("step") != 400
            or config.get("steps") != 400 or config.get("validate_every") != 50):
        raise ValueError("Require the completed registered400-update run")
    if (not audit.get("complete") or not audit.get("full_400_update_log_and_validation_audited")
            or audit.get("through_step") != 400 or audit.get("logged_training_updates_checked") != 400
            or audit.get("prediction_grids_rescored") != 1672
            or audit.get("research_selection_tolerance") != 0 or not audit.get("original_artifacts_unchanged")):
        raise ValueError("Require the full unchanged raw-prediction audit, not a successful prefix")
    expected = list(range(0, 401, 50))
    if [report["step"] for report in reports] != expected:
        raise ValueError("Every validation boundary including initialization is mandatory")
    if len(audit.get("selection_trace", [])) != len(reports):
        raise ValueError("The audit must retain the complete selection trace")
    initial = {(group, metric): value for group, metric, value in retention_values(reports[0]["student"], config)}
    rows, boundaries = [], []
    best_step, best_score = 0, float("inf")
    for report, trace in zip(reports, audit["selection_trace"]):
        comparisons = []
        for group, metric, value in retention_values(report["student"], config):
            baseline = initial[group, metric]
            row = {"step": report["step"], "group": group, "metric": metric, "initialization": baseline,
                   "candidate": value, "delta": value - baseline,
                   "relative_percent": 100 * (value / baseline - 1) if baseline else None,
                   "retained": value <= baseline}
            comparisons.append(row)
            rows.append(row)
        eligible = all(row["retained"] for row in comparisons)
        if not math.isfinite(report["score"]) or report["score"] < 0:
            raise ValueError("Require a finite nonnegative minimax score")
        if eligible and report["score"] < best_score:
            best_step, best_score = report["step"], report["score"]
        decision = {"step": report["step"], "score": report["score"], "eligible": eligible, "best_step": best_step}
        if decision != trace or any(report[key] != value for key, value in decision.items()):
            raise ValueError("Reporting must reproduce the audited strict selection decisions")
        boundaries.append({**decision, "regressed_retention_metrics": sum(not row["retained"] for row in comparisons),
                           "regressions": [f'{row["group"]}/{row["metric"]}' for row in comparisons if not row["retained"]]})
    if status.get("best_step") != best_step:
        raise ValueError("Final training status disagrees with the audited selection")
    return {"complete": True, "scope": "Synthetic training selection only; no new benchmark results",
            "selected_step": best_step, "eligible_trained_steps": [row["step"] for row in boundaries if row["step"] and row["eligible"]],
            "selected_trained_checkpoint": best_step != 0, "boundaries": boundaries,
            "retention_comparisons_per_boundary": 22, "research_selection_tolerance": 0,
            "goal_success_not_assessed": True, "optimizer_trajectory_replayed": False}, rows


def markdown(summary, rows):
    lines = ["# V9 structured-mixture training outcome", "", summary["scope"] + ".",
             "All400 updates and all nine registered validation boundaries are retained.",
             "The preceding independent CPU audit rescores1672 complete prediction grids; this report derives its tables from those audited results.",
             "", f'Selected update: **{summary["selected_step"]}**. Research selection tolerance: **zero**.',
             "Independent confirmation and full frozen real benchmarks are still required before any success claim.", ""]
    if not summary["selected_trained_checkpoint"]:
        lines.extend(["**No trained checkpoint was selected.** Step zero is the unchanged V8 initialization, not a new model.", ""])
    lines.extend(["## All validation boundaries", "", "| Update | Extended minimax | Retention regressions /22 | Eligible | Best update so far |",
                  "|---:|---:|---:|:---:|---:|"])
    for boundary in summary["boundaries"]:
        lines.append(f'| {boundary["step"]} | {boundary["score"]:.10f} | {boundary["regressed_retention_metrics"]} | {boundary["eligible"]} | {boundary["best_step"]} |')
    lines.extend(["", "## Final update versus initialization", "", "These are synthetic metrics, not MissBench/UCI/recommender benchmark measurements.",
                  "Positive changes are worse. All22 original retention constraints are shown, including related metric definitions; none are replaced with a pooled score.",
                  "", "| Group | Metric | Initialization | Update400 | Change (%) | Retained |", "|---|---|---:|---:|---:|:---:|"])
    for row in rows:
        if row["step"] == 400:
            change = "undefined (zero baseline)" if row["relative_percent"] is None else f'{row["relative_percent"]:+.6f}'
            lines.append(f'| {row["group"]} | {row["metric"]} | {row["initialization"]:.10f} | {row["candidate"]:.10f} | {change} | {row["retained"]} |')
    lines.extend(["", "Full-precision values and all earlier boundaries are in `retention.csv` and `summary.json`.",
                  "This report changes no checkpoint, training decision, reference score or benchmark protocol.", ""])
    return "\n".join(lines)


def report(run, audit_directory, output):
    if output.is_relative_to(run) or run.is_relative_to(output) or output.is_relative_to(audit_directory) or audit_directory.is_relative_to(output):
        raise ValueError("Write the derived report outside the original run and audit")
    if output.exists():
        raise FileExistsError("Never replace a prior report")
    audit = read(audit_directory / "audit.json")
    hashes = dict(audit["original_artifact_sha256"])
    paths = [run / name for name in ("config.json", "status.json")]
    paths.extend(run / f"validation_{step}.json" for step in range(0, 401, 50))
    for path in paths:
        if path.name != "status.json" and str(path) not in hashes:
            raise ValueError("Every reported validation and its configuration must be bound by the original audit")
    paths.extend((audit_directory / "audit.json", Path(__file__).resolve()))
    verify_hashes(hashes)
    for path in paths:
        hashes.setdefault(str(path), sha256(path))
    reports = [read(run / f"validation_{step}.json") for step in range(0, 401, 50)]
    summary, rows = summarize(reports, read(run / "config.json"), audit, read(run / "status.json"))
    verify_hashes(hashes)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    with (output / "retention.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "RESULTS.md").write_text(markdown(summary, rows))
    write_json(output / "provenance.json", {"run": str(run), "audit": str(audit_directory), "input_sha256": hashes,
                                           "training_performed": False, "checkpoint_reselection_performed": False})
    print(json.dumps({key: value for key, value in summary.items() if key != "boundaries"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.audit.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
