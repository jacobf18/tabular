"""Compare saved synthetic teacher/reference scores without predicting, selecting or training."""

import argparse
import math
import os
from pathlib import Path

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from report_single_model_guard_contributions import matches, write_csv
import train_single_matrixformer_paired_decoder as training


POLICY = ("Descriptive comparison of all152 already-audited development cases and all22 original guard cohorts. "
          "Compare V8, original general, the original reference teacher and the current audited candidate. "
          "The actual training teacher is V8 for imputation and the original specialist for ratings; the "
          "reference teacher is the original observed-error imputation blend or rating specialist. Preserve "
          "every case, guard and truth-defined null. These score differences are not a teacher-induced causal "
          "effect, a hard distillation ceiling, a generalization result or a decision to change the live trial. "
          "Ground-truth NLL/CDF and point protection can already improve beyond the training teacher. "
          "No new predictions, training, selection, confirmation or real evaluation.")
ROLES = ("v8", "original_general", "reference_teacher", "candidate", "actual_training_teacher")


def metadata(row):
    excluded = {"student", "general", "teacher", "routing", "prediction_file", "prediction_sha256"}
    return {key: value for key, value in row.items() if key not in excluded}


def summarize(reference, initial, current, comparisons):
    if not initial or len(initial) != len(reference) or len(initial) != len(current):
        raise ValueError("Keep every aligned original development case")
    seen, aligned = set(), []
    for original, baseline, candidate in zip(reference, initial, current):
        identity = baseline["index"], baseline["seed"]
        if identity in seen or not (metadata(original) == metadata(baseline) == metadata(candidate)):
            raise ValueError("Case identity, order and all non-prediction metadata must match")
        seen.add(identity)
        actual = original["teacher"] if baseline["domain"] == "recsys" else baseline["student"]
        aligned.append((baseline, {"v8": baseline["student"], "original_general": original["general"],
                        "reference_teacher": original["teacher"], "candidate": candidate["student"],
                        "actual_training_teacher": actual}))
    summaries, cases, guards = [], [], set()
    for guard in comparisons:
        identity = guard["group"], guard["metric"]
        if identity in guards:
            raise ValueError("Never duplicate or overwrite a guard cohort")
        guards.add(identity)
        selected = [(row, roles) for row, roles in aligned if matches(row, guard["group"])]
        if not selected:
            raise ValueError("Every original guard cohort must retain its cases")
        values = {role: [] for role in ROLES}
        for row, roles in selected:
            measured = {role: scores[guard["metric"]] for role, scores in roles.items()}
            nulls = [value is None for value in measured.values()]
            if any(nulls) and (not all(nulls) or guard["metric"] != "common_nrmse"):
                raise ValueError("Truth-defined missing score coverage must remain identical across roles")
            for role, value in measured.items():
                if value is not None:
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("Every defined score must remain finite and nonnegative")
                    values[role].append(value)
            cases.append({"group": guard["group"], "metric": guard["metric"], "index": row["index"],
                          "seed": row["seed"], "domain": row["domain"], "family": row.get("family"),
                          "component": row.get("component", row.get("preference_family")), **measured})
        if not values["v8"]:
            raise ValueError("Never fabricate an all-undefined cohort mean")
        means = {role: math.fsum(scores) / len(scores) for role, scores in values.items()}
        if (not math.isclose(means["v8"], guard["initialization"], rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(means["candidate"], guard["candidate"], rel_tol=1e-12, abs_tol=1e-12)
                or guard["retained"] != (guard["candidate"] <= guard["initialization"])):
            raise ValueError("Audited baseline/candidate means and strict decisions must reproduce")
        row = {"group": guard["group"], "metric": guard["metric"], "cases": len(selected),
               "defined_cases": len(values["v8"]), "undefined_cases": len(selected) - len(values["v8"]), **means}
        for role in ROLES[1:]:
            row[f"{role}_change_vs_v8_percent"] = None if means["v8"] == 0 else 100 * (means[role] / means["v8"] - 1)
        summaries.append(row)
    return {"guards": summaries, "cases": cases, "case_identities": sorted(seen)}


def report(audit_path, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Use CPU-only analysis of saved audited scores")
    if output.exists():
        raise FileExistsError("Never overwrite a descriptive report")
    audit = training.read(audit_path)
    if (not audit["complete"] or not audit["decoder_only_training"] or audit["research_selection_tolerance"] != 0
            or audit["training_initialization"] != str(training.PARENT)
            or audit["retention_baseline"] != str(training.PARENT)
            or audit["through_step"] not in training.BOUNDARIES[1:]):
        raise ValueError("Require the completed unchanged V19 boundary audit")
    training.verify_registration(training.OUTPUT)
    verify_hashes(audit["artifact_sha256"])
    verify_hashes(audit["source_sha256"])
    producer = training.original.alignment.fixed.PRODUCER
    paths = [producer / "reference.json", producer / "validation_0.json",
             training.OUTPUT / f'validation_{audit["through_step"]}.json']
    if str(paths[2]) not in audit["artifact_sha256"]:
        raise ValueError("The requested candidate must have been independently audited")
    reference, initial, current = map(training.read, paths)
    if len(current["cases"]) != 152 or len(current["decision"]["comparisons"]) != 22:
        raise ValueError("Preserve the entire original development protocol")
    sources = [Path(__file__).resolve(), REPO / "scripts/report_single_model_guard_contributions.py",
               REPO / "tests/test_single_model_teacher_reference_gaps.py"]
    config = {"policy": POLICY, "through_step": audit["through_step"],
              "input_sha256": {str(path): sha256(path) for path in [audit_path, *paths]},
              "source_sha256": {str(path): sha256(path) for path in sources}}
    output.mkdir()
    write_json(output / "config.json", config)
    result = summarize(reference["cases"], initial["cases"], current["cases"], current["decision"]["comparisons"])
    write_csv(output / "guards.csv", result["guards"])
    write_csv(output / "cases.csv", result["cases"])
    write_json(output / "result.json", {**result, "complete": True, "prediction_performed": False,
               "selection_performed": False, "training_performed": False, "confirmation_evaluated": False,
               "goal_success_not_assessed": True, "config_sha256": sha256(output / "config.json")})
    lines = ["# Saved synthetic teacher/reference differences", "", POLICY, "",
             "Percentages below are relative to V8 on the same saved synthetic cases; negative is better.", "",
             "| Cohort | Metric | Cases | Original general | Reference teacher | Candidate | Actual training teacher |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["guards"]:
        changes = [row[f"{role}_change_vs_v8_percent"] for role in ROLES[1:]]
        text = ["undefined" if value is None else f"{value:+.6f}%" for value in changes]
        lines.append(f'| {row["group"]} | {row["metric"]} | {row["defined_cases"]}/{row["cases"]} | ' + " | ".join(text) + " |")
    lines.extend(["", "All152 identities and all case/cohort scores, including undefined slots, remain in the JSON/CSV files.",
                  "This report does not select a replacement teacher or modify the live experiment.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.audit.resolve(), arguments.output.resolve())
