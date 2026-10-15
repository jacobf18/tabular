"""Describe audited V22 development gaps without new prediction or model selection."""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from report_single_model_guard_contributions import matches, write_csv
from report_single_model_teacher_reference_gaps import metadata
import recover_single_model_cached_continuation as training


OUTPUT = training.ROOT / "teacher_continuation_v22_cached_recovery_v1_development_gaps"
POLICY = ("Descriptive CPU-only analysis of the already selected and fully audited V22 boundary64 on all152 "
    "original synthetic development cases. Compare original V8, V20 warm start, original general, original "
    "reference blend/specialist and actual domain-routed training teachers. Preserve every original22-guard "
    "cohort and truth-defined null. Training teacher is V8 for legacy, original general for structured and "
    "original specialist for ratings. These are same-case score differences, not causal effects, distillation "
    "ceilings, independent confirmation, real benchmark results or a new checkpoint-selection rule. No "
    "predictions, training, reselection, qualification changes or inference ensemble.")
ROLES = ("v8", "warm_start", "original_general", "reference_teacher", "candidate", "training_teacher")


def read(path):
    return json.loads(Path(path).read_text())


def summarize(reference, initial, parent, current, comparisons):
    if not initial or any(len(rows) != len(initial) for rows in (reference, parent, current)):
        raise ValueError("Preserve every aligned development case")
    aligned, seen = [], set()
    for original, baseline, warm, candidate in zip(reference, initial, parent, current):
        identity = baseline["index"], baseline["seed"]
        if identity in seen or any(metadata(row) != metadata(baseline) for row in (original, warm, candidate)):
            raise ValueError("Preserve unique case identity, ordering and complete metadata")
        seen.add(identity)
        target = (original["teacher"] if baseline["domain"] == "recsys" else
                  original["general"] if baseline.get("structured", False) else baseline["student"])
        roles = {"v8": baseline["student"], "warm_start": warm["student"], "original_general": original["general"],
                 "reference_teacher": original["teacher"], "candidate": candidate["student"], "training_teacher": target}
        aligned.append((baseline, roles))
    summaries, cases, guards = [], [], set()
    for guard in comparisons:
        identity = guard["group"], guard["metric"]
        if identity in guards:
            raise ValueError("Never duplicate an original guard")
        guards.add(identity)
        selected = [(row, roles) for row, roles in aligned if matches(row, guard["group"])]
        values = {role: [] for role in ROLES}
        for row, roles in selected:
            measured = {role: scores[guard["metric"]] for role, scores in roles.items()}
            nulls = [value is None for value in measured.values()]
            if any(nulls) and (not all(nulls) or guard["metric"] != "common_nrmse"):
                raise ValueError("Preserve identical truth-defined null coverage")
            for role, value in measured.items():
                if value is not None:
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("Require finite nonnegative defined scores")
                    values[role].append(value)
            cases.append({"group": guard["group"], "metric": guard["metric"], "index": row["index"], "seed": row["seed"],
                          "domain": row["domain"], "family": row.get("family"), **measured})
        if not values["v8"]:
            raise ValueError("Never omit or fabricate an all-undefined cohort")
        means = {role: math.fsum(scores) / len(scores) for role, scores in values.items()}
        if (not math.isclose(means["v8"], guard["initialization"], rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(means["candidate"], guard["candidate"], rel_tol=1e-12, abs_tol=1e-12)
                or guard["retained"] != (guard["candidate"] <= guard["initialization"])):
            raise ValueError("Reproduce the actual audited guard scores and decisions")
        result = {"group": guard["group"], "metric": guard["metric"], "cases": len(selected),
                  "defined_cases": len(values["v8"]), "undefined_cases": len(selected) - len(values["v8"]), **means}
        for role in ROLES:
            if role != "candidate":
                result[f"candidate_change_vs_{role}_percent"] = None if means[role] == 0 else 100 * (means["candidate"] / means[role] - 1)
                result[f"candidate_minus_{role}"] = means["candidate"] - means[role]
        summaries.append(result)
    return {"guards": summaries, "cases": cases, "case_identities": sorted(seen)}


def report():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Use CPU-only saved-score analysis")
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite a descriptive report")
    training_config = training.verify_registration(training.OUTPUT)
    audit_path = training.AUDIT / "audit.json"
    audit = read(audit_path)
    if (not audit["complete"] or not audit["complete_training_audit"] or audit["through_step"] != 64
            or audit["best"]["step"] != 64 or audit["prediction_grids"] != 1368
            or audit["research_selection_tolerance"] != 0):
        raise ValueError("Require the completed unchanged training audit and selected64 boundary")
    verify_hashes(audit["artifact_sha256"])
    verify_hashes(audit["source_sha256"])
    producer = training.original.alignment.fixed.PRODUCER
    paths = [producer / "reference.json", producer / "validation_0.json", training.INITIAL_RUN / "validation_64.json",
             training.OUTPUT / "validation_64.json"]
    audited_inputs = {**training_config["input_sha256"], **audit["artifact_sha256"]}
    for path in paths:
        if audited_inputs.get(str(path)) != sha256(path):
            raise ValueError("Every source score must belong to the independent training audit")
    reference, initial, parent, current = map(read, paths)
    if len(current["cases"]) != 152 or len(current["decision"]["comparisons"]) != 22:
        raise ValueError("All152 original development cases and22 guards remain mandatory")
    config = {"policy": POLICY, "input_sha256": {str(path): sha256(path) for path in [audit_path, *paths]},
              "source_sha256": {str(path): sha256(path) for path in (Path(__file__).resolve(),
                  REPO / "scripts/report_single_model_guard_contributions.py",
                  REPO / "scripts/report_single_model_teacher_reference_gaps.py",
                  REPO / "tests/test_single_model_cached_continuation_progress.py")}}
    result = summarize(reference["cases"], initial["cases"], parent["cases"], current["cases"], current["decision"]["comparisons"])
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    write_csv(OUTPUT / "guards.csv", result["guards"])
    write_csv(OUTPUT / "cases.csv", result["cases"])
    write_json(OUTPUT / "result.json", {**result, "complete": True, "prediction_performed": False, "selection_performed": False,
        "confirmation_evaluated": False, "goal_success_not_assessed": True, "config_sha256": sha256(OUTPUT / "config.json")})
    lines = ["# Audited synthetic development gaps", "", POLICY, "",
             "Candidate percentage changes; negative is better. All baselines use the identical synthetic cases.", "",
             "| Cohort | Metric | Defined cases | vs V8 | vs warm start | vs general | vs reference teacher | vs training teacher |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["guards"]:
        values = [row[f"candidate_change_vs_{role}_percent"] for role in ROLES if role != "candidate"]
        labels = ["undefined" if value is None else f"{value:+.6f}%" for value in values]
        lines.append(f'| {row["group"]} | {row["metric"]} | {row["defined_cases"]}/{row["cases"]} | ' + " | ".join(labels) + " |")
    lines.extend(["", "CSV/JSON retain all cases, truth-defined nulls, absolute deltas and reference scores.",
                  "Independent qualification and real-benchmark results remain separate and unchanged.", ""])
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines))
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report()
