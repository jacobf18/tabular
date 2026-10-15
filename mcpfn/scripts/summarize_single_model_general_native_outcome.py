"""Describe the closed V47 trial without selecting an ineligible checkpoint."""

import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

import run_single_model_general_native_retention_pipeline as pipeline
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_csv, write_json


ROOT = pipeline.ROOT
OUTPUT = ROOT / "general_native_retention_v47_synthetic_outcome_v1"
RUN_SHA256 = "1a5410dda6c87086a35801e646688f408751eec0d99a0c8d01865d57f6d9c0df"
BOUNDARIES = (0, 10, 25, 50, 64)
POLICY = (
    "Read-only closure of the complete V47 native-general retention trial. Require the original "
    "64-attempt full task/teacher/component/reference audit, complete endpoint replay and original "
    "no-selection gate. Preserve all boundaries,22 comparisons, rejected updates and adverse "
    "actual-initializer changes. V8 is the historical development guard reference, not the actual "
    "V29 initializer. No new inference, confirmation, checkpoint selection, training, seeds, "
    "real benchmarks, guard changes or success claim. An unselected final state is not a model "
    "candidate. Descriptive consumed-bank results are not fresh confirmation or real benchmarks."
)


def comparison_rows(boundaries):
    if [record["step"] for record in boundaries] != list(BOUNDARIES):
        raise ValueError("Preserve every original boundary including the ineligible initializer")
    keys = [(row["group"], row["metric"]) for row in boundaries[0]["decision"]["comparisons"]]
    if len(keys) != 22 or len(set(keys)) != 22:
        raise ValueError("Preserve all22 original distinct development comparisons")
    limits = [row["initialization"] for row in boundaries[0]["decision"]["comparisons"]]
    parent_values = [row["candidate"] for row in boundaries[0]["decision"]["comparisons"]]
    rows = []
    for boundary in boundaries:
        decision = boundary["decision"]
        comparisons = decision["comparisons"]
        if ([(row["group"], row["metric"]) for row in comparisons] != keys
                or [row["initialization"] for row in comparisons] != limits
                or decision["research_selection_tolerance"] != 0
                or boundary["best_step"] != 0 or boundary["selected_checkpoint_sha256"] is not None
                or (boundary["step"] != 0 and decision["passed"] is not False)):
            raise ValueError("Never relabel or reselect a boundary or change its frozen comparisons")
        for comparison, parent in zip(comparisons, parent_values):
            value, limit = comparison["candidate"], comparison["initialization"]
            if (not all(isinstance(number, (int, float)) and math.isfinite(number)
                        for number in (value, limit, parent))
                    or comparison["retained"] is not (value <= limit)):
                raise ValueError("Retain finite values and the original exact zero-tolerance decision")
            rows.append({"step": boundary["step"], "group": comparison["group"], "metric": comparison["metric"],
                "V8_limit": limit, "actual_V29_initializer": parent, "candidate": value,
                "retained_vs_V8": comparison["retained"], "delta_vs_actual_V29": value - parent,
                "relative_percent_vs_actual_V29": 100 * (value / parent - 1) if parent else None})
    return rows


def validate_closed(documents):
    audit, replay, gate, result = (documents[name] for name in ("training_audit", "endpoint_audit", "gate", "result"))
    retention, components = documents["retention"], documents["components"]
    if (audit["complete"] is not True or audit["complete_training_audit"] is not True
            or audit["through_step"] != 64 or audit["attempted_batches"] != 64 or audit["accepted_updates"] != 58
            or audit["training_episodes"] != 1792 or audit["prediction_grids"] != 1368
            or audit["independently_regenerated_training_tasks"] != 1792
            or audit["independently_regenerated_rating_contexts"] != 128
            or audit["independently_regenerated_structured_draws"] != 1536
            or audit["best"]["step"] != 0 or audit["confirmation_evaluated"] is not False
            or audit["research_selection_tolerance"] != 0 or audit["full_network_training"] is not True
            or audit["fixed_half_training"] is not True or audit["displacement_fraction"] != .5
            or audit["autograd_replayed"] is not False or audit["full_optimizer_trajectory_replayed"] is not False):
        raise ValueError("Require the complete unchanged original training audit, not a prefix or replay claim")
    if (retention["complete"] is not True or retention["eligibility_decisions"] != 1792
            or retention["reference_grids"] != 1664 or retention["reference_role_counts"] != {"general": 128, "v8": 1536}
            or components["complete"] is not True or components["blend_tasks"] != 1664
            or components["raw_component_grids"] != 3328 or components["observed_weights_refitted_on_cpu"] is not True):
        raise ValueError("Require every independently reproduced reference and imputation teacher component")
    if (replay["complete"] is not True or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 0
            or replay["cuda_initialized"] is not False or replay["autograd_replayed"] is not False
            or gate["passed"] is not False or gate["selected_step"] != 0
            or gate["confirmation_evaluated"] is not False or gate["real_benchmark_success"] is not False
            or result["complete"] is not True or result["gate"] != gate
            or result["campaign"] is not None or result["review"] is not None
            or result["no_inference_ensemble"] is not True or result["goal_success_not_assessed"] is not True):
        raise ValueError("Close only the original audited no-selection outcome without consuming confirmation")


def require_closed():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("The closure is CPU-only and must never initialize a neural model")
    directories = [pipeline.training.OUTPUT, pipeline.OUTPUT,
                   pipeline.AUDIT.with_name(pipeline.AUDIT.name + "_task_replay"), pipeline.REPLAY, pipeline.CONFIRMATION]
    for directory in directories:
        state = pipeline.read(directory / "status.json")
        expected = "training_complete" if directory == pipeline.training.OUTPUT else "complete"
        if state["state"] != expected or pipeline.process_identity(state["pid"]) is not None:
            raise ValueError("Require all exact original workers to be complete and stopped")
    verify_hashes({str(pipeline.training.OUTPUT / "config.json"): RUN_SHA256})
    audit = pipeline.prerequisite(pipeline.OUTPUT)
    retention = pipeline.retention_training.require_report(pipeline.training, audit)
    components = pipeline.blend_replay.require_report(pipeline.training, audit)
    documents = {"training_audit": audit, "retention": retention, "components": components,
        "endpoint_audit": pipeline.read(pipeline.REPLAY / "audit.json"),
        "gate": pipeline.read(pipeline.CONFIRMATION / "gate.json"),
        "result": pipeline.read(pipeline.OUTPUT / "result.json")}
    validate_closed(documents)
    replay_gate = pipeline.read(pipeline.REPLAY / "gate.json")
    if (replay_gate["endpoint_reproduced"] is not True or replay_gate["selected_validation_passed"] is not False
            or replay_gate["selected_step"] != 0
            or replay_gate["audit_sha256"] != sha256(pipeline.REPLAY / "audit.json")
            or documents["gate"]["reason"] != pipeline.no_confirmation_reason(audit["best"], replay_gate)
            or {path.name for path in pipeline.CONFIRMATION.iterdir()} != {
                "config.json", "registration.json", "status.json", "run.lock", "gate.json"}):
        raise ValueError("Require the exact no-selection gate and an otherwise unconsumed confirmation directory")
    if documents["result"]["training_audit_sha256"] != sha256(pipeline.AUDIT / "audit.json"):
        raise ValueError("Bind the exact original training audit used by the coordinator")
    for report in (audit, retention, components, documents["endpoint_audit"]):
        verify_hashes(report["artifact_sha256"])
    forbidden = [pipeline.CAMPAIGN, pipeline.REVIEW,
                 *(pipeline.training.OUTPUT / name for name in ("selected_checkpoint.pth", "best_validation_checkpoint.pth")),
                 *(pipeline.CONFIRMATION / name for name in ("manifest.json", "synthetic_bank.pt", "student.json", "reference.json"))]
    if any(path.exists() for path in forbidden):
        raise ValueError("Never summarize absent selection or unconsumed confirmation when contradictory artifacts exist")
    boundaries = [pipeline.read(pipeline.training.OUTPUT / f"validation_{step}.json") for step in BOUNDARIES]
    reference = pipeline.read(pipeline.training.original.alignment.fixed.PRODUCER / "reference.json")["references"]
    baseline = pipeline.read(pipeline.training.original.alignment.fixed.PRODUCER / "validation_0.json")["student"]
    config = pipeline.read(pipeline.training.OUTPUT / "config.json")
    for record in boundaries:
        recomputed = pipeline.training.original.alignment.fixed.selection_decision(record["student"], baseline, reference, config["base"])
        if recomputed != record["decision"]:
            raise ValueError("Independently reproduce every original development decision without reselection")
    rows = comparison_rows(boundaries)
    trace = [json.loads(line) for line in (pipeline.training.OUTPUT / "training.jsonl").read_text().splitlines()]
    if ([row["step"] for row in trace] != list(range(1, 65)) or sum(row["accepted"] for row in trace) != 58
            or [row["step"] for row in trace if not row["accepted"]] != [18, 27, 41, 42, 57, 58]):
        raise ValueError("Preserve every scheduled attempt and all six rejected updates")
    paths = [pipeline.training.OUTPUT / name for name in ("config.json", "registration.json", "training.jsonl", "validation.jsonl")]
    paths.extend(pipeline.training.OUTPUT / f"validation_{step}.json" for step in BOUNDARIES)
    paths.extend(directory / "status.json" for directory in directories)
    paths.extend([pipeline.AUDIT / "audit.json", pipeline.REPLAY / "audit.json", pipeline.REPLAY / "gate.json",
                  pipeline.CONFIRMATION / "gate.json", pipeline.OUTPUT / "config.json", pipeline.OUTPUT / "result.json"])
    return documents, boundaries, rows, trace, {str(path): sha256(path) for path in paths}


def render(summary, boundaries, rows):
    lines = ["# V47 native-general retention: audited negative result", "", POLICY, "",
        "The actual initializer is unchanged V29. Only native-imputation training retention changes from V8 to the original general parent; "
        "all structured families retain V8. Mean20, mixed priors, observed-blend imputation targets, specialist rating targets and all original guards stay fixed. "
        "Seeds differ from V46, so this is not a seed-controlled causal comparison.", "",
        "## Outcome", "", "64 attempts, 58 accepted updates and six rejected updates. **No trained boundary qualifies.** "
        "The final step64 state is not a selected model. No fresh confirmation, real benchmark campaign or review runs; "
        "the single-checkpoint benchmark goal remains unmet.", "",
        "The independent full audit reproduces 1,792 original tasks, 128 rating contexts, 3,328 imputation teacher components, "
        "128 general-native and 1,536 V8-structured reference grids. The CPU endpoint audit checks 56 grids and no selected-model grids. "
        "These checks are not historical full-autograd or full optimizer-trajectory replay.", "",
        "## Synthetic Development Boundaries", "", "These are synthetic results, not real MissBench/UCI/recommender scores. "
        "All original 22 comparisons remain below; step0 is the actual unchanged V29 initializer and is ineligible by design.", "",
        "| Step | Accepted updates | Minimax score | Failed guards |", "|---|---:|---:|---:|"]
    for record in boundaries:
        decision = record["decision"]
        lines.append(f'| {record["step"]} | {record["accepted_updates"]} | {decision["score"]:.12g} | '
                     f'{sum(not row["retained"] for row in decision["comparisons"])}/22 |')
    lines.extend(["", "## All Development Comparisons", "", "The frozen V8 limit is not the actual V29 training initializer. "
        "No tolerance is added and no aggregate gain overrides an individual guard failure.", "",
        "| Group | Metric | V8 limit | V29 step0 | Step10 | Step25 | Step50 | Step64 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"])
    for index in range(22):
        values = [rows[boundary_index * 22 + index] for boundary_index in range(len(BOUNDARIES))]
        first = values[0]
        lines.append(f'| {first["group"]} | {first["metric"]} | {first["V8_limit"]:.12g} | '
                     + " | ".join(f'{row["candidate"]:.12g}' for row in values) + " |")
    lines.extend(["", "## Provenance", "", "Full numeric values, actual-initializer changes and all adverse outcomes are in "
        "`development_comparisons.csv` and `summary.json`. This report performs no neural inference or new selection.", ""])
    for path, digest in summary["artifact_sha256"].items():
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart a closed-outcome report")
    for path, reserve in ((ROOT, 18 * 1024**3 + 16 * 1024**2), (Path("/"), 6 * 1024**3),
                          (pipeline.training.SCRATCH, 50 * 1024**3)):
        if shutil.disk_usage(path).free < reserve:
            raise OSError("Preserve original reserves while writing the small read-only report")
    documents, boundaries, rows, trace, inputs = require_closed()
    source_paths = [Path(__file__).resolve(), REPO / "tests/test_single_model_general_native_outcome.py"]
    sources = {str(path): sha256(path) for path in source_paths}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"policy": POLICY, "input_sha256": inputs, "source_sha256": sources})
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
               "registered_unix": time.time(), "scientific_stage_repeated": False})
    summary = {"complete": True, "outcome": "no_eligible_trained_boundary", "goal_achieved": False,
        "actual_initializer": str(pipeline.training.PARENT), "training_attempts": 64, "accepted_updates": 58,
        "selected_step": None, "confirmation_evaluated": False, "real_benchmarks_evaluated": False,
        "new_inference_performed": False, "checkpoint_selection_performed": False, "cuda_initialized": False,
        "boundaries": [{"step": record["step"], "score": record["decision"]["score"],
                        "comparisons": record["decision"]["comparisons"]} for record in boundaries],
        "rejected_updates": [{"step": row["step"], "reason": row["reason"], "finite": row["finite"]}
                             for row in trace if not row["accepted"]],
        "gate": documents["gate"], "artifact_sha256": inputs, "source_sha256": sources}
    write_json(OUTPUT / "summary.json", summary)
    write_csv(OUTPUT / "development_comparisons.csv", rows)
    (OUTPUT / "RESULTS.md").write_text(render(summary, boundaries, rows))
    verify_hashes(inputs)
    verify_hashes(sources)
    write_json(OUTPUT / "audit.json", {"complete": True, "config_sha256": sha256(OUTPUT / "config.json"),
        "comparison_rows": len(rows), "all_original_guards_preserved": True, "new_inference_performed": False,
        "checkpoint_selection_performed": False, "cuda_initialized": torch.cuda.is_initialized(),
        "artifact_sha256": {str(OUTPUT / name): sha256(OUTPUT / name) for name in
                            ("summary.json", "development_comparisons.csv", "RESULTS.md")}})
    print(json.dumps({"complete": True, "report": str(OUTPUT / "RESULTS.md"), "goal_achieved": False}), flush=True)


if __name__ == "__main__":
    with pipeline.arithmetic_recovery.previous.original_arithmetic():
        main()
