"""Describe audited targeted-decoder progress separately from its inherited V14 errors."""

import argparse
import os
from pathlib import Path

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from report_single_model_guard_contributions import write_csv
import train_single_matrixformer_targeted_decoder as training
from report_single_model_decoder_progress import paired_progress


POLICY = ("Paired descriptive decomposition of already-audited synthetic development scores. "
          "Keep V8 as the sole retention baseline; distinguish V14-inherited gaps from decoder-only changes. "
          "Retain every guard, component and case. No new prediction, selection, training, confirmation, "
          "real benchmark, extrapolated learning curve or change to qualification.")


def report(audit_path, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Use CPU-only saved-score analysis")
    if output.exists():
        raise FileExistsError("Preserve earlier descriptive reports")
    audit = training.read(audit_path)
    if (not audit["complete"] or not audit["decoder_only_training"] or audit["research_selection_tolerance"] != 0
            or audit["training_initialization"] != str(training.PARENT)
            or audit["retention_baseline"] != str(training.original.alignment.fixed.PARENT)):
        raise ValueError("Require a completed V15 audit with distinct original baselines")
    training.verify_registration(training.OUTPUT)
    verify_hashes(audit["artifact_sha256"])
    verify_hashes(audit["source_sha256"])
    baseline_path = training.original.alignment.fixed.PRODUCER / "validation_0.json"
    parent_path = training.INITIAL_RUN / "validation_50.json"
    baseline, parent = training.read(baseline_path), training.read(parent_path)
    inputs = {str(path): sha256(path) for path in (audit_path, baseline_path, parent_path)}
    all_rows = {}
    for step in training.BOUNDARIES:
        if step > audit["through_step"]:
            continue
        path = training.OUTPUT / f"validation_{step}.json"
        if str(path) not in audit["artifact_sha256"]:
            raise ValueError("Every analyzed boundary must be independently audited")
        inputs[str(path)] = audit["artifact_sha256"][str(path)]
        current = training.read(path)
        if len(current["cases"]) != 152 or len(current["decision"]["comparisons"]) != 22:
            raise ValueError("Keep every original synthetic case and guard")
        for name, rows in paired_progress(baseline, parent, current).items():
            all_rows.setdefault(name, []).extend({"step": step, "accepted_updates": current["accepted_updates"], **row} for row in rows)
    output.mkdir(parents=True)
    sources = [Path(__file__).resolve(), REPO / "scripts/report_single_model_guard_contributions.py",
               REPO / "tests/test_single_model_decoder_progress.py",
               REPO / "scripts/report_single_model_decoder_progress.py",
               REPO / "tests/test_single_model_targeted_decoder_progress.py"]
    config = {"policy": POLICY, "audit": str(audit_path), "through_step": audit["through_step"],
              "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sources}}
    write_json(output / "config.json", config)
    for name, rows in all_rows.items():
        write_csv(output / f"{name}.csv", rows)
    write_json(output / "result.json", {"complete": True, "config_sha256": sha256(output / "config.json"),
               "boundaries": sorted({row["step"] for row in all_rows["guards"]}),
               "prediction_performed": False, "selection_performed": False, "research_selection_tolerance": 0,
               "goal_success_not_assessed": True, **all_rows})
    lines = ["# Audited decoder repair progress", "", POLICY, "",
             "Positive changes worsen the score. Repair percentages describe only gaps already present at V14; they are not qualification thresholds.",
             "", "| Boundary | Guard | V8 reference | V14 input | Candidate | Change vs V8 (%) | Inherited gap repaired (%) |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for row in all_rows["guards"]:
        change = "undefined" if row["retention_change_percent"] is None else f'{row["retention_change_percent"]:+.6f}'
        repair = "not an inherited deficit" if row["inherited_gap_repaired_percent"] is None else f'{row["inherited_gap_repaired_percent"]:.6f}'
        lines.append(f'| {row["step"]} | {row["group"]}/{row["metric"]} | {row["initialization"]:.10f} | '
                     f'{row["parent"]:.10f} | {row["candidate"]:.10f} | {change} | {repair} |')
    lines.extend(["", "All component and case contributions, including improvements and regressions, are retained in the CSV files.",
                  "These development diagnostics neither consume fresh confirmation nor establish real benchmark improvements.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.audit.resolve(), arguments.output.resolve())
