"""Report the complete audited grid and compare its known quarter-step control."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from confirm_single_model_weight_average import inventory, ready
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_csv, write_json
import single_model_displacement_grid as grid


def summarize(decisions, selection, gate):
    if grid.choose(decisions) != selection or gate["selected_candidate"] != selection["selected"]:
        raise ValueError("Reported selection differs from the audited grid")
    if gate["confirmation_evaluated"] != selection["passed"] or (gate["passed"] and not selection["passed"]):
        raise ValueError("Confirmation cannot bypass original development eligibility")
    candidates, retention = [], []
    expected_keys = set(grid.mixture.column.RETENTION_METRICS)
    expected_keys.update(("structured_overall", metric) for metric in grid.mixture.STRUCTURED_METRICS)
    expected_keys.update((family, "common_nrmse") for family in grid.mixture.FAMILIES)
    for specification in grid.GRID:
        decision = decisions[specification["id"]]
        comparisons = decision["comparisons"]
        keys = [(row["group"], row["metric"]) for row in comparisons]
        if len(keys) != 22 or set(keys) != expected_keys:
            raise ValueError("Keep all22 distinct retention comparisons for every candidate")
        for row in comparisons:
            if row["retained"] != (row["candidate"] <= row["initialization"]):
                raise ValueError("Never waive or round away a regression")
            retention.append({"candidate_id": specification["id"], "alpha": specification["alpha"], "beta": specification["beta"],
                              **row, "relative_percent": None if row["initialization"] == 0 else
                              100 * (row["candidate"] / row["initialization"] - 1)})
        retained = all(row["retained"] for row in comparisons)
        if retained != decision["retained"] or decision["passed"] != (retained and decision["score"] < decision["initialization_score"]):
            raise ValueError("Original strict eligibility does not reproduce")
        selected = selection["selected"] == specification
        lookup = {(row["group"], row["metric"]): row["candidate"] for row in comparisons}
        candidates.append({**specification, "minimax": decision["score"], "initialization_minimax": decision["initialization_score"],
                           "failed_guards": sum(not row["retained"] for row in comparisons), "eligible": decision["passed"],
                           "selected": selected, "synthetic_confirmed": selected and gate["passed"],
                           "legacy_common_nrmse": lookup["imputation", "common_nrmse"],
                           "rating_rmse": lookup["recsys", "rmse"], "rating_mae": lookup["recsys", "mae"],
                           "structured_common_nrmse": lookup["structured_overall", "common_nrmse"]})
    return candidates, retention


def compare_predictions(previous, current):
    if len(previous["cases"]) != 152 or len(current["cases"]) != 152:
        raise ValueError("Known control comparison requires all152 original cases")
    records, hashes = [], {}
    for index, (old, new) in enumerate(zip(previous["cases"], current["cases"])):
        if old["index"] != index or new["index"] != index or old["seed"] != new["seed"] or old["domain"] != new["domain"]:
            raise ValueError("Known control cases do not match")
        paths = [Path(record["prediction_file"]) for record in (old, new)]
        fingerprints = {str(path): record["prediction_sha256"] for path, record in zip(paths, (old, new))}
        verify_hashes(fingerprints)
        arrays = []
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != {"student"}:
                    raise ValueError("Require one full student prediction grid")
                arrays.append(saved["student"])
        first, second = arrays
        if first.shape != second.shape or first.dtype != second.dtype or not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("Known control grids must have the same finite shape and dtype")
        records.append({"index": index, "seed": old["seed"], "domain": old["domain"], "elements": first.size,
                        "different_elements": int(np.count_nonzero(first != second)),
                        "max_absolute_difference": float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))})
        hashes.update(fingerprints)
    return {"cases": len(records), "elements": sum(row["elements"] for row in records),
            "different_elements": sum(row["different_elements"] for row in records),
            "max_absolute_difference": max(row["max_absolute_difference"] for row in records),
            "all_prediction_values_equal": all(row["different_elements"] == 0 for row in records),
            "case_metrics_identical": all(old["student"] == new["student"] for old, new in zip(previous["cases"], current["cases"])),
            "aggregate_metrics_identical": previous["student"] == current["student"]}, records, hashes


def markdown(summary, candidates, retention):
    lines = ["# Complete single-network displacement-grid results", "",
             "Synthetic evidence only: these are not MissBench, UCI, recommender, or LLM benchmark measurements.",
             "All twelve original candidates and all264 zero-tolerance retention comparisons are retained.",
             "Each candidate is a single original network constructed from two already-trained parameter directions; no new gradient training or prediction ensemble.", "",
             "| Candidate | V10 alpha | V11 beta | Minimax | Failed guards /22 | Eligible | Selected |",
             "|---|---:|---:|---:|---:|:---:|:---:|"]
    for row in candidates:
        lines.append(f'| {row["id"]} | {row["alpha"]} | {row["beta"]} | {row["minimax"]:.10f} | {row["failed_guards"]} | {row["eligible"]} | {row["selected"]} |')
    lines += ["", f'Fresh confirmation evaluated: **{summary["confirmation_evaluated"]}**.',
              f'Final synthetic gate passed: **{summary["synthetic_gate_passed"]}**.', "",
              "## Every failed development guard", "", "Positive changes are worse; no numerical selection tolerance is introduced.", "",
              "| Candidate | Group | Metric | Initial | Candidate | Change (%) |", "|---|---|---|---:|---:|---:|"]
    for row in retention:
        if not row["retained"]:
            relative = "undefined (zero baseline)" if row["relative_percent"] is None else f'{row["relative_percent"]:+.8f}'
            lines.append(f'| {row["candidate_id"]} | {row["group"]} | {row["metric"]} | {row["initialization"]:.12g} | {row["candidate"]:.12g} | {relative} |')
    control = summary["known_control"]
    lines += ["", "## Previously evaluated quarter-step control", "",
              "Candidate04 (alpha=.25,beta=0) is an explicitly known control, not a blinded candidate.",
              f'All parameter values identical: **{control["state_values_identical"]}**.',
              f'All152 complete prediction grids identical in value: **{control["all_prediction_values_equal"]}**.',
              f'Maximum absolute prediction difference: {control["max_absolute_difference"]:.12g}.',
              f'Aggregate metrics identical: **{control["aggregate_metrics_identical"]}**.', "",
              "`retention.csv` contains every guard; `candidates.csv` contains all four main synthetic metric summaries.",
              "Reporting changes no checkpoint, selected candidate, gate, source configuration, or benchmark protocol.",
              "Real-benchmark parity and goal completion are not established by this report.", ""]
    return "\n".join(lines)


def report(run, qualification, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Reporting must remain CPU-only")
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output) for path in (run, qualification)):
        raise ValueError("Write a new report separate from immutable originals")
    grid.verify_registration(run)
    if grid.read(qualification / "status.json")["state"] != "complete":
        raise ValueError("Wait for the original qualifier; never report a partial grid")
    grid.require_audit(qualification, "selection")
    gate = grid.read(qualification / "gate.json")
    if gate["confirmation_evaluated"]:
        grid.require_audit(qualification, "confirmation")
        confirmation = grid.read(qualification / "confirmation/decision.json")
        if any(gate.get(key) != value for key, value in confirmation.items()):
            raise ValueError("Final gate differs from the audited confirmation decision")
    audit = grid.read(qualification / "selection/audit.json")
    hashes = dict(audit["artifact_sha256"])
    paths = [run / "config.json", qualification / "gate.json", qualification / "selection/audit.json", Path(__file__).resolve()]
    if gate["confirmation_evaluated"]:
        hashes.update(grid.read(qualification / "confirmation/audit.json")["artifact_sha256"])
        paths.append(qualification / "confirmation/audit.json")
    known = grid.ROOT / "fixed_contraction_v1_qualification"
    original_audit = grid.read(known / "selection/audit.json")
    if not original_audit["complete"] or original_audit["cases"] != 152 or original_audit["prediction_grids_rescored"] != 608:
        raise ValueError("Require the full original quarter-step audit")
    hashes.update(original_audit["artifact_sha256"])
    paths += [known / "selection/audit.json", known / "selection/student.json", known / "selection/decision.json"]
    bank_path = str(grid.PRODUCER / "synthetic_bank.pt")
    if audit["artifact_sha256"][bank_path] != original_audit["bank_sha256"]:
        raise ValueError("Known control must use the identical saved development bank")
    decisions = {spec["id"]: grid.read(qualification / "selection" / spec["id"] / "decision.json") for spec in grid.GRID}
    candidates, retention = summarize(decisions, grid.read(qualification / "selection/decision.json"), gate)
    specification = next(spec for spec in grid.GRID if spec["alpha"] == .25 and spec["beta"] == 0.)
    previous_path = grid.ROOT / "fixed_contraction_v1/contracted_checkpoint.pth"
    verify_hashes({str(previous_path): original_audit["checkpoint_sha256"]})
    previous_state = torch.load(previous_path, map_location="cpu", weights_only=True)["model"]
    current_state = grid.load_candidate(run, specification)["model"]
    details = inventory(current_state, previous_state)
    control, control_rows, prediction_hashes = compare_predictions(grid.read(known / "selection/student.json"),
        grid.read(qualification / "selection" / specification["id"] / "student.json"))
    control.update(state_values_identical=details["changed_tensors"] == 0, state_inventory=details,
                   strict_decision_identical=decisions[specification["id"]] == grid.read(known / "selection/decision.json"))
    if not control["state_values_identical"]:
        raise ValueError("Registered quarter-step control parameters differ from the previous fixed construction")
    hashes.update(prediction_hashes)
    hashes.update({str(path): sha256(path) for path in paths})
    verify_hashes(hashes)
    summary = {"complete": True, "candidates": len(candidates), "retention_comparisons": len(retention),
               "selected_candidate": gate["selected_candidate"], "synthetic_gate_passed": gate["passed"],
               "confirmation_evaluated": gate["confirmation_evaluated"], "known_control": control,
               "cuda_initialized": torch.cuda.is_initialized(), "training_performed": False,
               "checkpoint_reselection_performed": False, "goal_success_not_assessed": True}
    output.mkdir(parents=True)
    write_csv(output / "candidates.csv", candidates)
    write_csv(output / "retention.csv", retention)
    write_csv(output / "known_control_cases.csv", control_rows)
    write_json(output / "summary.json", summary)
    (output / "RESULTS.md").write_text(markdown(summary, candidates, retention))
    write_json(output / "provenance.json", {"run": str(run), "qualification": str(qualification), "input_sha256": hashes})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    run, qualification, output = args.run.resolve(), args.qualification.resolve(), args.output.resolve()
    if args.wait:
        while not ready(qualification):
            time.sleep(30)
    report(run, qualification, output)


if __name__ == "__main__":
    main()
