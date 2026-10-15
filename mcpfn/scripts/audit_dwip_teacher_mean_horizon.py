"""Terminal128k and full-panel audits for the single registered horizon trial."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from analyze_dwip_spectral_evidence import cached_comparison
from audit_dwip_continuous_completion import fixed_selection, full_panel, read_records, save_new
from dwip_distribution_compact_artifacts import BEST, LATEST
from dwip_teacher_mean_horizon import HORIZON_STEPS, PREVIOUS_QUEUE, horizon_recipe
from dwip_teacher_mean_training import read_json, verify_hashes
from evaluate_dwip_retention_followup import compare_records
from evaluate_dwip_runtime_followup import summarize
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK


DATASETS = ("mteb", "mmlu", "merged", "benchpress")
MODEL_FILES, ROLLING = (BEST,), LATEST


def expected_lr(step):
    elapsed = step + 1
    if elapsed <= 600:
        return 1e-5 * elapsed / 600
    progress = min(1., (elapsed - 600) / 127400)
    return 1e-5 * (.05 + .95 * .5 * (1 + math.cos(math.pi * progress)))


def training_records(records):
    if [row["step"] for row in records] != [1, *range(25, HORIZON_STEPS + 1, 25)]:
        raise ValueError("Require all5121 ordered sampled training records through128k")
    required = {"loss_missing", "nll_missing", "teacher_point_loss_missing", "cdf_loss_missing",
                "rmse_missing_mean", "teacher_mean_rmse_missing", "lr", "grad_norm"}
    for row in records:
        if not required.issubset(row) or any(isinstance(value, (int, float)) and not math.isfinite(value) for value in row.values()):
            raise ValueError("Incomplete or nonfinite training metrics")
        if not math.isclose(row["lr"], expected_lr(row["step"]), rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Sampled learning rate differs from the declared128k cosine")
    return len(records)


def check_model_state(payload, reference):
    state = payload["model"]
    if state.keys() != reference.keys() or len(state) != 254:
        raise ValueError("Require the unchanged254-tensor student without teacher weights")
    if any(value.shape != reference[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all()
           for name, value in state.items()):
        raise ValueError("Student state has changed shapes, non-FP32 or nonfinite tensors")


def matched_data_records(candidate, previous):
    expected_steps = [1, *range(25, 64001, 25)]
    prefix = [row for row in candidate if row["step"] <= 64000]
    if [row["step"] for row in prefix] != expected_steps or [row["step"] for row in previous] != expected_steps:
        raise ValueError("Require all2561 shared data records through64k")
    fields = {"matrix_rows", "matrix_cols", "batch_size", "sampled_prior_index", "sampled_prior_name"}
    for actual, expected in zip(prefix, previous):
        support = {name for name in expected if name.startswith("support_")}
        if not support or support != {name for name in actual if name.startswith("support_")}:
            raise ValueError("Support-normalization record fields differ")
        if not fields.issubset(expected) or any(name not in actual or actual[name] != expected[name] for name in fields | support):
            raise ValueError("The horizon changed sampled data shapes, components or support normalization")
    return len(prefix)


def check_optimizer(payload):
    if payload["epoch"] != HORIZON_STEPS or payload["scheduler"]["last_epoch"] != HORIZON_STEPS:
        raise ValueError("Require terminal128k optimizer and cosine scheduler")
    states = payload["optimizer"]["state"]
    if len(states) != 238:
        raise ValueError("Require all238 FP32 Adam moment pairs")
    for state in states.values():
        if float(state["step"]) != HORIZON_STEPS or any(state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all()
                                               for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Invalid terminal AdamW state")
    return len(states)


def audit_training(root):
    output = root / "training_stage_audit.json"
    if output.exists():
        raise ValueError("Preserve the completed terminal audit")
    registration = read_json(root / "training_registration.json")
    verify_hashes(registration["input_sha256"])
    status = read_json(root / "training_status.json")
    if status["state"] != "complete" or status["step"] != HORIZON_STEPS or not 0 < status["elapsed_seconds"] <= 86400:
        raise ValueError("Training is not a completed128k run within24h")
    work = WORK / f"tabimpute-{root.name}-dwip1e5"
    names = {path.name for path in work.glob("*checkpoint*.pth*")}
    if names != {*MODEL_FILES, ROLLING}:
        raise ValueError("Unexpected or incomplete compact checkpoint/temporary inventory")
    selected = read_json(work / "best_validation_metrics.json")
    validation = read_records(work / "validation_stream.jsonl")
    fixed_selection(validation, selected, final_step=HORIZON_STEPS)
    count = training_records(read_records(work / "metrics_stream.jsonl"))
    config = read_json(work / "config.json")
    if config["teacher_serialized_in_student"] or config["point_target_kind"] != "detached_equal_frozen_teacher_mean":
        raise ValueError("Invalid saved teacher/student semantics")
    if (config["nll_loss_weight"], config["point_loss_weight"], config["cdf_loss_weight"]) != (1., 2., .05):
        raise ValueError("The previously successful teacher-mean objective must remain unchanged")
    args = horizon_recipe()
    if (config["total_steps"], config["warmup_steps"], config["stable_steps"], config["decay_steps"]) != (HORIZON_STEPS, 600, 0, 127400):
        raise ValueError("Saved horizon/cosine configuration differs")
    if config["phases"] != json.loads(args.phases_json) or config["terminal_checkpoint"] != LATEST:
        raise ValueError("Saved phase or terminal-artifact policy differs")
    candidate_records = read_records(work / "metrics_stream.jsonl")
    previous_records = read_records(WORK / f"tabimpute-{PREVIOUS_QUEUE}-dwip1e5/metrics_stream.jsonl")
    matched_data_records(candidate_records, previous_records)
    initial = read_json(root / "manifest.json")["initial_checkpoint"]
    reference = torch.load(initial, map_location="cpu", weights_only=True)["model"]
    checkpoint_hashes, moments = {}, None
    budget = read_json(root / "artifact_budget.json")
    for name in (*MODEL_FILES, ROLLING):
        path = work / name
        ceiling = budget["latest_file_ceiling_bytes" if name == ROLLING else "model_file_ceiling_bytes"]
        if path.stat().st_size > ceiling:
            raise ValueError("Saved checkpoint exceeds its registered padded ceiling")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        check_model_state(payload, reference)
        if payload["epoch"] != (selected["step"] if name == MODEL_FILES[0] else HORIZON_STEPS):
            raise ValueError("Saved checkpoint step disagrees with selector or terminal step")
        if name == ROLLING:
            moments = check_optimizer(payload)
        elif "optimizer" in payload:
            raise ValueError("Model-only checkpoint unexpectedly contains an optimizer")
        checkpoint_hashes[name] = sha256(path)
        del payload
    paths = [work / name for name in ("config.json", "metrics_stream.jsonl", "validation_stream.jsonl",
             "best_validation_metrics.json", "validation_predictions_best.pth", "metrics.json")]
    paths += [root / "training_status.json", root / "training_registration.json"]
    result = {"utc": datetime.now(timezone.utc).isoformat(), "state": "complete", "selected": selected,
              "checkpoint_sha256": checkpoint_hashes, "sampled_training_records": count,
              "validation_records": len(validation), "finite_fp32_state_tensors": 254,
              "finite_fp32_adam_moment_pairs": moments, "elapsed_seconds": status["elapsed_seconds"],
              "input_sha256": {str(path): sha256(path) for path in paths}, "goal_status": "active"}
    verify_hashes(registration["input_sha256"])
    save_new(output, result)
    return result


def load_arrays(path):
    with np.load(path, allow_pickle=True) as cache:
        return {key: cache[key] for key in cache.files}


def complete_panels(references, selected):
    if set(selected) != set(DATASETS) or set(references) != {"original", "primary_mean64"}:
        raise ValueError("Require all four selected datasets and both registered references")
    reports, rows = {}, []
    for label, panels in references.items():
        if set(panels) != set(DATASETS):
            raise ValueError("Reference dataset coverage is incomplete")
        reports[label] = {}
        for dataset in DATASETS:
            report, panel_rows = full_panel(panels[dataset], selected[dataset])
            reports[label][dataset] = report
            rows.extend({"reference": label, "dataset": dataset, **row} for row in panel_rows)
    return reports, rows


def audit_complete(root):
    if (root / "completed_stage_audit.json").exists():
        raise ValueError("Preserve the completed full audit")
    manifest = read_json(root / "manifest.json")
    admission = read_json(root / "admission.json")
    verify_hashes(admission["input_sha256"])
    training = read_json(root / "training_stage_audit.json")
    verify_hashes(training["input_sha256"])
    stages = [root, *(root / f"{name}_evaluation" for name in ("transfer", "retention", "runtime"))]
    for stage in stages:
        if read_json(stage / "status.json")["state"] != "complete" or not (stage / "results.json").exists():
            raise ValueError(f"Stage is incomplete: {stage}")
    work = WORK / f"tabimpute-{root.name}-dwip1e5"
    checkpoint = work / "best_validation_checkpoint.pth"
    selected_sha = sha256(checkpoint)
    if selected_sha != training["checkpoint_sha256"][MODEL_FILES[0]]:
        raise ValueError("Selected checkpoint changed after terminal audit")
    for name, expected in training["checkpoint_sha256"].items():
        if sha256(work / name) != expected:
            raise ValueError("A saved terminal artifact changed during follow-ups")
    if selected_sha != read_json(root / "synthetic_evaluation.json")["checkpoint_sha256"]:
        raise ValueError("Synthetic evaluation used different selected weights")
    if selected_sha != read_json(root / "transfer_evaluation/checkpoint_sha256.json")["selected"]:
        raise ValueError("Transfer evaluated different selected weights")
    if selected_sha != read_json(root / "retention_evaluation/status.json")["checkpoint_sha256"]:
        raise ValueError("Retention evaluated different selected weights")
    if selected_sha != read_json(root / "runtime_evaluation/checkpoint_sha256.json")["selected"]:
        raise ValueError("Runtime evaluated different selected weights")
    inputs = dict(admission["input_sha256"])
    inputs.update(training["input_sha256"])

    def record(path):
        path = Path(path)
        actual = sha256(path)
        if str(path) in inputs and inputs[str(path)] != actual:
            raise ValueError(f"Input changed during the completion audit: {path}")
        inputs[str(path)] = actual
        return path

    references = {label: {dataset: load_arrays(record(path)) for dataset, path in paths.items()}
                  for label, paths in manifest["reference_panels"].items()}
    selected = {dataset: load_arrays(record(path)) for dataset, path in manifest["selected_panels"].items()}
    reports, rows = complete_panels(references, selected)
    transfer = read_json(record(root / "transfer_evaluation/results.json"))
    for dataset in DATASETS[1:]:
        if transfer[dataset] != reports["original"][dataset]["standard_budgets"]:
            raise ValueError("Transfer aggregates do not reproduce from all folds")
    mteb = read_json(record(root / "dwip1e5_mteb.json"))
    expected_mteb = {f"{method}_r2_{budget}": reports["original"]["mteb"]["standard_budgets"][f"{method}_r2_{budget}"]["selected"]
                     for method in ("entropy", "mi", "random") for budget in (5, 10, 15)}
    if mteb != expected_mteb:
        raise ValueError("MTEB aggregates do not reproduce")
    retention = read_json(record(root / "retention_evaluation/results.json"))
    original_retention = read_json(record(manifest["original_retention"]))["original"]["records"]
    primary_retention = read_json(record(manifest["primary_retention"]))["records"]
    retention_comparisons = {"original": compare_records(original_retention, retention["records"]),
                            "primary_mean64": compare_records(primary_retention, retention["records"])}
    if len(retention["records"]) != 256 or any(len(groups) != 8 for groups in retention_comparisons.values()):
        raise ValueError("Incomplete retention case/group coverage")
    runtime = read_json(record(root / "runtime_evaluation/results.json"))
    runtime_comparisons = summarize(runtime["rows"])
    if runtime_comparisons != runtime["comparisons"] or len(runtime["rows"]) != 6:
        raise ValueError("Incomplete or inconsistent matched runtime panel")
    synthetic = {}
    for label, sources in manifest["synthetic_references"].items():
        synthetic[label] = {family: cached_comparison(record(path), record(root / f"dwip1e5_{'selected' if family == 'spectral' else 'general'}_predictions.pth"))
                            for family, path in sources.items()}
    inventory = read_json(record(manifest["previous_inventory"]))
    verify_hashes(inventory["input_sha256"])
    score = training["selected"]["macro_rmse"]
    candidate = {"queue": root.name, "kind": "fixed_selected_training_checkpoint", "state": "complete",
                 "step": training["selected"]["step"], "fixed_macro_rmse": score, "checkpoint": str(checkpoint),
                 "sha256": selected_sha, "case_count": 72, "case_digest": manifest["fixed_case_digest"],
                 "checkpoint_mtime_utc": datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat()}
    if any(item["queue"] == root.name for item in inventory["inventory"]):
        raise ValueError("Candidate already appears in prior inventory")
    candidates = sorted([*inventory["inventory"], candidate], key=lambda item: (item["fixed_macro_rmse"], item["queue"]))
    totals = {label: sum(panel["mean_setting_counts"]["improved"] for panel in panels.values()) for label, panels in reports.items()}
    for path in (checkpoint, root / "manifest.json", root / "admission.json", root / "training_stage_audit.json",
                 root / "synthetic_evaluation.json", root / "launch.json", root / "retention_evaluation/launch.json"):
        record(path)
    verify_hashes(inputs)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "queue": root.name, "selected": candidate,
              "previous_fixed_winner": inventory["winner"], "candidate_beats_previous_fixed_winner": score < inventory["winner"]["fixed_macro_rmse"],
              "reports": reports, "improved_mean_r2_settings_out_of_180": totals, "synthetic": synthetic,
              "retention": retention_comparisons, "runtime": runtime_comparisons, "input_sha256": inputs,
              "goal_status": "active", "scope": "Complete selected-single-model panels; all non-fixed outcomes are report-only. Correlated mean-setting counts are descriptive, not significance tests or automatic objective completion."}
    save_new(root / "teacher_full_comparison.json", result)
    with (root / "full_benchmark_r2.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inventory_inputs = {**inventory["input_sha256"], str(checkpoint): selected_sha,
                        str(work / "best_validation_metrics.json"): sha256(work / "best_validation_metrics.json")}
    save_new(root / "final_saved_model_inventory.json", {"utc": result["utc"], "inventory": candidates,
             "winner": candidates[0], "selection": "Identical fixed synthetic macro RMSE only", "input_sha256": inventory_inputs})
    lines = ["# Teacher-mean trial: full benchmark R2", "", f"Completed audit: {result['utc']}", "",
             f"Selected step{candidate['step']}, fixed macro RMSE{score:.12f}. Beats the prior fixed winner: {result['candidate_beats_previous_fixed_winner']}.", "",
             "| Dataset | Original MI R2@5 | Mean64 MI R2@5 | Selected MI R2@5 | Gains vs original /45 | Gains vs Mean64 /45 |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        original, primary = reports["original"][dataset], reports["primary_mean64"][dataset]
        metric = original["standard_budgets"]["mi_r2_5"]
        lines.append(f"| {dataset.upper()} | {metric['baseline']:.6f} | {primary['standard_budgets']['mi_r2_5']['baseline']:.6f} | {metric['selected']:.6f} | {original['mean_setting_counts']['improved']} | {primary['mean_setting_counts']['improved']} |")
    lines += ["", f"Improved mean settings: {totals}. Full4×10×3×15 coverage; no benchmark-driven checkpoint selection.",
              "", "## Largest regressions versus original"]
    for row in sorted((row for row in rows if row["reference"] == "original" and row["delta_r2"] < 0), key=lambda row: row["delta_r2"])[:12]:
        lines.append(f"- {row['dataset']} {row['method']}@{row['budget']}: {row['delta_r2']:+.6f}; {row['folds_improved']}/10 folds improve.")
    lines += ["", "Full independent synthetic, paired256-case retention, matched100/500/1000 runtime, hashes and all fold arrays are in `teacher_full_comparison.json`.",
              "No saved teacher weights or deployed ensemble. The broad objective requires a separate assessment; this trial audit does not declare it achieved."]
    with (root / "FULL_BENCHMARK_R2.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    save_new(root / "completed_stage_audit.json", {"state": "complete", "utc": result["utc"],
             "comparison_sha256": sha256(root / "teacher_full_comparison.json"),
             "inventory_sha256": sha256(root / "final_saved_model_inventory.json"),
             "selected_sha256": selected_sha, "goal_status": "active"})
    return {"selected": candidate, "totals": totals, "goal_status": "active"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    if Path(args.queue).name != args.queue:
        raise ValueError("Require a single queue name")
    root = REPO / "runs/objective_queues" / args.queue
    print(json.dumps(audit_training(root) if args.training_only else audit_complete(root), indent=2))


if __name__ == "__main__":
    main()
