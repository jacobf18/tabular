"""Read-only completion verification and full paired panels for the registered prior trial."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import psutil
import torch

from analyze_dwip_spectral_evidence import cached_comparison
from diagnose_dwip_matched_objective_gradients import require_completed_stages
from evaluate_dwip_retention_followup import compare_records
from evaluate_dwip_runtime_followup import summarize
from run_dwip_anchor_mix_huber_continuous import PROPOSAL, QUEUE, verify_continuous_only_change
from run_dwip_budgeted_queue_v1 import storage_guard
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import compare_fold_metrics, sha256
from run_rowcol_controlled_experiments import BENCH, REPO, WORK


def read_json(path):
    return json.loads(path.read_text())


def read_records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.endswith("\n")]


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def fixed_selection(validation, selected, final_step=64000):
    if [row["step"] for row in validation] != list(range(0, final_step + 1, 2000)):
        raise ValueError("Require the complete ordered fixed-validation schedule")
    values = np.asarray([row["macro_rmse"] for row in validation])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid fixed-validation RMSE")
    minimum = validation[int(values.argmin())]
    if any(selected[key] != minimum[key] for key in ("step", "macro_rmse")):
        raise ValueError("Selected checkpoint is not the fixed-validation minimum")
    return minimum


def full_panel(baseline, selected):
    standard = compare_fold_metrics(baseline, selected)
    rows = []
    for method in ("entropy", "mi", "random"):
        for index in range(15):
            before, after = baseline[f"{method}_r2"][:, index], selected[f"{method}_r2"][:, index]
            delta = after - before
            rows.append({"method": method, "budget": index + 1, "baseline_r2": float(before.mean()),
                         "selected_r2": float(after.mean()), "delta_r2": float(delta.mean()),
                         "folds_improved": int((delta > 0).sum()), "fold_count": 10})
    arrays = {f"{method}_{metric}": {"baseline": baseline[f"{method}_{metric}"].tolist(),
              "selected": selected[f"{method}_{metric}"].tolist()}
              for method in ("entropy", "mi", "random") for metric in ("r2", "rmse")}
    counts = {"improved": sum(row["delta_r2"] > 0 for row in rows),
              "regressed": sum(row["delta_r2"] < 0 for row in rows),
              "tied": sum(row["delta_r2"] == 0 for row in rows)}
    return {"standard_budgets": standard, "mean_setting_counts": counts, "fold_arrays": arrays,
            "benchmark_names": baseline["benchmark_names"].tolist()}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default=QUEUE)
    args = parser.parse_args()
    if Path(args.queue).name != args.queue or args.queue in {"", ".", ".."}:
        raise ValueError("Require a single registered queue name")
    root = REPO / "runs/objective_queues" / args.queue
    require_completed_stages(root)
    outputs = ("completed_stage_audit.json", "continuous_full_comparison.json", "full_benchmark_r2.csv", "FULL_BENCHMARK_R2.md")
    if any((root / name).exists() for name in outputs):
        raise ValueError("Completion artifacts already exist; inspect rather than overwrite")
    work = WORK / f"tabimpute-{root.name}-dwip1e5"
    handles = {"training": read_json(root / "training_launcher.json"), "trainer": read_json(root / "trainer_identity.json"),
               **{stage: read_json(root / f"{stage}_evaluation/launch.json") for stage in ("transfer", "retention", "runtime")}}
    for stage, handle in handles.items():
        try:
            process = psutil.Process(handle["pid"])
            if process.create_time() == handle["created"] and process.status() != psutil.STATUS_ZOMBIE:
                raise RuntimeError(f"Original {stage} handle is still live; preserve jobs and wait")
        except psutil.NoSuchProcess:
            pass
    verified = {}

    def verify(path, expected=None):
        path = (REPO / path).resolve()
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Registered input changed: {path}")
        verified[str(path)] = actual
        return actual

    registration = read_json(root / "registration_audit.json")
    audit_registration = read_json(root / "completion_audit_registration.json")
    for source, expected in audit_registration["source_sha256"].items():
        verify(source, expected)
    control = root.parent / registration["fixed_control"]
    require_completed_stages(control)
    for name in ("continuous_huber", "baseline_reuse"):
        verify(root / f"{name}_manifest.json", registration[f"{name}_manifest_sha256"])
    for source, expected in registration["training_source_sha256"].items():
        verify(source, expected)
    manifest = read_json(root / "manifest.json")
    verify(manifest["initial_checkpoint"], manifest["initial_checkpoint_sha256"])
    for source, expected in manifest["source_sha256"].items():
        verify(source, expected)
    prior = read_json(root / "continuous_huber_manifest.json")
    verify(control / "manifest.json", prior["control_manifest_sha256"])
    verify(root.parent / PROPOSAL, prior["proposal_sha256"])
    commands = [read_json(folder / "dwip1e5_command.json") for folder in (control, root)]
    for command in commands:
        for flag in ("--run-name", "--fixed-validation-cache"):
            command[command.index(flag) + 1] = "<output-specific>"
    verify_continuous_only_change(*commands)
    for name, cache in prior["cache_equivalence"].items():
        for folder, key in ((root, "candidate_sha256"), (control, "source_sha256")):
            verify(folder / name, cache[key])
            cases = torch.load(folder / name, map_location="cpu", weights_only=True)["cases"]
            if len(cases) != 72 or case_digest(cases) != cache["identical_case_sha256"]:
                raise ValueError("Fixed case contents differ from the registered control")
    reuse = read_json(root / "baseline_reuse_manifest.json")
    for destination, artifact in reuse["artifacts"].items():
        verify(destination, artifact["sha256"])
        verify(artifact["source"], artifact["sha256"])
    for source, expected in {**reuse["mteb_inputs"], **reuse["transfer_source_sha256"]}.items():
        verify(source, expected)
    health = read_json(root / "health_at64k.json")
    verify(root / "audit_continuous_progress.py", health["audit_source_sha256"])
    if health["last_sampled_step"] != 64000 or health["snapshot_step"] != 64000:
        raise ValueError("Require the terminal64k precision/optimizer/schedule audit")
    for source, expected in health["input_sha256"].items():
        verify(source, expected)
    training = read_records(work / "metrics_stream.jsonl")
    validation = read_records(work / "validation_stream.jsonl")
    json.dumps([training, validation], allow_nan=False)
    selected = read_json(work / "best_validation_metrics.json")
    fixed_selection(validation, selected)
    if training[-1]["step"] != 64000 or len(training) != health["finite_sampled_records"]:
        raise ValueError("Training records changed after terminal audit")
    reference_training = {row["step"]: row for row in read_records(
        WORK / f"tabimpute-{control.name}-dwip1e5/metrics_stream.jsonl")}
    if len({row["step"] for row in training}) != len(training) or not all(
            all(row.get(field) == reference_training[row["step"]].get(field) and
                (field in row) == (field in reference_training[row["step"]])
                for field in health["matched_schedule_fields"]) for row in training):
        raise ValueError("Declared shape/component/LR scheduling does not match control")
    checkpoint = work / "best_validation_checkpoint.pth"
    selected_sha = verify(checkpoint)
    original = torch.load(manifest["initial_checkpoint"], map_location="cpu", weights_only=True)["model"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = payload["model"]
    if payload["epoch"] != selected["step"] or model.keys() != original.keys() or not all(
            value.shape == original[name].shape and value.dtype == torch.float32 and torch.isfinite(value).all()
            for name, value in model.items()):
        raise ValueError("Selected checkpoint identity, architecture, precision or finiteness differs")
    del original, payload, model
    for stage in ("retention", "runtime"):
        status = read_json(root / f"{stage}_evaluation/status.json")
        if status["checkpoint_sha256"] != selected_sha:
            raise ValueError(f"Selected weights differ in {stage}")
    for name in ("retention_evaluation/registration.json", "retention_evaluation/manifest.json"):
        for source, expected in read_json(root / name)["input_sha256"].items():
            verify(source, expected)
    runtime_registration = read_json(root / "runtime_evaluation/registration.json")
    verify(root / "manifest.json", runtime_registration["queue_manifest_sha256"])
    for source, expected in runtime_registration["source_sha256"].items():
        verify(source, expected)
    config = read_json(work / "config.json")
    if config["model_family"] != "new_stable" or config["model"] != runtime_registration["model_config"]:
        raise ValueError("Architecture differs from original DWIP")
    for stage, original_key in (("transfer", "baseline"), ("runtime", "original")):
        hashes = read_json(root / f"{stage}_evaluation/checkpoint_sha256.json")
        if hashes["selected"] != selected_sha or hashes[original_key] != manifest["initial_checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hashes differ in {stage}")
    source_paths = {"evaluator": BENCH / "code/eval_tabimpute.py", "interface": REPO / "src/tabimpute/interface.py",
                    "model": REPO / "src/tabimpute/model/model_new_stable.py", "driver": REPO / "scripts/run_dwip_transfer_evaluation.py"}
    transfer_manifest = read_json(root / "transfer_evaluation/manifest.json")
    for folder in (root, control):
        panel = read_json(folder / "transfer_evaluation/manifest.json")
        for key in ("datasets", "data_sha256", "source_sha256", "protocol", "precision"):
            if panel[key] != transfer_manifest[key]:
                raise ValueError(f"Paired transfer protocol differs: {key}")
        for dataset, expected in panel["data_sha256"].items():
            verify(BENCH / "data" / f"{dataset}.matrix.npz", expected)
        for name, expected in panel["source_sha256"].items():
            verify(source_paths[name], expected)
        hashes = read_json(folder / "transfer_evaluation/checkpoint_sha256.json")
        for label, path in panel["checkpoints"].items():
            verify(path, hashes[label])

    def load_cache(folder, dataset, baseline=False):
        if dataset == "mteb":
            label = "dwip_spectral_20260915_baseline" if baseline else f"{folder.name}_dwip1e5"
            path = BENCH / "logs" / f"tabimpute_{label}_mteb.npz"
        else:
            label = "baseline" if baseline else "selected"
            path = folder / "transfer_evaluation/logs" / f"tabimpute_dwip_transfer_{label}_{dataset}.npz"
        verify(path)
        with np.load(path, allow_pickle=True) as cache:
            return {key: cache[key] for key in cache.files}

    transfer_results = read_json(root / "transfer_evaluation/results.json")
    if set(transfer_results) != {"mmlu", "merged", "benchpress"}:
        raise ValueError("Incomplete transfer dataset coverage")
    reports, rows = {}, []
    for label, folder in (("original", root), ("primary_huber2", control)):
        reports[label] = {}
        for dataset in ("mteb", "mmlu", "merged", "benchpress"):
            panel, dataset_rows = full_panel(load_cache(folder, dataset, baseline=label == "original"), load_cache(root, dataset))
            if label == "original" and dataset != "mteb" and panel["standard_budgets"] != transfer_results[dataset]:
                raise ValueError(f"Transfer aggregates do not reproduce: {dataset}")
            if dataset == "mteb":
                recorded = read_json(root / "dwip1e5_mteb.json")
                expected_scores = {f"{method}_r2_{budget}" for method in ("entropy", "mi", "random") for budget in (5, 10, 15)}
                if set(recorded) != expected_scores:
                    raise ValueError("Incomplete saved MTEB score coverage")
                if any(panel["standard_budgets"][key]["selected"] != value for key, value in recorded.items()):
                    raise ValueError("MTEB aggregate does not reproduce")
            reports[label][dataset] = panel
            rows.extend({"reference": label, "dataset": dataset, **row} for row in dataset_rows)
    retention = read_json(root / "retention_evaluation/results.json")
    reference_retention = root.parent.parent / "information_flow_diagnostics/dwip_latent_retention_20260916/results.json"
    retention_comparisons = {"original": compare_records(read_json(reference_retention)["original"]["records"], retention["records"])}
    verify(control / "retention_evaluation/results.json", registration["retention_control_sha256"][control.name])
    retention_comparisons["primary_huber2"] = compare_records(read_json(control / "retention_evaluation/results.json")["records"], retention["records"])
    if len(retention["records"]) != 256 or len(retention_comparisons["original"]) != 8:
        raise ValueError("Incomplete frozen retention coverage")
    synthetic = {}
    for label, folder in (("original", root), ("primary_huber2", control)):
        prefix = "baseline" if label == "original" else "dwip1e5"
        synthetic[label] = {}
        for family, before_suffix, after_suffix in (("spectral", "spectral" if label == "original" else "selected", "selected"), ("general", "general", "general")):
            before, after = folder / f"{prefix}_{before_suffix}_predictions.pth", root / f"dwip1e5_{after_suffix}_predictions.pth"
            verify(before)
            verify(after)
            synthetic[label][family] = cached_comparison(before, after)
    runtime = read_json(root / "runtime_evaluation/results.json")
    runtime_comparisons = summarize(runtime["rows"])
    if runtime_comparisons != runtime["comparisons"]:
        raise ValueError("Runtime aggregates do not reproduce")
    old_inventory = read_json(root.parent / "dwip_spectral_anchor_mix25_huber1_point4_64k_20260917/final_saved_model_inventory.json")
    for source, expected in old_inventory["input_sha256"].items():
        verify(source, expected)
    previous_winner = old_inventory["winner"]
    wins_fixed = selected["macro_rmse"] < previous_winner["fixed_macro_rmse"]
    for path in (root / "health_at64k.json", root / "registration_audit.json", root / "launch.json",
                 work / "metrics_stream.jsonl", work / "validation_stream.jsonl", work / "best_validation_metrics.json",
                 root / "retention_evaluation/results.json", root / "runtime_evaluation/results.json", Path(__file__)):
        verify(path)
    totals = {label: sum(panel["mean_setting_counts"]["improved"] for panel in datasets.values()) for label, datasets in reports.items()}
    comparison = {"utc": datetime.now(timezone.utc).isoformat(), "queue": root.name, "selected_step": selected["step"],
                  "selected_fixed_macro_rmse": selected["macro_rmse"], "selected_sha256": selected_sha,
                  "previous_fixed_winner": previous_winner, "candidate_beats_previous_fixed_winner": wins_fixed,
                  "reports": reports, "improved_mean_r2_settings_out_of_180": totals, "synthetic": synthetic,
                  "retention": retention_comparisons, "runtime": runtime_comparisons, "input_sha256": verified,
                  "scope": "Identical fixed case contents select weights. Independent synthetic,10-fold/3-method/all15-budget benchmarks, retention and runtime are report-only. Counts are correlated descriptive settings, not independent tests or proof of the full objective."}
    storage = storage_guard(root, work, read_json(root / "artifact_budget.json"))
    for path, expected in verified.items():
        if sha256(path) != expected:
            raise ValueError(f"Input changed during read-only audit: {path}")
    lines = ["# Continuous-prior trial: completed benchmark breadth", "", f"Audit UTC: {comparison['utc']}", "",
             f"Selected step{selected['step']}; fixed synthetic macro RMSE{selected['macro_rmse']:.12f}; checkpoint SHA256 `{selected_sha}`.",
             f"Beats the previous global fixed-validation winner: **{wins_fixed}**. No benchmark-directed promotion.", "",
             "| Dataset | Original MI R2@5 | Huber2 MI R2@5 | Candidate MI R2@5 | Gains vs original /45 | Gains vs Huber2 /45 |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("mteb", "mmlu", "merged", "benchpress"):
        original, primary = reports["original"][dataset], reports["primary_huber2"][dataset]
        values = original["standard_budgets"]["mi_r2_5"]
        lines.append(f"| {dataset.upper()} | {values['baseline']:.6f} | {primary['standard_budgets']['mi_r2_5']['baseline']:.6f} | {values['selected']:.6f} | {original['mean_setting_counts']['improved']} | {primary['mean_setting_counts']['improved']} |")
    for label in reports:
        lines.extend(["", f"## Versus {label}", "", f"Improved mean R2 settings: {totals[label]}/180. All10 folds,three methods,budgets1–15 are included.", "", "Largest mean R2 regressions:"])
        regressions = sorted((row for row in rows if row["reference"] == label and row["delta_r2"] < 0), key=lambda row: row["delta_r2"])
        lines.extend(f"- {row['dataset'].upper()} {row['method']}@{row['budget']}: {row['baseline_r2']:.6f} → {row['selected_r2']:.6f} ({row['delta_r2']:+.6f}); {row['folds_improved']}/10 folds improve." for row in regressions[:10])
        if not regressions:
            lines.append("- No mean R2 regressions in this panel; this is not a significance claim.")
        lines.append(f"- Frozen retention: {sum(group['macro_case_rmse_delta'] < 0 for group in retention_comparisons[label].values())}/8 group macro RMSEs improve; independent samples, not OOD families.")
        for family, values in synthetic[label].items():
            scores = values["scores"]
            lines.append(f"- Independent {family} macro RMSE: {scores['baseline_macro_rmse']:.9f} → {scores['candidate_macro_rmse']:.9f}; {scores['groups_improved']}/{scores['group_count']} groups improve.")
    lines.extend(["", "## Runtime", ""])
    for size, values in runtime_comparisons.items():
        lines.append(f"- Public{size}×{size}: original{values['original_seconds']:.6f}s; selected{values['selected_seconds']:.6f}s; ratio{values['latency_ratio']:.4f}×. One warmup/three timed calls.")
    lines.extend(["", "Full fold R2/RMSE arrays, provenance and paired retention are in `continuous_full_comparison.json`; every mean R2 setting is in `full_benchmark_r2.csv`. A completed trial or a single threshold crossing does not automatically complete the broad research objective."])
    save_new(root / "continuous_full_comparison.json", comparison)
    with (root / "full_benchmark_r2.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "FULL_BENCHMARK_R2.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    save_new(root / "completed_stage_audit.json", {"utc": comparison["utc"], "all_four_stages_complete": True,
             "original_handles_absent": handles, "selected_sha256": selected_sha, "storage": storage,
             "input_sha256": verified, "comparison_sha256": sha256(root / "continuous_full_comparison.json"),
             "scope": "Trial completion and paired evidence only; broad objective requires separate assessment."})
    print(json.dumps({"queue": root.name, "selected_fixed_macro_rmse": selected["macro_rmse"], "improved_settings": totals,
                      "candidate_beats_previous_fixed_winner": wins_fixed}, indent=2))


if __name__ == "__main__":
    main()
