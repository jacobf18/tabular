"""One equal parameter average of the two lowest fixed-selected raw checkpoints."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.resources
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch

from audit_dwip_continuous_completion import full_panel
import diagnose_dwip_latent_retention as retention
from diagnose_dwip_top2_fixed_predictions import INVENTORY, top_two
from diagnose_dwip_matched_objective_gradients import require_completed_stages
from evaluate_dwip_checkpoint_average import average_states
import evaluate_dwip_cross_seed_average as fixed
from evaluate_dwip_retention_followup import compare_records
from evaluate_dwip_runtime_followup import SIZES, VARIANTS, summarize, worker_command
import run_dwip_average_transfer as transfer
from run_dwip_budgeted_queue_v1 import check_storage, immutable_json, regular_bytes, run_guarded
from run_dwip_continuous_spectrum import case_digest
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import gpu_processes, sha256, write_json
from run_rowcol_controlled_experiments import BENCH, REPO, WORK
from tabimpute.interface import ImputePFN
from train_direct_pure_rabel_curriculum import atomic_torch_save


EXPERIMENT = "dwip_top2_fixed_parameter_average_20260917"
DIAGNOSTIC = "dwip_top2_fixed_complementarity_20260917"
RETENTION = REPO / "runs/information_flow_diagnostics/dwip_latent_retention_20260916"
SOURCES = ("scripts/evaluate_dwip_top2_fixed_average.py", "tests/test_dwip_top2_fixed_average.py",
           "scripts/diagnose_dwip_top2_fixed_predictions.py", "scripts/audit_dwip_continuous_completion.py",
           "scripts/evaluate_dwip_checkpoint_average.py", "scripts/evaluate_dwip_cross_seed_average.py",
           "scripts/run_dwip_average_transfer.py", "scripts/run_dwip_transfer_evaluation.py",
           "scripts/run_dwip_budgeted_queue_v1.py", "scripts/rowcol_fixed_validation.py",
           "scripts/evaluate_dwip_retention_followup.py", "scripts/evaluate_dwip_runtime_followup.py",
           "scripts/diagnose_dwip_latent_retention.py", "scripts/benchmark_dwip_geometry_runtime.py",
           "scripts/train_direct_pure_rabel_curriculum.py", "src/tabimpute/interface.py",
           "src/tabimpute/model/model_new_stable.py", "src/tabimpute/model/bar_distribution.py",
           "src/tabimpute/model/prediction_distribution.py")


def read_json(path):
    return json.loads(path.read_text())


def average_budget(source_sizes):
    if len(source_sizes) != 2 or min(source_sizes) <= 0:
        raise ValueError("Require two positive checkpoint sizes")
    mib = 1024**2
    ceiling = ((max(source_sizes) * 5 + 4 * mib - 1) // (4 * mib)) * mib
    peak = 2 * ceiling + 2 * 1024**3
    return {"model_file_ceiling_bytes": ceiling, "persistent_checkpoint_bytes": ceiling,
            "atomic_temporary_bytes": ceiling, "other_artifacts_bytes": 2 * 1024**3,
            "peak_additional_bytes": peak, "reserve_bytes": 20 * 1024**3,
            "required_free_bytes_at_launch": peak + 20 * 1024**3,
            "scope": "One model with25% size padding, one atomic temporary,2GiB other artifacts and20GiB reserve. No optimizer or training snapshots; all outputs are under the experiment root."}


def guard(root, launch=False):
    if sha256(root / "artifact_budget.json") != read_json(root / "registration_integrity.json")["artifact_budget.json"]:
        raise ValueError("Registered artifact budget changed")
    result = check_storage(read_json(root / "artifact_budget.json"), shutil.disk_usage(root).free,
                           regular_bytes(root), launch=launch)
    write_json(root / "storage_status.json", {**result, "updated_unix": time.time()})
    return result


def verify(root):
    for name, expected in read_json(root / "registration_integrity.json").items():
        if sha256(root / name) != expected:
            raise ValueError(f"Average registration changed: {name}")
    registration = read_json(root / "prespecification.json")
    for path, expected in registration["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen average input changed: {path}")
    return registration


def strict_gate(selection, members):
    scores = selection["macro_rmse"]
    if set(scores) != {"seed1", "seed2", "average"} or not all(math.isfinite(value) and value >= 0 for value in scores.values()):
        raise ValueError("Invalid synthetic score set")
    for label, item in zip(("seed1", "seed2"), members):
        if not math.isclose(scores[label], item["fixed_macro_rmse"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Fresh raw fixed score does not reproduce closely enough")
    accepted = scores["average"] < min(scores["seed1"], scores["seed2"])
    expected_variant = "average" if accepted else min(("seed1", "seed2"), key=scores.get)
    if selection["average_accepted"] is not accepted or selection.get("selected_variant") != expected_variant:
        raise ValueError("Inconsistent synthetic gate")
    if accepted and scores["average"] >= min(item["fixed_macro_rmse"] for item in members):
        raise ValueError("Average must also beat the registered fixed winner")
    return accepted


def register(root):
    inventory_path = root.parent / INVENTORY
    inventory = read_json(inventory_path)
    for path, expected in inventory["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Fixed inventory changed: {path}")
    members = top_two(inventory["inventory"])
    diagnostic_root = root.parent / DIAGNOSTIC
    diagnostic = read_json(diagnostic_root / "results.json")
    if read_json(diagnostic_root / "status.json")["state"] != "complete" or not diagnostic["equal_prediction_mean_beats_both_raw"]:
        raise ValueError("Require the completed fixed-only complementarity evidence")
    if read_json(diagnostic_root / "prespecification.json")["members"] != members:
        raise ValueError("Fixed-ranked members differ from the registered diagnostic")
    queues = dict(zip(("seed1", "seed2"), (item["queue"] for item in members)))
    inputs = {str(REPO / path): sha256(REPO / path) for path in SOURCES}
    inputs.update({str(path): sha256(path) for path in (inventory_path, diagnostic_root / "results.json", diagnostic_root / "prespecification.json")})
    caches = {"validation": root.parent / members[0]["queue"] / "validation.pth",
              "spectral": root.parent / members[0]["queue"] / "holdout.pth",
              "general": root.parent / "rowcol_output_remedies_paired_20260907/independent_validation.pth"}
    configs, checkpoint_hashes, sizes = {}, {}, []
    spec = dwip_spec()
    for label, item in zip(queues, members):
        folder = root.parent / item["queue"]
        require_completed_stages(folder)
        work = WORK / f"tabimpute-{item['queue']}-dwip1e5"
        config = read_json(work / "config.json")
        if config["model_family"] != "new_stable" or config["model"] != spec["model_config"]:
            raise ValueError("Require unchanged DWIP architecture")
        training_manifest = read_json(folder / "manifest.json")
        if training_manifest["initial_checkpoint_sha256"] != sha256(spec["checkpoint_path"]):
            raise ValueError("Constituents must share the original initialization")
        for path, expected in training_manifest["source_sha256"].items():
            if sha256(REPO / path) != expected:
                raise ValueError("Frozen constituent training source changed")
        checkpoint = Path(item["checkpoint"])
        if sha256(checkpoint) != item["sha256"]:
            raise ValueError("Constituent checkpoint changed")
        for name in ("validation.pth", "holdout.pth"):
            cases = torch.load(folder / name, map_location="cpu", weights_only=True)["cases"]
            reference = torch.load(caches["validation" if name == "validation.pth" else "spectral"], map_location="cpu", weights_only=True)["cases"]
            if len(cases) != 72 or case_digest(cases) != case_digest(reference):
                raise ValueError("Case contents differ; metadata hashes may differ")
        configs[label] = config["model"]
        checkpoint_hashes[label] = item["sha256"]
        sizes.append(checkpoint.stat().st_size)
        for path in (checkpoint, work / "config.json", work / "best_validation_metrics.json", folder / "manifest.json",
                     folder / "validation.pth", folder / "holdout.pth", folder / "retention_evaluation/results.json"):
            inputs[str(path)] = sha256(path)
    borders = Path(importlib.resources.files("tabimpute") / "data/borders.pt")
    for path in (*caches.values(), borders, Path(spec["checkpoint_path"]), RETENTION / "cases.pth", RETENTION / "results.json", RETENTION / "manifest.json"):
        inputs[str(path)] = sha256(path)
    reuse_path = root.parent / members[1]["queue"] / "baseline_reuse_manifest.json"
    reuse = read_json(reuse_path)
    inputs[str(reuse_path)] = sha256(reuse_path)
    for dataset in transfer.DATASETS:
        if dataset == "mteb":
            expected_baseline = reuse["mteb_inputs"][str(transfer.baseline_cache(dataset))]
        else:
            matches = [artifact["sha256"] for path, artifact in reuse["artifacts"].items()
                       if Path(path).name == f"tabimpute_dwip_transfer_baseline_{dataset}.npz"]
            if len(matches) != 1:
                raise ValueError("Missing verified original baseline")
            expected_baseline = matches[0]
        if sha256(transfer.baseline_cache(dataset)) != expected_baseline:
            raise ValueError("Original benchmark baseline changed")
        for path in (BENCH / "data" / f"{dataset}.matrix.npz", transfer.baseline_cache(dataset),
                     *(transfer.raw_cache(item["queue"], dataset) for item in members)):
            inputs[str(path)] = sha256(path)
    inputs[str(BENCH / "code/eval_tabimpute.py")] = sha256(BENCH / "code/eval_tabimpute.py")
    registration = {"members": members, "queues": queues, "input_sha256": inputs,
        "legacy_labels": "seed1 and seed2 are compatibility artifact names for fixed ranks1 and2. Both models use seed1201; these are NOT independent seed replicas.",
        "selection_rule": "Deterministic two lowest completed raw fixed72-case macro RMSE models. One equal parameter average, FP64 accumulation into FP32. No pair/weight/seed grid and no benchmark-driven member choices.",
        "acceptance": "Fresh raw fixed scores must reproduce; average must strictly beat both fresh raw scores and the registered fixed winner. Otherwise stop without benchmark evaluation.",
        "basis": "Same original initialization and architecture; fixed-only equal prediction mean0.9621256601336974 improves both raw models and8/12 selector groups. Test whether a SINGLE parameter-averaged model retains any benefit without doubled inference cost.",
        "risks": "Prediction complementarity does not imply parameter-average benefit. Three older saved averages failed their gates. Fixed validation has been reused repeatedly; it is not independent evidence. Constituent benchmark outcomes are known, not used in input ranking; no blinded claim.",
        "followups": "If accepted: independent spectral/general, all4 benchmark datasets with10 folds/3 methods/budgets1-15, frozen256-case retention versus original/both members, then matched public100/500/1000 runtime. All report-only; never undo the fixed gate based on benchmarks.",
        "limits": "One candidate, no training or optimizer changes, outer24h timeout, preserve sources/checkpoints/caches,20GiB reserve, no repeat or alternate weighting if rejected."}
    immutable_json(root / "prespecification.json", registration)
    immutable_json(root / "artifact_budget.json", average_budget(sizes))
    manifest = {"queues": queues, "prespecification_sha256": sha256(root / "prespecification.json"),
                "checkpoint_sha256": checkpoint_hashes, "source_sha256": {str(REPO / path): sha256(REPO / path) for path in SOURCES},
                "caches": {key: str(path) for key, path in caches.items()}, "borders": str(borders),
                "model_config": configs["seed1"], "legacy_labels": registration["legacy_labels"],
                "initial_checkpoint": spec["checkpoint_path"]}
    immutable_json(root / "manifest.json", manifest)
    immutable_json(root / "registration_integrity.json", {name: sha256(root / name) for name in
                   ("prespecification.json", "manifest.json", "artifact_budget.json")})
    return registration, manifest


def synthetic_worker(root):
    registration = verify(root)
    manifest = read_json(root / "manifest.json")
    work = {label: WORK / f"tabimpute-{queue}-dwip1e5" for label, queue in manifest["queues"].items()}
    checkpoints = {label: path / "best_validation_checkpoint.pth" for label, path in work.items()}
    checkpoints["average"] = root / "candidate/checkpoint.pth"
    guard(root)
    fixed.evaluate(root, work, checkpoints, manifest["model_config"], Path(manifest["borders"]),
                   {label: Path(path) for label, path in manifest["caches"].items()})
    strict_gate(read_json(root / "selection.json"), registration["members"])
    verify(root)
    guard(root)


def retention_worker(root):
    registration = verify(root)
    output = root / "retention_evaluation"
    output.mkdir(exist_ok=True)
    if gpu_processes() or not torch.backends.cuda.cudnn_sdp_enabled():
        raise RuntimeError("Require idle GPU and default public SDPA")
    manifest = read_json(root / "manifest.json")
    checkpoint = root / "candidate/checkpoint.pth"
    if sha256(checkpoint) != read_json(root / "candidate_sha256.json")["sha256"]:
        raise ValueError("Average checkpoint changed")
    cases = torch.load(RETENTION / "cases.pth", map_location="cpu", weights_only=True)
    imputer = ImputePFN(device="cuda", checkpoint_path=str(checkpoint), entry_wise_features=False,
                       json_config={"config": manifest["model_config"]}, model_family="new_stable",
                       precision_mode="amp_bfloat16", prediction_stat="mean", max_num_rows=None, max_num_cols=None)
    if any(parameter.dtype != torch.float32 for parameter in imputer.model.parameters()):
        raise ValueError("Expected FP32 model parameters")
    records, predictions = [], {}
    started = time.monotonic()
    for case in cases:
        guard(root)
        prediction = imputer.impute(case["source"].numpy().copy())
        records.append(retention.score_prediction(case, prediction))
        predictions[case["id"]] = torch.from_numpy(prediction[case["mask"].numpy()].copy())
    torch.cuda.synchronize()
    references = {"original": read_json(RETENTION / "results.json")["original"]["records"],
                  **{label: read_json(root.parent / queue / "retention_evaluation/results.json")["records"] for label, queue in registration["queues"].items()}}
    paired = {label: compare_records(reference, records) for label, reference in references.items()}
    if len(records) != 256 or any(len(groups) != 8 for groups in paired.values()):
        raise ValueError("Incomplete retention population")
    torch.save(predictions, output / "predictions.pth")
    immutable_json(output / "results.json", {"records": records, "groups": retention.summarize(records),
                   "comparisons": paired, "seconds": time.monotonic() - started,
                   "scope": "Frozen independent-sample retention, not OOD families; no selection from retention."})
    verify(root)
    immutable_json(output / "status.json", {"state": "complete", "checkpoint_sha256": sha256(checkpoint)})


def finish(root):
    manifest = read_json(root / "manifest.json")
    results = read_json(root / "transfer_evaluation/results.json")
    reports, rows = {}, []
    for dataset in transfer.DATASETS:
        selected = transfer.load_cache(root / "transfer_evaluation/logs" / f"tabimpute_dwip_cross_seed_average_{dataset}.npz")
        reports[dataset] = {}
        references = {"original": transfer.baseline_cache(dataset), **{label: transfer.raw_cache(queue, dataset) for label, queue in manifest["queues"].items()}}
        for label, path in references.items():
            panel, settings = full_panel(transfer.load_cache(path), selected)
            if panel["standard_budgets"] != results[dataset]["baseline" if label == "original" else label]:
                raise ValueError("Full panel does not reproduce saved aggregates")
            reports[dataset][label] = panel
            rows.extend({"dataset": dataset, "reference": label, **item} for item in settings)
    totals = {label: sum(datasets[label]["mean_setting_counts"]["improved"] for datasets in reports.values()) for label in ("original", "seed1", "seed2")}
    output = {"utc": datetime.now(timezone.utc).isoformat(), "reports": reports,
              "improved_mean_r2_settings_out_of_180": totals, "selection": read_json(root / "selection.json"),
              "retention": read_json(root / "retention_evaluation/results.json")["comparisons"],
              "runtime": read_json(root / "runtime_evaluation/results.json")["comparisons"],
              "legacy_labels": manifest["legacy_labels"], "checkpoint_sha256": read_json(root / "candidate_sha256.json")["sha256"],
              "scope": "Full descriptive panels after fixed-only acceptance. No goal-completion claim or benchmark-based change to fixed selection."}
    immutable_json(root / "full_comparison.json", output)
    with (root / "full_benchmark_r2.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--worker", choices=("synthetic", "retention"))
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.experiment in {"", ".", ".."}:
        raise ValueError("Require one experiment directory")
    root = REPO / "runs/objective_queues" / args.experiment
    root.mkdir(exist_ok=True)
    if args.worker:
        (synthetic_worker if args.worker == "synthetic" else retention_worker)(root)
        return
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "pipeline_status.json").exists():
            raise ValueError("Pipeline was already started; inspect, never restart automatically")
        registration, manifest = register(root)
        admission = guard(root, launch=True)
        if args.register_only:
            print(json.dumps({"state": "registered", "required_free_gib": admission["required_free_bytes"] / 1024**3,
                              "free_gib": admission["free_bytes"] / 1024**3, "members": manifest["queues"]}, indent=2))
            return
        if gpu_processes():
            raise RuntimeError("GPU is busy; preserve other jobs")

        def status(state, **extra):
            write_json(root / "pipeline_status.json", {"state": state, "updated_unix": time.time(), **extra})

        try:
            status("materializing")
            candidate = root / "candidate"
            candidate.mkdir()
            states = [torch.load(item["checkpoint"], map_location="cpu", weights_only=True, mmap=True)["model"] for item in registration["members"]]
            averaged = average_states(states)
            atomic_torch_save({"model": averaged, "artifact_kind": "one_equal_parameter_average_of_fixed_top_two_raw_models",
                               "source_checkpoint_sha256": manifest["checkpoint_sha256"]}, candidate / "checkpoint.pth")
            immutable_json(candidate / "config.json", {"model_family": "new_stable", "model": manifest["model_config"],
                           "artifact_kind": "fixed_top_two_parameter_average"})
            immutable_json(root / "candidate_sha256.json", {"sha256": sha256(candidate / "checkpoint.pth")})
            del states, averaged
            guard(root)
            status("synthetic_evaluation")
            base_command = [sys.executable, "scripts/evaluate_dwip_top2_fixed_average.py", "--experiment", root.name]
            run_guarded([*base_command, "--worker", "synthetic"], root / "synthetic.log", lambda: guard(root))
            accepted = strict_gate(read_json(root / "selection.json"), registration["members"])
            if not accepted:
                status("complete_rejected_by_fixed_selector", candidate_promoted=False, benchmark_evaluation_started=False)
                print(json.dumps(read_json(root / "selection.json"), indent=2))
                return
            verify(root)
            status("transfer_evaluation")
            identity = psutil.Process()
            command = [sys.executable, "scripts/run_dwip_average_transfer.py", "--experiment", root.name,
                       "--dependency-pid", str(identity.pid), "--dependency-created", str(identity.create_time())]
            run_guarded(command, root / "transfer.log", lambda: guard(root))
            if read_json(root / "transfer_evaluation/status.json")["state"] != "complete":
                raise ValueError("Accepted average did not complete its full panel")
            status("retention_evaluation")
            run_guarded([*base_command, "--worker", "retention"], root / "retention.log", lambda: guard(root))
            status("runtime_evaluation")
            output = root / "runtime_evaluation"
            output.mkdir()
            checkpoints = {"original": manifest["initial_checkpoint"], "selected": str(candidate / "checkpoint.pth")}
            for variant, checkpoint in checkpoints.items():
                immutable_json(output / f"{variant}_models.json", {"models": [{"name": "dwip-v2-2", "checkpoint_path": checkpoint,
                               "model_config": manifest["model_config"], "model_family": "new_stable"}]})
            runtime_rows = []
            for index, size in enumerate(SIZES):
                for variant in VARIANTS if index % 2 == 0 else reversed(VARIANTS):
                    verify(root)
                    if gpu_processes():
                        raise RuntimeError("GPU became occupied; preserve external work")
                    command = ["timeout", "--signal=TERM", "--kill-after=120s", "15m", *worker_command(output, variant, size)]
                    run_guarded(command, output / f"{variant}_{size}.log", lambda: guard(root))
                    runtime_rows.append({**read_json(output / f"{variant}_{size}.json"), "variant": variant})
            immutable_json(output / "results.json", {"rows": runtime_rows, "comparisons": summarize(runtime_rows)})
            immutable_json(output / "status.json", {"state": "complete"})
            verify(root)
            finish(root)
            guard(root)
            status("complete", candidate_promoted_by_fixed_only=True, checkpoint_sha256=sha256(candidate / "checkpoint.pth"))
            print(json.dumps(read_json(root / "full_comparison.json")["improved_mean_r2_settings_out_of_180"], indent=2))
        except Exception as error:
            status("error", error=repr(error), no_automatic_restart=True)
            raise


if __name__ == "__main__":
    main()
