"""Evaluate one synthetic-selected checkpoint on already-frozen retention cases."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

import psutil
import torch

import diagnose_dwip_latent_retention as diagnostic
from run_dwip_transfer_evaluation import gpu_processes, sha256, write_json
from run_rowcol_controlled_experiments import REPO, WORK
from tabimpute.interface import ImputePFN


def read_json(path):
    return json.loads(path.read_text())


def dependency_ready(root, handle):
    status = read_json(root / "transfer_evaluation/status.json")
    if status["state"] == "complete":
        if not (root / "transfer_evaluation/results.json").exists() or read_json(root / "status.json")["state"] != "complete":
            raise RuntimeError("Transfer completion lacks its results or completed training queue")
        return True
    if status["state"] in {"error", "cancelled_by_user"}:
        raise RuntimeError(f"Transfer dependency failed: {status}")
    try:
        process = psutil.Process(handle["pid"])
        if process.create_time() != handle["created"] or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("Original transfer handle is no longer live; no restart attempted")
    except psutil.NoSuchProcess as error:
        raise RuntimeError("Original transfer handle is missing; no restart attempted") from error
    return False


def compare_records(reference, candidate):
    before = {record["id"]: record for record in reference}
    after = {record["id"]: record for record in candidate}
    if not before or before.keys() != after.keys() or len(before) != len(reference) or len(after) != len(candidate):
        raise ValueError("Require identical nonempty unique case IDs")
    for name, record in before.items():
        other = after[name]
        if record["group"] != other["group"] or record["n"] != other["n"] or record["n"] <= 0:
            raise ValueError("Paired groups or scored populations differ")
        if any(not math.isfinite(value[key]) or value[key] < 0
               for value in (record, other) for key in ("sse", "raw_sse")):
            raise ValueError("Invalid squared error")
    baseline, selected = diagnostic.summarize(reference), diagnostic.summarize(candidate)
    return {name: {"baseline_macro_case_rmse": group["macro_case_rmse"],
                   "selected_macro_case_rmse": selected[name]["macro_case_rmse"],
                   "macro_case_rmse_delta": selected[name]["macro_case_rmse"] - group["macro_case_rmse"],
                   "macro_case_rmse_ratio": selected[name]["macro_case_rmse"] / group["macro_case_rmse"] if group["macro_case_rmse"] else None,
                   "baseline_pooled_rmse": group["pooled_rmse"], "selected_pooled_rmse": selected[name]["pooled_rmse"],
                   "cases_improved": sum(record["sse"] < before[record["id"]]["sse"] for record in candidate if record["group"] == name),
                   "case_count": group["case_count"]} for name, group in baseline.items()}


def verify_hashes(hashes):
    if any(sha256(path) != expected for path, expected in hashes.items()):
        raise ValueError("Frozen evaluation input changed")


def immutable_json(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError("Registered evaluation changed; do not reuse artifacts")
    write_json(path, value)


def verify_selected_checkpoint(root, checkpoint):
    recorded = read_json(root / "transfer_evaluation/checkpoint_sha256.json")
    if sha256(checkpoint) != recorded["selected"]:
        raise ValueError("Selected weights differ from the completed transfer panel")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="dwip_spectral_anchor_mix25_48k_20260916")
    parser.add_argument("--diagnostic", default="dwip_latent_retention_20260916")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue
    baseline_root = REPO / "runs/information_flow_diagnostics" / args.diagnostic
    output = root / "retention_evaluation"
    output.mkdir(exist_ok=True)
    lock = (output / "evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "results.json").exists():
        raise ValueError("Results already exist; inspect rather than overwrite")
    launch = read_json(root / "launch.json")
    handle = {"pid": launch["transfer_launcher_pid"], "created": launch["transfer_launcher_created"]}

    def status(state, **extra):
        write_json(output / "status.json", {"state": state, "updated_unix": time.time(), **extra})

    try:
        if read_json(baseline_root / "status.json")["state"] != "complete":
            raise RuntimeError("Original retention diagnostic must be complete")
        baseline_manifest = read_json(baseline_root / "manifest.json")
        case_path = baseline_root / "cases.pth"
        if sha256(case_path) != read_json(baseline_root / "cases_sha256.json")["sha256"]:
            raise ValueError("Frozen case cache integrity mismatch")
        verify_hashes(baseline_manifest["source_sha256"])
        paths = [Path(__file__), baseline_root / "manifest.json", baseline_root / "cases_sha256.json",
                 baseline_root / "results.json", case_path, root / "manifest.json", root / "launch.json"]
        hashes = {**baseline_manifest["source_sha256"], **{str(path): sha256(path) for path in paths}}
        registration = {"queue": args.queue, "diagnostic": args.diagnostic, "dependency": handle,
            "input_sha256": hashes,
            "selection": "Only the training queue's fixed-synthetic-selected checkpoint; this diagnostic is strictly report-only, with no checkpoint or mixture-weight choices.",
            "scope": "Exact frozen diagnostic cases and existing reference scores. The new latent/shape training support overlaps these families, so this is independent-sample retention, not OOD-family evaluation."}
        immutable_json(output / "registration.json", registration)
        while not dependency_ready(root, handle):
            status("waiting_for_transfer", verified_live_launcher=handle["pid"])
            time.sleep(120)
        work = WORK / f"tabimpute-{args.queue}-dwip1e5"
        checkpoint = work / "best_validation_checkpoint.pth"
        verify_selected_checkpoint(root, checkpoint)
        config = read_json(work / "config.json")
        if config["model_family"] != "new_stable" or config["model"] != baseline_manifest["model_config"]:
            raise ValueError("Selected checkpoint architecture differs from original DWIP")
        for path in (checkpoint, work / "config.json", work / "best_validation_metrics.json",
                     root / "transfer_evaluation/checkpoint_sha256.json"):
            hashes[str(path)] = sha256(path)
        immutable_json(output / "manifest.json", {"input_sha256": hashes,
            "checkpoint": str(checkpoint), "selection_metrics": read_json(work / "best_validation_metrics.json")})
        while active := gpu_processes():
            status("waiting_for_idle_gpu", observed_gpu_pids=active)
            time.sleep(60)
        verify_hashes(hashes)
        torch.set_num_threads(4)
        if not torch.backends.cuda.cudnn_sdp_enabled():
            raise RuntimeError("Expected default public-interface SDPA backend settings")
        cases = torch.load(case_path, map_location="cpu", weights_only=True)
        references = read_json(baseline_root / "results.json")
        started = time.monotonic()
        imputer = ImputePFN(device="cuda", checkpoint_path=str(checkpoint), entry_wise_features=False,
                           json_config={"config": config["model"]}, model_family="new_stable",
                           precision_mode="amp_bfloat16", prediction_stat="mean", max_num_rows=None, max_num_cols=None)
        if any(parameter.dtype != torch.float32 for parameter in imputer.model.parameters()):
            raise ValueError("Expected FP32 model weights")
        torch.cuda.reset_peak_memory_stats()
        records, predictions = [], {}
        for case in cases:
            status("evaluating", case=case["id"])
            prediction = imputer.impute(case["source"].numpy().copy())
            records.append(diagnostic.score_prediction(case, prediction))
            predictions[case["id"]] = torch.from_numpy(prediction[case["mask"].numpy()].copy())
        torch.cuda.synchronize()
        result = {"records": records, "groups": diagnostic.summarize(records),
                  "comparisons": {name: compare_records(reference["records"], records) for name, reference in references.items()},
                  "seconds": time.monotonic() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "scope": registration["scope"], "selection": registration["selection"]}
        verify_hashes(hashes)
        torch.save(predictions, output / "predictions.pth")
        write_json(output / "results.json", result)
        status("complete", case_count=len(cases), checkpoint_sha256=hashes[str(checkpoint)])
        print(json.dumps({"state": "complete", "case_count": len(cases), "seconds": result["seconds"]}), flush=True)
    except Exception as error:
        status("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
