"""Evaluate the exact historical model checkpoints on the approved frozen imputation cases."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from evaluate_dwip_missbench_uci import REPO, score_predictions, sha256, standardize_uci, write_json


ROOT = REPO / "runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_missbench_uci_20260918"
ANCHORS = REPO / "benchmark/movielens/configs/tabimpute_models.json"
V2_CHECKPOINT = Path("/root/.cache/huggingface/hub/models--Tabimpute--TabImpute/snapshots/96d7a0f160184cfb361a03559430baf1cf31aec5/tabimputev2.ckpt")


def model_spec(name):
    if name == "tabimpute_v2":
        return {"name": name, "label": "TabImpute v2 (paper)", "checkpoint": str(V2_CHECKPOINT),
                "prediction_stat": "median", "precision": "full_bfloat16", "model_family": "tabimpute_v2"}
    anchor = next(model for model in json.loads(ANCHORS.read_text())["models"] if model["name"] == "dwip-v2-2")
    return {"name": name, "label": "DWIP v2-2 (original)", "checkpoint": anchor["checkpoint_path"],
            "config": anchor["model_config"], "prediction_stat": "mean", "precision": "amp_bfloat16",
            "model_family": anchor["model_family"], "anchor_manifest_sha256": sha256(ANCHORS)}


def build_model(spec):
    if spec["name"] == "tabimpute_v2":
        from tabimpute.tabimpute_v2 import TabImputeV2
        return TabImputeV2(device="cuda", checkpoint_path=spec["checkpoint"], max_num_rows=None)
    from tabimpute.interface import ImputePFN
    return ImputePFN(device="cuda", checkpoint_path=spec["checkpoint"], entry_wise_features=False,
                     json_config={"config": spec["config"]}, model_family=spec["model_family"],
                     prediction_stat=spec["prediction_stat"], precision_mode=spec["precision"],
                     max_num_rows=None, max_num_cols=None)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args):
    import torch

    output = args.output or args.root / "baselines" / args.model
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_manifest = json.loads((args.root / "manifest.json").read_text())
    cases = source_manifest["protocol"]["cases"]
    if args.case_id:
        cases = [case for case in cases if case["case_id"] in args.case_id]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No selected cases")
    spec = model_spec(args.model)
    spec["checkpoint_sha256"] = sha256(Path(spec["checkpoint"]))
    protocol = {"model": spec, "cases": cases, "source_manifest_sha256": sha256(args.root / "manifest.json"),
                "script_sha256": sha256(Path(__file__)), "seed": 42, "cudnn_sdpa": False,
                "mha_fastpath": False, "max_num_rows": None, "max_num_cols": None,
                "preprocessors": None, "normalization": "Same as newest-model run, including zero-std UCI guard",
                "source_code_sha256": {str(path): sha256(path) for path in (
                    REPO / "scripts/evaluate_dwip_missbench_uci.py", REPO / "src/tabimpute/interface.py",
                    REPO / "src/tabimpute/tabimpute_v2.py")}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["protocol"] != protocol:
            raise ValueError("Output protocol differs; refusing to mix results")
    else:
        write_json(manifest_path, {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                                  "torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name()})
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.manual_seed(42)
    np.random.seed(42)
    imputer = build_model(spec)
    started = time.time()
    rows = []
    for index, case in enumerate(cases):
        if shutil.disk_usage(output).free < 20 * 1024**3:
            raise RuntimeError("20 GiB disk reserve reached")
        result_path = output / "cases" / (case["case_id"] + ".json")
        prediction_path = output / "predictions" / (case["case_id"] + ".npz")
        source = Path(case["source_directory"])
        reference = json.loads((args.root / "cases" / (case["case_id"] + ".json")).read_text())
        hashes = {"missing_sha256": sha256(source / "missing.npy"), "true_sha256": sha256(source / "true.npy")}
        if reference["status"] != "ok" or any(reference[key] != value for key, value in hashes.items()):
            raise ValueError(f"Data do not match newest-model evaluation: {case['case_id']}")
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior["status"] == "ok":
                if any(prior[key] != value for key, value in hashes.items()) or sha256(prediction_path) != prior["prediction_sha256"]:
                    raise ValueError(f"Invalid cached result: {case['case_id']}")
                rows.append(prior)
                continue
        write_json(output / "status.json", {"state": "running", "pid": os.getpid(), "completed": len(rows),
                   "expected": len(cases), "current": case["case_id"], "elapsed_seconds": time.time() - started})
        row = {**case, **hashes, "model": args.model, "status": "ok"}
        try:
            missing = np.load(source / "missing.npy", allow_pickle=False)
            truth = np.load(source / "true.npy", allow_pickle=False)
            if case["suite"] == "uci":
                model_input, mean, scale, constant_columns = standardize_uci(missing)
                row["zero_std_columns"] = constant_columns
            else:
                model_input = missing.copy()
            torch.manual_seed(42)
            np.random.seed(42)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            case_started = time.perf_counter()
            prediction = imputer.impute(model_input.copy())
            torch.cuda.synchronize()
            row["inference_seconds"] = time.perf_counter() - case_started
            row["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if case["suite"] == "uci":
                prediction = prediction * scale + mean
            row.update(score_predictions(truth, prediction, missing, case["suite"]))
            row.update({"rows": int(truth.shape[0]), "columns": int(truth.shape[1])})
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = prediction_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, missing_values=prediction[np.isnan(missing)], shape=np.asarray(missing.shape))
            temporary.replace(prediction_path)
            row.update({"prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256(prediction_path)})
        except Exception as error:
            row.update({"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
            torch.cuda.empty_cache()
        write_json(result_path, row)
        rows.append(row)
        print(f"[{index + 1}/{len(cases)}] {args.model} {case['case_id']} {row['status']} "
              f"seconds={row.get('inference_seconds', 0):.3f} nrmse={row.get('nrmse_original')}", flush=True)
        if (index + 1) % 25 == 0:
            write_csv(output / "per_case_metrics.csv", rows)
    write_csv(output / "per_case_metrics.csv", rows)
    if sha256(Path(spec["checkpoint"])) != spec["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    failures = sum(row["status"] != "ok" for row in rows)
    write_json(output / "status.json", {"state": "complete" if failures == 0 else "complete_with_failures", "completed": len(rows),
               "expected": len(cases), "successful": len(rows) - failures, "failed": failures,
               "elapsed_seconds": time.time() - started, "checkpoint_sha256": spec["checkpoint_sha256"],
               "free_disk_gib": shutil.disk_usage(output).free / 1024**3})
    if failures:
        raise RuntimeError(f"{failures} baseline cases failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=("dwip_v2_2", "tabimpute_v2"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception as error:
        destination = arguments.output or arguments.root / "baselines" / arguments.model
        write_json(destination / "fatal_error.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
