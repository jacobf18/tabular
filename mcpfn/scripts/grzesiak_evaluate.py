"""Fresh full-matrix evaluations; imported reference scores are never pooled here."""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from evaluate_dwip_missbench_uci import CHECKPOINT, score_predictions
from grzesiak_benchmark import COMMIT, REPO, RSCRIPT, WORK, sha256, write_json


CHECKPOINT_SHA256 = "1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d"
CORE_METHODS = ("column_mean", "mice_pmm", "mice_cart", "mice_rf", "mice_drf")
EXTRA_METHODS = ("mice_norm", "mice_norm_predict", "mice_norm_nob", "mice_midastouch", "mice_boot",
                 "missforest", "missmda_pca_em", "missmda_pca_reg", "missmda_MIPCA_em", "missmda_MIPCA_reg")


def stable_seed(case_id):
    return 56135 + int(hashlib.sha256(case_id.encode()).hexdigest()[:7], 16)


def distance_mean(first, second, block_size=256):
    total = 0.0
    for start in range(0, first.shape[0], block_size):
        total += cdist(first[start:start + block_size], second, metric="euclidean").sum(dtype=np.float64)
    return float(total / (first.shape[0] * second.shape[0]))


def standardized_energy(truth, prediction, self_distance=None):
    scale = truth.std(axis=0, ddof=1)
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("Undefined energy standardization for constant columns")
    center = truth.mean(axis=0)
    standardized_truth = (truth - center) / scale
    standardized_prediction = (prediction - center) / scale
    if self_distance is None:
        self_distance = distance_mean(standardized_truth, standardized_truth)
    distance = (2 * distance_mean(standardized_truth, standardized_prediction) - self_distance
                - distance_mean(standardized_prediction, standardized_prediction))
    if distance < -1e-9:
        raise ValueError("Negative energy distance beyond numerical tolerance")
    distance = max(0.0, distance)
    return {"energy_distance_std": distance, "energy_statistic_std": distance * truth.shape[0] / 2}


def metrics(truth, missing, prediction, suite, energy=False, self_distance=None):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid completed matrix")
    observed = ~np.isnan(missing)
    if not np.allclose(prediction[observed], missing[observed], rtol=1.5e-5, atol=1.5e-5):
        raise ValueError("Observed values modified")
    mask = ~observed
    result = score_predictions(truth, prediction, missing, suite)
    result = {"nrmse": result["nrmse_original"], "rmse": result["rmse"],
              "nrmse_skipped_columns": result["nrmse_skipped_constant_columns"]}
    scale = np.std(truth[mask], ddof=1)
    result["nrmse_upstream_pooled"] = float(np.sqrt(np.mean((truth[mask] - prediction[mask]) ** 2)) / scale) if scale > 0 else None
    if energy:
        result.update(standardized_energy(truth, prediction, self_distance))
    return result


def load_matrix(path):
    return np.load(path, allow_pickle=False) if str(path).endswith(".npy") else pd.read_csv(path).to_numpy(dtype=np.float64)


def cases_for(work, scope, datasets=None, limit=None, include_mixed=False):
    if scope == "fresh":
        frame = pd.read_csv(work / "fresh/cases.csv")
        if not include_mixed:
            frame = frame[frame.numeric_only]
    else:
        frame = pd.read_csv(Path("/root/MatrixFormer/results/imputation_cases.csv")).fillna("")
        if scope != "broad":
            frame = frame[frame.suite.eq(scope)]
        frame["truth_path"] = frame.source_directory + "/true.npy"
        frame["missing_path"] = frame.source_directory + "/missing.npy"
        frame["numeric_only"] = True
    if datasets:
        frame = frame[frame.dataset.isin(datasets)]
    if limit:
        frame = frame.head(limit)
    return frame.to_dict("records")


def result_path(work, case, method):
    return work / "evaluations" / case["suite"] / case["case_id"] / f"{method}.json"


def prediction_path(work, case, method):
    return result_path(work, case, method).with_suffix(".npz")


def base_record(case, method):
    for source_key, expected_key in (("truth_path", "true_sha256"), ("missing_path", "missing_sha256")):
        if case.get(expected_key) and sha256(case[source_key]) != case[expected_key]:
            raise ValueError(f"Frozen benchmark input changed: {case[source_key]}")
    return {key: case[key] for key in ("case_id", "suite", "dataset", "pattern", "missingness")} | {
        "method": method, "seed": stable_seed(case["case_id"]), "upstream_commit": COMMIT,
        "truth_sha256": sha256(case["truth_path"]), "missing_sha256": sha256(case["missing_path"]),
        "numeric_only": case["numeric_only"], "runner_sha256": sha256(Path(__file__)),
        "bridge_sha256": sha256(Path(__file__).with_name("grzesiak_bridge.R"))}


def existing_valid(work, case, method, retry=False):
    path = result_path(work, case, method)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    for source_key, checksum_key in (("truth_path", "truth_sha256"), ("missing_path", "missing_sha256")):
        if record[checksum_key] != sha256(case[source_key]):
            raise ValueError(f"Resume input checksum mismatch: {path}")
    if record["status"] == "ok":
        predicted = prediction_path(work, case, method)
        if sha256(predicted) != record["prediction_sha256"]:
            raise ValueError(f"Resume prediction checksum mismatch: {predicted}")
        return True
    return not retry


def store_prediction(work, case, method, prediction, missing, record):
    if shutil.disk_usage(work).free < 1024 ** 3:
        raise RuntimeError("Less than 1 GiB disk reserve; no existing artifacts removed")
    path = prediction_path(work, case, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, missing_values=prediction[np.isnan(missing)], shape=np.array(prediction.shape))
    temporary.replace(path)
    record["prediction_sha256"] = sha256(path)
    record["prediction_path"] = str(path.relative_to(work))


def evaluate_r(case, method, work, rscript, timeout, energy, retry):
    if existing_valid(work, case, method, retry):
        return json.loads(result_path(work, case, method).read_text())
    record = base_record(case, method)
    record["timeout_seconds"] = timeout
    started = time.monotonic()
    path = result_path(work, case, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        truth, missing = load_matrix(case["truth_path"]), load_matrix(case["missing_path"])
        with tempfile.TemporaryDirectory(prefix="grzesiak-", dir="/dev/shm") as temporary:
            temporary = Path(temporary)
            rds = Path(case["source_directory"]) / "missing.RDS"
            if rds.exists():
                input_path = rds
            else:
                input_path = temporary / "missing.csv"
                pd.DataFrame(missing, columns=[f"V{index}" for index in range(missing.shape[1])]).to_csv(input_path, index=False, na_rep="NA")
            request = {"input": str(input_path), "output": str(temporary / "prediction.csv"),
                       "result": str(temporary / "result.json"), "upstream": str(work / "upstream"),
                       "method": method, "seed": record["seed"]}
            write_json(temporary / "request.json", request)
            with path.with_suffix(".log").open("w") as log:
                process = subprocess.Popen([str(rscript), "--vanilla", str(Path(__file__).with_name("grzesiak_bridge.R")),
                                            str(temporary / "request.json")], stdout=log, stderr=subprocess.STDOUT,
                                           env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                                                             "MKL_NUM_THREADS": "1", "RCPP_PARALLEL_NUM_THREADS": "1"}, start_new_session=True)
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise TimeoutError(f"Method exceeded {timeout} seconds")
            if returncode != 0:
                raise RuntimeError(f"R subprocess exit {returncode}; see {path.with_suffix('.log')}")
            outcome = json.loads((temporary / "result.json").read_text())
            if outcome["status"] != "ok":
                raise RuntimeError(outcome["error"])
            prediction = load_matrix(temporary / "prediction.csv")
            record["imputation_seconds"] = outcome["seconds"]
        record.update(metrics(truth, missing, prediction, case["suite"], energy=energy))
        store_prediction(work, case, method, prediction, missing, record)
        record["status"] = "ok"
    except Exception as error:
        record.update(status="timeout" if isinstance(error, TimeoutError) else "error", error=str(error))
    record["total_seconds"] = time.monotonic() - started
    write_json(path, record)
    return record


def evaluate_model(cases, work, energy, retry):
    import torch
    from tabimpute.interface import ImputePFN

    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint checksum mismatch")
    config = json.loads((CHECKPOINT.parent / "config.json").read_text())
    torch.manual_seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.mha.set_fastpath_enabled(False)
    model = ImputePFN(device="cuda", checkpoint_path=str(CHECKPOINT), entry_wise_features=False,
                      json_config={"config": config.get("model", config.get("config"))},
                      model_family="new_stable", precision_mode="amp_bfloat16", prediction_stat="mean",
                      max_num_rows=None, max_num_cols=None)
    for index, case in enumerate(cases):
        if existing_valid(work, case, "matrixformer", retry):
            continue
        record = base_record(case, "matrixformer")
        record.update(checkpoint_sha256=CHECKPOINT_SHA256, checkpoint_config_sha256=sha256(CHECKPOINT.parent / "config.json"))
        started = time.monotonic()
        try:
            if not case["numeric_only"]:
                raise ValueError("Current continuous predictive-mean model has no categorical output adapter")
            truth, missing = load_matrix(case["truth_path"]), load_matrix(case["missing_path"])
            center, scale = np.nanmean(missing, axis=0), np.nanstd(missing, axis=0)
            scale = np.where(scale == 0, 1, scale)
            prediction = model.impute(((missing - center) / scale).copy()) * scale + center
            record["imputation_seconds"] = time.monotonic() - started
            record.update(metrics(truth, missing, prediction, case["suite"], energy=energy))
            store_prediction(work, case, "matrixformer", prediction, missing, record)
            record["status"] = "ok"
        except Exception as error:
            record.update(status="error", error=str(error))
        record["total_seconds"] = time.monotonic() - started
        write_json(result_path(work, case, "matrixformer"), record)
        print(f"model {index + 1}/{len(cases)} {case['case_id']} {record['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--rscript", type=Path, default=RSCRIPT)
    parser.add_argument("--scope", choices=("fresh", "broad", "missbench", "uci"), default="fresh")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--methods", nargs="+", default=list(CORE_METHODS))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=36000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--include-mixed", action="store_true")
    parser.add_argument("--retry", action="store_true")
    arguments = parser.parse_args()
    cases = cases_for(arguments.work, arguments.scope, arguments.datasets, arguments.limit, arguments.include_mixed)
    if not cases:
        raise ValueError("No selected cases")
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    manifest = {"scope": arguments.scope, "methods": arguments.methods, "cases": [case["case_id"] for case in cases],
                "timeout": arguments.timeout, "workers": arguments.workers, "energy": arguments.energy,
                "row_split": False, "runner_sha256": sha256(Path(__file__)),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "torch")}}
    write_json(arguments.work / "runs" / f"{run_id}.json", manifest)
    r_methods = [method for method in arguments.methods if method != "matrixformer"]
    jobs = [(case, method) for case in cases for method in r_methods]
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = [executor.submit(evaluate_r, case, method, arguments.work, arguments.rscript,
                                   arguments.timeout, arguments.energy, arguments.retry) for case, method in jobs]
        if "matrixformer" in arguments.methods:
            evaluate_model(cases, arguments.work, arguments.energy, arguments.retry)
        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            print(f"R {index + 1}/{len(jobs)} {result['case_id']} {result['method']} {result['status']}", flush=True)
    manifest["state"] = "finished_all_requested_attempts"
    write_json(arguments.work / "runs" / f"{run_id}.json", manifest)


if __name__ == "__main__":
    main()
