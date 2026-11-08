"""Fixed, resumable evaluation of all 13 TabArena regression-Lite tasks."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import torch

import evaluate_cylinder_completion as shared
from evaluate_tabarena_smoke import scores
from matrixformer_supervised import SupervisedMatrixFormer


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("MatrixFormer", "RandomForest", "Dummy")


def regression_tasks(root):
    tasks = []
    for path in (root / "data/tasks").glob("*/metadata.json"):
        record = json.loads(path.read_text())
        if record["problem_type"] == "regression":
            tasks.append((path.parent, record))
    widths = pd.read_csv(root / "preprocessing_preflight/inventory.csv").set_index("task_id")
    return sorted(tasks, key=lambda item: (
        int(widths.loc[item[1]["task_id"], "test_rows"]) *
        min(1024, int(widths.loc[item[1]["task_id"], "train_rows"]) + 1) *
        (int(widths.loc[item[1]["task_id"], "encoded_features"]) + 1), item[1]["task_id"]))


def chunk_ranges(count, batch_size, batches_per_chunk=16):
    if count < 1 or batch_size < 1 or batches_per_chunk < 1:
        raise ValueError("Positive test size and batch sizes required")
    size = batch_size * batches_per_chunk
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def checked_prediction(values, count):
    values = np.asarray(values)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("Incomplete or nonfinite regression predictions")
    return values


def predict_in_chunks(predictor, features, case):
    chunks = case / "chunks"
    chunks.mkdir(exist_ok=True)
    predictions, records = [], []
    for start, stop in chunk_ranges(len(features), predictor.batch_size_):
        path = chunks / f"{start:07d}_{stop:07d}.npz"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["sha256"] != shared.sha256(path):
                raise ValueError(f"Corrupt prediction chunk: {path}")
            if (record["start"], record["stop"], record["batch_size"]) != (start, stop, predictor.batch_size_):
                raise ValueError("Prediction chunk layout changed")
            with np.load(path, allow_pickle=False) as saved:
                prediction = checked_prediction(saved["prediction"], stop - start)
        else:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = checked_prediction(predictor.predict(features.iloc[start:stop]), stop - start)
            torch.cuda.synchronize()
            record = {"start": start, "stop": stop, "batch_size": predictor.batch_size_,
                      "seconds": time.perf_counter() - started,
                      "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, prediction=prediction)
            temporary.replace(path)
            record["sha256"] = shared.sha256(path)
            shared.write_json(receipt, record)
        predictions.append(prediction)
        records.append(record)
        shared.write_json(case / "progress.json", {"predicted_rows": stop, "test_rows": len(features),
                          "prediction_seconds": sum(item["seconds"] for item in records)})
        print(f"  {case.name}: {stop}/{len(features)} query rows ({record['seconds']:.2f}s/chunk)", flush=True)
    return np.concatenate(predictions), records


def summary_rows(records):
    rows = []
    for record in records:
        dummy_rmse = record["metrics"]["Dummy"]["rmse"]
        if dummy_rmse <= 0:
            raise ValueError("Cannot aggregate ratios against a zero-error dummy")
        for method in METHODS:
            rows.append({"dataset": record["dataset"], "task_id": record["task_id"], "method": method,
                         **record["metrics"][method],
                         "rmse_over_dummy": record["metrics"][method]["rmse"] / dummy_rmse,
                         "train_rows": record["train_rows"], "test_rows": record["test_rows"],
                         "context_rows": record["context_rows"],
                         "encoded_features": record["preprocessing"]["encoded_features"],
                         "fit_seconds": record["timings"][method]["fit_seconds"],
                         "predict_seconds": record["timings"][method]["predict_seconds"]})
    return pd.DataFrame(rows)


def summarize(output, total, failed):
    records = [json.loads(path.read_text()) for path in sorted(output.glob("*/results.json"))]
    frame = summary_rows(records)
    lines = ["# MatrixFormer: TabArena regression-Lite", "",
             f"Completed {len(records)}/{total} datasets; {failed} failures.", "",
             "Official repeat 0 / fold 0; every test row. Frozen single checkpoint; no HPO or ensembling.",
             "Train-only preprocessing; at most 1023 labeled context rows and one independent query.",
             "RandomForest (200 trees) and training-mean Dummy use all training rows.",
             "This standalone comparison is not the official bagged/HPO TabArena leaderboard.", "",
             "## RMSE in original target units (lower is better)", "",
             "| Dataset | Train / test | Context | Encoded features | MatrixFormer | RandomForest | Mean dummy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        values = [record["metrics"][method]["rmse"] for method in METHODS]
        cells = [f"**{value:.6g}**" if value == min(values) else f"{value:.6g}" for value in values]
        lines.append(f"| {record['dataset']} | {record['train_rows']} / {record['test_rows']} | "
                     f"{record['context_rows']} | {record['preprocessing']['encoded_features']} | " + " | ".join(cells) + " |")
    if len(frame):
        frame.to_csv(output / "summary.csv", index=False)
        aggregate = frame.groupby("method", sort=False).agg(
            datasets=("dataset", "count"), mean_rmse_over_dummy=("rmse_over_dummy", "mean"),
            median_rmse_over_dummy=("rmse_over_dummy", "median"))
        aggregate.to_csv(output / "aggregate.csv")
        lines += ["", "## Unitless cross-dataset summary", "",
                  "Each dataset has equal weight; divide each RMSE by that dataset's mean-dummy RMSE.",
                  "Do not average raw RMSE across different target units.", "",
                  "| Method | Datasets | Mean RMSE / dummy RMSE | Median RMSE / dummy RMSE |",
                  "|---|---:|---:|---:|"]
        for method, row in aggregate.iterrows():
            lines.append(f"| {method} | {int(row['datasets'])} | {row['mean_rmse_over_dummy']:.6f} | {row['median_rmse_over_dummy']:.6f} |")
        comparison = frame.pivot(index="dataset", columns="method", values="rmse")
        for baseline in ["RandomForest", "Dummy"]:
            wins = int((comparison["MatrixFormer"] < comparison[baseline]).sum())
            ties = int((comparison["MatrixFormer"] == comparison[baseline]).sum())
            lines.append(f"\nMatrixFormer vs {baseline}: {wins} wins, {ties} ties, {len(comparison) - wins - ties} losses.")
    lines += ["", "No uncertainty intervals from this single split/seed. Any partial-suite aggregate is provisional.",
              "Predictions, split/context indices, preprocessing, timings, RMSE/MAE/R² and hashes are saved per task.",
              "Dataset overlap with earlier checkpoint-selection benchmarks is not yet audited; no clean holdout claim."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return len(records)


def evaluate_task(folder, record, output, model, distribution):
    for filename, digest in record["files_sha256"].items():
        if shared.sha256(folder / filename) != digest:
            raise ValueError(f"Data hash mismatch: {folder / filename}")
    case = output / str(record["task_id"])
    case.mkdir(exist_ok=True)
    features = pd.read_parquet(folder / "features.parquet")
    with np.load(folder / "splits.npz") as splits:
        train, test = splits["repeat0_fold0_train"], splits["repeat0_fold0_test"]
    train_features, test_features = features.iloc[train], features.iloc[test]
    train_target = pd.read_parquet(folder / "target.parquet").iloc[train, 0]
    started = time.perf_counter()
    predictor = SupervisedMatrixFormer(model, distribution, "regression")
    predictor.fit(train_features, train_target, record["categorical_columns"])
    fit_seconds = time.perf_counter() - started
    print(f"Predicting {record['dataset_name']}: train={len(train)} test={len(test)} "
          f"context={predictor.context_.shape} batch={predictor.batch_size_}", flush=True)
    prediction, chunks = predict_in_chunks(predictor, test_features, case)
    predictions = {"MatrixFormer": prediction}
    timings = {"MatrixFormer": {"fit_seconds": fit_seconds,
               "predict_seconds": sum(chunk["seconds"] for chunk in chunks),
               "peak_cuda_allocated_gib": max(chunk["peak_cuda_allocated_gib"] for chunk in chunks)}}
    started = time.perf_counter()
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    baseline_train = imputer.fit_transform(predictor.preprocessor_.transform(train_features))
    baseline_test = imputer.transform(predictor.preprocessor_.transform(test_features))
    baseline_preprocessing_seconds = time.perf_counter() - started
    for method, baseline in [("Dummy", DummyRegressor()),
                             ("RandomForest", RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0))]:
        started = time.perf_counter()
        baseline.fit(baseline_train, np.asarray(train_target))
        fitted = time.perf_counter()
        predictions[method] = checked_prediction(baseline.predict(baseline_test), len(test))
        timings[method] = {"fit_seconds": fitted - started, "predict_seconds": time.perf_counter() - fitted}
    np.savez_compressed(case / "predictions.npz", **predictions, train_indices=train, test_indices=test,
                        context_indices=train[predictor.context_indices_])
    joblib.dump(predictor.preprocessor_, case / "preprocessor.joblib")
    joblib.dump(imputer, case / "baseline_imputer.joblib")
    target = pd.read_parquet(folder / "target.parquet").iloc[test, 0]
    np.save(case / "scoring_targets.npy", np.asarray(target, dtype=np.float64))
    metrics = {method: scores(target, values, "regression") for method, values in predictions.items()}
    result = {"dataset": record["dataset_name"], "task_id": record["task_id"], "problem_type": "regression",
              "primary_metric": "rmse", "train_rows": len(train), "test_rows": len(test),
              "context_rows": len(predictor.context_indices_), "batch_size": predictor.batch_size_,
              "target_mean": predictor.target_mean_, "target_scale": predictor.target_scale_,
              "preprocessing": predictor.preprocessor_.summary(), "metrics": metrics, "timings": timings,
              "baseline_preprocessing_seconds": baseline_preprocessing_seconds,
              "timing_note": "Model-only baseline timing; shared encoding is included in MatrixFormer fit/predict, extra baseline preprocessing is separate. Not a runtime benchmark.",
              "data_files_sha256": record["files_sha256"],
              "artifacts_sha256": {name: shared.sha256(case / name) for name in
                                   ["predictions.npz", "preprocessor.joblib", "baseline_imputer.joblib", "scoring_targets.npy"]},
              "chunks": chunks}
    shared.write_json(case / "results.json", result)
    print({method: values["rmse"] for method, values in metrics.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/matrixformer_scratch/tabarena_20261028"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output or args.root / "regression_lite"
    tasks = regression_tasks(args.root)
    if len(tasks) != 13:
        raise ValueError(f"Expected all 13 regression tasks, found {len(tasks)}")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    sources = [Path(__file__).resolve(), ROOT / "scripts/matrixformer_supervised.py",
               ROOT / "scripts/evaluate_tabarena_smoke.py", ROOT / "SUPERVISED_TABARENA_REGRESSION.md",
               Path(shared.__file__).resolve(), shared.SINGLE, shared.SINGLE.with_name("config.json"),
               ROOT / "src/tabimpute/model/model_new_stable.py", ROOT / "src/tabimpute/model/bar_distribution.py",
               ROOT / "src/tabimpute/data/borders.pt", args.root / "data/manifest.json",
               args.root / "preprocessing_preflight/inventory.csv"]
    sources += [folder / "metadata.json" for folder, record in tasks]
    manifest = {"datasets": [record["dataset_name"] for folder, record in tasks],
                "task_ids": [record["task_id"] for folder, record in tasks],
                "repeat": 0, "fold": 0, "sample": 0, "context_rows": 1023, "seed": 0,
                "max_batch_cells": 131072, "max_batch_size": 8, "batches_per_chunk": 16,
                "transductive": False, "hpo": False, "bagging": False, "all_test_rows": True,
                "file_sha256": {str(path): shared.sha256(path) for path in sources},
                "packages": {name: metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "torch", "openml"]}}
    if manifest["file_sha256"][str(shared.SINGLE)] != shared.SINGLE_SHA256:
        raise ValueError("Unexpected checkpoint")
    manifest_path = output / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        previous.pop("created_utc")
        if previous != manifest:
            raise ValueError("Cannot resume: sources, data, settings or environment changed")
    else:
        manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
        shared.write_json(manifest_path, manifest)
    models, distribution = shared.load_models({"single": shared.SINGLE})
    failed = 0
    for folder, record in tasks:
        case = output / str(record["task_id"])
        if not (case / "results.json").exists():
            shared.write_json(output / "status.json", {"status": "running", "current": record["dataset_name"],
                              "completed": len(list(output.glob("*/results.json"))), "failed": failed, "total": len(tasks)})
            try:
                evaluate_task(folder, record, output, models["single"], distribution)
            except Exception as error:
                failed += 1
                failure = {"dataset": record["dataset_name"], "error": str(error), "traceback": traceback.format_exc()}
                shared.write_json(case / f"failure_{time.time_ns()}.json", failure)
                print(failure["traceback"], flush=True)
            finally:
                torch.cuda.empty_cache()
        completed = summarize(output, len(tasks), failed)
    shared.write_json(output / "status.json", {"status": "complete" if not failed else "complete_with_failures",
                      "completed": completed, "failed": failed, "total": len(tasks)})


if __name__ == "__main__":
    main()
