"""Cache the 51 TabArena-v0.1 datasets and their official evaluation splits."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import openml
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def download_task(record, output):
    task_id = int(record["task_id"])
    folder = output / "tasks" / str(task_id)
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        cached = json.loads(metadata_path.read_text())
        if all(sha256(folder / name) == digest for name, digest in cached["files_sha256"].items()):
            return cached
        raise ValueError(f"Changed cached data: {folder}")
    started = time.perf_counter()
    task = openml.tasks.get_task(task_id, download_splits=True, download_data=True,
                                download_qualities=False, download_features_meta_data=True)
    dataset = task.get_dataset(download_data=True, download_qualities=False)
    features, target, categorical, names = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    if target.isna().any() or len(features) != int(record["num_instances"]):
        raise ValueError("Unexpected missing labels or row count")
    if task.dataset_id != int(record["dataset_id"]) or str(task.target_name) != record["target_feature"]:
        raise ValueError("Task identity does not match pinned metadata")
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)
    features.to_parquet(folder / "features.parquet", index=False)
    target.to_frame("target").to_parquet(folder / "target.parquet", index=False)
    repeats, folds = int(record["tabarena_num_repeats"]), int(record["num_folds"])
    available_repeats, available_folds, samples = task.get_split_dimensions()
    if repeats > available_repeats or folds != available_folds or samples != 1:
        raise ValueError("Unexpected split dimensions")
    splits = {}
    for repeat in range(repeats):
        test_counts = np.zeros(len(features), dtype=int)
        for fold in range(folds):
            train, test = task.get_train_test_split_indices(repeat=repeat, fold=fold, sample=0)
            np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(len(features)))
            test_counts[test] += 1
            splits[f"repeat{repeat}_fold{fold}_train"] = train
            splits[f"repeat{repeat}_fold{fold}_test"] = test
        np.testing.assert_array_equal(test_counts, np.ones(len(features)))
    np.savez_compressed(folder / "splits.npz", **splits)
    result = {"task_id": task_id, "dataset_id": task.dataset_id, "dataset_name": record["dataset_name"].strip(),
              "openml_name": dataset.name, "target": task.target_name, "problem_type": record["problem_type"],
              "rows": len(features), "features": features.shape[1], "categorical_columns": [name for name, flag in zip(names, categorical) if flag],
              "dtypes": features.dtypes.astype(str).to_dict(), "missing_feature_cells": int(features.isna().sum().sum()),
              "cardinality": {name: int(features[name].nunique(dropna=True)) for name, flag in zip(names, categorical) if flag},
              "repeat_count": repeats, "fold_count": folds, "split_count": repeats * folds,
              "license_from_curated_metadata": str(record["licence"]), "source": f"https://www.openml.org/t/{task_id}",
              "files_sha256": {name: sha256(folder / name) for name in ["features.parquet", "target.parquet", "splits.npz"]},
              "download_seconds": time.perf_counter() - started}
    write_json(metadata_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/matrixformer_scratch/tabarena_20261028"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    upstream = args.root / "upstream"
    metadata_path = upstream / "packages/tabarena/src/tabarena/benchmark/task/metadata/data/curated_tabarena_dataset_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != 51 or metadata.task_id.nunique() != 51:
        raise ValueError("Expected the 51 distinct TabArena-v0.1 tasks")
    output = args.root / "data"
    output.mkdir(exist_ok=True)
    (output / "tasks").mkdir(exist_ok=True)
    copied_metadata = output / "curated_metadata.csv"
    if copied_metadata.exists() and sha256(copied_metadata) != sha256(metadata_path):
        raise ValueError("Upstream metadata changed; use a new cache root")
    shutil.copy2(metadata_path, copied_metadata)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "suite": "TabArena-v0.1", "datasets": 51,
                "official_splits": int((metadata.num_folds * metadata.tabarena_num_repeats).sum()),
                "upstream": "https://github.com/autogluon/tabarena", "upstream_commit": subprocess.check_output(
                    ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
                "metadata_sha256": sha256(metadata_path), "downloader_sha256": sha256(__file__),
                "openml_version": openml.__version__, "split_policy": "Official OpenML repeat/fold/sample=0; TabArena repeat counts"}
    write_json(output / "manifest.json", manifest)
    openml.config.set_root_cache_directory(str(args.root / "openml_cache"))
    priority = ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"]
    records = metadata.to_dict("records")
    records.sort(key=lambda record: priority.index(record["dataset_name"].strip()) if record["dataset_name"].strip() in priority else 3)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_task, record, output): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Downloaded {len(results)}/51: {result['dataset_name']} {result['rows']}×{result['features']}", flush=True)
            except Exception:
                failure = {"task_id": int(record["task_id"]), "dataset_name": record["dataset_name"], "traceback": traceback.format_exc()}
                failures.append(failure)
                write_json(output / "tasks" / str(record["task_id"]) / "download_failure.json", failure)
                print(failure, flush=True)
            write_json(output / "status.json", {"status": "running", "completed": len(results), "failed": len(failures), "total": 51})
    inventory = [{key: result[key] for key in ["task_id", "dataset_id", "dataset_name", "problem_type", "rows", "features",
                 "missing_feature_cells", "split_count"]} | {"categorical_features": len(result["categorical_columns"]),
                 "max_categorical_cardinality": max(result["cardinality"].values(), default=0)} for result in results]
    pd.DataFrame(inventory).sort_values("task_id").to_csv(output / "inventory.csv", index=False)
    write_json(output / "status.json", {"status": "complete" if not failures else "finished_with_failures",
               "completed": len(results), "failed": len(failures), "total": 51,
               "splits_downloaded": sum(result["split_count"] for result in results)})
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to retry without replacing verified data")


if __name__ == "__main__":
    main()
