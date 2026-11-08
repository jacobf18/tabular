"""Check train-fitted feature processing on every TabArena-Lite split, without scoring."""

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from download_tabarena import sha256, write_json
from matrixformer_supervised import TabularPreprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/matrixformer_scratch/tabarena_20261028"))
    args = parser.parse_args()
    output = args.root / "preprocessing_preflight"
    output.mkdir(exist_ok=True)
    rows, failures = [], []
    paths = sorted((args.root / "data/tasks").glob("*/metadata.json"))
    if len(paths) != 51:
        raise ValueError("Download all 51 datasets first")
    for path in paths:
        record = json.loads(path.read_text())
        started = time.perf_counter()
        try:
            folder = path.parent
            for name in ["features.parquet", "splits.npz"]:
                assert sha256(folder / name) == record["files_sha256"][name]
            features = pd.read_parquet(folder / "features.parquet")
            with np.load(folder / "splits.npz") as splits:
                train, test = splits["repeat0_fold0_train"], splits["repeat0_fold0_test"]
            preprocessor = TabularPreprocessor().fit(features.iloc[train], record["categorical_columns"])
            training = preprocessor.transform(features.iloc[train])
            testing = preprocessor.transform(features.iloc[test])
            assert training.shape[1] == testing.shape[1] and not np.isinf(training).any() and not np.isinf(testing).any()
            rows.append({"dataset": record["dataset_name"], "task_id": record["task_id"],
                         "train_rows": len(train), "test_rows": len(test), "input_features": features.shape[1],
                         "encoded_features": training.shape[1], "dropped_features": len(preprocessor.dropped_columns_),
                         "categorical_features": len(preprocessor.categorical_columns_),
                         "inferred_categorical_features": len(preprocessor.inferred_categorical_columns_),
                         "datetime_features": len(preprocessor.datetime_columns_),
                         "train_missing_cells": int(np.isnan(training).sum()), "test_missing_cells": int(np.isnan(testing).sum()),
                         "seconds": time.perf_counter() - started})
            print(f"Preprocessed {record['dataset_name']}: {features.shape[1]} -> {training.shape[1]} features", flush=True)
        except Exception:
            failures.append({"dataset": record["dataset_name"], "traceback": traceback.format_exc()})
            print(failures[-1], flush=True)
    pd.DataFrame(rows).to_csv(output / "inventory.csv", index=False)
    write_json(output / "audit.json", {"status": "passed" if not failures else "failed", "datasets": len(rows),
               "test_labels_read": False, "failures": failures,
               "source_sha256": {str(source): sha256(source) for source in
                                 [Path(__file__), Path(__file__).with_name("matrixformer_supervised.py")]}})
    if failures:
        raise RuntimeError("Preprocessing failures recorded")


if __name__ == "__main__":
    main()
