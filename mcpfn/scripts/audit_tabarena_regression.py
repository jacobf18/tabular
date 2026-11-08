"""Independently audit saved regression-Lite predictions, provenance and metrics."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import evaluate_cylinder_completion as shared
from evaluate_tabarena_regression import METHODS, chunk_ranges, regression_tasks
from matrixformer_supervised import TabularPreprocessor


def regression_metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.ndim != 1 or prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Bad prediction shape or values")
    error = prediction - target
    residual_sum = np.sum(error ** 2)
    total_sum = np.sum((target - target.mean()) ** 2)
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "r2": float(1 - residual_sum / total_sum) if total_sum else float(residual_sum == 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/matrixformer_scratch/tabarena_20261028"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = args.output or args.root / "regression_lite"
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    if not args.allow_partial:
        assert status == {"status": "complete", "completed": 13, "failed": 0, "total": 13}, status
    expected = {record["task_id"] for folder, record in regression_tasks(args.root)}
    assert len(expected) == 13 and set(manifest["task_ids"]) == expected
    assert manifest["all_test_rows"] and not manifest["transductive"] and not manifest["hpo"]
    for path, digest in manifest["file_sha256"].items():
        assert shared.sha256(path) == digest, f"Source changed: {path}"
    records = [json.loads(path.read_text()) for path in sorted(output.glob("*/results.json"))]
    completed = {record["task_id"] for record in records}
    assert completed <= expected
    if not args.allow_partial:
        assert completed == expected
    rows, metric_count, query_count = [], 0, 0
    smoke_match, smoke_differences = None, {}
    for record in records:
        case = output / str(record["task_id"])
        folder = args.root / "data/tasks" / str(record["task_id"])
        assert record["problem_type"] == "regression"
        for name, digest in record["data_files_sha256"].items():
            assert shared.sha256(folder / name) == digest
        for name, digest in record["artifacts_sha256"].items():
            assert shared.sha256(case / name) == digest
        features = pd.read_parquet(folder / "features.parquet")
        target = pd.read_parquet(folder / "target.parquet").iloc[:, 0].to_numpy(dtype=np.float64)
        with np.load(case / "predictions.npz") as arrays, np.load(folder / "splits.npz") as splits:
            train, test = arrays["train_indices"], arrays["test_indices"]
            np.testing.assert_array_equal(train, splits["repeat0_fold0_train"])
            np.testing.assert_array_equal(test, splits["repeat0_fold0_test"])
            assert not np.intersect1d(train, test).size
            assert len(train) == record["train_rows"] and len(test) == record["test_rows"]
            assert len(train) + len(test) == len(features)
            context = np.arange(len(train)) if len(train) <= 1023 else np.sort(
                np.random.default_rng(0).choice(len(train), 1023, replace=False))
            np.testing.assert_array_equal(arrays["context_indices"], train[context])
            assert record["context_rows"] == len(context)
            np.testing.assert_equal(record["target_mean"], target[train].mean())
            np.testing.assert_equal(record["target_scale"], target[train].std() or 1.0)
            np.testing.assert_array_equal(np.load(case / "scoring_targets.npy"), target[test])
            np.testing.assert_array_equal(arrays["Dummy"], np.full(len(test), target[train].mean()))
            chunks = []
            ranges = chunk_ranges(len(test), record["batch_size"])
            assert len(ranges) == len(record["chunks"])
            for (start, stop), chunk in zip(ranges, record["chunks"]):
                path = case / "chunks" / f"{start:07d}_{stop:07d}.npz"
                assert chunk["start"] == start and chunk["stop"] == stop
                assert chunk["sha256"] == shared.sha256(path)
                assert chunk == json.loads(path.with_suffix(".json").read_text())
                with np.load(path) as values:
                    assert values["prediction"].shape == (stop - start,)
                    chunks.append(values["prediction"])
            np.testing.assert_array_equal(np.concatenate(chunks), arrays["MatrixFormer"])
            for method in METHODS:
                metrics = regression_metrics(target[test], arrays[method])
                for metric, value in metrics.items():
                    np.testing.assert_allclose(value, record["metrics"][method][metric], rtol=1e-12, atol=1e-12)
                    metric_count += 1
                rows.append({"dataset": record["dataset"], "task_id": record["task_id"], "method": method,
                             **metrics, "rmse_over_dummy": metrics["rmse"] / record["metrics"]["Dummy"]["rmse"]})
            prior = args.root / "supervised_smoke" / case.name / "predictions.npz"
            if prior.exists():
                with np.load(prior) as previous:
                    for method in METHODS:
                        smoke_differences[method] = float(np.max(np.abs(arrays[method] - previous[method])))
                        if method == "RandomForest":
                            np.testing.assert_allclose(arrays[method], previous[method], rtol=1e-12, atol=1e-12)
                        else:
                            np.testing.assert_array_equal(arrays[method], previous[method])
                smoke_match = record["dataset"]
        saved_preprocessor = joblib.load(case / "preprocessor.joblib")
        schema = json.loads((folder / "metadata.json").read_text())
        refitted = TabularPreprocessor().fit(features.iloc[train], schema["categorical_columns"])
        assert saved_preprocessor.summary() == refitted.summary() == record["preprocessing"]
        np.testing.assert_array_equal(saved_preprocessor.transform(features), refitted.transform(features))
        query_count += len(test)
        print(f"Audited {record['dataset']}: {len(test)} predictions per method", flush=True)
    rebuilt = pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)
    saved = pd.read_csv(output / "summary.csv")[rebuilt.columns]
    if args.allow_partial:
        saved = saved[saved.task_id.isin(completed)]
    saved = saved.sort_values(["dataset", "method"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, saved, rtol=1e-12, atol=1e-12)
    aggregate = rebuilt.groupby("method").agg(datasets=("dataset", "count"),
                mean_rmse_over_dummy=("rmse_over_dummy", "mean"), median_rmse_over_dummy=("rmse_over_dummy", "median"))
    if not args.allow_partial:
        saved_aggregate = pd.read_csv(output / "aggregate.csv").set_index("method").sort_index()
        pd.testing.assert_frame_equal(aggregate, saved_aggregate, rtol=1e-12, atol=1e-12)
    audit = {"status": "passed_partial" if args.allow_partial else "passed", "datasets": len(records),
             "total_datasets": len(expected), "test_rows": query_count, "prediction_sets": len(rows),
             "metric_values_checked": metric_count, "source_data_artifact_hashes_verified": True,
             "official_splits_and_seeded_train_only_context_verified": True,
             "train_only_preprocessing_refit_verified": True,
             "smoke_matrixformer_predictions_bitwise_unchanged": smoke_match,
             "smoke_method_max_absolute_differences": smoke_differences,
             "aggregate_csv_verified": not args.allow_partial,
             "audit_source_sha256": shared.sha256(__file__)}
    shared.write_json(output / ("audit_partial.json" if args.allow_partial else "audit.json"), audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
