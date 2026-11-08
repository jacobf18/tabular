"""Fixed supervised smoke test on the three official TabArena quickstart tasks."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error
import torch

import evaluate_cylinder_completion as shared
from matrixformer_supervised import SupervisedMatrixFormer


DATASETS = ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"]
ROOT = Path(__file__).resolve().parents[1]


def scores(target, prediction, problem_type, classes=None):
    if problem_type == "regression":
        return {"rmse": float(root_mean_squared_error(target, prediction)),
                "mae": float(mean_absolute_error(target, prediction)), "r2": float(r2_score(target, prediction))}
    encoded = pd.Index(classes).get_indexer(np.asarray(target))
    if (encoded < 0).any():
        raise ValueError("Test class absent from the training vocabulary")
    np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-6)
    result = {"log_loss": float(log_loss(encoded, prediction, labels=np.arange(len(classes)))),
              "accuracy": float(accuracy_score(encoded, prediction.argmax(axis=1)))}
    if problem_type == "binary":
        result["roc_auc"] = float(roc_auc_score(encoded, prediction[:, 1]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/matrixformer_scratch/tabarena_20261028"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "supervised_smoke"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    task_records = {}
    for path in (args.root / "data/tasks").glob("*/metadata.json"):
        record = json.loads(path.read_text())
        task_records[record["dataset_name"]] = (path.parent, record)
    if any(name not in task_records for name in DATASETS):
        raise ValueError("Quickstart datasets have not all downloaded")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    sources = [Path(__file__).resolve(), ROOT / "scripts/matrixformer_supervised.py", ROOT / "SUPERVISED_TABARENA.md",
               Path(shared.__file__).resolve(), shared.SINGLE, shared.SINGLE.with_name("config.json"),
               ROOT / "src/tabimpute/model/model_new_stable.py", ROOT / "src/tabimpute/model/bar_distribution.py",
               ROOT / "src/tabimpute/data/borders.pt"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "datasets": DATASETS,
                "repeat": 0, "fold": 0, "sample": 0, "context_rows": 1023, "seed": 0,
                "transductive": False, "hpo": False, "bagging": False,
                "file_sha256": {str(path): shared.sha256(path) for path in sources},
                "packages": {name: metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "torch", "openml"]}}
    if manifest["file_sha256"][str(shared.SINGLE)] != shared.SINGLE_SHA256:
        raise ValueError("Unexpected checkpoint")
    shared.write_json(output / "manifest.json", manifest)
    shared.write_json(output / "status.json", {"status": "running", "completed": 0, "total": len(DATASETS)})
    models, distribution = shared.load_models({"single": shared.SINGLE})
    rows = []
    for dataset_index, name in enumerate(DATASETS):
        folder, record = task_records[name]
        for filename, digest in record["files_sha256"].items():
            assert shared.sha256(folder / filename) == digest
        features = pd.read_parquet(folder / "features.parquet")
        with np.load(folder / "splits.npz") as splits:
            train, test = splits["repeat0_fold0_train"], splits["repeat0_fold0_test"]
        train_features, test_features = features.iloc[train], features.iloc[test]
        train_target = pd.read_parquet(folder / "target.parquet").iloc[train, 0]
        case = output / str(record["task_id"])
        case.mkdir()
        started = time.perf_counter()
        predictor = SupervisedMatrixFormer(models["single"], distribution, record["problem_type"])
        predictor.fit(train_features, train_target, record["categorical_columns"])
        fit_seconds = time.perf_counter() - started
        print(f"Predicting {name}: train={len(train)} test={len(test)} context={predictor.context_.shape} batch={predictor.batch_size_}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = predictor._predict(test_features)
        torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - started
        predictions = {"MatrixFormer": prediction}
        timings = {"MatrixFormer": {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                    "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}}
        transformed_train = predictor.preprocessor_.transform(train_features)
        transformed_test = predictor.preprocessor_.transform(test_features)
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        baseline_train = imputer.fit_transform(transformed_train)
        baseline_test = imputer.transform(transformed_test)
        if record["problem_type"] == "regression":
            baselines = {"Dummy": DummyRegressor(), "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0)}
            baseline_target = np.asarray(train_target)
        else:
            baselines = {"Dummy": DummyClassifier(strategy="prior"),
                         "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0)}
            baseline_target = pd.Index(predictor.classes_).get_indexer(np.asarray(train_target))
        for method, baseline in baselines.items():
            started = time.perf_counter()
            baseline.fit(baseline_train, baseline_target)
            fitted = time.perf_counter()
            predictions[method] = baseline.predict(baseline_test) if record["problem_type"] == "regression" else baseline.predict_proba(baseline_test)
            timings[method] = {"fit_seconds": fitted - started, "predict_seconds": time.perf_counter() - fitted}
        np.savez_compressed(case / "predictions.npz", **predictions, train_indices=train, test_indices=test,
                            context_indices=train[predictor.context_indices_],
                            classes=np.array([], dtype=str) if predictor.classes_ is None else predictor.classes_.astype(str))
        prediction_hash = shared.sha256(case / "predictions.npz")
        joblib.dump(predictor.preprocessor_, case / "preprocessor.joblib")
        target = pd.read_parquet(folder / "target.parquet").iloc[test, 0]
        np.save(case / "scoring_targets.npy", np.asarray(target, dtype=np.float64 if record["problem_type"] == "regression" else str))
        primary = {"regression": "rmse", "binary": "roc_auc", "multiclass": "log_loss"}[record["problem_type"]]
        metrics = {method: scores(target, values, record["problem_type"], predictor.classes_) for method, values in predictions.items()}
        case_record = {"dataset": name, "task_id": record["task_id"], "problem_type": record["problem_type"],
                       "primary_metric": primary, "train_rows": len(train), "test_rows": len(test),
                       "context_rows": len(predictor.context_indices_), "target_columns": predictor.target_columns_,
                       "target_mean": getattr(predictor, "target_mean_", None), "target_scale": getattr(predictor, "target_scale_", None),
                       "preprocessing": predictor.preprocessor_.summary(), "metrics": metrics, "timings": timings,
                       "data_files_sha256": record["files_sha256"], "prediction_sha256": prediction_hash,
                       "preprocessor_sha256": shared.sha256(case / "preprocessor.joblib"),
                       "scoring_targets_sha256": shared.sha256(case / "scoring_targets.npy")}
        shared.write_json(case / "results.json", case_record)
        for method, values in metrics.items():
            rows.append({"dataset": name, "task_id": record["task_id"], "problem_type": record["problem_type"],
                         "method": method, "primary_metric": primary, "score": values[primary],
                         "fit_seconds": timings[method]["fit_seconds"], "predict_seconds": timings[method]["predict_seconds"]})
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
        shared.write_json(output / "status.json", {"status": "running", "completed": dataset_index + 1, "total": len(DATASETS)})
        print({method: values[primary] for method, values in metrics.items()}, flush=True)
    lines = ["# MatrixFormer supervised TabArena smoke test", "",
             "Official repeat 0 / fold 0, three predeclared official quickstart datasets; all test rows.",
             "No HPO or bagging; this is not a full TabArena leaderboard result.", "",
             "| Dataset | Metric | MatrixFormer | RandomForest (200 trees) | Dummy |", "|---|---|---:|---:|---:|"]
    frame = pd.DataFrame(rows)
    for name in DATASETS:
        selected = frame[frame.dataset == name].set_index("method")
        metric = selected.loc["MatrixFormer", "primary_metric"]
        direction = "↑" if metric == "roc_auc" else "↓"
        entries = [f"{selected.loc[method, 'score']:.5f}" for method in ["MatrixFormer", "RandomForest", "Dummy"]]
        lines.append(f"| {name} | {metric} {direction} | " + " | ".join(entries) + " |")
    lines.extend(["", "Inputs are transformed using training-only empirical Gaussianization and nominal encoding.",
                  "Regression decodes means; binary classification reads CDF mass; multiclass normalizes one-vs-rest CDF masses.",
                  "One independent query per matrix prevents test-to-test attention. All training rows fit the context for these three tasks.",
                  "Saved predictions, class order, split/context indices, fitted transformations, timings, and provenance accompany this report.",
                  f"Protocol and limitations: `{ROOT / 'SUPERVISED_TABARENA.md'}`.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    shared.write_json(output / "status.json", {"status": "complete", "completed": len(DATASETS), "total": len(DATASETS)})


if __name__ == "__main__":
    main()
