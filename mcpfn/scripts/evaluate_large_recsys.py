"""Resumable rating-only MovieLens 1M/10M evaluation with sparse block assembly."""

import argparse
from datetime import datetime, timezone
import fcntl
import inspect
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from recsys_evaluation_common import CHECKPOINT, REPO, atomic_npz, sha256, user_rating_metrics, write_csv, write_json
from evaluate_matrixformer_recsys_ratings import CHECKPOINT_SHA256
from tabimpute.streamed_inference import StreamedImputePFN, overlapping_spans


RATINGS = {"1M": Path.home() / ".cornac/ml-1m/ratings.dat", "10M": Path.home() / ".cornac/ml-10M100K/ratings.dat"}
DEFAULT_OUTPUT = REPO / "runs/benchmark_evaluations/matrixformer_large_recsys_20260930"
METHODS = ("MatrixFormer-global", "MatrixFormer-column", "MF", "PMF", "SVD", "BPR", "UserKNN", "ItemKNN")
BASELINE_CONFIGS = {
    "MF": dict(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123),
    "PMF": dict(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
    "SVD": dict(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=123),
    "BPR": dict(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
    "UserKNN": dict(k=20, similarity="pearson", mean_centered=True, seed=123),
    "ItemKNN": dict(k=20, similarity="pearson", mean_centered=True, seed=123),
}


def rating_statistics(items, ratings, columns, mode):
    if mode == "global":
        scale = float(ratings.std(ddof=0))
        return np.full(columns, ratings.mean()), np.full(columns, scale if scale > 1e-4 else 1.0)
    if mode != "column":
        raise ValueError("Unknown normalization")
    counts = np.bincount(items, minlength=columns)
    if (counts == 0).any():
        raise ValueError("Unknown item in training mapping")
    means = np.bincount(items, weights=ratings, minlength=columns) / counts
    residuals = ratings - means[items]
    scales = np.sqrt(np.bincount(items, weights=residuals**2, minlength=columns) / counts)
    scales[np.isclose(scales, 0, atol=1e-4)] = 1.0
    return means, scales


def predict_blocks(imputer, train, users, items, means, stds, status_path, context):
    row_spans = overlapping_spans(train.shape[0], imputer.max_num_rows)
    col_spans = overlapping_spans(train.shape[1], imputer.max_num_cols)
    sums = np.zeros(len(users), dtype=np.float64)
    counts = np.zeros(len(users), dtype=np.uint8)
    for row_index, (row_start, row_stop) in enumerate(row_spans):
        selected_rows = (users >= row_start) & (users < row_stop)
        for col_index, (col_start, col_stop) in enumerate(col_spans):
            selected = np.flatnonzero(selected_rows & (items >= col_start) & (items < col_stop))
            if not selected.size:
                continue
            write_json(status_path, {"state": "running", **context, "row_block": row_index + 1,
                                    "row_blocks": len(row_spans), "item_block": col_index + 1, "item_blocks": len(col_spans)})
            observed = train[row_start:row_stop, col_start:col_stop].toarray().astype(np.float64)
            observed[observed == 0] = np.nan
            local_users, local_items = users[selected] - row_start, items[selected] - col_start
            if np.isfinite(observed[local_users, local_items]).any():
                raise ValueError("A held-out query is visible in its training context")
            completed = imputer.impute(observed, means=means[col_start:col_stop], stds=stds[col_start:col_stop])
            visible = np.isfinite(observed)
            if not np.isfinite(completed).all() or not np.array_equal(completed[visible], observed[visible]):
                raise ValueError("Invalid imputed block")
            sums[selected] += completed[local_users, local_items]
            counts[selected] += 1
    if (counts == 0).any():
        raise ValueError("Uncovered held-out queries")
    return sums / counts, counts


def score_independently(users, truth, prediction):
    order = np.argsort(users, kind="stable")
    _, starts, counts = np.unique(users[order], return_index=True, return_counts=True)
    errors = prediction[order] - truth[order]
    return {"MAE": float((np.add.reduceat(np.abs(errors), starts) / counts).mean()),
            "RMSE": float(np.sqrt(np.add.reduceat(errors**2, starts) / counts).mean())}


def run(arguments):
    import cornac
    from cornac.eval_methods import RatioSplit
    import torch

    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Unexpected checkpoint")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.9)
    if shutil.disk_usage(output).free < 512 * 1024**2:
        raise RuntimeError("Need at least 512 MiB for compact split and prediction artifacts")
    source_paths = [Path(__file__), Path(__file__).with_name("recsys_evaluation_common.py"),
                    REPO / "src/tabimpute/streamed_inference.py", REPO / "src/tabimpute/model/model_new_stable.py",
                    REPO / "src/tabimpute/model/bar_distribution.py"]
    manifest = {"checkpoint_sha256": CHECKPOINT_SHA256, "datasets": arguments.datasets, "methods": list(METHODS),
                "block_shape": [10000, 1000], "split_seed": 123, "model_seed": 42, "test_fraction": 0.2,
                "exclude_unknowns": True, "precision": "amp_bfloat16", "normalization": "Observed training ratings only",
                "metric": "Equal-user mean MAE and mean within-user RMSE on original rating scale",
                "clipping": "Observed training rating range", "neural_training": "None",
                "geometry_selection": "Fixed before evaluation; 10000-user windows, no dataset or user subsampling",
                "baseline_configs": BASELINE_CONFIGS, "cornac": cornac.__version__, "torch": torch.__version__,
                "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
                "source_sha256": {str(path): sha256(path) for path in source_paths},
                "dataset_sources": {variant: {"path": str(RATINGS[variant]), "sha256": sha256(RATINGS[variant]),
                                               "url": f"https://files.grouplens.org/datasets/movielens/ml-{variant.lower()}.zip"}
                                    for variant in arguments.datasets}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Refusing to resume with changed sources or protocol")
    else:
        write_json(manifest_path, manifest)
        shutil.copyfile(Path(__file__), output / "evaluator_snapshot.py")
        shutil.copyfile(REPO / "src/tabimpute/streamed_inference.py", output / "streamed_inference_snapshot.py")
    records = []
    model_config = json.loads((CHECKPOINT.parent / "config.json").read_text())["model"]
    imputer = StreamedImputePFN(device="cuda", checkpoint_path=str(CHECKPOINT), json_config={"config": model_config},
                               max_num_rows=10000, max_num_cols=1000, precision_mode="amp_bfloat16", prediction_stat="mean")
    for variant in arguments.datasets:
        dataset = "MovieLens " + variant
        directory = output / variant
        directory.mkdir(exist_ok=True)
        write_json(output / "status.json", {"state": "running", "dataset": dataset, "stage": "prepare split"})
        feedback = cornac.datasets.movielens.load_feedback(variant=variant)
        split = RatioSplit(data=feedback, test_size=0.2, seed=123, exclude_unknowns=True, rating_threshold=4.0, verbose=False)
        del feedback
        train = split.train_set
        train_users, train_items, train_ratings = train.uir_tuple
        test_users, test_items, truth = split.test_set.uir_tuple
        shape = (train.num_users, train.num_items)
        if train_ratings.min() <= 0 or np.unique(train_users.astype(np.int64) * shape[1] + train_items).size != len(train_ratings):
            raise ValueError("Expected positive, unique rating entries")
        frozen = dict(shape=np.asarray(shape), train_users=train_users.astype(np.int32), train_items=train_items.astype(np.int32),
                      train_ratings=train_ratings.astype(np.float32), test_users=test_users.astype(np.int32),
                      test_items=test_items.astype(np.int32), test_ratings=truth.astype(np.float32))
        split_path = directory / "split.npz"
        if split_path.exists():
            with np.load(split_path, allow_pickle=False) as original:
                for key, value in frozen.items():
                    np.testing.assert_array_equal(value, original[key])
        else:
            atomic_npz(split_path, **frozen)
            write_json(directory / "id_maps.json", {"users": list(train.uid_map), "items": list(train.iid_map)})
        del frozen
        split_hash = sha256(split_path)
        write_json(directory / "dataset.json", {"dataset": dataset, "shape": list(shape), "train_ratings": len(train_ratings),
                                               "test_ratings": len(truth), "test_users": int(np.unique(test_users).size),
                                               "split_sha256": split_hash})
        for method in METHODS:
            case = directory / method
            case.mkdir(exist_ok=True)
            metrics_path = case / "metrics.json"
            if metrics_path.exists():
                record = json.loads(metrics_path.read_text())
                if record["status"] == "ok":
                    assert record["split_sha256"] == split_hash
                    assert sha256(case / "predictions.npz") == record["predictions_sha256"]
                records.append(record)
                write_csv(output / "results.csv", records)
                continue
            context = {"dataset": dataset, "method": method}
            write_json(output / "status.json", {"state": "running", **context})
            started = time.perf_counter()
            record = {"dataset": dataset, "method": method, "status": "ok", "MAE": None, "RMSE": None,
                      "users": int(np.unique(test_users).size), "test_ratings": len(truth), "train_ratings": len(train_ratings),
                      "seconds": None, "peak_gpu_allocated_bytes": None, "split_sha256": split_hash,
                      "predictions_sha256": None, "error": None}
            try:
                if method.startswith("MatrixFormer"):
                    mode = method.split("-")[1]
                    means, stds = rating_statistics(train_items, train_ratings, train.num_items, mode)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    prediction, counts = predict_blocks(imputer, train.csr_matrix, test_users, test_items, means, stds,
                                                        output / "status.json", context)
                    torch.cuda.synchronize()
                    record["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    atomic_npz(case / "normalization.npz", means=means, stds=stds)
                    atomic_npz(case / "query_coverage.npz", counts=counts)
                else:
                    model_class = getattr(cornac.models, method)
                    kwargs = dict(BASELINE_CONFIGS[method], verbose=False)
                    if "num_threads" in inspect.signature(model_class).parameters:
                        kwargs["num_threads"] = 4
                    baseline = model_class(**kwargs).fit(train)
                    prediction = np.fromiter((baseline.rate(int(user), int(item)) for user, item in zip(test_users, test_items)),
                                             dtype=np.float64, count=len(truth))
                    del baseline
                prediction = np.clip(prediction, train_ratings.min(), train_ratings.max())
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite rating predictions")
                scores = user_rating_metrics(test_users, truth, prediction)
                independent = score_independently(test_users, truth, prediction)
                for metric in ("MAE", "RMSE"):
                    np.testing.assert_allclose(scores[metric], independent[metric], rtol=0, atol=1e-12)
                    record[metric] = scores[metric]
                atomic_npz(case / "predictions.npz", prediction=prediction)
                record["predictions_sha256"] = sha256(case / "predictions.npz")
                write_json(case / "audit.json", {"independent_user_macro_scores": independent, "finite": True,
                                                "clipped_to_training_range": True, "test_ratings": len(truth)})
                del prediction
            except Exception as error:
                record["status"], record["error"] = "error", f"{type(error).__name__}: {error}"
                (case / "error.log").write_text(traceback.format_exc())
                torch.cuda.empty_cache()
            record["seconds"] = time.perf_counter() - started
            write_json(metrics_path, record)
            records.append(record)
            write_csv(output / "results.csv", records)
            print(json.dumps(record), flush=True)
        del split, train
    write_json(output / "status.json", {"state": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                                       "successful": sum(row["status"] == "ok" for row in records), "cases": len(records)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(RATINGS), default=list(RATINGS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())
