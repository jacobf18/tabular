"""Compare a new checkpoint against archived, identically configured rating evaluations."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from scipy.sparse import csr_matrix

from evaluate_large_recsys import predict_blocks, rating_statistics, score_independently
from evaluate_matrixformer_recsys_ratings import build_observed, normalization_parameters
from recsys_evaluation_common import REPO, atomic_npz, sha256, user_rating_metrics, write_csv, write_json
from tabimpute.streamed_inference import StreamedImputePFN


ROOT = REPO / "runs/benchmark_evaluations"
SMALL_REFERENCE = ROOT / "matrixformer_recsys_streamed_allusers_20260930"
LARGE_REFERENCE = ROOT / "matrixformer_large_recsys_20260930"
FROZEN_SMALL = ROOT / "dwip_teacher_mean128k_selected98k_recsys_20260919"
DATASETS = {
    "100K": ("MovieLens 100K", "movielens_100k", 9976, "dense"),
    "netflix": ("Netflix small", "netflix_small", 9976, "dense"),
    "1M": ("MovieLens 1M", "1M", 10000, "sparse"),
    "10M": ("MovieLens 10M", "10M", 10000, "sparse"),
}


def reference_paths(dataset, normalization):
    label, folder, row_limit, assembly = DATASETS[dataset]
    if assembly == "dense":
        case = SMALL_REFERENCE / folder / normalization
        split = FROZEN_SMALL / folder / "split.npz"
        prediction = case / "test_predictions.npz"
    else:
        case = LARGE_REFERENCE / folder / ("MatrixFormer-" + normalization)
        split = LARGE_REFERENCE / folder / "split.npz"
        prediction = case / "predictions.npz"
    return {"label": label, "case": case, "split": split, "prediction": prediction,
            "rows": row_limit, "assembly": assembly}


def comparison_record(dataset, normalization, previous, current):
    row = {"dataset": dataset, "normalization": normalization}
    for metric in ("RMSE", "MAE"):
        row[f"previous_{metric}"] = float(previous[metric])
        row[f"new_{metric}"] = float(current[metric])
        row[f"delta_{metric}"] = float(current[metric] - previous[metric])
        row[f"change_percent_{metric}"] = 100 * row[f"delta_{metric}"] / previous[metric]
    return row


def audit_scores(split, prediction, expected=None):
    if prediction.shape != split["test_ratings"].shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid held-out predictions")
    scores = user_rating_metrics(split["test_users"], split["test_ratings"], prediction)
    independent = score_independently(split["test_users"], split["test_ratings"], prediction)
    for metric in ("RMSE", "MAE"):
        np.testing.assert_allclose(scores[metric], independent[metric], rtol=0, atol=1e-12)
        if expected is not None:
            np.testing.assert_allclose(scores[metric], expected[metric], rtol=0, atol=1e-12)
    return scores


def summarize(output, comparisons, total):
    write_csv(output / "comparison.csv", comparisons)
    lines = ["# Larger-matrix checkpoint: recommender comparison", "",
             f"Completed {len(comparisons)} of {total} checkpoint/normalization comparisons.",
             "Same frozen rating splits, observations, normalization arrays, context sizes, and scoring as the previous model.",
             "Raw RMSE and MAE are macro-averaged over test users. Negative changes are improvements.",
             "No recommender fitting or test-based checkpoint/geometry selection. These are previously inspected splits.", ""]
    for normalization in ("global", "column"):
        lines += [f"## {normalization.title()} normalization", "",
                  "| Dataset | Previous RMSE | New RMSE | RMSE change | Previous MAE | New MAE |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in comparisons:
            if row["normalization"] == normalization:
                lines.append(f"| {row['dataset']} | {row['previous_RMSE']:.6f} | {row['new_RMSE']:.6f} | "
                             f"{row['change_percent_RMSE']:+.2f}% | {row['previous_MAE']:.6f} | {row['new_MAE']:.6f} |")
        lines.append("")
    lines += ["Global normalization is the preselected primary setting; column normalization is a control.",
              "Existing external baseline results are unchanged and remain in the referenced evaluations.",
              "Each case includes compact held-out predictions, normalization arrays, source hashes, and rescoring audits."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def run(arguments):
    import torch

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = arguments.checkpoint.resolve()
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    parent_hash = sha256(Path(config["parent_checkpoint"]))
    reference_manifests = [SMALL_REFERENCE / "manifest.json", LARGE_REFERENCE / "manifest.json"]
    for path in reference_manifests:
        if json.loads(path.read_text())["checkpoint_sha256"] != parent_hash:
            raise ValueError("Archived reference is not the new checkpoint's parent")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.8)
    sources = [Path(__file__), REPO / "src/tabimpute/streamed_inference.py",
               *sorted((REPO / "src/tabimpute/model").rglob("*.py")),
               *[Path(__file__).with_name(name) for name in
                 ("evaluate_large_recsys.py", "evaluate_matrixformer_recsys_ratings.py", "recsys_evaluation_common.py")]]
    manifest = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path), "parent_checkpoint_sha256": parent_hash,
        "datasets": arguments.datasets, "normalizations": ["global", "column"],
        "primary_normalization": "global", "precision": "amp_bfloat16", "prediction_stat": "mean",
        "columns": 1000, "small_rows": 9976, "large_rows": 10000,
        "normalization": "Recompute from observed training ratings and verify against archived arrays",
        "scoring": "Mean per-user RMSE and MAE, training-range clipping, identical frozen query order",
        "selection": "User-specified checkpoint, originally selected on synthetic validation; no recommender-based selection",
        "reference_manifests_sha256": {str(path): sha256(path) for path in reference_manifests},
        "source_sha256": {str(path): sha256(path) for path in sources},
        "torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Refusing to resume with changed checkpoint, sources, or protocol")
    else:
        write_json(manifest_path, manifest)
        for path in sources:
            relative = path.relative_to(REPO)
            destination = output / "sources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    imputer = StreamedImputePFN(
        device="cuda", checkpoint_path=str(checkpoint), json_config={"config": config["model"]},
        max_num_rows=9976, max_num_cols=1000, precision_mode="amp_bfloat16", prediction_stat="mean",
    )
    comparisons = []
    for dataset in arguments.datasets:
        paths = reference_paths(dataset, "global")
        with np.load(paths["split"], allow_pickle=False) as stored:
            split = dict(stored)
        split_hash = sha256(paths["split"])
        if paths["assembly"] == "dense":
            train = build_observed(split)
        else:
            train = csr_matrix((split["train_ratings"], (split["train_users"], split["train_items"])),
                               shape=tuple(split["shape"]))
            if train.nnz != len(split["train_ratings"]) or split["train_ratings"].min() <= 0:
                raise ValueError("Expected positive unique observed ratings")
        imputer.max_num_rows = paths["rows"]
        for normalization in ("global", "column"):
            paths = reference_paths(dataset, normalization)
            previous = json.loads((paths["case"] / "metrics.json").read_text())
            if split_hash != previous["split_sha256"]:
                raise ValueError("Frozen split hash mismatch")
            expected_hash = previous.get("prediction_sha256", previous.get("predictions_sha256"))
            if sha256(paths["prediction"]) != expected_hash:
                raise ValueError("Archived predictions hash mismatch")
            with np.load(paths["prediction"], allow_pickle=False) as saved:
                audit_scores(split, saved["prediction"], previous)
                if "users" in saved:
                    for saved_key, split_key in (("users", "test_users"), ("items", "test_items"), ("truth", "test_ratings")):
                        np.testing.assert_array_equal(saved[saved_key], split[split_key])
            if paths["assembly"] == "dense":
                means, stds = normalization_parameters(train, normalization)
            else:
                means, stds = rating_statistics(split["train_items"], split["train_ratings"].astype(np.float64),
                                                train.shape[1], normalization)
            with np.load(paths["case"] / "normalization.npz", allow_pickle=False) as stored:
                for actual, name in ((means, "means"), (stds, "stds")):
                    np.testing.assert_allclose(actual, stored[name], rtol=1e-7, atol=1e-7)
                means, stds = stored["means"].copy(), stored["stds"].copy()
            case = output / dataset / normalization
            case.mkdir(parents=True, exist_ok=True)
            if (case / "metrics.json").exists():
                record = json.loads((case / "metrics.json").read_text())
                if (record["split_sha256"] != split_hash
                        or record["checkpoint_sha256"] != manifest["checkpoint_sha256"]
                        or sha256(case / "normalization.npz") != record["normalization_sha256"]
                        or sha256(case / "predictions.npz") != record["predictions_sha256"]):
                    raise ValueError("Completed case artifact mismatch")
                with np.load(case / "predictions.npz", allow_pickle=False) as stored:
                    scores = audit_scores(split, stored["prediction"], record)
            else:
                context = {"dataset": paths["label"], "normalization": normalization}
                write_json(output / "status.json", {"state": "running", **context})
                torch.manual_seed(42)
                np.random.seed(42)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                if paths["assembly"] == "dense":
                    completed = imputer.impute(train.copy(), means=means, stds=stds)
                    if not np.isfinite(completed).all():
                        raise ValueError("Nonfinite completed matrix")
                    np.testing.assert_array_equal(completed[split["train_users"], split["train_items"]], split["train_ratings"])
                    prediction = completed[split["test_users"], split["test_items"]]
                    del completed
                else:
                    prediction, counts = predict_blocks(
                        imputer, train, split["test_users"], split["test_items"], means, stds,
                        output / "status.json", context,
                    )
                    atomic_npz(case / "query_coverage.npz", counts=counts)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                prediction = np.clip(prediction, split["train_ratings"].min(), split["train_ratings"].max())
                scores = audit_scores(split, prediction)
                atomic_npz(case / "predictions.npz", prediction=prediction)
                atomic_npz(case / "normalization.npz", means=means, stds=stds)
                record = {
                    "dataset": paths["label"], "normalization": normalization, **scores,
                    "split": str(paths["split"]), "split_sha256": split_hash,
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "predictions_sha256": sha256(case / "predictions.npz"),
                    "normalization_sha256": sha256(case / "normalization.npz"),
                    "reference_metrics_sha256": sha256(paths["case"] / "metrics.json"),
                    "test_ratings": len(prediction), "train_ratings": len(split["train_ratings"]),
                    "seconds": elapsed, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "block_rows": imputer.max_num_rows, "block_cols": imputer.max_num_cols,
                }
                write_json(case / "audit.json", {"reference_rescored": True, "new_predictions_independently_rescored": True,
                                                "same_split": True, "same_normalization_arrays": True,
                                                "same_context_geometry": True, "finite": True})
                write_json(case / "metrics.json", record)
                del prediction
            row = comparison_record(paths["label"], normalization, previous, scores)
            comparisons.append(row)
            summarize(output, comparisons, 2 * len(arguments.datasets))
            print(json.dumps(row), flush=True)
        del train, split
    if sha256(checkpoint) != manifest["checkpoint_sha256"] or any(parameter.grad is not None for parameter in imputer.model.parameters()):
        raise ValueError("Checkpoint changed or unexpected gradients computed")
    write_json(output / "status.json", {"state": "complete", "completed": len(comparisons),
                                       "completed_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_unchanged": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    arguments = parser.parse_args()
    try:
        run(arguments)
    except BlockingIOError:
        raise
    except Exception:
        if arguments.output.exists():
            write_json(arguments.output / "status.json", {"state": "failed", "error": traceback.format_exc()})
        raise
