"""Frozen MatrixFormer+ MovieLens evaluation with compact prediction storage."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from compare_recsys_checkpoint import audit_scores, reference_paths
from dwip_recsys_experiment import inference_components, inference_row_scales
from evaluate_two_expert_router import CHECKPOINT, GENERAL, RECSYS, ROOT, check_reserve, make_model
from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from tabimpute.routing import fit_routing
from tabimpute.streamed_inference import overlapping_spans


OUTPUT = ROOT / "matrixformer_plus_large_recsys_20261004"


def collect_predictions(model, observed, users, items, means, scales, offsets=None,
                        row_scales=None, progress=None):
    sums = np.zeros(len(users), dtype=np.float64)
    counts = np.zeros(len(users), dtype=np.uint8)
    row_spans = overlapping_spans(observed.shape[0], model.max_num_rows)
    column_spans = overlapping_spans(observed.shape[1], model.max_num_cols)
    for row_index, (row_start, row_stop) in enumerate(row_spans):
        selected_rows = (users >= row_start) & (users < row_stop)
        for column_index, (column_start, column_stop) in enumerate(column_spans):
            selected = np.flatnonzero(selected_rows & (items >= column_start) & (items < column_stop))
            if not selected.size:
                continue
            if progress is not None:
                progress(row_index + 1, len(row_spans), column_index + 1, len(column_spans))
            block = observed[row_start:row_stop, column_start:column_stop].copy()
            block_means, block_scales = means[column_start:column_stop], scales[column_start:column_stop]
            if offsets is not None:
                block_offsets = offsets[row_start:row_stop, None]
                block_row_scales = row_scales[row_start:row_stop, None]
                block = ((block - block_offsets - block_means) / block_scales) / block_row_scales
                _, raw = model.impute(block, means=np.zeros_like(block_means),
                                      stds=np.ones_like(block_scales), return_full=True)
                raw = raw * block_row_scales * block_scales + block_means + block_offsets
            else:
                _, raw = model.impute(block, means=block_means, stds=block_scales, return_full=True)
            if not np.isfinite(raw).all():
                raise ValueError("Nonfinite raw predictions")
            sums[selected] += raw[users[selected] - row_start, items[selected] - column_start]
            counts[selected] += 1
    if (counts == 0).any():
        raise ValueError("Uncovered evidence or query cells")
    return sums / counts, counts


def fit_observed_blend(truth, general, specialist):
    scale = max(float(np.std(truth)), 1e-8)
    return fit_routing(truth[:, None], general[:, None], specialist[:, None],
                       np.ones((len(truth), 1), dtype=bool), np.array([scale]))


def run(arguments):
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    sources = [Path(__file__), Path(__file__).with_name("evaluate_two_expert_router.py"),
               Path(__file__).with_name("dwip_recsys_experiment.py"),
               *sorted((ROOT.parents[1] / "src/tabimpute").rglob("*.py"))]
    manifest = {
        "datasets": arguments.datasets, "general": str(GENERAL), "specialist": str(CHECKPOINT),
        "checkpoint_sha256": {str(path): sha256(path) for path in (GENERAL, CHECKPOINT)},
        "source_sha256": {str(path.resolve()): sha256(path) for path in sources},
        "split_sha256": {dataset: sha256(reference_paths(dataset, "global")["split"])
                         for dataset in arguments.datasets},
        "protocol": {
            "general": "Global observed normalization; streamed 10000x1000 windows",
            "specialist": "two_way_user_scale; original dense kernel; 2000x1000 windows",
            "routing": "One global observed-only least-squares convex weight; raw predictions; no holdout tuning",
            "clipping": "After overlap averaging, clip each expert to observed training rating range (1M:1-5;10M:0.5-5)",
            "scoring": "Frozen seed123 80/20 split; mean per-user RMSE and MAE, not pooled",
            "storage": "Dense observed context in RAM, only observed/query predictions on disk",
            "limitation": "Observed reconstruction is in-sample evidence, not unbiased validation",
        },
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen manifest differs")
    else:
        write_json(manifest_path, manifest)
    rows = []
    for dataset in arguments.datasets:
        check_reserve(output)
        case = output / dataset
        case.mkdir(exist_ok=True)
        paths = reference_paths(dataset, "global")
        with np.load(paths["split"], allow_pickle=False) as stored:
            split = dict(stored)
        observed = np.full(tuple(split["shape"]), np.nan, dtype=np.float64)
        observed[split["train_users"], split["train_items"]] = split["train_ratings"]
        if np.isfinite(observed[split["test_users"], split["test_items"]]).any():
            raise ValueError("Query leakage")
        users = np.concatenate([split["train_users"], split["test_users"]])
        items = np.concatenate([split["train_items"], split["test_items"]])
        evidence_count = len(split["train_ratings"])
        predictions = {}
        for expert, checkpoint in (("general", GENERAL), ("specialist", CHECKPOINT)):
            artifact = case / f"{expert}.npz"
            record_path = case / f"{expert}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if sha256(artifact) != record["prediction_sha256"]:
                    raise ValueError("Prediction artifact changed")
                with np.load(artifact, allow_pickle=False) as stored:
                    predictions[expert] = stored["prediction"]
                continue
            model = make_model(checkpoint, streamed=expert == "general")
            model.max_num_rows = 10000 if expert == "general" else 2000
            model.max_num_cols = 1000
            if expert == "general":
                with np.load(RECSYS / dataset / "global/normalization.npz") as stored:
                    means, scales = stored["means"], stored["stds"]
                offsets = row_scales = None
            else:
                means, scales, offsets = inference_components(observed, "two_way_user_scale")
                row_scales = inference_row_scales(observed, "two_way_user_scale", means, scales, offsets)
            normalization = {"means": means, "scales": scales}
            if offsets is not None:
                normalization.update(offsets=offsets, row_scales=row_scales)
            atomic_npz(case / f"{expert}_normalization.npz", **normalization)

            def progress(row, row_count, column, column_count):
                status = {"state": "running", "dataset": dataset, "expert": expert,
                          "row_block": row, "row_blocks": row_count,
                          "column_block": column, "column_blocks": column_count}
                write_json(output / "status.json", status)
                print(json.dumps(status), flush=True)

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            prediction, counts = collect_predictions(model, observed, users, items, means, scales,
                                                    offsets, row_scales, progress)
            prediction = np.clip(prediction, split["train_ratings"].min(), split["train_ratings"].max())
            atomic_npz(artifact, prediction=prediction, counts=counts)
            record = {"seconds": time.perf_counter() - started, "prediction_sha256": sha256(artifact),
                      "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "normalization_sha256": sha256(case / f"{expert}_normalization.npz")}
            write_json(record_path, record)
            predictions[expert] = prediction
            del model
            torch.cuda.empty_cache()
        reference = RECSYS / dataset / "global"
        reference_metrics = json.loads((reference / "metrics.json").read_text())
        assert reference_metrics["split_sha256"] == manifest["split_sha256"][dataset]
        assert sha256(reference / "predictions.npz") == reference_metrics["predictions_sha256"]
        with np.load(reference / "predictions.npz") as stored:
            difference = float(np.max(np.abs(predictions["general"][evidence_count:] - stored["prediction"])))
            np.testing.assert_allclose(predictions["general"][evidence_count:], stored["prediction"], rtol=0, atol=1e-12)
        decision = fit_observed_blend(split["train_ratings"].astype(np.float64),
                                     predictions["general"][:evidence_count], predictions["specialist"][:evidence_count])
        weight = decision["specialist_weight"]
        query_predictions = {name: prediction[evidence_count:] for name, prediction in predictions.items()}
        query_predictions["observed_blend"] = ((1 - weight) * query_predictions["general"]
                                                + weight * query_predictions["specialist"])
        for variant, prediction in query_predictions.items():
            rows.append({"dataset": paths["label"], "variant": variant, **audit_scores(split, prediction)})
        atomic_npz(case / "query_predictions.npz", **query_predictions)
        write_json(case / "audit.json", {"decision": decision, "general_reference_max_abs_difference": difference,
                   "independently_rescored": True, "no_query_in_context": True,
                   "rating_range": [float(split["train_ratings"].min()), float(split["train_ratings"].max())],
                   "train_ratings": evidence_count, "test_ratings": len(split["test_ratings"]),
                   "query_predictions_sha256": sha256(case / "query_predictions.npz")})
        write_csv(output / "summary.csv", rows)
        del observed, predictions, query_predictions, split
    for path, expected in manifest["checkpoint_sha256"].items():
        assert sha256(Path(path)) == expected
    write_json(output / "final_audit.json", {"state": "complete", "datasets": arguments.datasets,
               "summary_sha256": sha256(output / "summary.csv"), "checkpoints_unchanged": True,
               "completed_utc": datetime.now(timezone.utc).isoformat()})
    write_json(output / "status.json", {"state": "complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=("1M", "10M"), default=["1M", "10M"])
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        if arguments.output.exists():
            write_json(arguments.output / "status.json", {"state": "failed", "error": traceback.format_exc()})
        raise
