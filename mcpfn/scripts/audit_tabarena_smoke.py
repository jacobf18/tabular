"""Audit saved supervised predictions and verify independent-query GPU batching."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

import evaluate_cylinder_completion as shared
from matrixformer_supervised import SupervisedMatrixFormer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/matrixformer_scratch/tabarena_20261028"))
    args = parser.parse_args()
    output = args.root / "supervised_smoke"
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    assert status == {"status": "complete", "completed": 3, "total": 3}
    for path, digest in manifest["file_sha256"].items():
        assert shared.sha256(path) == digest, f"Source changed: {path}"
    checks, rows, unchanged = 0, [], 0
    records = []
    for path in sorted(output.glob("*/results.json")):
        record = json.loads(path.read_text())
        records.append(record)
        case = path.parent
        folder = args.root / "data/tasks" / str(record["task_id"])
        for name, digest in record["data_files_sha256"].items():
            assert shared.sha256(folder / name) == digest
        assert shared.sha256(case / "predictions.npz") == record["prediction_sha256"]
        assert shared.sha256(case / "preprocessor.joblib") == record["preprocessor_sha256"]
        assert shared.sha256(case / "scoring_targets.npy") == record["scoring_targets_sha256"]
        with np.load(case / "predictions.npz") as arrays, np.load(folder / "splits.npz") as splits:
            train, test = arrays["train_indices"], arrays["test_indices"]
            np.testing.assert_array_equal(train, splits["repeat0_fold0_train"])
            np.testing.assert_array_equal(test, splits["repeat0_fold0_test"])
            assert not np.intersect1d(train, test).size
            assert np.isin(arrays["context_indices"], train).all()
            assert len(np.unique(arrays["context_indices"])) == record["context_rows"] == len(train)
            target = pd.read_parquet(folder / "target.parquet").iloc[test, 0]
            if record["problem_type"] == "regression":
                truth = np.asarray(target, dtype=np.float64)
            else:
                truth = pd.Index(arrays["classes"]).get_indexer(np.asarray(target, dtype=str))
                assert (truth >= 0).all()
            saved_truth = np.load(case / "scoring_targets.npy", allow_pickle=False)
            np.testing.assert_array_equal(saved_truth, np.asarray(target, dtype=saved_truth.dtype))
            for method in ["MatrixFormer", "RandomForest", "Dummy"]:
                prediction = arrays[method]
                assert np.isfinite(prediction).all()
                if record["problem_type"] == "regression":
                    error = prediction - truth
                    scores = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
                              "r2": float(1 - np.sum(error ** 2) / np.sum((truth - truth.mean()) ** 2))}
                else:
                    assert (prediction >= 0).all() and (prediction <= 1).all()
                    np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-12)
                    epsilon = np.finfo(prediction.dtype).eps
                    scores = {"log_loss": float(-np.log(np.clip(prediction[np.arange(len(truth)), truth], epsilon, 1 - epsilon)).mean()),
                              "accuracy": float(np.mean(truth == prediction.argmax(axis=1)))}
                    if record["problem_type"] == "binary":
                        scores["roc_auc"] = float(roc_auc_score(truth, prediction[:, 1]))
                for metric, value in scores.items():
                    np.testing.assert_allclose(value, record["metrics"][method][metric], rtol=1e-10, atol=1e-12)
                    checks += 1
                rows.append({"dataset": record["dataset"], "method": method, "score": scores[record["primary_metric"]]})
            initial = args.root / "supervised_smoke_initial" / case.name / "predictions.npz"
            if initial.exists():
                with np.load(initial) as previous:
                    np.testing.assert_array_equal(arrays["MatrixFormer"], previous["MatrixFormer"])
                    unchanged += 1
    assert {record["dataset"] for record in records} == set(manifest["datasets"])
    rebuilt = pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)
    saved = pd.read_csv(output / "summary.csv")[rebuilt.columns].sort_values(["dataset", "method"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, saved, rtol=1e-10, atol=1e-12)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    models, distribution = shared.load_models({"single": shared.SINGLE})
    binary = next(record for record in records if record["problem_type"] == "binary")
    folder = args.root / "data/tasks" / str(binary["task_id"])
    features = pd.read_parquet(folder / "features.parquet")
    with np.load(folder / "splits.npz") as splits:
        train, test = splits["repeat0_fold0_train"], splits["repeat0_fold0_test"]
    training_target = pd.read_parquet(folder / "target.parquet").iloc[train, 0]
    predictor = SupervisedMatrixFormer(models["single"], distribution, "binary")
    predictor.fit(features.iloc[train], training_target, binary["preprocessing"]["categorical_columns"])
    alone = predictor.predict_proba(features.iloc[test[:1]])
    combined = predictor.predict_proba(features.iloc[test[:5]])[:1]
    changed_queries = features.iloc[test[:5]].copy()
    changed_queries.iloc[1:] = features.iloc[train[-4:]].to_numpy()
    changed = predictor.predict_proba(changed_queries)[:1]
    np.testing.assert_array_equal(combined, changed)
    torch.set_float32_matmul_precision("highest")
    fp32_predictions = []
    with torch.inference_mode():
        for count in [1, 5]:
            transformed = predictor.preprocessor_.transform(features.iloc[test[:count]])
            matrix = torch.from_numpy(predictor._query_matrices(transformed)).cuda()
            encoded = models["single"](matrix, return_embeddings=True)
            logits = models["single"].decoder(encoded[:, -1, -1])
            probability = 1 - distribution.cdf(logits, torch.tensor([0.], device="cuda"))
            fp32_predictions.append(probability[:1].cpu().numpy())
    np.testing.assert_allclose(fp32_predictions[0], fp32_predictions[1], atol=1e-6, rtol=1e-6)
    bf16_difference = float(np.max(np.abs(alone - combined)))
    shared.write_json(output / "batch_precision_diagnostic.json", {
        "initial_1e_minus_4_cross_batch_size_check_passed": bool(np.allclose(alone, combined, atol=1e-4, rtol=1e-4)),
        "bf16_alone": alone.tolist(), "bf16_batched": combined.tolist(),
        "bf16_max_difference_across_batch_sizes": bf16_difference,
        "fixed_batch_changed_unrelated_queries_max_difference": float(np.max(np.abs(combined - changed))),
        "fp32_max_difference_across_batch_sizes": float(np.max(np.abs(fp32_predictions[0] - fp32_predictions[1]))),
        "interpretation": "BF16 batching changes numerical rounding. Changing unrelated queries at fixed batch shape does not change the first prediction. FP32 cross-batch-size check passes 1e-6. No inference or scoring setting was selected from benchmark scores."
    })
    result = {"status": "passed", "datasets": 3, "prediction_sets": len(rows), "metric_values_checked": checks,
              "official_splits_and_train_only_context_verified": True, "source_and_data_hashes_verified": True,
              "pilot_prediction_sets_unchanged_by_schema_bugfix": unchanged,
              "changed_unrelated_queries_max_difference": float(np.max(np.abs(combined - changed))),
              "bf16_cross_batch_size_roundoff_max_difference": bf16_difference,
              "fp32_cross_batch_size_max_difference": float(np.max(np.abs(fp32_predictions[0] - fp32_predictions[1]))),
              "audit_source_sha256": shared.sha256(__file__)}
    shared.write_json(output / "audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
