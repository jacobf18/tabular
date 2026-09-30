"""Independently rescore the full-user run and archive its capacity comparison."""

import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from recsys_evaluation_common import REPO, sha256, write_csv, write_json


ROOT = REPO / "runs/benchmark_evaluations"
SOURCE = ROOT / "matrixformer_recsys_streamed_allusers_20260930"
REFERENCE = ROOT / "matrixformer_recsys_rows2000_20260930"
PROBE = ROOT / "matrixformer_streaming_layout_probe_20260930"
DATASETS = {"MovieLens 100K": "movielens_100k", "Netflix small": "netflix_small"}


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def without_module_docstring(path):
    tree = ast.parse(path.read_text())
    if ast.get_docstring(tree) is not None:
        tree.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main():
    assert json.loads((SOURCE / "status.json").read_text()) == {"state": "complete", "completed": 4}
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    reference_manifest = json.loads((REFERENCE / "manifest.json").read_text())
    settings = ("checkpoint_sha256", "config_sha256", "frozen_manifest_sha256", "splits_sha256", "model_seed",
                "precision", "prediction_stat", "normalization_data", "global_definition", "split_seed",
                "test_fraction", "exclude_unknowns", "clipping", "max_num_chunks", "preprocessors", "training")
    for setting in settings:
        assert manifest[setting] == reference_manifest[setting]
    for path, value in reference_manifest["source_files_sha256"].items():
        if "/src/" in path:
            assert manifest["source_files_sha256"][path] == value
    streaming_path = REPO / "src/tabimpute/streamed_inference.py"
    snapshot = SOURCE / "streamed_inference_snapshot.py"
    expected = manifest["source_files_sha256"][str(streaming_path)]
    if not snapshot.exists():
        probe_text = (PROBE / "streamed_inference_snapshot.py").read_text()
        module_docstring = streaming_path.read_text().split('"""', 2)[1]
        restored = '"""' + module_docstring + '"""' + probe_text.split('"""', 2)[2]
        assert hashlib.sha256(restored.encode()).hexdigest() == expected
        snapshot.write_text(restored)
    assert sha256(snapshot) == expected
    assert without_module_docstring(snapshot) == without_module_docstring(PROBE / "streamed_inference_snapshot.py")
    assert manifest["block_shape"] == [9976, 1000] and manifest["inference_execution"] == "streamed"
    probe_status = json.loads((PROBE / "status.json").read_text())
    parity = json.loads((PROBE / "parity.json").read_text())
    assert probe_status["state"] == "complete" and probe_status["max_verified_rows"] == 9976 and parity["passed"]
    assert probe_status["weights_unchanged"] and probe_status["gradients_absent"]
    rows = read_rows(SOURCE / "results.csv")
    reference_rows = read_rows(REFERENCE / "results.csv")
    assert len(rows) == 4
    assert {(row["dataset"], row["normalization"]) for row in rows} == {
        (dataset, mode) for dataset in DATASETS for mode in ("global", "column")
    }
    audits = []
    for row in rows:
        folder, mode = DATASETS[row["dataset"]], row["normalization"]
        previous = next(value for value in reference_rows if value["dataset"] == row["dataset"] and value["normalization"] == mode)
        assert row["split_sha256"] == previous["split_sha256"] == manifest["splits_sha256"][folder]
        prediction_path = SOURCE / folder / mode / "test_predictions.npz"
        assert sha256(prediction_path) == row["prediction_sha256"]
        with np.load(prediction_path, allow_pickle=False) as stored:
            predictions = dict(stored)
        reference_prediction_path = REFERENCE / folder / mode / "test_predictions.npz"
        assert sha256(reference_prediction_path) == previous["prediction_sha256"]
        with np.load(reference_prediction_path, allow_pickle=False) as original:
            for name in ("users", "items", "truth"):
                np.testing.assert_array_equal(predictions[name], original[name])
            max_difference = float(np.abs(predictions["prediction"] - original["prediction"]).max())
        with np.load(SOURCE / folder / mode / "normalization.npz", allow_pickle=False) as current:
            with np.load(REFERENCE / folder / mode / "normalization.npz", allow_pickle=False) as original:
                for name in ("means", "stds"):
                    np.testing.assert_array_equal(current[name], original[name])
        assert np.isfinite(predictions["prediction"]).all()
        assert predictions["prediction"].min() >= 1 and predictions["prediction"].max() <= 5
        order = np.argsort(predictions["users"], kind="stable")
        users, starts, counts = np.unique(predictions["users"][order], return_index=True, return_counts=True)
        errors = predictions["prediction"][order] - predictions["truth"][order]
        scores = {"MAE": float((np.add.reduceat(np.abs(errors), starts) / counts).mean()),
                  "RMSE": float(np.sqrt(np.add.reduceat(errors**2, starts) / counts).mean())}
        assert users.size == int(row["users"]) and len(errors) == int(row["test_ratings"])
        for metric, value in scores.items():
            np.testing.assert_allclose(value, float(row[metric]), rtol=0, atol=1e-12)
        if row["dataset"] == "MovieLens 100K":
            assert max_difference < 2e-6
        audits.append({"dataset": row["dataset"], "normalization": mode, "rescored": scores,
                       "same_test_cells_and_truth": True, "normalization_arrays_unchanged": True,
                       "max_prediction_difference_from_2000_rows": max_difference})
    write_json(SOURCE / "audit.json", {
        "status": "passed", "matched_settings": list(settings), "model_and_interface_unchanged": True,
        "streaming_code_matches_capacity_probe_except_module_docstring": True,
        "probe_status": probe_status, "parity": parity, "cases": audits,
        "audit_script_sha256": sha256(Path(__file__)),
    })
    columns = ("dataset", "normalization", "block_rows", "block_cols", "inference_execution", "RMSE", "MAE",
               "imputation_seconds", "peak_gpu_allocated_bytes")
    comparison = [{name: row.get(name, "dense") for name in columns} for row in reference_rows + rows]
    comparison.sort(key=lambda row: (row["dataset"], row["normalization"], int(row["block_rows"])))
    write_csv(SOURCE / "context_comparison.csv", comparison)
    capacity = json.loads((PROBE / "capacity.json").read_text())
    report = ["# Streamed MatrixFormer inference", "", "## Capacity and parity", "",
              "Same step-98,000 checkpoint; no retraining. Each attention sequence retains its full context.",
              "The full hidden grid stays on GPU; encoding, attention, feed-forward layers and decoding run in microbatches.",
              "Cell batch size is a target; decoding keeps at least two rows where available to preserve tensor strides.", "",
              "| Users × items | Execution | Peak allocated GiB | Seconds per probe block |",
              "|---|---|---:|---:|"]
    for mode in ("dense", "streamed"):
        record = parity[mode]
        report.append(f"| 2000 × 1000 | {mode} | {record['peak_allocated_gib']:.3f} | {record['seconds']:.3f} |")
    for record in capacity:
        if record["status"] == "passed":
            report.append(f"| {record['rows']} × 1000 | streamed | {record['peak_allocated_gib']:.3f} | {record['seconds']:.3f} |")
    report.extend(["", f"At 2000 × 1000, maximum prediction difference is {parity['max_abs_difference']:.9g} rating units; RMS difference is {parity['rms_difference']:.9g}.",
                   "Checkpoint state is bitwise unchanged and gradients remain absent. Capacity probes use training observations only, not held-out labels.",
                   "The initial prototype failed the RMS parity gate (0.00207). Its artifacts remain in `matrixformer_streaming_probe_20260930`.",
                   "Preserving the original column-attention RMS normalization/transpose and strided decoder layouts fixes the BF16 fused-bias rounding discrepancy without relaxing that gate.",
                   "The corrected capacity artifacts are in `matrixformer_streaming_layout_probe_20260930`.", "",
                   "## Rating evaluation", "", "The largest training-only capacity-tested geometry (9976 × 1000) is scored under both normalizations.",
                   "This includes all Netflix training users in every item window, not all items in one block. MovieLens already fits all 943 users.",
                   "Errors are macro-averaged over users on the original 1–5 scale; predictions are clipped to [1,5].",
                   "Same frozen 80/20 entry holdout, seed 123, as the prior paper results; this is a previously inspected split, not a blind test.", "",
                   "| Dataset | Normalization | Row limit | RMSE | MAE |",
                   "|---|---|---:|---:|---:|"])
    for row in comparison:
        report.append(f"| {row['dataset']} | {row['normalization']} | {row['block_rows']} | {float(row['RMSE']):.6f} | {float(row['MAE']):.6f} |")
    report.extend(["", "`audit.json` independently rescores the held-out predictions and verifies frozen cells, normalization arrays, and checkpoint/configuration identity.",
                   "The paper is not modified by this experiment. Raw predictions and both failed/successful probe artifacts remain on disk.", "",
                   "## Reproduction", "", "From `/root/tabular/mcpfn` with the original checkpoint, splits and CUDA environment:", "", "```bash",
                   "PYTHONPATH=src:scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled \\",
                   "  python scripts/probe_matrixformer_streaming.py --output runs/benchmark_evaluations/streaming_probe_repeat",
                   "PYTHONPATH=src:scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled \\",
                   "  python scripts/evaluate_matrixformer_recsys_ratings.py --streamed --max-num-rows 9976 --max-num-cols 1000 \\",
                   "  --output runs/benchmark_evaluations/streaming_allusers_repeat", "```", ""])
    (SOURCE / "RESULTS.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
