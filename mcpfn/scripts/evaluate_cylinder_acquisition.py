"""One-batch uncertainty-guided acquisition on fixed, disjoint cylinder test cells."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from evaluate_cylinder_completion import (
    METHODS, SINGLE, load_models, normalize_observed, point_metrics, sha256, write_json,
)


def choose_acquisitions(pool, uncertainty, predictions, budget, seed):
    if not 0 < budget < len(pool) or len(np.unique(pool)) != len(pool):
        raise ValueError("Require a unique acquisition pool larger than the budget")
    return {
        "random": np.sort(np.random.default_rng(seed + 524287).choice(pool, budget, replace=False)),
        "uncertainty": np.sort(pool[np.argsort(uncertainty, kind="stable")[-budget:]]),
        "magnitude": np.sort(pool[np.argsort(np.abs(predictions.ravel()[pool]), kind="stable")[-budget:]]),
    }


@torch.inference_mode()
def predict(model, distribution, matrix):
    normalized, mean, scale = normalize_observed(matrix)
    values = torch.from_numpy(normalized.astype(np.float32)).cuda().unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = model(values, return_embeddings=True)[0]
    features = encoded.reshape(-1, encoded.shape[-1])
    estimate = torch.empty(matrix.size, dtype=torch.float32, device="cuda")
    for start in range(0, matrix.size, 2048):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.decoder(features[start:start + 2048])
        estimate[start:start + len(logits)] = distribution.mean(logits.float())
    return estimate.cpu().numpy().reshape(matrix.shape).astype(np.float64) * scale + mean


def run(output, budget):
    source_manifest = json.loads((output / "manifest.json").read_text())
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    records = [json.loads(path.read_text()) for path in sorted((output / "cases").glob("*.json"))]
    cases = [record for record in records if record["rate"] == .8]
    destination = output / "acquisition"
    destination.mkdir(exist_ok=True)
    manifest = {"source_manifest_sha256": sha256(output / "manifest.json"),
                "checkpoint_sha256": sha256(SINGLE), "source_sha256": sha256(__file__),
                "case_ids": [record["case_id"] for record in cases], "budget": budget,
                "candidate_pool": "Previously saved uniform hidden-cell uncertainty subset, 10000 cells per case",
                "test_set": "All original outer hidden cells EXCEPT the entire candidate pool; fixed across all policies",
                "score_denominator": "Original outer observed SD, fixed before acquisition",
                "policies": ["random", "uncertainty", "magnitude"],
                "uncertainty_policy": "Largest original native predictive standard deviation",
                "magnitude_policy": "Largest absolute original predicted mean, a non-uncertainty control",
                "selection_uses_test_labels": False, "model_training": False,
                "normalization": "Refitted to augmented observations for each policy",
                "timing": "Acquisition policies registered after completion pilot, before acquisition outcomes; exploratory follow-up"}
    assert manifest["checkpoint_sha256"] == source_manifest["checkpoint_sha256"][str(SINGLE)]
    if (destination / "manifest.json").exists():
        assert json.loads((destination / "manifest.json").read_text()) == manifest
    write_json(destination / "manifest.json", manifest)
    with np.load(output / "dataset.npz") as data:
        truth = data["truth"]
    models, distribution = load_models({"single": SINGLE})
    model = models["single"]
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_num_threads(4)
    rows = []
    for record in cases:
        source = output / "predictions" / f"{record['case_id']}.npz"
        assert sha256(source) == record["prediction_sha256"]
        with np.load(source) as saved:
            missing, pool = saved["missing"], saved["uncertainty_indices"]
            initial = saved[f"prediction_{METHODS.index('MatrixFormer')}"]
            uncertainty = saved[f"uncertainty_{METHODS.index('MatrixFormer')}_std"]
        selection = choose_acquisitions(pool, uncertainty, initial, budget, record["seed"])
        fixed_test = missing.copy()
        fixed_test.ravel()[pool] = False
        before = point_metrics(truth, initial, fixed_test, record["observed_std"])
        original_observed = np.where(missing, np.nan, truth)
        for policy, indices in selection.items():
            case_id = f"{record['case_id']}_{policy}"
            json_path, array_path = destination / f"{case_id}.json", destination / f"{case_id}.npz"
            if json_path.exists():
                row = json.loads(json_path.read_text())
                assert sha256(array_path) == row["prediction_sha256"]
            else:
                assert len(indices) == budget and not fixed_test.ravel()[indices].any()
                augmented = original_observed.copy()
                augmented.ravel()[indices] = truth.ravel()[indices]
                assert np.isnan(augmented[fixed_test]).all()
                started = time.perf_counter()
                prediction = predict(model, distribution, augmented)
                after = point_metrics(truth, prediction, fixed_test, record["observed_std"])
                np.savez_compressed(array_path, acquired_indices=indices, fixed_test=fixed_test,
                                    prediction=prediction.astype(np.float32))
                row = {"case_id": record["case_id"], "pattern": record["pattern"], "seed": record["seed"],
                       "policy": policy, "budget": budget, "test_cells": int(fixed_test.sum()),
                       "nrmse_before": before["nrmse"], "nrmse_after": after["nrmse"],
                       "relative_reduction": 1 - after["nrmse"] / before["nrmse"],
                       "seconds": time.perf_counter() - started, "prediction_sha256": sha256(array_path)}
                write_json(json_path, row)
            with np.load(array_path) as archived:
                assert np.array_equal(archived["acquired_indices"], indices)
                assert np.array_equal(archived["fixed_test"], fixed_test)
                check = point_metrics(truth, archived["prediction"], fixed_test, record["observed_std"])
                np.testing.assert_allclose(check["nrmse"], row["nrmse_after"], rtol=2e-5, atol=1e-7)
            print(json.dumps(row), flush=True)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "per_case.csv", index=False)
    summary = frame.groupby(["pattern", "policy"])[["nrmse_before", "nrmse_after", "relative_reduction"]].agg(["mean", "std"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.to_csv(destination / "summary.csv")
    write_json(destination / "audit.json", {"passed": True, "cases": len(cases), "policy_evaluations": len(rows),
                                            "fixed_disjoint_test_sets": True, "equal_acquisition_budgets": True})
    write_json(destination / "status.json", {"state": "complete", "policy_evaluations": len(rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, default=1000)
    arguments = parser.parse_args()
    run(arguments.output.resolve(), arguments.budget)
