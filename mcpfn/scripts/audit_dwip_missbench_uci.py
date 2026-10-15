"""Independently rescore saved predictions with the original benchmark functions."""

import argparse
import ast
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from evaluate_dwip_missbench_uci import METRICS, REPO, discover_cases, sha256, write_json


def load_original(path, name):
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text())
    protocol = manifest["protocol"]
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "complete" and status["successful"] == 2262 and status["failed"] == 0
    assert protocol["cases"] == discover_cases(REPO / "benchmark/datasets")
    assert sha256(Path(protocol["checkpoint"])) == protocol["checkpoint_sha256"]
    assert sha256(REPO / "scripts/evaluate_dwip_missbench_uci.py") == protocol["script_sha256"]
    paths = {"missbench": (REPO / "benchmark/plot_options.py", REPO / "benchmark/openml/plot_r_squared.py"),
             "uci": (REPO / "benchmark/uci/plot_uci_negative_rmse.py", REPO / "benchmark/uci/plot_uci_r_squared.py")}
    scorers = {suite: (load_original(paths[suite][0], "compute_normalized_rmse_columnwise"),
                       load_original(paths[suite][1], "compute_r_squared")) for suite in paths}
    maximum_difference = defaultdict(float)
    counts = defaultdict(int)
    repaired_cases = 0
    dataset_rows = defaultdict(list)
    predictions_bytes = 0
    artifacts = {}
    for case in protocol["cases"]:
        result_path = output / "cases" / (case["case_id"] + ".json")
        row = json.loads(result_path.read_text())
        assert row["status"] == "ok" and all(row[key] == value for key, value in case.items())
        source = Path(case["source_directory"])
        for filename, field in (("missing.npy", "missing_sha256"), ("true.npy", "true_sha256")):
            assert sha256(source / filename) == row[field], str(source / filename)
        prediction_path = output / row["prediction_path"]
        assert sha256(prediction_path) == row["prediction_sha256"]
        missing = np.load(source / "missing.npy", allow_pickle=False)
        truth = np.load(source / "true.npy", allow_pickle=False)
        if row.get("zero_std_columns"):
            assert row["zero_std_columns"] == np.flatnonzero(np.nanstd(missing, axis=0) == 0).tolist()
            assert (output / "initial_attempt/cases" / (case["case_id"] + ".json")).is_file()
            repaired_cases += 1
        mask = np.isnan(missing)
        with np.load(prediction_path, allow_pickle=False) as saved:
            values = saved["missing_values"]
            assert tuple(saved["shape"]) == truth.shape and values.shape == (int(mask.sum()),)
            assert np.isfinite(values).all()
            prediction = missing.astype(values.dtype)
            prediction[mask] = values
        assert np.isfinite(truth).all() and np.isfinite(prediction).all()
        original_nrmse, original_corr2 = scorers[case["suite"]]
        expected_nrmse = float(original_nrmse(truth, prediction, mask))
        expected_corr2 = float(original_corr2(truth, prediction, mask))
        conventional = [r2_score(truth[mask[:, column], column].astype(np.float64),
                                 prediction[mask[:, column], column].astype(np.float64))
                        for column in range(truth.shape[1]) if mask[:, column].sum() >= 2]
        expected = {"nrmse_original": expected_nrmse, "corr2_original": expected_corr2,
                    "r2_columnwise": float(np.mean(conventional))}
        for metric, value in expected.items():
            difference = abs(value - row[metric])
            maximum_difference[metric] = max(maximum_difference[metric], difference)
            relative_tolerance, absolute_tolerance = (5e-7, 1e-7) if metric == "nrmse_original" else (1e-10, 1e-10)
            assert np.isclose(value, row[metric], rtol=relative_tolerance, atol=absolute_tolerance), (case["case_id"], metric, value, row[metric])
        counts[case["suite"]] += 1
        dataset_rows[case["suite"], case["dataset"]].append(row)
        predictions_bytes += prediction_path.stat().st_size
        artifacts[str(prediction_path.relative_to(output))] = row["prediction_sha256"]
        artifacts[str(result_path.relative_to(output))] = sha256(result_path)
    with (output / "per_case_metrics.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {row["case_id"] for row in csv_rows} == {case["case_id"] for case in protocol["cases"]}
    assert len(csv_rows) == 2262
    assert repaired_cases == 151
    with (output / "initial_attempt/per_case_metrics.csv").open(newline="") as handle:
        initial_rows = list(csv.DictReader(handle))
    final_rows = {row["case_id"]: row for row in csv_rows}
    preserved_cases = 0
    for initial in initial_rows:
        if initial["status"] == "ok":
            final = final_rows[initial["case_id"]]
            assert initial["prediction_sha256"] == final["prediction_sha256"]
            assert all(initial[metric] == final[metric] for metric in METRICS)
            preserved_cases += 1
    assert preserved_cases == 2111
    for row in csv_rows:
        recorded = json.loads((output / "cases" / (row["case_id"] + ".json")).read_text())
        for metric in METRICS:
            assert float(row[metric]) == recorded[metric]
    summaries = []
    for (suite, dataset), rows in sorted(dataset_rows.items()):
        summary = {"suite": suite, "dataset": dataset, "cases": len(rows)}
        summary.update({f"mean_{metric}": float(np.mean([row[metric] for row in rows])) for metric in METRICS})
        summary["inference_seconds"] = sum(row["inference_seconds"] for row in rows)
        summaries.append(summary)
    with (output / "per_dataset_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "summary_metrics.csv").open(newline="") as handle:
        for aggregate in csv.DictReader(handle):
            if aggregate["level"] != "overall":
                continue
            dataset_group = [row for row in summaries if row["suite"] == aggregate["suite"]]
            for metric in METRICS:
                expected = float(np.mean([row[f"mean_{metric}"] for row in dataset_group]))
                assert np.isclose(float(aggregate[f"mean_{metric}"]), expected, rtol=1e-12, atol=1e-12)
    sections = ["# Per-dataset imputation results\n",
                "Means over the approved frozen settings/repeats. NRMSE and Corr² preserve the original scorers; "
                "R² is conventional 1 − SSE/SST, averaged over columns.\n",
                "| Suite | Dataset | Cases | NRMSE ↓ | Original Corr² ↑ | Conventional R² ↑ |",
                "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        sections.append(f"| {row['suite']} | {row['dataset']} | {row['cases']} | {row['mean_nrmse_original']:.6f} | "
                        f"{row['mean_corr2_original']:.6f} | {row['mean_r2_columnwise']:.6f} |")
    (output / "PER_DATASET.md").write_text("\n".join(sections) + "\n")
    for filename in ("manifest.json", "status.json", "per_case_metrics.csv", "summary_metrics.csv", "per_dataset_metrics.csv", "RESULTS.md", "PER_DATASET.md"):
        artifacts[filename] = sha256(output / filename)
    for path in sorted((output / "benchmark_selection").iterdir()):
        if path.is_file():
            artifacts[str(path.relative_to(output))] = sha256(path)
    write_json(output / "artifact_sha256.json", artifacts)
    report = {"status": "passed", "cases": dict(counts), "checkpoint_sha256": protocol["checkpoint_sha256"],
              "input_hashes_verified": 2 * sum(counts.values()), "prediction_hashes_verified": sum(counts.values()),
              "maximum_independent_score_difference": dict(maximum_difference),
              "nrmse_comparison_tolerance": {"relative": 5e-7, "absolute": 1e-7,
                  "reason": "Original function averages a float32 array of column scores; evaluator averages the same scores in float64"},
              "constant_column_cases_repaired": repaired_cases, "initial_successes_preserved": preserved_cases,
              "total_evaluation_seconds": status["elapsed_seconds"] + json.loads((output / "initial_attempt/status.json").read_text())["elapsed_seconds"],
              "prediction_bytes": predictions_bytes, "artifact_hashes": len(artifacts),
              "source_metric_functions_sha256": {str(path): sha256(path) for pair in paths.values() for path in pair},
              "r2_verification": "sklearn.metrics.r2_score per column; mean across columns",
              "summary_verification": "CSV equals case JSON; overall summaries independently recomputed from dataset means"}
    write_json(output / "completion_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output.resolve())
