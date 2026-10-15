"""Audit the baseline predictions and export matched, dataset-balanced NRMSE tables."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from audit_dwip_missbench_uci import load_original
from evaluate_dwip_missbench_uci import REPO, sha256, write_json
from evaluate_imputation_baseline import ROOT, write_csv


MODELS = ("new", "tabimpute_v2", "dwip_v2_2")
LABELS = {"new": "New (selected 98k)", "tabimpute_v2": "TabImpute v2 (paper)", "dwip_v2_2": "DWIP v2-2 (original)"}


def aggregate(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result = []
    for group_key, cases in sorted(groups.items()):
        summary = dict(zip(fields, group_key))
        summary.update({"cases": len(cases), "datasets": len({row["dataset"] for row in cases})})
        for model in MODELS:
            datasets = defaultdict(list)
            for case in cases:
                datasets[case["dataset"]].append(case[model])
            summary[model] = float(np.mean([np.mean(values) for values in datasets.values()]))
        result.append(summary)
    return result


def markdown_table(rows, fields):
    headers = [field.title() for field in fields] + ["Cases"] + [LABELS[model] for model in MODELS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(fields) + ["---:"] * 4) + "|"]
    for row in rows:
        best = min(row[model] for model in MODELS)
        cells = [str(row[field]) for field in fields] + [str(row["cases"])]
        for model in MODELS:
            number = f"{row[model]:.6f}"
            cells.append(f"**{number}**" if row[model] == best else number)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare(root):
    new_manifest = json.loads((root / "manifest.json").read_text())
    cases = new_manifest["protocol"]["cases"]
    assert len(cases) == 2262
    folders = {"new": root, **{model: root / "baselines" / model for model in MODELS if model != "new"}}
    hashes = {}
    checkpoint_hashes = {}
    for model, folder in folders.items():
        status = json.loads((folder / "status.json").read_text())
        assert status["state"] == "complete" and status["successful"] == 2262 and status["failed"] == 0
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["protocol"]["cases"] == cases
        spec = manifest["protocol"] if model == "new" else manifest["protocol"]["model"]
        assert sha256(Path(spec["checkpoint"])) == spec["checkpoint_sha256"]
        checkpoint_hashes[model] = spec["checkpoint_sha256"]
        for filename in ("manifest.json", "status.json", "per_case_metrics.csv"):
            path = folder / filename
            hashes[str(path.relative_to(root))] = sha256(path)
    originals = {"missbench": load_original(REPO / "benchmark/plot_options.py", "compute_normalized_rmse_columnwise"),
                 "uci": load_original(REPO / "benchmark/uci/plot_uci_negative_rmse.py", "compute_normalized_rmse_columnwise")}
    maximum_error = defaultdict(float)
    source_hashes = {}
    comparison = []
    for case in cases:
        source = Path(case["source_directory"])
        missing_path, truth_path = source / "missing.npy", source / "true.npy"
        input_hashes = {"missing_sha256": sha256(missing_path), "true_sha256": sha256(truth_path)}
        source_hashes[str(missing_path)] = input_hashes["missing_sha256"]
        source_hashes[str(truth_path)] = input_hashes["true_sha256"]
        missing, truth = np.load(missing_path, allow_pickle=False), np.load(truth_path, allow_pickle=False)
        mask = np.isnan(missing)
        compared = {key: case[key] for key in ("case_id", "suite", "dataset", "pattern", "missingness", "repeat")}
        for model, folder in folders.items():
            result_path = folder / "cases" / (case["case_id"] + ".json")
            row = json.loads(result_path.read_text())
            assert row["status"] == "ok" and all(row[key] == value for key, value in input_hashes.items())
            prediction_path = folder / row["prediction_path"]
            assert sha256(prediction_path) == row["prediction_sha256"]
            with np.load(prediction_path, allow_pickle=False) as saved:
                assert tuple(saved["shape"]) == truth.shape
                values = saved["missing_values"]
                assert values.shape == (int(mask.sum()),) and np.isfinite(values).all()
                prediction = missing.astype(values.dtype)
                prediction[mask] = values
            expected = float(originals[case["suite"]](truth, prediction, mask))
            actual = float(row["nrmse_original"])
            assert np.isfinite(actual) and np.isclose(expected, actual, rtol=5e-7, atol=1e-7), (case["case_id"], model, expected, actual)
            maximum_error[model] = max(maximum_error[model], abs(expected - actual))
            compared[model] = actual
            hashes[str(result_path.relative_to(root))] = sha256(result_path)
            hashes[str(prediction_path.relative_to(root))] = row["prediction_sha256"]
        comparison.append(compared)
    summaries = aggregate(comparison, ("suite",))
    datasets = aggregate(comparison, ("suite", "dataset"))
    patterns = aggregate(comparison, ("suite", "pattern"))
    pattern_rates = aggregate(comparison, ("suite", "pattern", "missingness"))
    for name, rows in (("nrmse_comparison.csv", summaries), ("nrmse_comparison_per_dataset.csv", datasets),
                       ("nrmse_comparison_per_pattern.csv", patterns), ("nrmse_comparison_per_pattern_rate.csv", pattern_rates),
                       ("nrmse_comparison_per_case.csv", comparison)):
        write_csv(root / name, rows)
        hashes[name] = sha256(root / name)
    sections = ["# Matched imputation NRMSE comparison\n",
                "Lower is better. Every model is evaluated on the identical 462 MissBench cases and 1800 UCI cases. "
                "No MissBench MCAR rate sweep is included. Bold marks the best value in each row.\n",
                "New: selected step-98000 checkpoint from the completed 128000-step run. "
                "DWIP v2-2: the exact original anchor used in the preceding comparisons, not another DWIP trial. "
                "TabImpute v2: the paper checkpoint `tabimputev2.ckpt`, not `masters_mcar` or a test-time-trained variant.\n",
                "NRMSE matches the original scorer: average columnwise masked-cell RMSE divided by masked ground-truth standard deviation. "
                "Case scores are averaged within each dataset, then datasets receive equal weight. "
                "All models use the same observed-data-only normalization, with scale=1 for exactly constant UCI columns. "
                "No rows/columns are truncated and no permutation ensemble is used. "
                "The new and DWIP models use predictive means with FP32 weights/BF16 autocast; "
                "the paper v2 wrapper retains its native predictive median and BF16 weights. "
                "These settings are fixed, not selected on benchmark scores.\n",
                "## Overall\n", markdown_table(summaries, ("suite",)), "\n## By mechanism\n",
                markdown_table(patterns, ("suite", "pattern")), "\n## UCI by mechanism and missingness\n",
                markdown_table([row for row in pattern_rates if row["suite"] == "uci"], ("pattern", "missingness")),
                "\n## By dataset\n", markdown_table(datasets, ("suite", "dataset")),
                "\nFull-precision overall, per-dataset, per-mechanism/rate, and per-case CSV files accompany this table. "
                "Checkpoint identities, original scorer parity, input/prediction integrity, and full coverage are recorded in "
                "`baseline_comparison_audit.json`. Original newest-model results and their audit are preserved unchanged."]
    (root / "NRMSE_COMPARISON.md").write_text("\n".join(sections) + "\n")
    hashes["NRMSE_COMPARISON.md"] = sha256(root / "NRMSE_COMPARISON.md")
    write_json(root / "baseline_comparison_artifact_sha256.json", hashes)
    write_json(root / "baseline_comparison_input_sha256.json", source_hashes)
    audit = {"status": "passed", "case_count_per_model": len(cases), "models": list(MODELS),
             "checkpoint_sha256": checkpoint_hashes, "input_files_verified": len(source_hashes),
             "predictions_verified": len(cases) * len(MODELS), "maximum_original_nrmse_difference": dict(maximum_error),
             "nrmse_tolerance": {"relative": 5e-7, "absolute": 1e-7, "reason": "Float32 versus float64 averaging of the same columnwise scores"},
             "aggregation": "Mean of dataset means over the same complete case grid", "overall": summaries,
             "script_sha256": sha256(Path(__file__))}
    write_json(root / "baseline_comparison_audit.json", audit)
    print(markdown_table(summaries, ("suite",)))
    print("All 6786 saved predictions independently rescored and verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    compare(parser.parse_args().root.resolve())
