import ast
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import r2_score


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_dwip_missbench_uci import PATTERNS, discover_cases, score_predictions, standardize_uci, summarize


def original_function(path, name):
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[definition], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("suite", ["missbench", "uci"])
def test_scores_match_original_functions_and_conventional_r2(suite):
    rng = np.random.default_rng(2026)
    true = rng.normal(size=(30, 6)).astype(np.float32)
    true[:, 3] = 1
    true[:, 4] = 1
    mask = rng.uniform(size=true.shape) < 0.4
    prediction = true.astype(np.float64) + rng.normal(scale=0.2, size=true.shape)
    prediction[:, 4] = 1
    prediction[:, 5] = 1
    missing = np.where(mask, np.nan, true)
    result = score_predictions(true, prediction, missing, suite)
    nrmse_path = REPO / ("benchmark/plot_options.py" if suite == "missbench" else "benchmark/uci/plot_uci_negative_rmse.py")
    corr_path = REPO / ("benchmark/openml/plot_r_squared.py" if suite == "missbench" else "benchmark/uci/plot_uci_r_squared.py")
    assert result["nrmse_original"] == pytest.approx(original_function(nrmse_path, "compute_normalized_rmse_columnwise")(true, prediction, mask))
    assert result["corr2_original"] == pytest.approx(original_function(corr_path, "compute_r_squared")(true, prediction, mask))
    expected = np.mean([r2_score(true[mask[:, column], column].astype(np.float64), prediction[mask[:, column], column])
                        for column in range(true.shape[1])])
    assert result["r2_columnwise"] == pytest.approx(expected)
    assert result["nrmse_skipped_constant_columns"] == 1


def test_only_missing_cells_are_scored():
    true = np.arange(12, dtype=np.float64).reshape(6, 2)
    missing = true.copy()
    missing[:3] = np.nan
    prediction = true.copy()
    prediction[3:] += 1000
    result = score_predictions(true, prediction, missing, "missbench")
    assert result["rmse"] == 0
    assert result["r2_columnwise"] == 1
    assert result["missing_cells"] == 6


def test_nonfinite_predictions_are_rejected():
    true = np.ones((3, 2))
    with pytest.raises(ValueError, match="Nonfinite"):
        score_predictions(true, true * np.nan, true * np.nan, "uci")


def test_constant_uci_columns_preserve_missing_mask_and_observed_values():
    missing = np.array([[0.0, 1.0], [np.nan, 3.0], [0.0, np.nan]], dtype=np.float32)
    normalized, mean, scale, constant_columns = standardize_uci(missing)
    assert constant_columns == [0]
    assert scale[0] == 1
    assert np.array_equal(np.isnan(normalized), np.isnan(missing))
    assert np.allclose((normalized * scale + mean)[~np.isnan(missing)], missing[~np.isnan(missing)])
    assert np.isfinite(normalized[~np.isnan(missing)]).all()


def test_nonconstant_uci_preprocessing_is_unchanged():
    missing = np.array([[2.0, 1.0], [np.nan, 3.0], [4.0, np.nan]], dtype=np.float32)
    normalized, mean, scale, constant_columns = standardize_uci(missing)
    expected = (missing - np.nanmean(missing, axis=0)) / np.nanstd(missing, axis=0)
    assert not constant_columns
    assert np.array_equal(normalized, expected, equal_nan=True)


def test_manifest_scope_is_exact(tmp_path):
    for folder, count in (("openml", 42), ("uci", 12)):
        for dataset in range(count):
            base = tmp_path / folder / f"dataset-{dataset}"
            settings = [base / f"{pattern}_0.4" for pattern in PATTERNS] if folder == "openml" else [
                base / pattern / f"missingness-{rate / 10}" / f"repeat-{repeat}"
                for pattern in ("MCAR", "MAR", "MNAR") for rate in range(1, 6) for repeat in range(10)]
            for setting in settings:
                setting.mkdir(parents=True)
                (setting / "missing.npy").touch()
                (setting / "true.npy").touch()
    cases = discover_cases(tmp_path)
    missbench = [case for case in cases if case["suite"] == "missbench"]
    uci = [case for case in cases if case["suite"] == "uci"]
    assert len(missbench) == 462
    assert len(uci) == 1800
    assert len({case["case_id"] for case in cases}) == 2262
    assert {case["pattern"] for case in missbench} == set(PATTERNS)
    assert {case["missingness"] for case in missbench} == {0.4}
    assert {case["repeat"] for case in uci} == set(range(10))


def test_summary_weights_datasets_equally(tmp_path):
    from evaluate_dwip_missbench_uci import METRICS

    rows = []
    for dataset, values in [("first", [1.0, 3.0]), ("second", [10.0])]:
        for value in values:
            rows.append({"suite": "uci", "dataset": dataset, "pattern": "MCAR", "missingness": 0.4,
                         "status": "ok", **dict.fromkeys(METRICS, value)})
    summaries = summarize(tmp_path, rows, 3)
    overall = next(row for row in summaries if row["level"] == "overall")
    assert overall["mean_nrmse_original"] == 6
    assert overall["cases"] == 3
    assert (tmp_path / "RESULTS.md").is_file()


def test_export_retains_all_tables_and_missing_fold_counts(tmp_path, monkeypatch):
    import csv
    import export_dwip_comparison_tables as exporter

    logs = tmp_path / "benchmark-selection/logs"
    logs.mkdir(parents=True)
    queue = tmp_path / "queue"
    queue.mkdir()
    references = []
    for dataset in exporter.DATASETS:
        for reference, baseline in (("original", 0.4), ("primary_mean64", 0.6)):
            for method in exporter.METHODS:
                for budget in range(1, 16):
                    references.append({"reference": reference, "dataset": dataset, "method": method,
                                       "budget": budget, "baseline_r2": baseline, "selected_r2": 0.7, "fold_count": 10})
        gaussian = {f"{method}_r2": np.full((10, 15), 0.5) for method in exporter.METHODS}
        v2 = {f"{method}_r2": np.full((10, 15), 0.3) for method in exporter.METHODS}
        if dataset == "merged":
            gaussian["random_r2"][5, 0] = np.nan
        if dataset == "benchpress":
            gaussian["entropy_r2"][0, 0] = np.nan
            gaussian["mi_r2"][1, :2] = np.nan
            gaussian["random_r2"][[6, 7], 0] = np.nan
            gaussian = {f"emi_{key}": value for key, value in gaussian.items()}
        name = "benchpress_results.npz" if dataset == "benchpress" else f"entropy_vs_mi_{dataset}.npz"
        np.savez(logs / name, benchmark_names=np.array(["benchmark"]), **gaussian)
        np.savez(logs / f"tabimpute_v2_{dataset}.npz", benchmark_names=np.array(["benchmark"]), **v2)
    with (queue / "full_benchmark_r2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(references[0]))
        writer.writeheader()
        writer.writerows(references)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    monkeypatch.setattr(exporter, "QUEUE", queue)
    output = tmp_path / "tables"
    exporter.export(output)
    with (output / "full_r2.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 180
    assert sum(int(row["gaussian_folds"]) < 10 for row in rows) == 5
    assert all(int(row["gaussian_folds"]) == 10 for row in rows if int(row["budget"]) in (5, 10, 15))
    assert "(8 folds)" in (output / "full_new_original_gaussian_tabimpute_v2.md").read_text()
    assert "**0.7000**" in (output / "summary_r2_5_10_15.md").read_text()
    assert (output / "full_new_original_mean64.md").is_file()
