"""Independently audit orientation predictions and export benchmark tables."""

import argparse
import csv
from pathlib import Path

import numpy as np

from audit_dwip_missbench_uci import load_original
from compare_recsys_checkpoint import audit_scores, reference_paths
from evaluate_orientation_ensemble import BASE_VARIANTS, OUTPUT, PARENT, VARIANTS, campaign, summarize_tabular
from recsys_evaluation_common import sha256, write_json


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_average(values):
    for name in BASE_VARIANTS:
        np.testing.assert_array_equal(values[name + "_average"], (values[name] + values[name + "_transpose"]) / 2)


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + " |" for row in rows], ""]


def audit(output):
    manifest = campaign.read_json(output / "manifest.json")
    assert manifest["pilot_limit"] is None
    for source, expected in manifest["input_sha256"].items():
        assert sha256(Path(source)) == expected, source
    for source, expected in manifest["source_sha256"].items():
        snapshot = output / "source_snapshot" / Path(source).relative_to(campaign.REPO.parent)
        assert sha256(snapshot) == expected, source
    for stage in ("tabular", "llm", "recsys"):
        assert campaign.read_json(output / stage / "status.json")["state"] == "complete", stage
    scorers = {suite: load_original(campaign.REPO / path, "compute_normalized_rmse_columnwise") for suite, path in (
        ("missbench", "benchmark/plot_options.py"), ("uci", "benchmark/uci/plot_uci_negative_rmse.py"))}
    records = []
    for case in manifest["cases"]:
        record = campaign.read_json(output / "tabular/cases" / (case["case_id"] + ".json"))
        path = output / "tabular/predictions" / (case["case_id"] + ".npz")
        assert sha256(path) == record["prediction_sha256"]
        source = Path(case["source_directory"])
        assert sha256(source / "missing.npy") == record["missing_sha256"]
        assert sha256(source / "true.npy") == record["true_sha256"]
        missing, truth = (np.load(source / name, allow_pickle=False) for name in ("missing.npy", "true.npy"))
        mask = np.isnan(missing)
        with np.load(path, allow_pickle=False) as stored:
            check_average(stored)
            for variant in VARIANTS:
                prediction = missing.astype(stored[variant].dtype)
                prediction[mask] = stored[variant]
                np.testing.assert_allclose(scorers[case["suite"]](truth, prediction, mask),
                                           record["nrmse"][variant], atol=1e-5, rtol=1e-5)
        records.append(record)
    assert len(records) == 2262
    summarize_tabular(output / "tabular", records, manifest["cases"])
    recsys_rows = read_rows(output / "recsys/summary.csv")
    for dataset in ("100K", "netflix", "1M", "10M"):
        paths = reference_paths(dataset, "global")
        record = campaign.read_json(output / "recsys" / dataset / "audit.json")
        assert sha256(paths["split"]) == record["split_sha256"]
        path = output / "recsys" / dataset / "query_predictions.npz"
        assert sha256(path) == record["prediction_sha256"]
        with np.load(paths["split"], allow_pickle=False) as split, np.load(path, allow_pickle=False) as stored:
            check_average(stored)
            for variant in VARIANTS:
                row = next(row for row in recsys_rows if row["dataset"] == paths["label"] and row["variant"] == variant)
                audit_scores(split, stored[variant], {metric: float(row[metric]) for metric in ("RMSE", "MAE")})
    llm_rows = read_rows(output / "llm/summary.csv")
    llm_lookup = {(row['dataset'], row['variant'], row['method'], row['metric'], int(row['budget'])): float(row['value']) for row in llm_rows}
    assert len(llm_lookup) == 4 * 9 * 3 * 2 * 15
    for dataset in campaign.DATASETS:
        record = campaign.read_json(output / "llm" / f"{dataset}_audit.json")
        assert sha256(output / "llm" / f"{dataset}.npz") == record["panel_sha256"]
        assert sha256(output / "llm" / f"{dataset}_routing.jsonl") == record["routing_sha256"]
        with np.load(output / "llm" / f"{dataset}.npz", allow_pickle=True) as current, np.load(PARENT / "llm" / f"{dataset}.npz", allow_pickle=True) as previous:
            for variant in VARIANTS:
                for method in campaign.METHODS:
                    for metric in ("r2", "rmse"):
                        values = current[f"{variant}_{method}_{metric}"]
                        assert values.shape == (10, 15) and np.isfinite(values).all()
                        if variant in BASE_VARIANTS:
                            np.testing.assert_array_equal(values, previous[f"{variant}_{method}_{metric}"])
                        for budget in range(1, 16):
                            assert values[:, budget - 1].mean() == llm_lookup[dataset, variant, method, metric, budget]
    displayed = ("general", "general_transpose", "general_average", "observed_blend", "observed_blend_transpose", "observed_blend_average")
    headers = ["MatrixFormer", "MF transpose", "MF average", "MatrixFormer+", "MF+ transpose", "MF+ average"]
    lines = ["# MatrixFormer orientation evaluations", "",
             "Same frozen masks/splits, checkpoints and scoring. Transpose predictions are mapped back before scoring.",
             "Average means the arithmetic mean of original and transposed predictions, not the mean of scores.",
             "Each orientation refits preprocessing and the observed-only blend; no test-based orientation selection.",
             "Recommender row-residual scaling acts on items in the transposed recipe. No new causal or R-method reruns.", "",
             "## Tabular NRMSE (lower is better)", ""]
    tabular_rows = read_rows(output / "tabular/summary.csv")
    lines += table(["Suite", "Pattern", *headers], [[row['suite'], row['pattern'], *[float(row[name]) for name in displayed]] for row in tabular_rows])
    lines += ["## Rating errors (user-macro; lower is better)", ""]
    recsys_lookup = {(row['dataset'], row['variant']): row for row in recsys_rows}
    lines += table(["Dataset", "Metric", *headers], [[dataset, metric, *[float(recsys_lookup[dataset, name][metric]) for name in displayed]]
                    for dataset in ('MovieLens 100K', 'Netflix small', 'MovieLens 1M', 'MovieLens 10M') for metric in ('RMSE', 'MAE')])
    lines += ["## LLM R² (higher is better)", "", "All 180 settings, ten folds each.", ""]
    lines += table(["Dataset", "Selection", "Budget", *headers], [[dataset, method, budget, *[llm_lookup[dataset, name, method, 'r2', budget] for name in displayed]]
                    for dataset in campaign.DATASETS for method in campaign.METHODS for budget in range(1, 16)])
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    artifacts = ['manifest.json', 'RESULTS.md', 'tabular/per_case.csv', 'tabular/summary.csv',
                 'recsys/summary.csv', 'llm/summary.csv', 'llm/per_fold.csv', 'llm/orientation_audit.json']
    artifacts += [f'{stage}/status.json' for stage in ('tabular', 'llm', 'recsys')]
    artifacts += [f'recsys/{dataset}/audit.json' for dataset in ('100K', 'netflix', '1M', '10M')]
    artifacts += [f'llm/{dataset}_audit.json' for dataset in campaign.DATASETS]
    write_json(output / 'final_audit.json', {'state': 'complete', 'tabular_cases': len(records), 'llm_settings': 180,
               'rating_datasets': 4, 'original_predictions_or_panels_preserved': True,
               'tabular_and_recsys_prediction_averages_verified': True,
               'llm_original_panel_parity_verified': True,
               'sha256': {name: sha256(output / name) for name in artifacts}})
    print('Complete orientation evaluation audited.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    audit(parser.parse_args().output)
