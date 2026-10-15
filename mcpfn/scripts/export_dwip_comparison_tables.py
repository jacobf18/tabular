"""Export the benchmark-selection tables reported in the evaluation conversation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918"
DATASETS = {"mteb": "MTEB", "mmlu": "MMLU", "merged": "Merged", "benchpress": "BenchPress"}
METHODS = ("entropy", "mi", "random")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_table(rows, models, summary=False):
    labels = {"new": "New (selected 98k)", "original": "Original DWIP", "mean64": "Previous mean64",
              "gaussian": "Gaussian", "tabimpute_v2": "TabImpute v2 (paper)"}
    columns = ["Dataset", "Selection"] if summary else ["Budget"]
    lines = ["| " + " | ".join(columns + [labels[model] for model in models]) + " |",
             "|" + "|".join(["---"] * len(columns) + ["---:"] * len(models)) + "|"]
    for row in rows:
        cells = [DATASETS[row["dataset"]], row["method"]] if summary else [str(row["budget"])]
        best = max(row[model] for model in models)
        for model in models:
            value = f"{row[model]:.4f}"
            if summary and row[model] == best:
                value = f"**{value}**"
            if row[f"{model}_folds"] != 10:
                value += f" ({row[f'{model}_folds']} folds)"
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    source = QUEUE / "full_benchmark_r2.csv"
    with source.open(newline="") as handle:
        references = {(row["reference"], row["dataset"], row["method"], int(row["budget"])): row
                      for row in csv.DictReader(handle)}
    assert len(references) == 360
    sources = {str(source): file_hash(source)}
    rows = []
    for dataset in DATASETS:
        logs = ROOT / "benchmark-selection/logs"
        gaussian_path = logs / ("benchpress_results.npz" if dataset == "benchpress" else f"entropy_vs_mi_{dataset}.npz")
        v2_path = logs / f"tabimpute_v2_{dataset}.npz"
        sources.update({str(path): file_hash(path) for path in (gaussian_path, v2_path)})
        with np.load(gaussian_path, allow_pickle=False) as gaussian, np.load(v2_path, allow_pickle=False) as v2:
            assert np.array_equal(gaussian["benchmark_names"], v2["benchmark_names"])
            for method in METHODS:
                gaussian_key = f"emi_{method}_r2" if dataset == "benchpress" else f"{method}_r2"
                gaussian_values, v2_values = gaussian[gaussian_key], v2[f"{method}_r2"]
                assert gaussian_values.shape == v2_values.shape == (10, 15)
                assert not np.isinf(gaussian_values).any() and np.isfinite(v2_values).all()
                for budget in range(1, 16):
                    original = references["original", dataset, method, budget]
                    mean64 = references["primary_mean64", dataset, method, budget]
                    assert original["selected_r2"] == mean64["selected_r2"]
                    rows.append({"dataset": dataset, "method": method, "budget": budget,
                                 "new": float(original["selected_r2"]), "original": float(original["baseline_r2"]),
                                 "mean64": float(mean64["baseline_r2"]),
                                 "gaussian": float(np.nanmean(gaussian_values[:, budget - 1])),
                                 "tabimpute_v2": float(v2_values[:, budget - 1].mean()),
                                 "new_folds": int(original["fold_count"]), "original_folds": int(original["fold_count"]),
                                 "mean64_folds": int(mean64["fold_count"]),
                                 "gaussian_folds": int(np.isfinite(gaussian_values[:, budget - 1]).sum()),
                                 "tabimpute_v2_folds": 10})
    for filename, selected in [("full_r2.csv", rows), ("summary_r2_5_10_15.csv", [row for row in rows if row["budget"] in (5, 10, 15)])]:
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    intro = ("Mean benchmark-selection R² over ten folds; higher is better. Values rounded to four decimals.\n\n"
             "New is the synthetic-validation-selected step-98000 checkpoint from the completed 128000-step run. "
             "Original is the DWIP anchor; mean64 is the preceding mean-Huber64k model. "
             "Gaussian is the conditional-mean baseline; TabImpute v2 uses the paper checkpoint. "
             "Missing Gaussian folds are excluded with their available fold counts shown; those entries are not fully paired. "
             "These are saved evaluations, not new runs.\n")
    for filename, models in [("full_new_original_mean64.md", ("original", "mean64", "new")),
                             ("full_new_original_gaussian_tabimpute_v2.md", ("new", "original", "gaussian", "tabimpute_v2"))]:
        sections = ["# Full benchmark-selection R²\n", intro]
        for dataset in DATASETS:
            for method in METHODS:
                selected = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
                sections.extend([f"\n## {DATASETS[dataset]} — {method}\n", render_table(selected, models)])
        (output / filename).write_text("\n".join(sections) + "\n")
    sections = ["# Summary benchmark-selection R²\n", intro, "Bold denotes the best model within each row.\n"]
    for budget in (5, 10, 15):
        sections.extend([f"\n## R² at budget {budget}\n", render_table([row for row in rows if row["budget"] == budget],
                         ("new", "original", "gaussian", "tabimpute_v2"), summary=True)])
    (output / "summary_r2_5_10_15.md").write_text("\n".join(sections) + "\n")
    improvements = []
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        improvements.append({"dataset": dataset, "settings": len(selected),
                             "new_better_than_original": sum(row["new"] > row["original"] for row in selected),
                             "new_better_than_mean64": sum(row["new"] > row["mean64"] for row in selected)})
    with (output / "improvement_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(improvements[0]))
        writer.writeheader()
        writer.writerows(improvements)
    counts_table = ["# New-model improvement counts\n",
                    "All 180 dataset/method/budget settings, compared at unrounded precision. "
                    "These counts are descriptive, not significance tests.\n",
                    "| Dataset | Better than original | Better than mean64 |", "|---|---:|---:|"]
    for row in improvements:
        counts_table.append(f"| {DATASETS[row['dataset']]} | {row['new_better_than_original']} / {row['settings']} | "
                            f"{row['new_better_than_mean64']} / {row['settings']} |")
    counts_table.append(f"| Total | {sum(row['new_better_than_original'] for row in improvements)} / {len(rows)} | "
                        f"{sum(row['new_better_than_mean64'] for row in improvements)} / {len(rows)} |")
    (output / "improvement_counts.md").write_text("\n".join(counts_table) + "\n")
    (output / "provenance.json").write_text(json.dumps({"sources_sha256": sources, "settings": len(rows),
                                                       "fold_counts": "Explicit per model and setting in CSV", "improvements": improvements}, indent=2) + "\n")
    print(f"Exported {len(rows)} settings and all previous table variants to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args().output)
