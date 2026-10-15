"""Summarize paired captured predictions without changing official benchmark scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json


def paired_error_summary(before, after, test_indices):
    for key in ("target", "evaluation_mask", "mean", "scale"):
        if not np.array_equal(before[key], after[key], equal_nan=True):
            raise ValueError(f"Paired prediction inputs differ: {key}")
    target, mask, mean, scale = (before[key] for key in ("target", "evaluation_mask", "mean", "scale"))
    if len(test_indices) != len(target):
        raise ValueError("Test row identities do not match predictions")
    truth = np.clip((target - mean) / scale, -10, 10)
    errors = {}
    for label, data in (("original", before), ("candidate", after)):
        prediction = data["prediction"]
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid predictions")
        errors[label] = np.where(mask, (np.clip((prediction - mean) / scale, -10, 10) - truth) ** 2, 0.)
    delta = errors["candidate"] - errors["original"]
    total = float(delta.sum())
    rows = [{"test_row": index, "dataset_row": int(test_indices[index]),
             "n_eval": int(mask[index].sum()), "scored_sse_original": float(errors["original"][index].sum()),
             "scored_sse_candidate": float(errors["candidate"][index].sum()),
             "scored_sse_delta": float(delta[index].sum()),
             "fraction_of_net_sse_increase": float(delta[index].sum() / total) if total > 0 else None}
            for index in range(len(target))]
    rows.sort(key=lambda row: row["scored_sse_delta"], reverse=True)
    return {"scored_sse_original": float(errors["original"].sum()),
            "scored_sse_candidate": float(errors["candidate"].sum()), "scored_sse_delta": total,
            "rows_with_increased_error": int((delta.sum(1) > 0).sum()), "rows": rows}


def summarize_root(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if json.loads((root / "status.json").read_text()).get("cached_scores_reproduced") is not True:
        raise ValueError("Only successfully reproduced diagnostics may be interpreted")
    results = json.loads((root / "results.json").read_text())
    summary = {"dataset": manifest["dataset"], "fold": manifest["fold"],
               "manifest_sha256": sha256(root / "manifest.json"), "prediction_sha256": {}, "comparisons": {}}
    for budget in manifest["budgets"]:
        original_path = root / f"original_k{budget}_predictions.npz"
        summary["prediction_sha256"][original_path.name] = sha256(original_path)
        with np.load(original_path) as original:
            for label, reports in results.items():
                if label == "original":
                    continue
                candidate_path = root / f"{label}_k{budget}_predictions.npz"
                summary["prediction_sha256"][candidate_path.name] = sha256(candidate_path)
                with np.load(candidate_path) as candidate:
                    paired = paired_error_summary(original, candidate, manifest["test_indices"])
                    sources = sorted(key for key in original.files if key.startswith("source_"))
                    if sources != sorted(key for key in candidate.files if key.startswith("source_")):
                        raise ValueError("Paired source batches differ")
                    for key in sources:
                        if not np.array_equal(original[key], candidate[key], equal_nan=True):
                            raise ValueError("Paired imputer sources differ")
                    context_count = min(len(manifest["train_indices"]), 3800)
                    queries = np.concatenate([candidate[key][context_count:] for key in sources])
                    if len(queries) != len(manifest["test_indices"]):
                        raise ValueError("Captured query rows do not match test rows")
                    for row in paired["rows"]:
                        query = queries[row["test_row"]]
                        revealed = np.isfinite(query)
                        standardized = (query[revealed] - candidate["mean"][revealed]) / candidate["scale"][revealed]
                        row["revealed_count"] = int(revealed.sum())
                        row["revealed_standardized_values"] = [
                            {"column": int(column), "value": float(value)}
                            for column, value in zip(np.flatnonzero(revealed), standardized)]
                for variant, report in (("original", results["original"][str(budget)]), ("candidate", reports[str(budget)])):
                    if not np.isclose(paired[f"scored_sse_{variant}"], report["scored_sse"], rtol=1e-10, atol=1e-10):
                        raise ValueError("Paired SSE disagrees with reproduced decomposition")
                    paired[f"metrics_{variant}"] = report["metrics"]
                    paired[f"clipped_predictions_{variant}"] = sum(column["predictions_clipped"] for column in report["columns"])
                    paired[f"clipped_targets_{variant}"] = sum(column["targets_clipped"] for column in report["columns"])
                    paired[f"raw_rmse_{variant}"] = report["raw_rmse_report_only"]
                summary["comparisons"][f"{label}_k{budget}"] = paired
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=REPO / "runs/information_flow_diagnostics/dwip_fold_regression_summary_20260916.json")
    args = parser.parse_args()
    roots = args.roots or [REPO / f"runs/information_flow_diagnostics/dwip_{dataset}_fold{fold}_regression_20260916_v2"
                          for dataset, fold in (("mteb", 8), ("merged", 5), ("merged", 8))]
    payload = {"purpose": "Post-hoc paired error attribution only; no row exclusions or new benchmark scores",
               "cautions": ["Fractions can exceed one because other rows improve.",
                            "Extreme revealed values do not establish incorrect data or a causal mechanism.",
                            "These results must not set training priors or select checkpoints."],
               "source_sha256": sha256(Path(__file__)), "folds": [summarize_root(root) for root in roots]}
    write_json(args.output, payload)
    for fold in payload["folds"]:
        for label in ("48k_seed1", "point2_48k"):
            for budget in (10, 15):
                paired = fold["comparisons"][f"{label}_k{budget}"]
                row = paired["rows"][0]
                print(fold["dataset"], fold["fold"], label, budget, "delta_sse", paired["scored_sse_delta"],
                      "top_dataset_row", row["dataset_row"], "fraction", row["fraction_of_net_sse_increase"])


if __name__ == "__main__":
    main()
