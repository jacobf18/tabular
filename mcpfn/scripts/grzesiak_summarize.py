"""Auditable summaries of author artifacts and independently generated evaluations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grzesiak_benchmark import WORK, sha256, write_json


CASE_KEYS = ["set_id", "mechanism", "ratio", "rep"]


def rank_scores(frame, case_keys, score="score"):
    frame = frame.copy()
    groups = frame.groupby(case_keys, dropna=False)[score]
    frame["rank"] = groups.rank(method="average", ascending=True)
    failures = groups.transform("count") + 1
    frame["rank"] = frame["rank"].fillna(failures)
    return frame


def reference(work):
    root = work / "reference"
    panels = {"numerical_artificial": ("numerical_per_case.csv", "energy_std"),
              "mixed_artificial": ("mixed_per_case.csv", "energy_std"),
              "numerical_natural": ("natural_numerical_per_case.csv", "IScore"),
              "mixed_natural_common_methods": ("natural_per_case.csv", "IScore")}
    all_rankings, all_cases, conflicts = [], [], []
    for panel, (filename, measure) in panels.items():
        original = pd.read_csv(root / filename)
        frame = original[original.measure.eq(measure)].copy()
        if panel == "mixed_natural_common_methods":
            names = original.loc[original["case"].eq("incomplete_categorical"), "set_id"].unique()
            frame = frame[frame.set_id.isin(names)]
        keys = ["method", *CASE_KEYS]
        for _, group in frame.groupby(keys, dropna=False):
            finite = group.score.dropna().unique()
            if len(finite) > 1:
                raise ValueError(f"Conflicting recovered scores in {panel}: {group[keys].iloc[0].to_dict()}")
            if len(group) > 1:
                conflicts.append({"panel": panel, "method": str(group.method.iloc[0]), "dataset": str(group.set_id.iloc[0]),
                                  "rows": len(group), "resolution": "prefer the unique finite score; retain missing if none"})
        frame = frame.sort_values("score", na_position="last").drop_duplicates(keys)
        for aggregation in ("per_replicate", "average_replicates_before_ranking"):
            selected = frame.copy()
            case_keys = CASE_KEYS
            if aggregation == "average_replicates_before_ranking":
                case_keys = CASE_KEYS[:-1]
                selected = selected.groupby(["method", *case_keys], dropna=False, as_index=False).agg(score=("score", "mean"))
            selected = rank_scores(selected, case_keys)
            selected["panel"], selected["aggregation"] = panel, aggregation
            all_cases.append(selected)
            summary = selected.groupby("method", as_index=False).agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"),
                mean_score=("score", "mean"), valid_cases=("score", "count"), cases=("score", "size"), datasets=("set_id", "nunique"))
            summary["panel"], summary["aggregation"], summary["metric"] = panel, aggregation, measure
            summary["source_status"] = "recomputed_from_author_artifacts_not_rerun"
            all_rankings.append(summary)
    rankings = pd.concat(all_rankings, ignore_index=True)
    rankings.to_csv(root / "rankings.csv", index=False)
    pd.concat(all_cases, ignore_index=True).to_csv(root / "ranked_cases.csv", index=False)
    numerical = pd.read_csv(root / "numerical_per_case.csv")
    nrmse = numerical[numerical.measure.eq("nrmse")].copy()
    for pattern in ("mcar", "mar"):
        subset = nrmse[nrmse.mechanism.eq(pattern)]
        table = subset.groupby(["method", "set_id"], as_index=False).score.mean().groupby("method").score.mean()
        nrmse.loc[nrmse.mechanism.eq(pattern), "dataset_balanced_method_mean"] = nrmse.loc[nrmse.mechanism.eq(pattern), "method"].map(table)
    nrmse.to_csv(root / "numerical_nrmse_per_case.csv", index=False)
    write_json(root / "ranking_audit.json", {
        "duplicate_resolutions": conflicts,
        "failure_rank": "number of finite scores in the same case + 1; average ranks for ties",
        "paper_vs_plot": "Paper describes per-replicate ranks; final plotting functions also contain replicate-averaged ranking. Both are exported.",
        "reference_scope": "Recovered numerical/mixed artificial panels and natural panels. Natural mixed uses the final common-method artifact, not a claimed exact reconstruction of every paper figure.",
        "historical_masks": "Not released in this snapshot; fresh model results must never be ranked alongside these historical scores.",
        "input_sha256": {filename: sha256(root / filename) for filename, _ in panels.values()}})
    print(rankings[rankings.aggregation.eq("per_replicate")].sort_values(["panel", "mean_rank"]).groupby("panel").head(6).to_string(index=False))


def dataset_balanced(frame, metric):
    return float(frame.groupby("dataset")[metric].mean().mean()) if len(frame) else None


def summarize_evaluations(work):
    paths = sorted((work / "evaluations").rglob("*.json"))
    if not paths:
        return
    frame = pd.DataFrame([json.loads(path.read_text()) for path in paths])
    output = work / "summaries"
    output.mkdir(exist_ok=True)
    frame.to_csv(output / "per_case.csv", index=False)
    prior = pd.read_csv("/root/MatrixFormer/results/imputation_nrmse_per_case.csv")
    prior = prior[prior.method.isin(["matrixformer", "column_mean"])].copy()
    prior["source_status"] = "previous_frozen_evaluation"
    existing = set(zip(frame.case_id, frame.method))
    prior = prior[[key not in existing for key in zip(prior.case_id, prior.method)]]
    combined = pd.concat([frame, prior], ignore_index=True)
    combined.to_csv(output / "per_case_with_existing_anchors.csv", index=False)
    rows = []
    for (suite, pattern, method), group in combined.groupby(["suite", "pattern", "method"]):
        valid = group[group.status.eq("ok") & np.isfinite(group.nrmse)]
        model = combined[(combined.suite.eq(suite)) & combined.pattern.eq(pattern) & combined.method.eq("matrixformer") & combined.status.eq("ok")]
        paired = valid.drop(columns="matrixformer_nrmse", errors="ignore").merge(
            model[["case_id", "nrmse"]].rename(columns={"nrmse": "matrixformer_nrmse"}), on="case_id", validate="one_to_one")
        row = {"suite": suite, "pattern": pattern, "method": method, "attempted_cases": len(group),
               "valid_cases": len(valid), "datasets": valid.dataset.nunique(), "nrmse": dataset_balanced(valid, "nrmse"),
               "paired_cases": len(paired), "matrixformer_matched": dataset_balanced(paired, "matrixformer_nrmse"),
               "baseline_matched": dataset_balanced(paired, "nrmse")}
        if len(paired):
            paired["difference"] = paired.matrixformer_nrmse - paired.nrmse
            row["matrixformer_minus_baseline"] = dataset_balanced(paired, "difference")
        for metric in ("nrmse_upstream_pooled", "energy_distance_std", "imputation_seconds"):
            if metric in valid and valid[metric].notna().any():
                row[metric] = dataset_balanced(valid, metric)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "per_pattern.csv", index=False)
    fresh = frame[frame.suite.eq("grzesiak")].copy()
    if len(fresh) and "energy_statistic_std" in fresh:
        fresh.loc[~fresh.status.eq("ok"), "energy_statistic_std"] = np.nan
        counts = fresh.groupby("case_id").method.nunique()
        complete_cases = counts[counts.eq(fresh.method.nunique())].index
        fresh = fresh[fresh.case_id.isin(complete_cases)]
        ranked = rank_scores(fresh, ["case_id"], "energy_statistic_std")
        ranked.to_csv(output / "fresh_ranked_cases.csv", index=False)
        ranked.groupby(["method", "pattern"], as_index=False).agg(mean_rank=("rank", "mean"),
            median_rank=("rank", "median"), cases=("case_id", "size"), datasets=("dataset", "nunique")).to_csv(output / "fresh_energy_ranks.csv", index=False)
    write_json(output / "status.json", {"attempts": len(frame), "statuses": {str(key): int(value) for key, value in frame.status.value_counts().items()},
        "reference_scores_in_fresh_comparisons": False, "all_requested_runs_complete": all(
            json.loads(path.read_text()).get("state") == "finished_all_requested_attempts" for path in (work / "runs").glob("*.json"))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--reference", action="store_true")
    arguments = parser.parse_args()
    if arguments.reference:
        reference(arguments.work)
    summarize_evaluations(arguments.work)


if __name__ == "__main__":
    main()
