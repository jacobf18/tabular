"""Audit frozen router artifacts and write the complete comparison report."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from audit_dwip_missbench_uci import load_original
from evaluate_two_expert_router import DATASETS, METHODS, OUTPUT, PATTERNS, REPO, VARIANTS, read_json
from recsys_evaluation_common import sha256, user_rating_metrics, write_json


LABELS = ("General", "Specialist", "Observed router", "Observed blend", "Holdout router", "Holdout blend")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers, rows, best=None):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        if best is not None:
            optimum = (min if best == "min" else max)(value for value in row if isinstance(value, float))
        cells = []
        for value in row:
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            if best is not None and isinstance(value, float) and value == optimum:
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def audit_tabular(output, manifest):
    scorers = {suite: load_original(REPO / path, "compute_normalized_rmse_columnwise") for suite, path in (
        ("missbench", "benchmark/plot_options.py"), ("uci", "benchmark/uci/plot_uci_negative_rmse.py"))}
    records, max_difference = [], 0.
    timings = defaultdict(float)
    for case in manifest["cases"]:
        record = read_json(output / "cases" / (case["case_id"] + ".json"))
        source = Path(case["source_directory"])
        for key, filename in (("missing_sha256", "missing.npy"), ("true_sha256", "true.npy")):
            if sha256(source / filename) != record[key]:
                raise ValueError("Tabular data or mask changed")
        path = output / "predictions" / (case["case_id"] + ".npz")
        if sha256(path) != record["prediction_sha256"]:
            raise ValueError("Prediction artifact changed")
        missing, truth = (np.load(source / filename, allow_pickle=False) for filename in ("missing.npy", "true.npy"))
        mask = np.isnan(missing)
        with np.load(path, allow_pickle=False) as saved:
            if tuple(saved["shape"]) != missing.shape:
                raise ValueError("Prediction shape mismatch")
            for variant in VARIANTS:
                if saved[variant].shape != (int(mask.sum()),) or not np.isfinite(saved[variant]).all():
                    raise ValueError("Incomplete query coverage")
                prediction = missing.astype(saved[variant].dtype)
                prediction[mask] = saved[variant]
                score = float(scorers[case["suite"]](truth, prediction, mask))
                np.testing.assert_allclose(score, record["nrmse"][variant], atol=1e-5, rtol=1e-5)
                max_difference = max(max_difference, abs(score - record["nrmse"][variant]))
            for mode, decision in record["decisions"].items():
                np.testing.assert_array_equal(saved[f"{mode}_hard"], saved[decision["selected"]])
                weight = decision["specialist_weight"]
                if not 0 <= weight <= 1:
                    raise ValueError("Invalid convex weight")
                np.testing.assert_array_equal(saved[f"{mode}_blend"],
                    (1. - weight) * saved["general"] + weight * saved["specialist"])
        records.append(record)
        for key, seconds in record["seconds"].items():
            timings[key] += seconds
    summaries = read_csv(output / "summary.csv")
    for row in summaries:
        group = [record for record in records if record["suite"] == row["suite"] and record["pattern"] == row["pattern"]]
        if row["complete"] != "True" or len(group) != int(row["cases"]):
            raise ValueError("Incomplete tabular aggregate")
        for variant in VARIANTS:
            by_dataset = defaultdict(list)
            for record in group:
                by_dataset[record["dataset"]].append(record["nrmse"][variant])
            expected = np.mean([np.mean(values) for values in by_dataset.values()])
            np.testing.assert_allclose(expected, float(row[variant]), rtol=1e-12, atol=1e-12)
    return summaries, {"cases": len(records), "max_original_scorer_difference": max_difference,
                       "inference_seconds": dict(timings), "hard_choices_and_blends_reconstructed": True,
                       "baseline_max_abs_difference": {name: max(row["baseline_prediction_max_abs_difference"][name] for row in records)
                                                        for name in ("general", "specialist")}}


def audit_recsys(output):
    rows = read_csv(output / "summary.csv")
    details = {}
    for dataset in ("movielens_100k", "netflix_small"):
        record = read_json(output / f"{dataset}.json")
        path = output / f"{dataset}.npz"
        if sha256(path) != record["prediction_sha256"]:
            raise ValueError("Recsys artifact changed")
        with np.load(path, allow_pickle=False) as stored:
            for variant in VARIANTS:
                row = next(row for row in rows if row["dataset"] == dataset and row["variant"] == variant)
                scores = user_rating_metrics(stored["users"], stored["truth"], stored[variant])
                for metric in ("RMSE", "MAE"):
                    np.testing.assert_allclose(scores[metric], float(row[metric]), rtol=0, atol=1e-12)
            for mode, decision in record["decisions"].items():
                np.testing.assert_array_equal(stored[f"{mode}_hard"], stored[decision["selected"]])
                weight = decision["specialist_weight"]
                np.testing.assert_array_equal(stored[f"{mode}_blend"], (1 - weight) * stored["general"] + weight * stored["specialist"])
        details[dataset] = record
    if len(rows) != 12:
        raise ValueError("Incomplete recsys table")
    return rows, details


def audit_llm(output):
    rows = read_csv(output / "summary.csv")
    improvements = defaultdict(lambda: [0, 0, 0])
    for dataset in DATASETS:
        record = read_json(output / f"{dataset}_audit.json")
        if sha256(output / f"{dataset}.npz") != record["panel_sha256"] or sha256(output / f"{dataset}_routing.jsonl") != record["routing_sha256"]:
            raise ValueError("LLM artifact changed")
        with np.load(output / f"{dataset}.npz", allow_pickle=True) as stored:
            for variant in VARIANTS:
                for method in METHODS:
                    for metric in ("r2", "rmse"):
                        panel = stored[f"{variant}_{method}_{metric}"]
                        if panel.shape != (10, 15) or not np.isfinite(panel).all():
                            raise ValueError("Incomplete LLM panel")
                        reference = stored[f"general_{method}_{metric}"]
                        for budget in range(1, 16):
                            row = next(row for row in rows if row["dataset"] == dataset and row["variant"] == variant and
                                       row["method"] == method and row["metric"] == metric and int(row["budget"]) == budget)
                            actual = float(panel[:, budget-1].mean())
                            np.testing.assert_allclose(actual, float(row["value"]), rtol=0, atol=1e-12)
                            delta = actual - float(reference[:, budget-1].mean())
                            direction = delta if metric == "r2" else -delta
                            category = 0 if direction > 1e-12 else 1 if direction < -1e-12 else 2
                            improvements[f"{variant}_{metric}"][category] += 1
    if len(rows) != 4 * 6 * 3 * 15 * 2:
        raise ValueError("Incomplete LLM summaries")
    return rows, dict(improvements)


def report(output):
    manifest = read_json(output / "manifest.json")
    if manifest["pilot_limit_per_suite"] is not None:
        raise ValueError("Final report requires the full campaign")
    for stage in ("tabular", "recsys", "llm"):
        if read_json(output / stage / "status.json")["state"] != "complete":
            raise ValueError(f"Stage not complete: {stage}")
    for filename, expected in manifest["input_sha256"].items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Frozen input changed: {filename}")
    for filename, expected in manifest["source_sha256"].items():
        snapshot = output / "source_snapshot" / Path(filename).relative_to(REPO.parent)
        if sha256(snapshot) != expected:
            raise ValueError(f"Frozen source snapshot changed: {filename}")
    for stage, reference in manifest.get("reused_stages", {}).items():
        if sha256(Path(reference["manifest"])) != reference["manifest_sha256"]:
            raise ValueError(f"Reused-stage provenance changed: {stage}")
        if (output / stage).resolve() != Path(reference["directory"]).resolve():
            raise ValueError(f"Reused-stage directory changed: {stage}")
    tabular, tabular_audit = audit_tabular(output / "tabular", manifest)
    recsys, recsys_audit = audit_recsys(output / "recsys")
    llm, improvements = audit_llm(output / "llm")
    lines = ["# Two-expert observed-only routing", "",
             "All metrics use the same frozen masks, splits and scorers as the earlier comparisons. No outer targets enter routing.", "",
             "## Main findings", "",
             f"- Observed hard routing improves {sum(float(row['observed_hard']) < float(row['general']) for row in tabular)}/{len(tabular)} tabular pattern summaries versus general alone.",
             f"- The observed-error blend improves {sum(float(row['observed_blend']) < float(row['general']) for row in tabular)}/{len(tabular)} tabular pattern summaries versus general alone. Inspect every pattern below; this is not a uniformly improving rule.",
             f"- Observed routing chooses {recsys_audit['movielens_100k']['decisions']['observed']['selected']} on MovieLens 100K and {recsys_audit['netflix_small']['decisions']['observed']['selected']} on Netflix small.",
             f"- Across 180 LLM R² settings, observed blending improves {improvements['observed_blend_r2'][0]}, worsens {improvements['observed_blend_r2'][1]}, and ties {improvements['observed_blend_r2'][2]} versus general alone.", "",
             "## Experts and fixed rules", "",
             f"- General checkpoint: `{manifest['general']}`.",
             f"- Specialist checkpoint: `{manifest['specialist']}`.",
             "- Observed router: choose the expert with lower squared error on raw observed-cell predictions, not copied observations.",
             "- Observed blend: fit a convex combination to those same predictions; the specialist weight is clipped to [0,1].",
             "- Holdout controls: hide 10% of observed entries (seed 42, retain row/column support), refit preprocessing, and select/fit on those held-out observations. Final outputs restore the complete original context.",
             "- One choice/weight per matrix, or per LLM inference batch. Ties/no evidence fall back to the general model.",
             "- Routing loss uses observed-column SD scaling for tabular/LLM and global SD scaling for ratings.",
             "- Raw observed predictions see their own values through the encoder and residual connections. Reconstruction loss is therefore in-sample, not unbiased validation. Random observed holdout also need not match MAR/MNAR targets or sparsely observed LLM test rows.",
             "- Four policies were fixed before the full campaign; none was tuned against these outer results. These are previously inspected datasets, not new independent confirmation.", "",
             "## Tabular NRMSE", "",
             "Lower is better. Equal dataset weighting after averaging cases within each dataset. MissBench: 42 datasets × 11 patterns at 40%, full context. UCI: 12 datasets × 3 patterns × 5 rates × 10 repeats, 2000×1000 windows. No row train/test split.", ""]
    for suite in ("missbench", "uci"):
        rows = sorted((row for row in tabular if row["suite"] == suite), key=lambda row: PATTERNS.index(row["pattern"]))
        lines += [f"### {suite.upper()}", ""]
        lines += table(["Pattern", *LABELS], [[row["pattern"], *[float(row[name]) for name in VARIANTS]] for row in rows], "min")
    lines += ["## Recommender ratings", "", "User-macro RMSE/MAE in original rating units; lower is better.",
              "The general expert retains global normalization and 9976×1000 context with streamed inference. The specialist retains two-way centering, user scaling and 2000×1000 context. Both clip to [1,5]. This routes deployed recipes, not isolated checkpoint weights.", ""]
    for dataset in ("movielens_100k", "netflix_small"):
        lines += [f"### {dataset}", ""]
        lines += table(["Metric", *LABELS], [[metric, *[float(next(row for row in recsys if row['dataset'] == dataset and row['variant'] == variant)[metric])
                                                        for variant in VARIANTS]] for metric in ("RMSE", "MAE")], "min")
        decisions = recsys_audit[dataset]["decisions"]
        for mode in ("observed", "holdout"):
            decision = decisions[mode]
            lines.append(f"- {mode.title()}: chose **{decision['selected']}**; specialist blend weight {decision['specialist_weight']:.6f}; {decision['evidence_cells']:,} routing cells.")
        lines.append("")
    lines += ["## LLM benchmark completion", "", "R² at five MI-selected tasks, averaged across ten folds; higher is better. Full entropy/MI/random budgets 1–15 and standardized RMSE are saved in `llm/summary.csv` and `llm/per_fold.csv`.", ""]
    lines += table(["Dataset", *LABELS], [[dataset, *[float(next(row for row in llm if row['dataset'] == dataset and row['variant'] == variant and
                          row['method'] == 'mi' and row['metric'] == 'r2' and row['budget'] == '5')['value']) for variant in VARIANTS]] for dataset in DATASETS], "max")
    lines += table(["Variant", "R² settings improved", "Worsened", "Tied"],
                   [[label, *improvements[f"{variant}_r2"]] for variant, label in zip(VARIANTS[2:], LABELS[2:])])
    lines += ["Counts compare with general over all 180 dataset/method/budget settings; settings/folds are dependent, not independent significance tests.", "",
              "## Routing and cost", ""]
    lines += table(["Suite", "Pattern", "Observed specialist fraction", "Holdout specialist fraction"],
                   [[row["suite"], row["pattern"], float(row['observed_specialist_fraction']), float(row['holdout_specialist_fraction'])] for row in tabular])
    timing = tabular_audit["inference_seconds"]
    lines += [f"Tabular expert inference seconds across all cases: general {timing['general']:.1f}, specialist {timing['specialist']:.1f}, masked general {timing['holdout_general']:.1f}, masked specialist {timing['holdout_specialist']:.1f}.",
              "Observed routing/blending requires two full expert passes, not one. The masked control adds two more passes; all six reported outputs share these four passes. Models run sequentially; both sets of weights remain resident.", "",
              "## Scope and audit", "",
              "- Completed 2,262 tabular cases, two recommender datasets, and 1,800 LLM fold-settings for each of six variants.",
              "- Tabular/query predictions and LLM fold panels are saved; hard choices and convex blends independently reconstructed and rescored.",
              "- Every tabular expert prediction and every LLM expert fold panel is checked against the archived standalone baseline. RecSys query vectors are also checked. Exact maximum differences are in `final_audit.json` and individual LLM audits.",
              "- LLM replay verifies identical masked inputs and invokes the unchanged historical scorer for every variant; it does not save per-entry LLM predictions.",
              "- No causal inference, R-method reruns, MovieLens 1M/10M, neural training, checkpoint selection, or manuscript edits in this campaign.",
              "- `manifest.json` freezes protocol/input/source hashes; `source_snapshot/` preserves runnable research sources.",
              "- Initial pilots exposed dtype/layout changes before UCI preprocessing. Those were fixed before the full run; failed pilots are retained separately and are not included in these results.", ""]
    if manifest.get("reused_stages"):
        lines += ["The first LLM attempt failed baseline parity because it disabled cuDNN SDPA, unlike the historical LLM harness. The corrected run enables that backend for LLM only; no tolerances or routing rules were changed. Completed tabular/recommender stages are reused through explicit links, with identical input hashes and AST-verified evaluator functions. Their original provenance is recorded in `manifest.json.reused_stages`; the unsuccessful LLM attempt is retained in the first campaign directory.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(output / "final_audit.json", {"state": "complete", "tabular": tabular_audit, "recsys": recsys_audit,
               "llm_improved_worsened_tied_vs_general": improvements, "report_sha256": sha256(output / "RESULTS.md"),
               "audit_script_sha256": sha256(Path(__file__))})
    print(output / "RESULTS.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    report(parser.parse_args().output)
