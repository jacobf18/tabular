"""Frozen evaluation references and unchanged public inference for the BPR goal."""
import argparse
import json
from pathlib import Path
import time

import cornac
from cornac.models import BPR
import numpy as np
import torch

from dwip_recsys_experiment import load_split, cornac_split
from dwip_recsys_ranking_audit import independent_ranking_metrics
from dwip_bpr_ranking_profile import ranking_profile
from evaluate_dwip_recsys import DATASETS, OUTPUT, METRIC_NAMES, FrozenRatings, metrics, sha256, user_rating_metrics, write_json, atomic_npz
from evaluate_dwip_recsys_candidate import impute_candidate_ratings, remap_prediction


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/recsys_bpr_goal_20260921"
MF_BASELINE = ROOT / "runs/recsys_goal_20260920/mf_baseline.json"
INITIAL = ROOT / "runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth"
INITIAL_SHA256 = "1fcadc0c5846f9c35fbbae4e3b519b9244b608272cb1eda68c0850132321f770"
RATING_METRICS = ("MAE", "RMSE")
TARGET_METRICS = ("NDCG@10", "Recall@10")


def split_hashes():
    return {spec["folder"]: sha256(OUTPUT / spec["folder"] / "split.npz") for spec in DATASETS.values()}


def compare_goal(results, mf, bpr):
    if set(results) != {spec["folder"] for spec in DATASETS.values()}:
        raise ValueError("Require both complete benchmark datasets")
    comparisons = {}
    for dataset, values in results.items():
        comparisons[dataset] = {}
        for metric in (*RATING_METRICS, *TARGET_METRICS):
            reference = mf[dataset][metric] if metric in RATING_METRICS else bpr[dataset][metric]
            actual = values[metric]
            if not np.isfinite(actual) or not np.isfinite(reference) or reference <= 0:
                raise ValueError("Require finite scores and positive reference metrics")
            comparisons[dataset][metric] = {
                "actual": actual, "baseline": reference,
                "better": bool(actual < reference if metric in RATING_METRICS else actual > reference),
                "ratio": float(actual / reference if metric in RATING_METRICS else reference / max(actual, 1e-12)),
            }
    return comparisons


def selection_key(results, mf, bpr):
    comparisons = compare_goal(results, mf, bpr)
    rating_ratio = max(values[metric]["ratio"] for values in comparisons.values() for metric in RATING_METRICS)
    ranking_ratios = [values[metric]["ratio"] for values in comparisons.values() for metric in TARGET_METRICS]
    feasible = all(values[metric]["better"] for values in comparisons.values() for metric in RATING_METRICS)
    return [0 if feasible else 1, max(ranking_ratios) if feasible else rating_ratio, float(np.mean(ranking_ratios))]


def run_bpr(root):
    root.mkdir(parents=True, exist_ok=True)
    output = root / "bpr_reference"
    output.mkdir(exist_ok=False)
    parameters = {"k": 10, "max_iter": 200, "learning_rate": .001, "lambda_reg": .01,
                  "seed": 123, "num_threads": 1, "verbose": False}
    manifest = {"purpose": "Isolated conventional evaluation reference; never a neural teacher",
                "parameters": parameters, "cornac_version": cornac.__version__,
                "script_sha256": sha256(Path(__file__)), "split_sha256": split_hashes(),
                "mf_sha256": sha256(MF_BASELINE), "candidate_protocol": "Existing Cornac; exclude observed positives only"}
    write_json(output / "manifest.json", manifest)
    results = {}
    for partition in ("validation", "test"):
        results[partition] = {}
        for spec in DATASETS.values():
            started = time.monotonic()
            _, context, queries = load_split(spec["folder"], partition)
            split = cornac_split(context, queries)
            model = BPR(**parameters)
            result, _ = split.evaluate(model=model, metrics=metrics(), user_based=True)
            values = {metric: float(result.metric_avg_results[metric]) for metric in METRIC_NAMES}
            prediction = np.stack([model.score(user) for user in range(split.train_set.num_users)])
            independent, per_user = independent_ranking_metrics(prediction, split.train_set.csr_matrix, split.test_set.csr_matrix)
            for metric, score in independent.items():
                if not np.isclose(score, values[metric], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Independent BPR metric mismatch: {metric}")
                if set(per_user) != set(result.metric_user_results[metric]):
                    raise ValueError("BPR ranking user coverage differs")
                if any(not np.isclose(scores[metric], result.metric_user_results[metric][user], rtol=1e-12, atol=1e-12)
                       for user, scores in per_user.items()):
                    raise ValueError("Independent BPR per-user metric mismatch")
            directory = output / partition / spec["folder"]
            directory.mkdir(parents=True)
            atomic_npz(directory / "factors.npz", user_factors=model.u_factors, item_factors=model.i_factors,
                       item_biases=model.i_biases)
            write_json(directory / "metrics.json", {"metrics": values, "independent_ranking_passed": True,
                       "seconds": time.monotonic() - started, "factor_sha256": sha256(directory / "factors.npz")})
            results[partition][spec["folder"]] = values
            print(json.dumps({"partition": partition, "dataset": spec["folder"], **values}), flush=True)
            del prediction
    write_json(root / "bpr_baseline.json", results)
    write_json(root / "baseline_registration.json", {**manifest, "bpr_sha256": sha256(root / "bpr_baseline.json")})
    return results


@torch.no_grad()
def evaluate_public(imputer, partition="validation", output=None, audit_cornac=False):
    if partition not in ("validation", "test"):
        raise ValueError("Unknown evaluation partition")
    if (imputer.max_num_rows, imputer.max_num_cols, imputer.prediction_stat, imputer.precision_mode) != (2000, 1000, "mean", "amp_bfloat16"):
        raise ValueError("Inference recipe is frozen for this goal")
    imputer.model.eval()
    results = {}
    for spec in DATASETS.values():
        started = time.monotonic()
        observed, context, queries = load_split(spec["folder"], partition)
        prediction = impute_candidate_ratings(imputer, observed, "two_way_user_scale")
        if not np.isfinite(prediction).all() or not np.allclose(prediction[context[0], context[1]], context[2], rtol=1e-12, atol=1e-12):
            raise ValueError("Nonfinite predictions or changed observed ratings")
        split = cornac_split(context, queries)
        mapped, original_users = remap_prediction(prediction, split.train_set)
        ranking, per_user = independent_ranking_metrics(mapped, split.train_set.csr_matrix, split.test_set.csr_matrix)
        rating = user_rating_metrics(queries[0], queries[2], prediction[queries[0], queries[1]])
        values = {**{metric: rating[metric] for metric in RATING_METRICS}, **ranking}
        original_items = np.asarray([int(raw) for raw, index in sorted(split.train_set.iid_map.items(), key=lambda entry: entry[1])])
        profile = ranking_profile(mapped, split.train_set.csr_matrix, split.test_set.csr_matrix, original_items)
        if not np.isclose(profile["all"]["recall10"], values["Recall@10"], rtol=1e-12, atol=1e-12):
            raise ValueError("Ranking profile changed the candidate protocol")
        if audit_cornac:
            result, _ = split.evaluate(model=FrozenRatings(prediction=mapped), metrics=metrics(), user_based=True)
            for metric, score in values.items():
                if not np.isclose(score, result.metric_avg_results[metric], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Independent final metric mismatch: {metric}")
            for metric in ranking:
                if set(per_user) != set(result.metric_user_results[metric]):
                    raise ValueError("Final ranking user coverage differs")
                if any(not np.isclose(scores[metric], result.metric_user_results[metric][user], rtol=1e-12, atol=1e-12)
                       for user, scores in per_user.items()):
                    raise ValueError("Independent final per-user ranking mismatch")
        results[spec["folder"]] = values
        if output is not None:
            directory = Path(output) / spec["folder"]
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "metrics.json", values)
            write_json(directory / "ranking_profile.json", profile)
            write_json(directory / "audit.json", {"finite_predictions": True, "observed_ratings_preserved": True,
                       "cornac_metrics_and_per_user_ranking_match": bool(audit_cornac), "gradients_enabled": torch.is_grad_enabled()})
            atomic_npz(directory / "query_predictions.npz", users=queries[0], items=queries[1], truth=queries[2],
                       prediction=prediction[queries[0], queries[1]])
            atomic_npz(directory / "ranking_metrics.npz", users=np.array([original_users[user] for user in per_user]),
                       **{metric: np.array([scores[metric] for scores in per_user.values()]) for metric in ranking})
        print(json.dumps({"partition": partition, "dataset": spec["folder"], "seconds": time.monotonic() - started, **values}), flush=True)
        print(json.dumps({"partition": partition, "dataset": spec["folder"], "ranking_profile": profile}), flush=True)
        del prediction, mapped, observed
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CAMPAIGN)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run_bpr(args.root)


if __name__ == "__main__":
    main()
