"""Inner-only baseline audit separating factor signal and observed-rating replay."""
import argparse
import json
from pathlib import Path

import numpy as np
from cornac.models import MF

from dwip_recsys_experiment import DEFAULT_ROOT, load_split, cornac_split, inference_components
from evaluate_dwip_recsys import DATASETS, FrozenRatings, METRIC_NAMES, metrics, user_rating_metrics, write_json, sha256
from evaluate_dwip_recsys_candidate import remap_prediction


def diagnose(root):
    registered = json.loads((root / "mf_baseline.json").read_text())["validation"]
    reports = {}
    for spec in DATASETS.values():
        observed, context, queries = load_split(spec["folder"], "validation")
        split = cornac_split(context, queries)
        model = MF(k=10, max_iter=25, learning_rate=.01, lambda_reg=.02, use_bias=True, seed=123, num_threads=1)
        model.fit(split.train_set)
        query_users = np.array([split.train_set.uid_map[int(user)] for user in queries[0]])
        query_items = np.array([split.train_set.iid_map[int(item)] for item in queries[1]])
        point = np.array([model.score(int(user), int(item)) for user, item in zip(query_users, query_items)])
        check = user_rating_metrics(queries[0], queries[2], np.clip(point, 1, 5))
        if any(abs(check[metric] - registered[spec["folder"]][metric]) > 1e-8 for metric in ("MAE", "RMSE")):
            raise ValueError("MF does not reproduce its registered baseline")
        factor = np.sum(model.u_factors[query_users] * model.i_factors[query_items], axis=1)
        mf_bias = model.global_mean + model.u_biases[query_users] + model.i_biases[query_items]
        scores = np.clip(np.stack([model.score(user) for user in range(split.train_set.num_users)]), 1, 5)
        train_users, train_items, train_ratings = split.train_set.uir_tuple
        scores[train_users, train_items] = train_ratings
        replay, _ = split.evaluate(FrozenRatings(prediction=scores), metrics=metrics(), user_based=True)
        means, _, row_offsets = inference_components(observed, "two_way")
        bias_prediction = np.clip(means[None, :] + row_offsets[:, None], 1, 5)
        bias_prediction[context[0], context[1]] = context[2]
        mapped, _ = remap_prediction(bias_prediction, split.train_set)
        bias_result, _ = split.evaluate(FrozenRatings(prediction=mapped), metrics=metrics(), user_based=True)
        reports[spec["folder"]] = {
            "MF_original": registered[spec["folder"]],
            "MF_observed_rating_replay": {metric: float(replay.metric_avg_results[metric]) for metric in METRIC_NAMES},
            "bias_only_observed_rating_replay": {metric: float(bias_result.metric_avg_results[metric]) for metric in METRIC_NAMES},
            "MF_bias_component_rating_metrics": user_rating_metrics(queries[0], queries[2], np.clip(mf_bias, 1, 5)),
            "MF_query_factor_std": float(factor.std()),
            "MF_query_factor_rms": float(np.sqrt(np.mean(factor**2))),
        }
        print(json.dumps({spec["folder"]: reports[spec["folder"]]}), flush=True)
    return {"datasets": reports, "source_sha256": sha256(Path(__file__)),
            "purpose": "Inner-only diagnostics; neither MF factors nor MF predictions are used by DWIP",
            "protocol_note": "Cornac ranking excludes training positives, not all training ratings. Standard DWIP imputation preserves observed ratings; the extra MF replay control applies the same preservation without changing held-out rating predictions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    write_json(args.root / "baseline_component_audit.json", diagnose(args.root))


if __name__ == "__main__":
    main()
