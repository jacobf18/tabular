"""Compare inner-held-out error by context support; never inspect outer labels."""
import argparse
import json
from pathlib import Path

import numpy as np
from cornac.models import MF

from dwip_recsys_experiment import DEFAULT_ROOT, load_split, cornac_split
from evaluate_dwip_recsys import DATASETS, atomic_npz, user_rating_metrics, write_json


def diagnose(root):
    baseline = json.loads((root / "mf_baseline.json").read_text())["validation"]
    report = {}
    for spec in DATASETS.values():
        observed, context, queries = load_split(spec["folder"], "validation")
        cache = root / f"mf_validation_{spec['folder']}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                if not all(np.array_equal(data[key], values) for key, values in zip(("users", "items", "truth"), queries)):
                    raise ValueError("Cached MF predictions belong to a different inner holdout")
                mf_prediction = data["prediction"]
        else:
            split = cornac_split(context, queries)
            model = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123, num_threads=1)
            model.fit(split.train_set)
            mf_prediction = np.clip(np.array([model.score(split.train_set.uid_map[int(user)], split.train_set.iid_map[int(item)])
                                              for user, item in zip(queries[0], queries[1])]), 1, 5)
            atomic_npz(cache, users=queries[0], items=queries[1], truth=queries[2], prediction=mf_prediction)
        check = user_rating_metrics(queries[0], queries[2], mf_prediction)
        if any(abs(check[metric] - baseline[spec["folder"]][metric]) > 1e-8 for metric in ("MAE", "RMSE")):
            raise ValueError("MF prediction replay does not reproduce its registered rating scores")
        predictions = {"MF": mf_prediction}
        for work in root.iterdir():
            path = work / "validation_full" / spec["folder"] / "query_predictions.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as data:
                    if not all(np.array_equal(data[key], values) for key, values in zip(("users", "items", "truth"), queries)):
                        raise ValueError("Candidate query IDs or labels changed")
                    predictions[work.name] = data["prediction"]
        supports = {"user": np.isfinite(observed).sum(1)[queries[0]],
                    "item": np.isfinite(observed).sum(0)[queries[1]]}
        dataset = {}
        for axis, counts in supports.items():
            for lower, upper in ((1, 5), (5, 20), (20, 100), (100, 1000000)):
                selected = (counts >= lower) & (counts < upper)
                if not selected.any():
                    continue
                dataset[f"{axis}_support_{lower}_{upper}"] = {
                    name: {**user_rating_metrics(queries[0][selected], queries[2][selected], values[selected]),
                           "bias": float((values[selected] - queries[2][selected]).mean()), "ratings": int(selected.sum())}
                    for name, values in predictions.items()}
        report[spec["folder"]] = dataset
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, type=Path)
    args = parser.parse_args()
    write_json(args.root / "support_error_diagnostics.json", diagnose(args.root))


if __name__ == "__main__":
    main()
