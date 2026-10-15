"""Inner-only, query-count-stratified diagnostics for eligible synthetic models."""
import argparse
import json
from pathlib import Path

import numpy as np

from dwip_recsys_experiment import DEFAULT_ROOT, load_split
from dwip_recsys_provenance import require_synthetic_checkpoint
from evaluate_dwip_recsys import DATASETS, sha256, write_json


def per_user_errors(users, truth, prediction):
    counts = np.bincount(users)
    residual = prediction - truth
    mse = np.bincount(users, weights=residual ** 2, minlength=len(counts)) / counts.clip(1)
    return counts, {"MAE": np.bincount(users, weights=np.abs(residual), minlength=len(counts)) / counts.clip(1),
                    "RMSE": np.sqrt(mse), "MSE": mse,
                    "bias": np.bincount(users, weights=residual, minlength=len(counts)) / counts.clip(1)}


def load_predictions(path, queries):
    with np.load(path, allow_pickle=False) as data:
        if not all(np.array_equal(data[key], values) for key, values in zip(("users", "items", "truth"), queries)):
            raise ValueError("Predictions do not match frozen inner queries")
        prediction = data["prediction"].copy()
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite cached prediction")
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"purpose": "Inner evaluation only; no neural training, outer labels, or real-trained neural candidates",
              "candidates": {}, "datasets": {}}
    for name in args.runs:
        run = args.root / name
        audit = require_synthetic_checkpoint(run / "best_validation_checkpoint.pth")
        manifest = json.loads((run / "validation_full/manifest.json").read_text())
        if manifest["checkpoint_sha256"] != audit["checkpoint_sha256"] or manifest["partition"] != "validation":
            raise ValueError("Evaluation manifest does not match eligible inner candidate")
        report["candidates"][name] = audit
    for spec in DATASETS.values():
        _, _, queries = load_split(spec["folder"], "validation")
        predictions = {"MF": load_predictions(args.root / f"mf_validation_{spec['folder']}.npz", queries)}
        predictions.update({name: load_predictions(args.root / name / "validation_full" / spec["folder"] / "query_predictions.npz", queries)
                            for name in args.runs})
        errors = {}
        for name, prediction in predictions.items():
            counts, errors[name] = per_user_errors(queries[0], queries[2], prediction)
        groups = {}
        for lower, upper in ((1, 1), (2, 2), (3, 5), (6, 10), (11, 1000000), (1, 1000000)):
            selected = (counts >= lower) & (counts <= upper)
            if not selected.any():
                continue
            groups[f"{lower}_{upper}"] = {"users": int(selected.sum()), "user_weight": float(selected.sum() / (counts > 0).sum()),
                "metrics": {name: {metric: float(values[selected].mean()) for metric, values in scores.items()} for name, scores in errors.items()},
                "delta_from_MF": {name: {metric: float((values[selected] - errors["MF"][metric][selected]).mean()) for metric, values in scores.items()}
                                  for name, scores in errors.items() if name != "MF"}}
        report["datasets"][spec["folder"]] = groups
    report["source_sha256"] = sha256(Path(__file__))
    write_json(args.output, report)
    for name, groups in report["datasets"].items():
        print(name, json.dumps({key: value["delta_from_MF"] for key, value in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
