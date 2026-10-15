"""Preserve the failed diagnostic and recover its original cases without changing canonical scorers."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import evaluate_single_model_missingness_probe as original
from recsys_evaluation_common import sha256, write_json


FAILURE = "ValueError('Metric coverage must be identical for every model')"
TRUTH_COVERAGE = ("queried_columns", "common_nonconstant_columns")


def score_predictions(source, target, query, predictions):
    if set(predictions) != set(original.ROLES) or not query.any() or not torch.isfinite(source).any():
        raise ValueError("Require complete roles and observed/query cells")
    if any(prediction.shape != source.shape or not torch.isfinite(prediction).all() for prediction in predictions.values()):
        raise ValueError("Require full finite prediction grids, including observed entries")
    episode = original.Episode(source.unsqueeze(0), target.unsqueeze(0), query.unsqueeze(0),
                               {"domain": "imputation", "gradient_data": "synthetic"})
    observed = torch.isfinite(source)
    records = {}
    for role, prediction in predictions.items():
        metrics = original.episode_metrics(prediction.unsqueeze(0), episode)
        metrics["observed_rmse"] = float((prediction[observed].double() - source[observed].double()).square().mean().sqrt())
        metrics["query_blend_rmse"] = float((prediction[query].double() - predictions["blend"][query].double()).square().mean().sqrt())
        records[role] = metrics
    if len({tuple(metrics["column_coverage"][name] for name in TRUTH_COVERAGE) for metrics in records.values()}) != 1:
        raise ValueError("Truth-defined coverage must remain identical for all roles")
    return records


def load_case(bank, case, prediction_path):
    with np.load(bank / f"context_{case['context']:02d}/{case['family']}.npz", allow_pickle=False) as stored:
        source, target, query = [torch.from_numpy(stored[name]) for name in ("model_source", "model_target", "query")]
    with np.load(prediction_path, allow_pickle=False) as stored:
        if set(stored.files) != set(original.ROLES):
            raise ValueError("Cached case must contain every original role")
        predictions = {role: torch.from_numpy(stored[role]) for role in original.ROLES}
    return source, target, query, predictions


def register(failed, output):
    if output.exists():
        raise FileExistsError("Never replace a recovery")
    config = original.read(failed / "config.json")
    status = original.read(failed / "status.json")
    if status["state"] != "failed" or status["error"] != FAILURE:
        raise ValueError("Only the specific canonical-coverage assumption failure may be recovered")
    try:
        os.kill(status["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("Never recover a live worker")
    original.verify_hashes(config["source_sha256"])
    original.verify_hashes(config["input_sha256"])
    metadata = original.read(Path(config["bank"]) / "per_case.json")
    paths = sorted((failed / "predictions").glob("case_*.npz"))
    if not paths or [path.name for path in paths] != [f"case_{index:03d}.npz" for index in range(len(paths))]:
        raise ValueError("Require the complete contiguous saved prefix, not a selected subset")
    progress = original.read(failed / "progress.json")
    if progress["cases"] != len(paths) - 1 or len(paths) >= len(metadata):
        raise ValueError("Require the original fully scored prefix plus its saved failure case")
    failures = []
    for index, path in enumerate(paths):
        values = load_case(Path(config["bank"]), metadata[index], path)
        corrected = score_predictions(*values)
        try:
            previous = original.score_predictions(*values)
        except ValueError as error:
            if repr(error) != FAILURE:
                raise
            failures.append(index)
        else:
            if previous != corrected:
                raise ValueError("A previously scored canonical metric changed")
    if failures != [len(paths) - 1]:
        raise ValueError("The original failure must reproduce only at the terminal saved case")
    extra_hashes = {str(path): sha256(path) for path in paths}
    extra_hashes.update({str(failed / name): sha256(failed / name) for name in
                        ("config.json", "status.json", "progress.json", "run.log")})
    config["input_sha256"].update(extra_hashes)
    config["source_sha256"].update(original.probe.source_hashes())
    config["recovery"] = {"failed": str(failed), "failure": FAILURE, "saved_prefix_cases": len(paths),
                          "previously_scored_cases": len(paths) - 1, "original_cases_and_models_unchanged": True,
                          "canonical_scores_unchanged": True, "truth_coverage": list(TRUTH_COVERAGE),
                          "clarification": "Canonical constant-column inclusion is prediction-dependent in the frozen scorers; only queried/common-nonconstant column counts are truth-only invariants",
                          "cache_policy": "Reuse every saved full prediction grid; infer only the unsaved suffix; no resampling or checkpoint reselection",
                          "registered_unix": time.time()}
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"state": "registered", "config_sha256": sha256(output / "config.json")})


@contextmanager
def recovery_callbacks(config):
    previous_score, previous_predict = original.score_predictions, original.streamed_point_predictions
    metadata = original.read(Path(config["bank"]) / "per_case.json")
    recovery = config["recovery"]
    counters = {"calls": 0, "cached_predictions": 0, "new_predictions": 0}
    network_roles = list(config["checkpoints"])
    if network_roles != [role for role in original.ROLES if role != "blend"]:
        raise ValueError("Original network role order changed")

    def predict(model, source, criterion, **options):
        case_index, role_index = divmod(counters["calls"], len(network_roles))
        if case_index >= len(metadata):
            raise ValueError("Unexpected extra prediction call")
        counters["calls"] += 1
        if case_index >= recovery["saved_prefix_cases"]:
            counters["new_predictions"] += 1
            return previous_predict(model, source, criterion, **options)
        prediction_path = Path(recovery["failed"]) / "predictions" / f"case_{case_index:03d}.npz"
        old_source, _, _, predictions = load_case(Path(config["bank"]), metadata[case_index], prediction_path)
        torch.testing.assert_close(source[0].cpu(), old_source, rtol=0., atol=0., equal_nan=True)
        counters["cached_predictions"] += 1
        return predictions[network_roles[role_index]].unsqueeze(0).to(source.device)

    original.score_predictions, original.streamed_point_predictions = score_predictions, predict
    try:
        yield counters
    finally:
        original.score_predictions, original.streamed_point_predictions = previous_score, previous_predict


def run(output):
    config = original.read(output / "config.json")
    with recovery_callbacks(config) as counters:
        original.evaluate(output)
    write_json(output / "status.json", {"state": "verifying_cache", "pid": os.getpid(), "updated_unix": time.time()})
    prefix = config["recovery"]["saved_prefix_cases"]
    if counters != {"calls": 768, "cached_predictions": prefix * 4, "new_predictions": (192 - prefix) * 4}:
        raise ValueError("Cached and fresh inference coverage changed")
    for index in range(prefix):
        old_path = Path(config["recovery"]["failed"]) / "predictions" / f"case_{index:03d}.npz"
        new_path = output / "predictions" / f"case_{index:03d}.npz"
        with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
            if set(old.files) != set(new.files) or any(not np.array_equal(old[role], new[role]) for role in original.ROLES):
                raise ValueError("A retained prediction grid was replaced")
    original.verify_hashes(config["input_sha256"])
    original.verify_hashes(config["source_sha256"])
    write_json(output / "recovery_audit.json", {"complete": True, **counters, "saved_cases": prefix,
                                               "all_cached_grids_identical": True, "failed_artifacts_unchanged": True,
                                               "canonical_metrics_unchanged": True, "config_sha256": sha256(output / "config.json")})
    write_json(output / "status.json", {"state": "complete", "cases": 192, "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--failed", type=Path, default=original.ROOT / "missingness_models_probe_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(args.failed.resolve(), output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if original.read(output / "status.json")["state"] != "registered":
            raise ValueError("Never overwrite a started recovery")
        try:
            run(output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
