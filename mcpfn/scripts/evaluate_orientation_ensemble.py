"""Frozen original, transposed and prediction-averaged MatrixFormer evaluations."""

import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np
import torch

import evaluate_two_expert_router as campaign
from audit_dwip_missbench_uci import load_original
from compare_recsys_checkpoint import audit_scores, reference_paths
from dwip_recsys_experiment import inference_components, inference_row_scales
from evaluate_matrixformer_plus_large_recsys import collect_predictions, fit_observed_blend
from evaluate_matrixformer_recsys_ratings import normalization_parameters
from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from tabimpute.routing import RoutingResult, TwoExpertRouter


PARENT = campaign.OUTPUT
LARGE = campaign.ROOT / "matrixformer_plus_large_recsys_20261004"
OUTPUT = campaign.ROOT / "matrixformer_orientation_v2_20261004"
BASE_VARIANTS = ("general", "specialist", "observed_blend")
VARIANTS = tuple(name + suffix for suffix in ("", "_transpose", "_average") for name in BASE_VARIANTS)


def standardize_orientation_uci(matrix):
    empty = ~np.isfinite(matrix).any(axis=0)
    if not empty.any():
        return campaign.standardize_uci(matrix)[:3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means, scales = np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)
    means = np.where(empty, 0., means)
    scales = np.where(empty | (scales == 0), 1., scales)
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("Invalid nonempty-column normalization")
    return (matrix - means) / scales, means, scales


class OrientationTabularExpert(campaign.TabularExpert):
    def impute(self, matrix, return_full=False):
        if self.suite != "uci":
            return super().impute(matrix, return_full=return_full)
        normalized, means, scales = standardize_orientation_uci(matrix)
        campaign.configure_windows(self.model, matrix.shape, 2000, 1000)
        completed, raw = self.model.impute(normalized.copy(), return_full=True)
        completed, raw = completed * scales + means, raw * scales + means
        return (completed, raw) if return_full else completed


def combine_predictions(original, transposed):
    result = {}
    for name in BASE_VARIANTS:
        before, after = original[name], transposed[name]
        if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("Invalid orientation prediction geometry or values")
        result[name] = before
        result[name + "_transpose"] = after
        result[name + "_average"] = (before + after) / 2
    return result


class OrientationRouter:
    def __init__(self, *args, **kwargs):
        self.router = TwoExpertRouter(*args, **kwargs)

    def predict_all(self, matrix):
        normal = self.router.predict_all(matrix, include_holdout=False)
        transposed = self.router.predict_all(matrix.T.copy(), include_holdout=False)
        predictions = combine_predictions(normal.predictions,
                                          {name: value.T for name, value in transposed.predictions.items()})
        predictions["observed_hard"] = normal.predictions["observed_hard"]
        seconds = {**normal.seconds, **{name + "_transpose": value for name, value in transposed.seconds.items()}}
        return RoutingResult(predictions, {"original": normal.decisions["observed"],
                                          "transpose": transposed.decisions["observed"]}, seconds)


def summarize_tabular(output, records, cases):
    write_csv(output / "per_case.csv", [{key: record[key] for key in
              ("case_id", "suite", "dataset", "pattern", "missingness", "repeat")} | record["nrmse"]
              for record in records])
    groups = defaultdict(list)
    for record in records:
        groups[record["suite"], record["pattern"]].append(record)
        groups[record["suite"], "overall"].append(record)
    rows = []
    for (suite, pattern), group in groups.items():
        expected = sum(case["suite"] == suite and (pattern == "overall" or case["pattern"] == pattern) for case in cases)
        datasets = sorted({record["dataset"] for record in group})
        row = {"suite": suite, "pattern": pattern, "cases": len(group), "expected": expected,
               "datasets": len(datasets), "complete": len(group) == expected}
        for variant in VARIANTS:
            row[variant] = float(np.mean([np.mean([record["nrmse"][variant] for record in group
                                                  if record["dataset"] == dataset]) for dataset in datasets])) if row["complete"] else None
        rows.append(row)
    write_csv(output / "summary.csv", rows)


def run_tabular(output, manifest):
    experts = [OrientationTabularExpert(campaign.make_model(path)) for path in (campaign.GENERAL, campaign.CHECKPOINT)]
    router = TwoExpertRouter(*experts)
    scorers = {suite: load_original(campaign.REPO / path, "compute_normalized_rmse_columnwise") for suite, path in (
        ("missbench", "benchmark/plot_options.py"), ("uci", "benchmark/uci/plot_uci_negative_rmse.py"))}
    records = []
    for case in manifest["cases"]:
        campaign.check_reserve(output)
        source = Path(case["source_directory"])
        parent_record = campaign.read_json(PARENT / "tabular/cases" / (case["case_id"] + ".json"))
        parent_prediction = PARENT / "tabular/predictions" / (case["case_id"] + ".npz")
        assert sha256(parent_prediction) == parent_record["prediction_sha256"]
        for key, filename in (("missing_sha256", "missing.npy"), ("true_sha256", "true.npy")):
            assert sha256(source / filename) == parent_record[key]
        record_path = output / "cases" / (case["case_id"] + ".json")
        prediction_path = output / "predictions" / (case["case_id"] + ".npz")
        if record_path.exists():
            record = campaign.read_json(record_path)
            assert sha256(prediction_path) == record["prediction_sha256"]
            records.append(record)
            continue
        write_json(output / "status.json", {"state": "running", "current": case["case_id"], "completed": len(records)})
        missing, truth = (np.load(source / filename, allow_pickle=False) for filename in ("missing.npy", "true.npy"))
        mask = np.isnan(missing)
        for expert in experts:
            expert.suite = case["suite"]
        result = router.predict_all(missing.T.copy(), include_holdout=False)
        with np.load(parent_prediction, allow_pickle=False) as stored:
            original = {name: stored[name] for name in BASE_VARIANTS}
        predictions = combine_predictions(original, {name: result.predictions[name].T[mask] for name in BASE_VARIANTS})
        scores = {}
        for variant, values in predictions.items():
            prediction = missing.astype(values.dtype)
            prediction[mask] = values
            score = campaign.score_predictions(truth, prediction, missing, case["suite"])["nrmse_original"]
            np.testing.assert_allclose(score, scorers[case["suite"]](truth, prediction, mask), atol=1e-5, rtol=1e-5)
            if variant in BASE_VARIANTS:
                np.testing.assert_allclose(score, parent_record["nrmse"][variant], atol=1e-12, rtol=0)
            scores[variant] = score
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(prediction_path, shape=np.array(missing.shape), **predictions)
        record = {**case, "nrmse": scores, "decisions": result.decisions, "seconds": result.seconds,
                  "prediction_sha256": sha256(prediction_path), "parent_prediction_sha256": sha256(parent_prediction),
                  "missing_sha256": parent_record["missing_sha256"], "true_sha256": parent_record["true_sha256"]}
        write_json(record_path, record)
        records.append(record)
        if len(records) % 25 == 0:
            summarize_tabular(output, records, manifest["cases"])
            print(f"Tabular {len(records)}/{len(manifest['cases'])}", flush=True)
    summarize_tabular(output, records, manifest["cases"])
    write_json(output / "status.json", {"state": "complete", "cases": len(records)})


def run_llm(output):
    original_router, original_variants = campaign.TwoExpertRouter, campaign.VARIANTS
    try:
        campaign.TwoExpertRouter = OrientationRouter
        campaign.VARIANTS = VARIANTS
        campaign.run_llm(output)
    finally:
        campaign.TwoExpertRouter, campaign.VARIANTS = original_router, original_variants
    differences = {}
    for dataset in campaign.DATASETS:
        with np.load(output / f"{dataset}.npz", allow_pickle=True) as current, np.load(PARENT / "llm" / f"{dataset}.npz", allow_pickle=True) as previous:
            for variant in BASE_VARIANTS:
                for method in campaign.METHODS:
                    for metric in ("r2", "rmse"):
                        key = f"{variant}_{method}_{metric}"
                        np.testing.assert_array_equal(current[key], previous[key])
                        differences[f"{dataset}_{key}"] = 0.
    write_json(output / "orientation_audit.json", {"original_panel_max_abs_difference": differences,
               "prediction_average_before_scoring": True, "no_query_targets_in_router": True})


def run_recsys(output):
    rows = []
    for dataset in ("100K", "netflix", "1M", "10M"):
        campaign.check_reserve(output)
        paths = reference_paths(dataset, "global")
        with np.load(paths["split"], allow_pickle=False) as stored:
            split = dict(stored)
        observed = np.full(tuple(split["shape"])[::-1], np.nan, dtype=np.float64)
        observed[split["train_items"], split["train_users"]] = split["train_ratings"]
        assert np.isnan(observed[split["test_items"], split["test_users"]]).all()
        users = np.concatenate([split["train_items"], split["test_items"]])
        items = np.concatenate([split["train_users"], split["test_users"]])
        evidence_count = len(split["train_ratings"])
        case = output / dataset
        case.mkdir(exist_ok=True)
        transposed = {}
        for expert, checkpoint in (("general", campaign.GENERAL), ("specialist", campaign.CHECKPOINT)):
            artifact, record_path = case / f"{expert}.npz", case / f"{expert}.json"
            if record_path.exists():
                assert sha256(artifact) == campaign.read_json(record_path)["prediction_sha256"]
                with np.load(artifact, allow_pickle=False) as stored:
                    transposed[expert] = stored["prediction"]
                continue
            model = campaign.make_model(checkpoint, streamed=expert == "general")
            model.max_num_rows = (10000 if dataset in ("1M", "10M") else 9976) if expert == "general" else 2000
            model.max_num_cols = 1000
            if expert == "general":
                means, scales = normalization_parameters(observed, "global")
                offsets = row_scales = None
            else:
                means, scales, offsets = inference_components(observed, "two_way_user_scale")
                row_scales = inference_row_scales(observed, "two_way_user_scale", means, scales, offsets)
            normalization = {"means": means, "scales": scales}
            if offsets is not None:
                normalization.update(offsets=offsets, row_scales=row_scales)
            atomic_npz(case / f"{expert}_normalization.npz", **normalization)

            def progress(row, row_count, column, column_count):
                record = {"state": "running", "dataset": dataset, "expert": expert, "orientation": "transpose",
                          "row_block": row, "row_blocks": row_count, "column_block": column, "column_blocks": column_count}
                write_json(output / "status.json", record)
                print(json.dumps(record), flush=True)

            started = time.perf_counter()
            prediction, counts = collect_predictions(model, observed, users, items, means, scales, offsets, row_scales, progress)
            prediction = np.clip(prediction, split["train_ratings"].min(), split["train_ratings"].max())
            atomic_npz(artifact, prediction=prediction, counts=counts)
            write_json(record_path, {"seconds": time.perf_counter() - started, "prediction_sha256": sha256(artifact),
                       "normalization_sha256": sha256(case / f"{expert}_normalization.npz")})
            transposed[expert] = prediction
            del model
            torch.cuda.empty_cache()
        decision = fit_observed_blend(split["train_ratings"].astype(np.float64),
                                     transposed["general"][:evidence_count], transposed["specialist"][:evidence_count])
        weight = decision["specialist_weight"]
        transposed = {name: prediction[evidence_count:] for name, prediction in transposed.items()}
        transposed["observed_blend"] = (1 - weight) * transposed["general"] + weight * transposed["specialist"]
        if dataset in ("1M", "10M"):
            parent_prediction = LARGE / dataset / "query_predictions.npz"
            expected_hash = campaign.read_json(LARGE / dataset / "audit.json")["query_predictions_sha256"]
        else:
            folder = "movielens_100k" if dataset == "100K" else "netflix_small"
            parent_prediction = PARENT / "recsys" / f"{folder}.npz"
            expected_hash = campaign.read_json(PARENT / "recsys" / f"{folder}.json")["prediction_sha256"]
        assert sha256(parent_prediction) == expected_hash
        with np.load(parent_prediction, allow_pickle=False) as stored:
            original = {name: stored[name] for name in BASE_VARIANTS}
        predictions = combine_predictions(original, transposed)
        for variant, prediction in predictions.items():
            rows.append({"dataset": paths["label"], "variant": variant, **audit_scores(split, prediction)})
        atomic_npz(case / "query_predictions.npz", **predictions)
        write_json(case / "audit.json", {"decision": decision, "split_sha256": sha256(paths["split"]),
                   "parent_prediction_sha256": expected_hash, "independently_rescored": True,
                   "prediction_sha256": sha256(case / "query_predictions.npz"),
                   "rating_range": [float(split["train_ratings"].min()), float(split["train_ratings"].max())]})
        write_csv(output / "summary.csv", rows)
        del observed, predictions, original, transposed, split
    write_json(output / "status.json", {"state": "complete", "datasets": 4})


def register(output, limit):
    cases = campaign.read_json(PARENT / "manifest.json")["cases"]
    if limit is not None:
        cases = [case for suite in ("missbench", "uci") for case in
                 [case for case in cases if case["suite"] == suite][:limit]]
    sources = [Path(__file__).resolve(), *sorted((campaign.REPO / "src/tabimpute").rglob("*.py")),
               *sorted((campaign.BENCH / "code").glob("*.py"))]
    sources += [campaign.REPO / "scripts" / name for name in (
        "evaluate_two_expert_router.py", "evaluate_matrixformer_plus_large_recsys.py", "summarize_orientation_ensemble.py",
        "evaluate_recsys_specialist_transfer.py", "dwip_recsys_experiment.py", "evaluate_chunked_uci.py",
        "evaluate_dwip_missbench_uci.py", "compare_recsys_checkpoint.py", "evaluate_large_recsys.py",
        "evaluate_matrixformer_recsys_ratings.py", "recsys_evaluation_common.py", "run_dwip_transfer_evaluation.py")]
    sources.append(campaign.REPO / "benchmark/eval_dwip_transpose.py")
    inputs = [campaign.GENERAL, campaign.CHECKPOINT, campaign.GENERAL.parent / "config.json",
              campaign.CHECKPOINT.parent / "config.json", PARENT / "manifest.json", PARENT / "final_audit.json"]
    inputs += [campaign.BENCH / "data" / f"{dataset}.matrix.npz" for dataset in campaign.DATASETS]
    inputs += [reference_paths(dataset, "global")["split"] for dataset in ("100K", "netflix", "1M", "10M")]
    if limit is None:
        inputs += [LARGE / "manifest.json", LARGE / "final_audit.json"]
        inputs += [LARGE / dataset / "query_predictions.npz" for dataset in ("1M", "10M")]
    manifest = {"cases": cases, "pilot_limit": limit, "variants": VARIANTS,
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "source_sha256": {str(path): sha256(path) for path in sources},
        "original_source": str(PARENT), "large_original_source": str(LARGE),
        "protocol": {
            "transpose": "Transpose observed matrix and mask together; recompute each deployed recipe on that orientation; map predictions back",
            "causal_consistency": "Matches benchmark/eval_dwip_transpose.py ordering: transpose first, then observed-column standardization",
            "empty_transposed_columns": "As in the causal harness, zero mean/unit scale for columns with no observations; no entries filled before inference",
            "average": "Arithmetic mean of original and mapped-back transpose predictions, before scoring; not a mean of metrics",
            "plus": "Fit one observed-only convex expert weight separately in each orientation; average final blended outputs",
            "tabular": "Original archived predictions reused; full MissBench; UCI up to2000x1000 in each oriented matrix",
            "llm": "Unchanged folds, masks, selection and scoring; inference transpose occurs after constructing each masked test batch",
            "recsys": "Original archived predictions reused; global general max9976/10000x1000; specialist max2000x1000; row-residual scales act on items when transposed",
            "clipping": "Both orientations clip each expert to observed training rating range before blending; 10M includes0.5",
            "selection": "All variants reported separately; no outer-label selection or tuning; causal experiments already separate",
            "backend": "cuDNN SDPA enabled only for LLM, as in frozen original evaluations",
        }}
    manifest = json.loads(json.dumps(manifest))
    path = output / "manifest.json"
    if path.exists():
        assert campaign.read_json(path) == manifest, "Frozen orientation manifest changed"
    else:
        write_json(path, manifest)
        for source in sources:
            target = output / "source_snapshot" / source.relative_to(campaign.REPO.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("all", "tabular", "llm", "recsys"), default="all")
    parser.add_argument("--pilot-limit", type=int)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f".{arguments.stage}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = register(output, arguments.pilot_limit)
    started = time.monotonic()
    try:
        if arguments.stage == "all":
            assert campaign.read_json(LARGE / "final_audit.json")["state"] == "complete"
            for stage in ("tabular", "llm", "recsys"):
                write_json(output / "status.json", {"state": "running", "stage": stage})
                command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--stage", stage]
                if arguments.pilot_limit is not None:
                    command += ["--pilot-limit", str(arguments.pilot_limit)]
                with (output / f"{stage}.log").open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            torch.set_num_threads(4)
            torch.manual_seed(42)
            np.random.seed(42)
            torch.backends.cuda.enable_cudnn_sdp(arguments.stage == "llm")
            destination = output / arguments.stage
            destination.mkdir(exist_ok=True)
            if arguments.stage == "tabular":
                run_tabular(destination, manifest)
            elif arguments.stage == "llm":
                run_llm(destination)
            else:
                run_recsys(destination)
        write_json(output / f"{arguments.stage}_completion.json", {"state": "complete", "seconds": time.monotonic() - started})
        if arguments.stage == "all":
            write_json(output / "status.json", {"state": "complete", "seconds": time.monotonic() - started})
    except BaseException:
        write_json(output / "status.json", {"state": "failed", "stage": arguments.stage, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
