"""Full public-interface rating/ranking evaluation of an inner-selected candidate."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from dwip_recsys_experiment import DEFAULT_ROOT, load_split, cornac_split, inference_components, inference_row_scales, selection_score
from dwip_recsys_provenance import require_synthetic_checkpoint
from dwip_recsys_ranking_audit import independent_ranking_metrics, RANKING_METRICS
from evaluate_dwip_recsys import (
    CHECKPOINT, OUTPUT, DATASETS, FrozenRatings, METRIC_NAMES, metrics,
    normalize_training_ratings, sha256, user_rating_metrics, write_json, write_csv, atomic_npz,
)
from tabimpute.interface import ImputePFN


def remap_prediction(prediction, train_set):
    users = np.asarray([int(raw) for raw, index in sorted(train_set.uid_map.items(), key=lambda entry: entry[1])])
    items = np.asarray([int(raw) for raw, index in sorted(train_set.iid_map.items(), key=lambda entry: entry[1])])
    return prediction[np.ix_(users, items)], users


def compare_metrics(values, baseline):
    return {metric: {"delta": values[metric] - baseline[metric],
                     "better": values[metric] < baseline[metric] if metric in ("MAE", "RMSE") else values[metric] > baseline[metric]}
            for metric in METRIC_NAMES}


def impute_candidate_ratings(imputer, observed, normalization):
    means, stds, row_offsets = inference_components(observed, normalization)
    row_scales = inference_row_scales(observed, normalization, means, stds, row_offsets)
    if row_scales is None:
        prediction = imputer.impute(observed - row_offsets[:, None], means=means, stds=stds) + row_offsets[:, None]
    else:
        source = (observed - row_offsets[:, None] - means) / stds / row_scales[:, None]
        normalized = imputer.impute(source, means=np.zeros_like(means), stds=np.ones_like(stds))
        prediction = normalized * row_scales[:, None] * stds + means + row_offsets[:, None]
    return np.clip(prediction, 1, 5)


def verify_selection(checkpoint):
    status = json.loads((checkpoint.parent / "status.json").read_text())
    if status["state"] != "complete" or checkpoint.name != "best_validation_checkpoint.pth":
        raise ValueError("Outer confirmation requires a completed, inner-selected training run")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    step = payload["step"]
    selected = json.loads((checkpoint.parent / f"validation_{step}.json").read_text())
    candidates = [json.loads(path.read_text()) for path in checkpoint.parent.glob("validation_[0-9]*.json")]
    if not selected["improved"] or selected["score"] != min(value["score"] for value in candidates):
        raise ValueError("Checkpoint does not reproduce inner-validation selection")
    if selected["score"] != status["best_score"]:
        raise ValueError("Selection does not match the completed run")
    return {"step": step, "score": selected["score"], "validation_sha256": sha256(checkpoint.parent / f"validation_{step}.json")}


def context_window_selection(checkpoint, screen_path=None, root=DEFAULT_ROOT):
    if screen_path is None:
        return [1000, 1000], None
    report = json.loads(screen_path.read_text())
    if report.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Context screen checkpoint hash mismatch")
    if not all(report.get(key) is True for key in ("weights_bitwise_unchanged", "gradients_absent", "legacy_scores_reproduced")):
        raise ValueError("Context screen lacks no-update and legacy-control verification")
    if report.get("outer_evaluated") is not False or report.get("axis") not in ("items", "users"):
        raise ValueError("Require a registered inner-only context screen")
    config = json.loads((checkpoint.parent / "config.json").read_text())
    if report.get("normalization") != config["arguments"]["normalization"]:
        raise ValueError("Context screen normalization mismatch")
    selected = verify_selection(checkpoint)
    if selected != report.get("selection"):
        raise ValueError("Context screen training-selection mismatch")
    baseline_path = root / "mf_baseline.json"
    if report.get("baseline_sha256") != sha256(baseline_path):
        raise ValueError("Context screen baseline hash mismatch")
    for spec in DATASETS.values():
        if report.get("split_sha256", {}).get(spec["folder"]) != sha256(OUTPUT / spec["folder"] / "split.npz"):
            raise ValueError("Context screen split hash mismatch")
    expected_sources = {"screen_dwip_recsys_context_windows.py", "dwip_recsys_experiment.py", "recsys_rating_episode.py",
                        "model_new_stable.py", "bar_distribution.py", "interface.py"}
    sources = report.get("source_sha256", {})
    if len(sources) != len(expected_sources) or {Path(source).name for source in sources} != expected_sources:
        raise ValueError("Context screen source manifest incomplete")
    for source, digest in sources.items():
        if sha256(screen_path.with_suffix(".source_snapshot") / Path(source).name) != digest:
            raise ValueError("Context screen source snapshot hash mismatch")
    windows = [[1000, 1000], [1000, 2000] if report["axis"] == "items" else [2000, 1000]]
    candidates = report.get("candidates", [])
    if report.get("registered_windows") != windows or [value["block_shape"] for value in candidates] != windows:
        raise ValueError("Context screen candidates differ from registration")
    baseline = json.loads(baseline_path.read_text())["validation"]
    legacy = json.loads((checkpoint.parent / f"validation_{selected['step']}.json").read_text())["datasets"]
    if candidates[0]["datasets"] != legacy:
        raise ValueError("Context screen legacy scores do not match selected training checkpoint")
    for candidate in candidates:
        if set(candidate["datasets"]) != set(baseline):
            raise ValueError("Context screen dataset coverage mismatch")
        if any(not np.isfinite(values[metric]) or values[metric] < 0
               for values in candidate["datasets"].values() for metric in ("MAE", "RMSE")):
            raise ValueError("Context screen has invalid rating metrics")
        score = selection_score(candidate["datasets"], baseline, "worst_rating")
        if not np.isfinite(score) or score != candidate["worst_inner_rating_ratio"]:
            raise ValueError("Context screen score mismatch")
    best = min(candidates, key=lambda value: value["worst_inner_rating_ratio"])
    if best["block_shape"] != report.get("best_inner_window") or best is candidates[0]:
        raise ValueError("Context screen does not select an improved wider window")
    return best["block_shape"], {"screen": str(screen_path.resolve()), "screen_sha256": sha256(screen_path),
                                 "axis": report["axis"], "worst_inner_rating_ratio": best["worst_inner_rating_ratio"],
                                 "selection": "Registered inner-only context window; weights selected independently on inner validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--context-screen", type=Path)
    args = parser.parse_args()
    provenance = require_synthetic_checkpoint(args.checkpoint)
    selection = verify_selection(args.checkpoint) if args.partition == "test" else None
    block_shape, context_selection = context_window_selection(args.checkpoint, args.context_screen, args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    run_config = json.loads((args.checkpoint.parent / "config.json").read_text())
    config = run_config["model"]
    normalization = run_config.get("arguments", {}).get("normalization", "column")
    if config != json.loads((CHECKPOINT.parent / "config.json").read_text())["model"]:
        raise ValueError("Candidate changed DWIP v2-2 architecture")
    baseline = json.loads((args.root / "mf_baseline.json").read_text())[args.partition]
    manifest = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
                "config_sha256": sha256(args.checkpoint.parent / "config.json"), "partition": args.partition,
                "evaluator_sha256": sha256(Path(__file__)), "split_helper_sha256": sha256(Path(__file__).with_name("dwip_recsys_experiment.py")),
                "mf_sha256": sha256(args.root / "mf_baseline.json"), "block_shape": block_shape,
                "context_selection": context_selection,
                "normalization": normalization,
                "training_data": run_config.get("training_data", "See training config"),
                "synthetic_provenance": provenance,
                "selection": selection,
                "inference": "Unchanged public interface; one model, predictive mean; FP32/BF16; no MF or post-hoc calibration",
                "split_sha256": {spec["folder"]: sha256(OUTPUT / spec["folder"] / "split.npz") for spec in DATASETS.values()}}
    write_json(args.output / "manifest.json", manifest)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("dwip_recsys_experiment.py").resolve(),
               Path(__file__).with_name("evaluate_dwip_recsys.py").resolve(),
               root / "src/tabimpute/interface.py", root / "src/tabimpute/model/model_new_stable.py",
               root / "src/tabimpute/model/bar_distribution.py",
               root / "src/tabimpute/prior/recsys_rating_episode.py",
               root / "src/tabimpute/prior/recsys_clustered_factors.py",
               Path(__file__).with_name("dwip_recsys_ranking_audit.py").resolve(),
               Path(__file__).with_name("dwip_recsys_provenance.py").resolve()]
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    manifest["source_sha256"] = {str(source): sha256(source) for source in sources}
    for source in sources:
        shutil.copyfile(source, snapshot / source.name)
    if args.context_screen is not None:
        shutil.copyfile(args.context_screen, args.output / "context_selection.json")
    write_json(args.output / "manifest.json", manifest)
    imputer = ImputePFN(device="cuda", checkpoint_path=str(args.checkpoint), entry_wise_features=False,
                        json_config={"config": config}, model_family="new_stable", prediction_stat="mean",
                        precision_mode="amp_bfloat16", max_num_rows=block_shape[0], max_num_cols=block_shape[1], max_num_chunks=1, verbose=False)
    summaries = []
    comparisons = {}
    for spec in DATASETS.values():
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        observed, context, queries = load_split(spec["folder"], args.partition)
        prediction = impute_candidate_ratings(imputer, observed, normalization)
        if not np.isfinite(prediction).all() or not np.allclose(prediction[context[0], context[1]], context[2], atol=1e-12, rtol=1e-12):
            raise ValueError("Nonfinite prediction or altered observed rating")
        split = cornac_split(context, queries)
        mapped, original_users = remap_prediction(prediction, split.train_set)
        result, _ = split.evaluate(model=FrozenRatings(prediction=mapped), metrics=metrics(), user_based=True)
        values = {metric: float(result.metric_avg_results[metric]) for metric in METRIC_NAMES}
        independent = user_rating_metrics(queries[0], queries[2], prediction[queries[0], queries[1]])
        for metric in ("MAE", "RMSE"):
            if not np.isclose(values[metric], independent[metric], atol=1e-12, rtol=1e-12):
                raise ValueError("Independent macro-over-users metric mismatch")
        ranking_values, ranking_users = independent_ranking_metrics(mapped, split.train_set.csr_matrix, split.test_set.csr_matrix)
        for metric in RANKING_METRICS:
            if not np.isclose(values[metric], ranking_values[metric], atol=1e-12, rtol=1e-12):
                raise ValueError("Independent ranking-average mismatch")
            if set(ranking_users) != set(result.metric_user_results[metric]):
                raise ValueError("Independent ranking-user coverage mismatch")
            if any(not np.isclose(scores[metric], result.metric_user_results[metric][user], atol=1e-12, rtol=1e-12)
                   for user, scores in ranking_users.items()):
                raise ValueError("Independent per-user ranking mismatch")
        comparison = compare_metrics(values, baseline[spec["folder"]])
        comparisons[spec["folder"]] = comparison
        summary = {"dataset": spec["folder"], **values, "seconds": time.monotonic() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                   "context_ratings": len(context[2]), "query_ratings": len(queries[2])}
        summaries.append(summary)
        directory = args.output / spec["folder"]
        directory.mkdir()
        write_json(directory / "metrics.json", summary)
        write_json(directory / "independent_ranking_audit.json", {
            "passed": True, "metrics": ranking_values, "users": len(ranking_users),
            "all_per_user_metrics_match": True, "source_sha256": sha256(Path(__file__).with_name("dwip_recsys_ranking_audit.py")),
            "protocol": "Known items; exclude context positives only; relevance >=4; strict AUC ties; max-rank AP ties; Cornac top-10 tie ordering"})
        write_csv(directory / "independent_ranking_per_user.csv", [
            {"user": int(original_users[user]), **scores} for user, scores in sorted(ranking_users.items())])
        users = sorted({user for scores in result.metric_user_results.values() for user in scores})
        write_csv(directory / "per_user_metrics.csv", [{"user": int(original_users[user]), **{
            metric: float(result.metric_user_results[metric][user]) if user in result.metric_user_results[metric] else None
            for metric in METRIC_NAMES}} for user in users])
        atomic_npz(directory / "query_predictions.npz", users=queries[0], items=queries[1], truth=queries[2],
                   prediction=prediction[queries[0], queries[1]])
        print(json.dumps(summary), flush=True)
    write_csv(args.output / "results.csv", summaries)
    write_json(args.output / "comparison.json", {"datasets": comparisons,
        "both_rating_metrics_win_on_both_datasets": all(value[metric]["better"] for value in comparisons.values() for metric in ("MAE", "RMSE")),
        "all_metrics_win_on_both_datasets": all(metric["better"] for value in comparisons.values() for metric in value.values())})


if __name__ == "__main__":
    main()
