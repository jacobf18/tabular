"""Frozen observed-only router comparison on imputation, ratings and LLM scores."""

import argparse
from collections import defaultdict
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from evaluate_chunked_uci import configure_windows
from evaluate_dwip_missbench_uci import PATTERNS, score_predictions, standardize_uci
from evaluate_recsys_specialist_transfer import (
    BENCH, CHECKPOINT, CHUNKED, DATASETS, FULL, GENERAL, LLM, REPO, ROOT,
    decode_prediction, read_json, reference_record,
)
from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from run_dwip_transfer_evaluation import METHODS, evaluation_environment
from tabimpute.routing import TwoExpertRouter


OUTPUT = ROOT / "two_expert_router_v2_20261004"
SPECIALIST = ROOT / "recsys_specialist_transfer_20261002"
RECSYS = ROOT / "matrixformer_large_trained_recsys_20260930"
RECHECK = ROOT / "recsys_specialist_checkpoint_recheck_20261002/fresh_evaluation"
VARIANTS = ("general", "specialist", "observed_hard", "observed_blend", "holdout_hard", "holdout_blend")


def register(output, limit, reuse_completed=None):
    cases = read_json(FULL / "manifest.json")["protocol"]["cases"]
    if limit is not None:
        cases = [case for suite in ("missbench", "uci") for case in
                 [item for item in cases if item["suite"] == suite][:limit]]
    sources = [Path(__file__).resolve(), *sorted((REPO / "src/tabimpute").rglob("*.py")),
               *sorted((BENCH / "code").glob("*.py"))]
    sources += [REPO / "scripts" / name for name in (
        "evaluate_recsys_specialist_transfer.py", "evaluate_chunked_uci.py", "evaluate_dwip_missbench_uci.py",
        "recsys_evaluation_common.py", "run_dwip_transfer_evaluation.py", "dwip_recsys_experiment.py",
        "evaluate_dwip_recsys.py", "evaluate_matrixformer_recsys_ratings.py", "audit_dwip_missbench_uci.py")]
    sources += [REPO / "benchmark/plot_options.py", REPO / "benchmark/uci/plot_uci_negative_rmse.py"]
    inputs = [GENERAL, CHECKPOINT, GENERAL.parent / "config.json", CHECKPOINT.parent / "config.json",
              FULL / "manifest.json", CHUNKED / "manifest.json", LLM / "manifest.json", SPECIALIST / "manifest.json"]
    inputs += [BENCH / "data" / f"{dataset}.matrix.npz" for dataset in DATASETS]
    for folder in ("movielens_100k", "netflix_small"):
        inputs.append(ROOT / "dwip_teacher_mean128k_selected98k_recsys_20260919" / folder / "split.npz")
    manifest = {
        "general": str(GENERAL), "specialist": str(CHECKPOINT), "cases": cases, "pilot_limit_per_suite": limit,
        "variants": VARIANTS, "seed": 42, "holdout_fraction": .1,
        "routing": "One decision per input matrix (per test batch for LLM); ties/no evidence select general",
        "observed": "Raw full predictions, NOT copied observations; in-sample reconstruction may see its own target",
        "blend": "Analytic nonnegative sum-to-one two-expert squared-error fit on routing evidence only",
        "holdout": "Mask 10% of observations, preserving row/column support; refit all preprocessing; restore full context for output",
        "loss": "Cell-mean squared error, observed-column SD normalized for tabular/LLM; global SD for ratings",
        "tabular": "Frozen 462 MissBench full-matrix and 1800 UCI 2000x1000 cases; original NRMSE, equal dataset weighting",
        "recsys": "MovieLens100K and Netflix small; general global 9976x1000 streamed, specialist two_way_user_scale 2000x1000; clip [1,5]; user-macro RMSE/MAE",
        "recsys_scope": "Routes complete deployed expert recipes, not checkpoint-only differences; excludes ML1M/10M",
        "llm": "Four suites, ten folds, entropy/MI/random, budgets1-15, max4000 rows/test batches200; unchanged historical scorer",
        "attention_backend": "cuDNN SDPA disabled for tabular/recsys, enabled for LLM to match each historical harness",
        "selection": "Fixed before this campaign; previously inspected frozen benchmarks, report-only; no outer-label tuning",
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "source_sha256": {str(path): sha256(path) for path in sources},
    }
    if reuse_completed is not None:
        previous = read_json(reuse_completed / "manifest.json")
        if previous["input_sha256"] != manifest["input_sha256"] or previous["cases"] != cases or limit is not None:
            raise ValueError("Completed-stage inputs or case coverage differ")
        previous_script = reuse_completed / "source_snapshot/mcpfn/scripts/evaluate_two_expert_router.py"
        import ast

        def definitions(path):
            return {node.name: ast.dump(node) for node in ast.parse(path.read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        before, after = definitions(previous_script), definitions(Path(__file__))
        for name in ("make_model", "TabularExpert", "RatingsExpert", "run_tabular", "run_recsys", "summarize_tabular"):
            if before[name] != after[name]:
                raise ValueError(f"Cannot reuse changed evaluator: {name}")
        for filename, digest in previous["source_sha256"].items():
            if filename != str(Path(__file__).resolve()) and manifest["source_sha256"].get(filename) != digest:
                raise ValueError(f"Cannot reuse changed inference/scoring dependency: {filename}")
        manifest["reused_stages"] = {}
        for stage in ("tabular", "recsys"):
            path = reuse_completed / stage
            if read_json(path / "status.json")["state"] != "complete":
                raise ValueError(f"Cannot reuse incomplete stage: {stage}")
            manifest["reused_stages"][stage] = {"directory": str(path), "manifest": str(reuse_completed / "manifest.json"),
                                              "manifest_sha256": sha256(reuse_completed / "manifest.json")}
            target = output / stage
            if not target.exists():
                target.symlink_to(path, target_is_directory=True)
            if target.resolve() != path.resolve():
                raise ValueError("Reuse directory conflicts with existing outputs")
    manifest = json.loads(json.dumps(manifest))
    path = output / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Frozen campaign changed; use a new output directory")
    else:
        write_json(path, manifest)
        for source in sources:
            target = output / "source_snapshot" / source.relative_to(REPO.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return manifest


def check_reserve(output):
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise RuntimeError("20 GiB free-space reserve reached")


def make_model(checkpoint, streamed=False):
    from tabimpute.interface import ImputePFN
    from tabimpute.streamed_inference import StreamedImputePFN

    factory = StreamedImputePFN if streamed else ImputePFN
    return factory(device="cuda", checkpoint_path=str(checkpoint), entry_wise_features=False,
                   json_config={"config": read_json(checkpoint.parent / "config.json")["model"]},
                   model_family="new_stable", precision_mode="amp_bfloat16", prediction_stat="mean",
                   max_num_rows=None, max_num_cols=None, max_num_chunks=1, verbose=False)


class TabularExpert:
    def __init__(self, model):
        self.model = model
        self.suite = None

    def impute(self, matrix, return_full=False):
        if self.suite == "uci":
            normalized, mean, scale, _ = standardize_uci(matrix)
            configure_windows(self.model, matrix.shape, 2000, 1000)
        else:
            normalized, mean, scale = matrix, 0., 1.
            self.model.max_num_rows = self.model.max_num_cols = None
        completed, raw = self.model.impute(normalized.copy(), return_full=True)
        completed, raw = completed * scale + mean, raw * scale + mean
        return (completed, raw) if return_full else completed


class RatingsExpert:
    def __init__(self, model, specialist):
        self.model, self.specialist = model, specialist
        self.model.max_num_rows = 2000 if specialist else 9976
        self.model.max_num_cols = 1000

    def impute(self, matrix, return_full=False):
        from dwip_recsys_experiment import inference_components, inference_row_scales
        from evaluate_matrixformer_recsys_ratings import normalization_parameters

        if self.specialist:
            means, scales, offsets = inference_components(matrix, "two_way_user_scale")
            row_scales = inference_row_scales(matrix, "two_way_user_scale", means, scales, offsets)
            source = ((matrix - offsets[:, None] - means) / scales) / row_scales[:, None]
            completed, raw = self.model.impute(source, means=np.zeros_like(means), stds=np.ones_like(scales), return_full=True)
            completed = completed * row_scales[:, None] * scales + means + offsets[:, None]
            raw = raw * row_scales[:, None] * scales + means + offsets[:, None]
        else:
            means, scales = normalization_parameters(matrix, "global")
            completed, raw = self.model.impute(matrix, means=means, stds=scales, return_full=True)
        completed, raw = np.clip(completed, 1, 5), np.clip(raw, 1, 5)
        completed[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
        return (completed, raw) if return_full else completed


def summarize_tabular(output, records, expected):
    groups = defaultdict(list)
    flat = []
    for record in records:
        groups[record["suite"], record["pattern"]].append(record)
        flat.append({key: record[key] for key in ("case_id", "suite", "dataset", "pattern", "missingness", "repeat")} |
                    record["nrmse"] | {f"{mode}_{key}": value for mode, decision in record["decisions"].items()
                                       for key, value in decision.items()})
    write_csv(output / "per_case.csv", flat)
    summaries = []
    for (suite, pattern), group in groups.items():
        datasets = {row["dataset"] for row in group}
        row = {"suite": suite, "pattern": pattern, "cases": len(group), "datasets": len(datasets),
               "complete": len(group) == sum(case["suite"] == suite and case["pattern"] == pattern for case in expected)}
        for variant in VARIANTS:
            row[variant] = float(np.mean([np.mean([item["nrmse"][variant] for item in group if item["dataset"] == dataset])
                                         for dataset in sorted(datasets)])) if row["complete"] else None
        for mode in ("observed", "holdout"):
            row[f"{mode}_specialist_fraction"] = float(np.mean([item["decisions"][mode]["selected"] == "specialist" for item in group]))
            row[f"{mode}_specialist_weight"] = float(np.mean([item["decisions"][mode]["specialist_weight"] for item in group]))
        summaries.append(row)
    write_csv(output / "summary.csv", summaries)


def run_tabular(output, manifest):
    from audit_dwip_missbench_uci import load_original

    experts = [TabularExpert(make_model(path)) for path in (GENERAL, CHECKPOINT)]
    router = TwoExpertRouter(*experts)
    scorers = {suite: load_original(REPO / path, "compute_normalized_rmse_columnwise") for suite, path in (
        ("missbench", "benchmark/plot_options.py"), ("uci", "benchmark/uci/plot_uci_negative_rmse.py"))}
    records = []
    for case in manifest["cases"]:
        check_reserve(output)
        source = Path(case["source_directory"])
        reference_root, reference = reference_record(case)
        hashes = {key: sha256(source / name) for key, name in (("missing_sha256", "missing.npy"), ("true_sha256", "true.npy"))}
        if any(reference[key] != value for key, value in hashes.items()):
            raise ValueError("Tabular frozen mask/target changed")
        result_path = output / "cases" / (case["case_id"] + ".json")
        prediction_path = output / "predictions" / (case["case_id"] + ".npz")
        write_json(output / "status.json", {"state": "running", "completed": len(records), "expected": len(manifest["cases"]), "current": case["case_id"]})
        if result_path.exists():
            record = read_json(result_path)
            if record["prediction_sha256"] != sha256(prediction_path) or any(record[key] != value for key, value in hashes.items()):
                raise ValueError("Resume artifact mismatch")
            records.append(record)
            continue
        missing, truth = (np.load(source / name, allow_pickle=False) for name in ("missing.npy", "true.npy"))
        for expert in experts:
            expert.suite = case["suite"]
        result = router.predict_all(missing)
        scores, differences = {}, {}
        for variant, prediction in result.predictions.items():
            score = score_predictions(truth, prediction, missing, case["suite"])["nrmse_original"]
            original = float(scorers[case["suite"]](truth, prediction, np.isnan(missing)))
            np.testing.assert_allclose(score, original, rtol=1e-5, atol=1e-5)
            scores[variant] = score
        references = {"general": (reference_root, reference), "specialist": (SPECIALIST / "tabular", read_json(SPECIALIST / "tabular/cases" / (case["case_id"] + ".json")))}
        for name, (root, record) in references.items():
            path = root / "predictions" / (case["case_id"] + ".npz")
            if sha256(path) != record["prediction_sha256"]:
                raise ValueError("Archived prediction changed")
            previous = decode_prediction(path, missing)
            difference = float(np.max(np.abs(previous[np.isnan(missing)] - result.predictions[name][np.isnan(missing)])))
            np.testing.assert_allclose(result.predictions[name][np.isnan(missing)], previous[np.isnan(missing)], atol=1e-6, rtol=1e-6)
            differences[name] = difference
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(prediction_path, shape=np.array(missing.shape), **{name: pred[np.isnan(missing)] for name, pred in result.predictions.items()})
        record = {**case, **hashes, "nrmse": scores, "decisions": result.decisions, "seconds": result.seconds,
                  "baseline_prediction_max_abs_difference": differences, "prediction_sha256": sha256(prediction_path)}
        write_json(result_path, record)
        records.append(record)
        if len(records) % 25 == 0:
            summarize_tabular(output, records, manifest["cases"])
            print(f"Tabular {len(records)}/{len(manifest['cases'])}", flush=True)
    summarize_tabular(output, records, manifest["cases"])
    write_json(output / "status.json", {"state": "complete", "completed": len(records), "expected": len(manifest["cases"])})


def run_recsys(output):
    from dwip_recsys_experiment import load_split
    from evaluate_dwip_recsys import user_rating_metrics

    router = TwoExpertRouter(RatingsExpert(make_model(GENERAL, streamed=True), False),
                             RatingsExpert(make_model(CHECKPOINT), True), normalization="global")
    rows = []
    for key, folder in (("100K", "movielens_100k"), ("netflix", "netflix_small")):
        check_reserve(output)
        write_json(output / "status.json", {"state": "running", "current": folder})
        missing, context, queries = load_split(folder, "test")
        result = router.predict_all(missing)
        selections = (queries[0], queries[1])
        predictions = {name: matrix[selections] for name, matrix in result.predictions.items()}
        references = {"general": RECSYS / key / "global/predictions.npz", "specialist": RECHECK / folder / "query_predictions.npz"}
        differences = {}
        for name, path in references.items():
            with np.load(path, allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["prediction"].shape, predictions[name].shape)
                np.testing.assert_allclose(stored["prediction"], predictions[name], rtol=0, atol=1e-7)
                differences[name] = float(np.max(np.abs(stored["prediction"] - predictions[name])))
        for name, prediction in predictions.items():
            rows.append({"dataset": folder, "variant": name, **user_rating_metrics(queries[0], queries[2], prediction)})
        path = output / f"{folder}.npz"
        atomic_npz(path, users=queries[0], items=queries[1], truth=queries[2], **predictions)
        write_json(output / f"{folder}.json", {"decisions": result.decisions, "seconds": result.seconds,
                   "context_ratings": len(context[2]), "query_ratings": len(queries[2]),
                   "baseline_max_abs_difference": differences, "prediction_sha256": sha256(path),
                   "reference_sha256": {str(path): sha256(path) for path in references.values()}})
        write_csv(output / "summary.csv", rows)
        print(f"Recsys complete: {folder}", flush=True)
    write_json(output / "status.json", {"state": "complete", "completed": 2, "expected": 2})


class BatchCollector:
    def __init__(self, router):
        self.router, self.batches = router, []

    def impute(self, matrix):
        result = self.router.predict_all(matrix)
        self.batches.append((matrix.copy(), result))
        return result.predictions["observed_hard"]


class Replay:
    def __init__(self, batches, variant):
        self.batches, self.variant, self.index = batches, variant, 0

    def impute(self, matrix):
        original, result = self.batches[self.index]
        np.testing.assert_array_equal(matrix, original)
        self.index += 1
        return result.predictions[self.variant].copy()


def run_llm(output):
    environment = evaluation_environment(output, GENERAL, "two_expert_router", ",".join(DATASETS))
    for key in list(os.environ):
        if key.startswith("BENCHSELECT_"):
            del os.environ[key]
    os.environ.update({key: value for key, value in environment.items() if key.startswith("BENCHSELECT_")})
    specification = importlib.util.spec_from_file_location("router_llm_harness", BENCH / "code/eval_tabimpute.py")
    harness = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(harness)
    original_score = harness.impute_tabimpute
    router = TwoExpertRouter(*(TabularExpert(make_model(path)) for path in (GENERAL, CHECKPOINT)))
    summaries, folds = [], []
    for dataset in DATASETS:
        check_reserve(output)
        write_json(output / "status.json", {"state": "running", "current": dataset})
        collected = []
        audit_path = output / f"{dataset}_routing.jsonl"
        with audit_path.open("w") as audit:
            def score_all(imputer, *arguments):
                collector = BatchCollector(router)
                values = {"observed_hard": original_score(collector, *arguments)}
                for variant in VARIANTS:
                    if variant != "observed_hard":
                        replay = Replay(collector.batches, variant)
                        values[variant] = original_score(replay, *arguments)
                        if replay.index != len(collector.batches):
                            raise ValueError("LLM replay did not consume all batches")
                for index, (_, result) in enumerate(collector.batches):
                    audit.write(json.dumps({"setting_index": len(collected), "batch": index,
                                           "decisions": result.decisions, "seconds": result.seconds}, allow_nan=False) + "\n")
                audit.flush()
                collected.append(values)
                return values["observed_hard"]
            harness.impute_tabimpute = score_all
            hard_metrics, names = harness.run_dataset(dataset, router)
        if len(collected) != 10 * len(METHODS) * 15:
            raise ValueError("Incomplete LLM setting coverage")
        arrays = {"benchmark_names": names}
        for variant in VARIANTS:
            for metric in ("r2", "rmse"):
                values = np.array([row[variant][metric] for row in collected]).reshape(10, len(METHODS), 15)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite LLM panel")
                for method_index, method in enumerate(METHODS):
                    panel = values[:, method_index, :]
                    arrays[f"{variant}_{method}_{metric}"] = panel
                    if variant == "observed_hard":
                        np.testing.assert_array_equal(panel, hard_metrics[method][metric])
                    for budget in range(1, 16):
                        summaries.append({"dataset": dataset, "variant": variant, "method": method, "budget": budget,
                                          "metric": metric, "value": float(panel[:, budget-1].mean())})
                        for fold in range(10):
                            folds.append({"dataset": dataset, "variant": variant, "method": method, "budget": budget,
                                          "metric": metric, "fold": fold, "value": float(panel[fold, budget-1])})
        differences = {}
        for variant, path in (("general", LLM / "logs" / f"tabimpute_matrixformer_large_trained_{dataset}.npz"),
                              ("specialist", SPECIALIST / "llm/logs" / f"tabimpute_recsys_specialist_transfer_{dataset}.npz")):
            with np.load(path, allow_pickle=True) as stored:
                np.testing.assert_array_equal(names, stored["benchmark_names"])
                for method in METHODS:
                    for metric in ("r2", "rmse"):
                        current, previous = arrays[f"{variant}_{method}_{metric}"], stored[f"{method}_{metric}"]
                        np.testing.assert_allclose(current, previous, atol=1e-6, rtol=1e-6)
                        differences[f"{variant}_{method}_{metric}"] = float(np.max(np.abs(current - previous)))
        atomic_npz(output / f"{dataset}.npz", **arrays)
        write_json(output / f"{dataset}_audit.json", {"baseline_panel_max_abs_difference": differences,
                   "routing_sha256": sha256(audit_path), "panel_sha256": sha256(output / f"{dataset}.npz")})
        write_csv(output / "summary.csv", summaries)
        write_csv(output / "per_fold.csv", folds)
    write_json(output / "status.json", {"state": "complete", "completed": 4, "expected": 4})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("all", "tabular", "recsys", "llm"), default="all")
    parser.add_argument("--pilot-limit", type=int)
    parser.add_argument("--reuse-completed", type=Path)
    arguments = parser.parse_args()
    if arguments.reuse_completed is not None and arguments.stage in {"tabular", "recsys"}:
        parser.error("Reused stages are immutable; use --stage all or --stage llm")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f".{arguments.stage}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reuse_completed = arguments.reuse_completed.resolve() if arguments.reuse_completed is not None else None
    manifest = register(output, arguments.pilot_limit, reuse_completed)
    check_reserve(output)
    try:
        execute(arguments, manifest, output)
    except BaseException as error:
        write_json(output / f"{arguments.stage}_failure.json", {"state": "failed", "error": repr(error)})
        write_json(output / "status.json", {"state": "failed", "stage": arguments.stage, "error": repr(error)})
        raise


def execute(arguments, manifest, output):
    started = time.monotonic()
    if arguments.stage == "all":
        for stage in ("tabular", "recsys", "llm"):
            if stage in manifest.get("reused_stages", {}):
                continue
            write_json(output / "status.json", {"state": "running", "current": stage})
            command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage, "--output", str(output)]
            if arguments.pilot_limit is not None:
                command += ["--pilot-limit", str(arguments.pilot_limit)]
            if arguments.reuse_completed is not None:
                command += ["--reuse-completed", str(arguments.reuse_completed.resolve())]
            with (output / f"{stage}.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        import torch

        torch.set_num_threads(4)
        torch.backends.cuda.enable_cudnn_sdp(arguments.stage == "llm")
        torch.cuda.set_per_process_memory_fraction(.9)
        torch.manual_seed(42)
        np.random.seed(42)
        destination = output / arguments.stage
        destination.mkdir(parents=True, exist_ok=True)
        if arguments.stage == "tabular":
            run_tabular(destination, manifest)
        elif arguments.stage == "recsys":
            run_recsys(destination)
        else:
            run_llm(destination)
    write_json(output / f"{arguments.stage}_completion.json", {"state": "complete", "seconds": time.monotonic() - started})
    if arguments.stage == "all":
        write_json(output / "status.json", {"state": "complete", "seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
