"""Single-checkpoint completion on the frozen, support-matched LLM campaign."""

import argparse
from collections import defaultdict
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from evaluate_matched_llm import (
    BATCH_ROWS, BENCH, DATASETS, GENERAL, METHODS, OUTPUT as REFERENCE,
    score_standardized, target_mask, verify_frozen, visible_input,
)
from evaluate_single_matrixformer_tabular import GOAL, REPO, check_reserve, make_model, read_json, verify_hashes
from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from train_streamed_large_matrices import wait_for_gpu


def setting_key(dataset, fold, method, budget):
    return f"{dataset}/fold_{fold:02d}/{method}_{budget:02d}"


def checked_metrics(target, prediction):
    metrics = score_standardized(target, prediction)
    truth = np.maximum(-10., np.minimum(10., target))
    estimate = np.maximum(-10., np.minimum(10., prediction))
    error = truth - estimate
    residual, total = float(np.dot(error, error)), float(np.dot(truth, truth))
    independent = {"r2": 1. - residual / total if total > 1e-12 else None,
                   "rmse": float(np.sqrt(residual / truth.size)) if truth.size else None,
                   "ss_res": residual, "ss_tot": total, "n_eval": int(truth.size)}
    for name in metrics:
        if independent[name] is None:
            if metrics[name] is not None:
                raise ValueError("Metric validity differs from independent scorer")
        else:
            np.testing.assert_allclose(metrics[name], independent[name], rtol=1e-12, atol=1e-12)
    return metrics


def setting_inputs(scores, observed, frozen, method, budget):
    context, test = frozen["context"], frozen["test"]
    if np.intersect1d(context, test).size:
        raise ValueError("Context/test row overlap")
    selected = frozen[f"order_{method}"][:budget]
    mask = frozen[f"mask_{method}_{budget:02d}"]
    np.testing.assert_array_equal(mask, target_mask(observed[test], selected))
    context_input, test_input = visible_input(scores[context], observed[context], scores[test], observed[test], selected)
    if np.isfinite(test_input[mask]).any():
        raise ValueError("Query labels leaked into input")
    truth = ((scores[test] - frozen["mean"]) / frozen["scale"])[mask]
    return context_input, test_input, mask, truth


def predict_setting(model, context_input, test_input, mask, mean, scale):
    prediction = np.empty(test_input.shape, dtype=np.float64)
    for start in range(0, len(test_input), BATCH_ROWS):
        stop = min(start + BATCH_ROWS, len(test_input))
        batch = np.vstack([context_input, test_input[start:stop]])
        completed, _ = model.impute(batch.copy(order="K"), return_full=True)
        if completed.shape != batch.shape or not np.isfinite(completed).all():
            raise ValueError("Invalid single-model prediction")
        prediction[start:stop] = completed[len(context_input):]
    return ((prediction - mean) / scale)[mask]


def frozen_settings():
    for dataset in DATASETS:
        with np.load(BENCH / "data" / f"{dataset}.matrix.npz", allow_pickle=False) as stored:
            scores, observed = stored["B"].astype(np.float64), stored["O"].astype(bool)
        for fold in range(10):
            fold_path = REFERENCE / "frozen" / dataset / f"fold_{fold:02d}.npz"
            with np.load(fold_path, allow_pickle=False) as stored:
                frozen = dict(stored)
            for method in METHODS:
                for budget in range(1, 16):
                    specification = {"dataset": dataset, "fold": fold, "method": method, "budget": budget,
                                     "case_id": setting_key(dataset, fold, method, budget), "fold_path": str(fold_path)}
                    yield specification, frozen, setting_inputs(scores, observed, frozen, method, budget)


def panel_summaries(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["dataset"], record["method"], record["budget"]].append(record)
    summaries = []
    for (dataset, method, budget), cases in sorted(grouped.items()):
        folds = [case["fold"] for case in cases]
        if len(set(folds)) != len(folds):
            raise ValueError("Duplicate LLM fold result")
        complete = sorted(folds) == list(range(10))
        row = {"dataset": dataset, "method": method, "budget": budget, "complete": complete,
               "folds": len(folds), "n_eval": sum(case["student"]["n_eval"] for case in cases)}
        for metric in ("r2", "rmse"):
            for role in ("student", "blend"):
                values = [case[role][metric] for case in cases if case[role][metric] is not None]
                row[f"{role}_{metric}"] = float(np.mean(values)) if complete and values else None
                row[f"{role}_{metric}_folds"] = len(values)
            row[f"delta_{metric}"] = row[f"student_{metric}"] - row[f"blend_{metric}"] if row[f"student_{metric}"] is not None else None
        summaries.append(row)
    return summaries


def check_reference(specification, frozen, inputs):
    path = REFERENCE / "predictions" / (specification["case_id"] + ".npz")
    record = read_json(path.with_suffix(".json"))
    if sha256(path) != record["prediction_sha256"] or sha256(Path(specification["fold_path"])) != record["fold_sha256"]:
        raise ValueError("Frozen LLM prediction/fold changed")
    for key in ("dataset", "fold", "method", "budget"):
        if record[key] != specification[key]:
            raise ValueError("Reference metadata mismatch")
    _, _, mask, truth = inputs
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["target_z"], truth)
        query_rows, query_columns = np.where(mask)
        np.testing.assert_array_equal(saved["query_rows"], query_rows)
        np.testing.assert_array_equal(saved["query_columns"], query_columns)
        metrics = checked_metrics(truth, saved["observed_blend_z"])
        if metrics != record["metrics"]["observed_blend"]:
            raise ValueError("Archived blend scores disagree with saved predictions")
    return metrics, {str(path): record["prediction_sha256"], str(path.with_suffix(".json")): sha256(path.with_suffix(".json"))}


def prepare(output):
    manifest = verify_frozen(REFERENCE)
    sources = {str(REPO.parent / relative): digest for relative, digest in manifest["source_sha256"].items()}
    verify_hashes(sources)
    for name in ("evaluate_single_matrixformer_llm.py", "evaluate_single_matrixformer_tabular.py",
                 "evaluate_two_expert_router.py", "recsys_evaluation_common.py", "train_streamed_large_matrices.py"):
        source = REPO / "scripts" / name
        sources[str(source)] = sha256(source)
    records, cases = [], []
    for specification, frozen, inputs in frozen_settings():
        metrics, hashes = check_reference(specification, frozen, inputs)
        cases.append({**specification, "reference_sha256": hashes})
        records.append({**specification, "student": metrics, "blend": metrics})
    panels = panel_summaries(records)
    paper_path = GOAL / "reference/llm_benchmark_r2.csv"
    with paper_path.open() as stream:
        paper = {(row["dataset"], row["method"], int(row["budget"])): row for row in csv.DictReader(stream)}
    if len(cases) != 1800 or len(panels) != 180 or set(paper) != {(row["dataset"], row["method"], row["budget"]) for row in panels}:
        raise ValueError("Incomplete matched LLM panel inventory")
    for row in panels:
        previous = paper[row["dataset"], row["method"], row["budget"]]
        np.testing.assert_allclose([row["blend_r2"], row["blend_rmse"]],
                                   [float(previous["matrixformer_plus"]), float(previous["matrixformer_plus_rmse"])], rtol=1e-12, atol=1e-12)
        if row["n_eval"] != int(previous["n_eval"]) or row["blend_r2_folds"] != int(previous["matrixformer_plus_folds"]):
            raise ValueError("Paper/query/fold coverage mismatch")
    references = {str(REFERENCE / name): sha256(REFERENCE / name) for name in ("manifest.json", "frozen_index.json", "final_audit.json")}
    references[str(paper_path)] = sha256(paper_path)
    protocol = {"version": 1, "cases": cases, "source_sha256": sources, "reference_sha256": references,
                "reference": str(REFERENCE), "protocol": manifest,
                "candidate": "One checkpoint; no routing; identical frozen context/test batches, queries, standardization, clipping and fold metrics",
                "selection": "Synthetic-only selected checkpoint; all 1800 fold settings reported, not used for training decisions"}
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol_hash.json", {"sha256": sha256(output / "protocol.json")})
    write_csv(output / "reference_panels.csv", panels)
    for source in sources:
        destination = output / "source_snapshot" / Path(source).relative_to(REPO.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / "status.json", {"state": "prepared", "fold_settings": len(cases), "panels": len(panels), "all_references_rescored": True})
    print(json.dumps({"prepared_fold_settings": len(cases), "panels": len(panels)}), flush=True)


def load_protocol(path):
    if sha256(path / "protocol.json") != read_json(path / "protocol_hash.json")["sha256"]:
        raise ValueError("LLM protocol hash mismatch")
    protocol = read_json(path / "protocol.json")
    verify_hashes(protocol["source_sha256"])
    verify_hashes(protocol["reference_sha256"])
    verify_frozen(REFERENCE)
    return protocol


def write_summaries(output, records):
    if not records:
        return
    panels = panel_summaries(records)
    write_csv(output / "panels.csv", panels)
    flattened = [{key: row[key] for key in ("dataset", "fold", "method", "budget")} |
                 {f"{role}_{metric}": row[role][metric] for role in ("student", "blend") for metric in ("r2", "rmse", "n_eval")}
                 for row in records]
    write_csv(output / "per_fold.csv", flattened)
    dataset_rows = []
    for dataset in DATASETS:
        group = [row for row in panels if row["dataset"] == dataset]
        complete = len(group) == 45 and all(row["complete"] for row in group)
        row = {"dataset": dataset, "complete": complete, "panels": len(group), "expected_panels": 45}
        for role in ("student", "blend"):
            for metric in ("r2", "rmse"):
                row[f"{role}_{metric}"] = float(np.mean([panel[f"{role}_{metric}"] for panel in group])) if complete else None
        dataset_rows.append(row)
    write_csv(output / "per_dataset.csv", dataset_rows)


def audit(output, protocol):
    manifest = read_json(output / "manifest.json")
    verify_hashes(manifest["candidate_sha256"])
    bound = Path(manifest["protocol"]) / "protocol.json"
    if sha256(bound) != manifest["protocol_sha256"] or read_json(bound) != protocol:
        raise ValueError("Evaluation protocol mismatch")
    parity = read_json(output / "parent_parity.json")
    if not parity["passed"] or len(parity["cases"]) != 12 or parity["parent_sha256"] != sha256(GENERAL):
        raise ValueError("Parent inference parity is required")
    cases = {case["case_id"]: case for case in protocol["cases"]}
    records = []
    for specification, frozen, inputs in frozen_settings():
        verify_hashes(cases[specification["case_id"]]["reference_sha256"])
        baseline, _ = check_reference(specification, frozen, inputs)
        path = output / "predictions" / (specification["case_id"] + ".npz")
        record = read_json(path.with_suffix(".json"))
        if record["manifest_sha256"] != sha256(output / "manifest.json") or sha256(path) != record["prediction_sha256"]:
            raise ValueError("Saved candidate result changed")
        if any(record[key] != value for key, value in specification.items()):
            raise ValueError("Candidate setting metadata mismatch")
        _, _, mask, truth = inputs
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["target_z"], truth)
            np.testing.assert_array_equal(saved["query_rows"], np.where(mask)[0])
            np.testing.assert_array_equal(saved["query_columns"], np.where(mask)[1])
            metrics = checked_metrics(truth, saved["student_z"])
        if record["student"] != metrics or record["blend"] != baseline:
            raise ValueError("Candidate scores or baseline changed")
        records.append(record)
    if len(records) != len(cases) or len(records) != 1800:
        raise ValueError("Incomplete LLM evaluation")
    write_summaries(output, records)
    write_json(output / "audit.json", {"complete": True, "fold_settings": 1800, "panels": 180,
                                       "single_checkpoint": True, "training_performed": False,
                                       "manifest_sha256": sha256(output / "manifest.json"),
                                       "parity_sha256": sha256(output / "parent_parity.json")})


def evaluate(output, protocol_path, checkpoint, confirmation):
    import torch

    protocol = load_protocol(protocol_path)
    gate = read_json(confirmation / "gate.json")
    if not gate["passed"] or gate["checkpoint_sha256"] != sha256(checkpoint) or gate["selected_step"] == 0:
        raise ValueError("Require the synthetic-confirmed frozen nonzero checkpoint")
    manifest = {"candidate_sha256": {str(checkpoint): sha256(checkpoint), str(checkpoint.parent / "config.json"): sha256(checkpoint.parent / "config.json")},
                "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path / "protocol.json"),
                "confirmation": str(confirmation), "confirmation_sha256": sha256(confirmation / "gate.json"),
                "selected_step": gate["selected_step"], "single_checkpoint": True}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Refusing changed resume manifest")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    wait_for_gpu(output, False, False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.cuda.set_per_process_memory_fraction(.9)
    if not (output / "parent_parity.json").exists():
        parent = make_model(GENERAL)
        parity = []
        for specification, frozen, inputs in frozen_settings():
            if specification["fold"] != 0 or (specification["method"], specification["budget"]) not in (("entropy", 1), ("mi", 8), ("random", 15)):
                continue
            context_input, test_input, mask, _ = inputs
            prediction = predict_setting(parent, context_input, test_input, mask, frozen["mean"], frozen["scale"])
            with np.load(REFERENCE / "predictions" / (specification["case_id"] + ".npz"), allow_pickle=False) as previous:
                np.testing.assert_allclose(prediction, previous["general_z"], rtol=1e-6, atol=1e-6)
                parity.append({"case_id": specification["case_id"], "max_abs_difference": float(np.max(np.abs(prediction - previous["general_z"])))})
        write_json(output / "parent_parity.json", {"passed": True, "cases": parity, "parent_sha256": sha256(GENERAL)})
        del parent
        gc.collect()
        torch.cuda.empty_cache()
    model = make_model(checkpoint)
    cases = {case["case_id"]: case for case in protocol["cases"]}
    records = []
    for specification, frozen, inputs in frozen_settings():
        check_reserve(output)
        verify_hashes(cases[specification["case_id"]]["reference_sha256"])
        path = output / "predictions" / (specification["case_id"] + ".npz")
        if path.with_suffix(".json").exists():
            record = read_json(path.with_suffix(".json"))
            if record["manifest_sha256"] != sha256(manifest_path) or sha256(path) != record["prediction_sha256"]:
                raise ValueError("Cached result changed")
        else:
            baseline, _ = check_reference(specification, frozen, inputs)
            context_input, test_input, mask, truth = inputs
            started = time.monotonic()
            prediction = predict_setting(model, context_input, test_input, mask, frozen["mean"], frozen["scale"])
            metrics = checked_metrics(truth, prediction)
            path.parent.mkdir(parents=True, exist_ok=True)
            query_rows, query_columns = np.where(mask)
            atomic_npz(path, target_z=truth, student_z=prediction, query_rows=query_rows, query_columns=query_columns)
            record = {**specification, "student": metrics, "blend": baseline, "manifest_sha256": sha256(manifest_path),
                      "prediction_sha256": sha256(path), "seconds": time.monotonic() - started}
            write_json(path.with_suffix(".json"), record)
        records.append(record)
        write_json(output / "status.json", {"state": "evaluating", "completed": len(records), "expected": len(cases),
                                            "pid": os.getpid(), "updated_unix": time.time()})
        if len(records) % 45 == 0:
            write_summaries(output, records)
            print(f"Single LLM checkpoint: {len(records)}/1800", flush=True)
    audit(output, protocol)
    write_json(output / "status.json", {"state": "complete", "fold_settings": 1800, "panels": 180})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "evaluate", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--confirmation", type=Path)
    arguments = parser.parse_args()
    if arguments.stage != "prepare" and arguments.protocol is None:
        parser.error("Evaluation/audit requires --protocol")
    if arguments.stage == "evaluate" and (arguments.checkpoint is None or arguments.confirmation is None):
        parser.error("Evaluation requires --checkpoint and --confirmation")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=arguments.stage != "prepare")
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if arguments.stage == "prepare":
            prepare(output)
        elif arguments.stage == "audit":
            audit(output, load_protocol(arguments.protocol.resolve()))
        else:
            evaluate(output, arguments.protocol.resolve(), arguments.checkpoint.resolve(), arguments.confirmation.resolve())


if __name__ == "__main__":
    main()
